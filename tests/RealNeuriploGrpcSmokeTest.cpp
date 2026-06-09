#if defined(NEURIPLO_RUNTIME_WITH_GRPC) && defined(NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO)

#include "GrpcServer.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include "kserve_grpc.grpc.pb.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int findFreePort() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(fd, reinterpret_cast<struct sockaddr *>(&address), sizeof(address)) < 0) {
        ::close(fd);
        throw std::runtime_error("bind failed");
    }

    socklen_t length = sizeof(address);
    if (::getsockname(fd, reinterpret_cast<struct sockaddr *>(&address), &length) < 0) {
        ::close(fd);
        throw std::runtime_error("getsockname failed");
    }
    const auto port = ntohs(address.sin_port);
    ::close(fd);
    return port;
}

std::string yoloModelPath() {
    if (const char *env = std::getenv("NEURIPLO_E2E_YOLO_MODEL")) {
        return env;
    }
    return "../neuriplo-infer/models/e2e/yolo26s.onnx";
}

RuntimeConfig yoloRuntimeConfig(const std::string &model_path) {
    RuntimeConfig config;
    config.model_name = "yolo";
    config.model_path = model_path;
    config.backend = "onnx_runtime";
    config.instances = 1;
    return config;
}

struct LiveGrpcServer {
    explicit LiveGrpcServer(RuntimeConfig config)
        : config_(std::move(config)), registry_(config_), grpc_server_("127.0.0.1", port_, registry_,
                                                                       metrics_) {
        grpc_thread_ = std::thread([this]() { grpc_server_.run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    ~LiveGrpcServer() {
        grpc_server_.stop();
        if (grpc_thread_.joinable()) {
            grpc_thread_.join();
        }
    }

    std::shared_ptr<grpc::Channel> channel() const {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(port_),
                                   grpc::InsecureChannelCredentials());
    }

    int port() const { return port_; }
    bool ready() const { return registry_.allReady(); }

    RuntimeConfig config_;
    int port_ = findFreePort();
    MetricsRegistry metrics_;
    ModelRegistry registry_;
    grpc_v2::GrpcServer grpc_server_;
    std::thread grpc_thread_;
};

std::shared_ptr<grpc::Channel> externalChannel() {
    const char *host = std::getenv("NEURIPLO_GRPC_E2E_HOST");
    const char *port = std::getenv("NEURIPLO_GRPC_E2E_PORT");
    if (host == nullptr || port == nullptr || host[0] == '\0' || port[0] == '\0') {
        return nullptr;
    }
    return grpc::CreateChannel(std::string(host) + ":" + port, grpc::InsecureChannelCredentials());
}

void fillYoloInput(inference::ModelInferRequest &request) {
    auto *input = request.add_inputs();
    input->set_name("images");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(3);
    input->add_shape(640);
    input->add_shape(640);

    constexpr size_t kElements = 1U * 3U * 640U * 640U;
    auto *contents = input->mutable_contents();
    contents->mutable_fp32_contents()->Reserve(static_cast<int>(kElements));
    for (size_t i = 0; i < kElements; ++i) {
        contents->add_fp32_contents(0.5F);
    }

    auto *output = request.add_outputs();
    output->set_name("output0");
}

void requireYoloMetadata(const inference::ModelMetadataResponse &metadata) {
    REQUIRE_EQ(metadata.name(), "yolo");
    REQUIRE(metadata.platform().find("neuriplo_onnx_runtime") != std::string::npos);
    REQUIRE(metadata.inputs_size() >= 1);
    REQUIRE(metadata.outputs_size() >= 1);
    REQUIRE_EQ(metadata.inputs(0).name(), "images");
    REQUIRE_EQ(metadata.outputs(0).name(), "output0");
}

void requireYoloInferResponse(const inference::ModelInferResponse &response) {
    REQUIRE_EQ(response.model_name(), "yolo");
    REQUIRE(response.outputs_size() >= 1);
    REQUIRE_EQ(response.outputs(0).name(), "output0");
    REQUIRE_EQ(response.outputs(0).datatype(), "FP32");
    REQUIRE_EQ(response.outputs(0).shape_size(), 3);
    REQUIRE_EQ(response.outputs(0).shape(0), 1);
    REQUIRE_EQ(response.outputs(0).shape(1), 300);
    REQUIRE_EQ(response.outputs(0).shape(2), 6);
}

} // namespace

TEST_CASE(grpc_real_neuriplo_yolo_infer) {
    std::unique_ptr<LiveGrpcServer> owned_server;
    std::shared_ptr<grpc::Channel> channel = externalChannel();
    if (channel == nullptr) {
        const auto model_path = yoloModelPath();
        if (!std::filesystem::exists(model_path)) {
            std::cout << "[SKIP] grpc_real_neuriplo_yolo_infer: YOLO model not found at "
                      << model_path << "\n";
            return;
        }

        owned_server = std::make_unique<LiveGrpcServer>(yoloRuntimeConfig(model_path));
        REQUIRE(owned_server->ready());
        channel = owned_server->channel();
    }

    auto stub = inference::GRPCInferenceService::NewStub(channel);

    grpc::ClientContext metadata_context;
    inference::ModelMetadataRequest metadata_request;
    metadata_request.set_name("yolo");
    inference::ModelMetadataResponse metadata_response;
    const auto metadata_status = stub->ModelMetadata(&metadata_context, metadata_request,
                                                     &metadata_response);
    REQUIRE(metadata_status.ok());
    requireYoloMetadata(metadata_response);

    grpc::ClientContext infer_context;
    infer_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(120));
    inference::ModelInferRequest infer_request;
    infer_request.set_model_name("yolo");
    infer_request.set_model_version("1");
    infer_request.set_id("grpc-yolo-e2e");
    fillYoloInput(infer_request);

    inference::ModelInferResponse infer_response;
    const auto infer_status = stub->ModelInfer(&infer_context, infer_request, &infer_response);
    REQUIRE(infer_status.ok());
    REQUIRE_EQ(infer_response.id(), "grpc-yolo-e2e");
    requireYoloInferResponse(infer_response);
}

#endif
