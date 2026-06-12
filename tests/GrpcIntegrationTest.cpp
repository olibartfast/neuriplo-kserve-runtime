#ifdef NEURIPLO_RUNTIME_WITH_GRPC

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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

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

struct TestGrpcContext {
    TestGrpcContext()
        : config(), registry(config), grpc_server("127.0.0.1", port, registry, metrics) {
        grpc_thread = std::thread([this]() { grpc_server.run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ~TestGrpcContext() {
        grpc_server.stop();
        if (grpc_thread.joinable()) {
            grpc_thread.join();
        }
    }

    std::shared_ptr<grpc::Channel> channel() {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                   grpc::InsecureChannelCredentials());
    }

    RuntimeConfig config;
    int port = findFreePort();
    MetricsRegistry metrics;
    ModelRegistry registry;
    grpc_v2::GrpcServer grpc_server;
    std::thread grpc_thread;
};

} // namespace

TEST_CASE(grpc_integration_server_live) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ServerLiveRequest request;
    inference::ServerLiveResponse response;

    const auto status = stub->ServerLive(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE(response.live());
}

TEST_CASE(grpc_integration_server_ready) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ServerReadyRequest request;
    inference::ServerReadyResponse response;

    const auto status = stub->ServerReady(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE(response.ready());
}

TEST_CASE(grpc_integration_server_metadata) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ServerMetadataRequest request;
    inference::ServerMetadataResponse response;

    const auto status = stub->ServerMetadata(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE_EQ(response.name(), "neuriplo-kserve-runtime");
    REQUIRE_EQ(response.version(), "0.2.0");
}

TEST_CASE(grpc_integration_model_ready) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelReadyRequest request;
    request.set_name("demo");
    inference::ModelReadyResponse response;

    const auto status = stub->ModelReady(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE(response.ready());
}

TEST_CASE(grpc_integration_model_ready_unknown_model) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelReadyRequest request;
    request.set_name("nonexistent");
    inference::ModelReadyResponse response;

    const auto status = stub->ModelReady(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE(!response.ready());
}

TEST_CASE(grpc_integration_model_metadata) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelMetadataRequest request;
    request.set_name("demo");
    inference::ModelMetadataResponse response;

    const auto status = stub->ModelMetadata(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE_EQ(response.name(), "demo");
    REQUIRE(response.versions_size() >= 1);
    REQUIRE(response.inputs_size() >= 1);
    REQUIRE(response.outputs_size() >= 1);
}

TEST_CASE(grpc_integration_model_metadata_unknown) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelMetadataRequest request;
    request.set_name("nonexistent");
    inference::ModelMetadataResponse response;

    const auto status = stub->ModelMetadata(&context, request, &response);
    REQUIRE(!status.ok());
    REQUIRE_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_CASE(grpc_integration_model_infer) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelInferRequest request;
    request.set_model_name("demo");
    request.set_model_version("1");
    request.set_id("grpc-integration-1");

    auto *input = request.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(3);
    input->add_shape(224);
    input->add_shape(224);

    auto *output = request.add_outputs();
    output->set_name("output");

    inference::ModelInferResponse response;

    const auto status = stub->ModelInfer(&context, request, &response);
    REQUIRE(status.ok());
    REQUIRE_EQ(response.model_name(), "demo");
    REQUIRE_EQ(response.model_version(), "1");
    REQUIRE_EQ(response.id(), "grpc-integration-1");
    REQUIRE(response.outputs_size() >= 1);
    REQUIRE_EQ(response.outputs(0).name(), "output");
    REQUIRE_EQ(response.outputs(0).datatype(), "FP32");
}

TEST_CASE(grpc_integration_model_infer_unknown_model) {
    TestGrpcContext ctx;
    auto stub = inference::GRPCInferenceService::NewStub(ctx.channel());

    grpc::ClientContext context;
    inference::ModelInferRequest request;
    request.set_model_name("nonexistent");

    inference::ModelInferResponse response;

    const auto status = stub->ModelInfer(&context, request, &response);
    REQUIRE(!status.ok());
    REQUIRE_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

#endif // NEURIPLO_RUNTIME_WITH_GRPC
