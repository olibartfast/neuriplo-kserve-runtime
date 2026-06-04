#ifdef NEURIPLO_RUNTIME_WITH_GRPC

#include "GrpcServer.hpp"

#include "GrpcV2Codec.hpp"
#include "KServeErrors.hpp"
#include "Logging.hpp"
#include "Scheduler.hpp"
#include "kserve_grpc.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace grpc_v2 {

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

class GrpcServiceImpl final : public inference::GRPCInferenceService::Service {
  public:
    explicit GrpcServiceImpl(ModelRegistry &registry, MetricsRegistry &metrics)
        : registry_(registry), metrics_(metrics) {}

    Status ServerLive(ServerContext * /*context*/, const inference::ServerLiveRequest * /*request*/,
                      inference::ServerLiveResponse *reply) override {
        reply->set_live(true);
        return Status::OK;
    }

    Status ServerReady(ServerContext * /*context*/,
                       const inference::ServerReadyRequest * /*request*/,
                       inference::ServerReadyResponse *reply) override {
        reply->set_ready(registry_.allReady());
        return Status::OK;
    }

    Status ModelReady(ServerContext * /*context*/, const inference::ModelReadyRequest *request,
                      inference::ModelReadyResponse *reply) override {
        const auto &version = request->version();
        bool ready = version.empty() ? registry_.ready(request->name())
                                     : registry_.readyVersion(request->name(), version);
        reply->set_ready(ready);
        return Status::OK;
    }

    Status ServerMetadata(ServerContext * /*context*/,
                          const inference::ServerMetadataRequest * /*request*/,
                          inference::ServerMetadataResponse *reply) override {
        reply->set_name("neuriplo-kserve-runtime");
        reply->set_version("0.1.0");
        return Status::OK;
    }

    Status ModelMetadata(ServerContext * /*context*/,
                         const inference::ModelMetadataRequest *request,
                         inference::ModelMetadataResponse *reply) override {
        const auto &version = request->version();
        const auto metadata = version.empty()
                                  ? registry_.find(request->name())
                                  : registry_.findVersion(request->name(), version);
        if (!metadata.has_value()) {
            return Status(StatusCode::NOT_FOUND, "model not found: " + request->name());
        }
        *reply = buildModelMetadataResponse(*metadata);
        return Status::OK;
    }

    Status ModelInfer(ServerContext *context, const inference::ModelInferRequest *request,
                      inference::ModelInferResponse *reply) override {
        const auto model_name = request->model_name();
        const auto model_version = request->model_version();
        const auto request_id =
            request->id().empty() ? std::optional<std::string>{} : std::make_optional(request->id());

        const auto *handle = model_version.empty()
                                 ? registry_.findHandle(model_name)
                                 : registry_.findHandleVersion(model_name, model_version);
        if (handle == nullptr) {
            return Status(StatusCode::NOT_FOUND, "model not found: " + model_name);
        }
        if (!handle->isReady()) {
            if (handle->scheduler != nullptr && handle->scheduler->isDraining()) {
                return Status(StatusCode::UNAVAILABLE, "model is draining");
            }
            return Status(StatusCode::UNAVAILABLE, "model is not ready");
        }
        if (handle->scheduler == nullptr) {
            return Status(StatusCode::UNAVAILABLE, "model scheduler is unavailable");
        }

        ExecutionRequest exec_request;
        try {
            exec_request = convertInferRequest(*request);
        } catch (const std::exception &error) {
            return Status(StatusCode::INVALID_ARGUMENT,
                          std::string("invalid request: ") + error.what());
        }

        {
            LogEvent span;
            span.severity = "info";
            span.message = "span: admit (grpc)";
            span.request_id = request_id.value_or("");
            span.model = model_name;
            span.model_version = model_version;
            span.route = "grpc://ModelInfer";
            defaultLogger().event(span);
        }

        {
            LogEvent span;
            span.severity = "info";
            span.message = "span: queue (grpc)";
            span.request_id = request_id.value_or("");
            span.model = model_name;
            span.model_version = model_version;
            defaultLogger().event(span);
        }

        auto scheduled = handle->scheduler->submit(std::move(exec_request));

        {
            LogEvent span;
            span.severity = "info";
            span.message = "span: batch form (grpc)";
            span.request_id = request_id.value_or("");
            span.model = model_name;
            span.model_version = model_version;
            span.batch_size = scheduled.batch_size;
            span.queue_latency_ns = scheduled.queue_latency_ns;
            defaultLogger().event(span);
        }

        {
            LogEvent span;
            span.severity = "info";
            span.message = "span: infer (grpc)";
            span.request_id = request_id.value_or("");
            span.model = model_name;
            span.model_version = model_version;
            span.batch_size = scheduled.batch_size;
            span.infer_latency_ns = scheduled.execution_latency_ns;
            defaultLogger().event(span);
        }

        if (!scheduled.ok) {
            const auto &code = scheduled.error_code.empty()
                                   ? std::string(KServeErrors::Internal)
                                   : scheduled.error_code;
            metrics_.recordInferRequest(model_name, "GRPC", gRpcStatusForCode(code), 0, 0, 0);

            {
                LogEvent span;
                span.severity = "warn";
                span.message = "span: respond (grpc, failed) - " + code;
                span.request_id = request_id.value_or("");
                span.model = model_name;
                span.model_version = model_version;
                span.route = "grpc://ModelInfer";
                span.status = gRpcStatusForCode(code);
                span.error_code = code;
                defaultLogger().event(span);
            }

            return gRpcErrorFromCode(code, scheduled.error_message);
        }

        if (!scheduled.response.ok) {
            const auto &code = scheduled.response.error_code.empty()
                                   ? std::string(KServeErrors::BackendError)
                                   : scheduled.response.error_code;
            metrics_.recordInferRequest(model_name, "GRPC", gRpcStatusForCode(code),
                                        scheduled.queue_latency_ns, scheduled.execution_latency_ns,
                                        scheduled.total_latency_ns, scheduled.batch_size);

            {
                LogEvent span;
                span.severity = "error";
                span.message = "span: respond (grpc, failed) - " + code;
                span.request_id = request_id.value_or("");
                span.model = model_name;
                span.model_version = model_version;
                span.route = "grpc://ModelInfer";
                span.status = gRpcStatusForCode(code);
                span.error_code = code;
                span.queue_latency_ns = scheduled.queue_latency_ns;
                span.infer_latency_ns = scheduled.execution_latency_ns;
                defaultLogger().event(span);
            }

            return gRpcErrorFromCode(code, scheduled.response.error_message);
        }

        metrics_.recordInferRequest(model_name, "GRPC", static_cast<int>(StatusCode::OK),
                                    scheduled.queue_latency_ns, scheduled.execution_latency_ns,
                                    scheduled.total_latency_ns, scheduled.batch_size);

        *reply = buildInferResponse(scheduled.response, model_name, model_version, request_id);

        {
            LogEvent span;
            span.severity = "info";
            span.message = "span: respond (grpc)";
            span.request_id = request_id.value_or("");
            span.model = model_name;
            span.model_version = model_version;
            span.route = "grpc://ModelInfer";
            span.status = static_cast<int>(StatusCode::OK);
            span.queue_latency_ns = scheduled.queue_latency_ns;
            span.infer_latency_ns = scheduled.execution_latency_ns;
            span.total_latency_ns = scheduled.total_latency_ns;
            span.batch_size = scheduled.batch_size;
            defaultLogger().event(span);
        }

        return Status::OK;
    }

  private:
    static int gRpcStatusForCode(const std::string &code) {
        if (code == KServeErrors::InvalidArgument)
            return static_cast<int>(StatusCode::INVALID_ARGUMENT);
        if (code == KServeErrors::ModelNotFound)
            return static_cast<int>(StatusCode::NOT_FOUND);
        if (code == KServeErrors::ModelNotReady)
            return static_cast<int>(StatusCode::UNAVAILABLE);
        if (code == KServeErrors::QueueFull)
            return static_cast<int>(StatusCode::RESOURCE_EXHAUSTED);
        if (code == KServeErrors::DeadlineExceeded)
            return static_cast<int>(StatusCode::DEADLINE_EXCEEDED);
        if (code == KServeErrors::BackendError || code == KServeErrors::Internal)
            return static_cast<int>(StatusCode::INTERNAL);
        if (code == KServeErrors::Unavailable)
            return static_cast<int>(StatusCode::UNAVAILABLE);
        if (code == KServeErrors::NotFound)
            return static_cast<int>(StatusCode::NOT_FOUND);
        return static_cast<int>(StatusCode::INTERNAL);
    }

    static Status gRpcErrorFromCode(const std::string &code, const std::string &message) {
        if (code == KServeErrors::InvalidArgument)
            return Status(StatusCode::INVALID_ARGUMENT, message);
        if (code == KServeErrors::ModelNotFound)
            return Status(StatusCode::NOT_FOUND, message);
        if (code == KServeErrors::ModelNotReady)
            return Status(StatusCode::UNAVAILABLE, message);
        if (code == KServeErrors::QueueFull)
            return Status(StatusCode::RESOURCE_EXHAUSTED, message);
        if (code == KServeErrors::DeadlineExceeded)
            return Status(StatusCode::DEADLINE_EXCEEDED, message);
        if (code == KServeErrors::BackendError || code == KServeErrors::Internal)
            return Status(StatusCode::INTERNAL, message);
        if (code == KServeErrors::Unavailable)
            return Status(StatusCode::UNAVAILABLE, message);
        if (code == KServeErrors::NotFound)
            return Status(StatusCode::NOT_FOUND, message);
        return Status(StatusCode::INTERNAL, message);
    }

    ModelRegistry &registry_;
    MetricsRegistry &metrics_;
};

struct GrpcServerImpl {
    std::unique_ptr<Server> server;
};

GrpcServer::GrpcServer(std::string host, int port, ModelRegistry &registry,
                       MetricsRegistry &metrics)
    : host_(std::move(host)), port_(port), registry_(registry), metrics_(metrics),
      service_(std::make_unique<GrpcServiceImpl>(registry_, metrics_)),
      impl_(std::make_unique<GrpcServerImpl>()) {}

GrpcServer::~GrpcServer() {
    stop();
}

void GrpcServer::run() {
    const auto listen_address = host_ + ":" + std::to_string(port_);

    ServerBuilder builder;
    builder.AddListeningPort(listen_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    builder.SetMaxReceiveMessageSize(67108864);
    builder.SetMaxSendMessageSize(67108864);

    impl_->server = builder.BuildAndStart();
    if (impl_->server == nullptr) {
        throw std::runtime_error("failed to start gRPC server on " + listen_address);
    }

    {
        LogEvent startup;
        startup.severity = "info";
        startup.message = "gRPC server listening on " + listen_address;
        defaultLogger().event(startup);
    }

    impl_->server->Wait();
}

void GrpcServer::stop() {
    if (impl_->server != nullptr) {
        impl_->server->Shutdown();
    }
}

} // namespace grpc_v2

#endif // NEURIPLO_RUNTIME_WITH_GRPC
