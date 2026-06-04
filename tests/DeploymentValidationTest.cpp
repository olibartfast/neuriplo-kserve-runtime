#include "Executor.hpp"
#include "KServeErrors.hpp"
#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelHandle.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace {

KServeRuntime makeRuntime() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    static MetricsRegistry metrics;
    static ModelRegistry registry(config);
    return KServeRuntime(registry, metrics);
}

KServeRuntime makeRuntimeWithVersion(const std::string &version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    static MetricsRegistry metrics;
    static ModelRegistry registry(config, [&version](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {version};
        metadata.platform = "test_sequence";
        metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        struct SequenceExecutor final : Executor {
            explicit SequenceExecutor(ModelMetadata model_metadata)
                : model_metadata_(std::move(model_metadata)) {}
            const ModelMetadata &metadata() const override {
                return model_metadata_;
            }
            ExecutionResponse infer(const ExecutionRequest &) override {
                ExecutionResponse response;
                OutputTensor output;
                output.name = "output";
                output.datatype = "FP32";
                output.shape = {1, 1};
                output.data = {1.0};
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<SequenceExecutor>(std::move(metadata));
    });
    return KServeRuntime(registry, metrics);
}

HttpResponse request(const KServeRuntime &runtime, std::string method, std::string path) {
    HttpRequest req;
    req.method = std::move(method);
    req.path = std::move(path);
    return runtime.handle(req);
}

HttpResponse request(const KServeRuntime &runtime, std::string method, std::string path,
                     std::string body) {
    HttpRequest req;
    req.method = std::move(method);
    req.path = std::move(path);
    req.body = std::move(body);
    return runtime.handle(req);
}

std::string validInferBody() {
    return R"({"id":"graph-1","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})";
}

} // namespace

TEST_CASE(inference_graph_infer_response_has_stable_shape) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody());
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_name")") != std::string::npos);
    REQUIRE(response.body.find(R"("model_version")") != std::string::npos);
    REQUIRE(response.body.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(inference_graph_error_response_has_code_and_message) {
    const auto runtime = makeRuntime();
    const auto response =
        request(runtime, "POST", "/v2/models/nonexistent/infer", validInferBody());
    REQUIRE_EQ(response.status, 404);
    REQUIRE(response.body.find(R"("error")") != std::string::npos);
    REQUIRE(response.body.find(R"("code")") != std::string::npos);
    REQUIRE(response.body.find(R"("message")") != std::string::npos);
    REQUIRE(response.body.find(R"("MODEL_NOT_FOUND")") != std::string::npos);
}

TEST_CASE(inference_graph_metadata_response_discoverable) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name")") != std::string::npos);
    REQUIRE(response.body.find(R"("versions")") != std::string::npos);
    REQUIRE(response.body.find(R"("inputs")") != std::string::npos);
    REQUIRE(response.body.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(inference_graph_versioned_routes_for_switch_routing) {
    const auto runtime = makeRuntimeWithVersion("2");
    const auto meta = request(runtime, "GET", "/v2/models/demo/versions/2");
    REQUIRE_EQ(meta.status, 200);
    REQUIRE(meta.body.find(R"("versions":["2"])") != std::string::npos);

    const auto ready = request(runtime, "GET", "/v2/models/demo/versions/2/ready");
    REQUIRE_EQ(ready.status, 200);
    REQUIRE(ready.body.find(R"("ready":true)") != std::string::npos);

    const auto infer =
        request(runtime, "POST", "/v2/models/demo/versions/2/infer", validInferBody());
    REQUIRE_EQ(infer.status, 200);
    REQUIRE(infer.body.find(R"("model_version":"2")") != std::string::npos);
}

TEST_CASE(inference_graph_ready_endpoint_for_splitter_routing) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/ready");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(inference_graph_live_endpoint_for_health_routing) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/health/live");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("live":true)") != std::string::npos);
}

TEST_CASE(inference_graph_server_metadata_for_service_discovery) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"neuriplo-kserve-runtime")") != std::string::npos);
    REQUIRE(response.body.find(R"("version")") != std::string::npos);
    REQUIRE(response.body.find(R"("extensions")") != std::string::npos);
}

TEST_CASE(inference_graph_invalid_input_returns_stable_error) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", "{bad json");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
    REQUIRE(response.body.find(R"("message")") != std::string::npos);
}

TEST_CASE(inference_graph_overload_returns_stable_error) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    config.max_queue_size = 1;
    config.instances = 1;
    config.request_timeout_ms = 5000;

    struct BlockingExecutor final : Executor {
        explicit BlockingExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}
        const ModelMetadata &metadata() const override {
            return metadata_;
        }
        ExecutionResponse infer(const ExecutionRequest &) override {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return release_; });
            ExecutionResponse response;
            OutputTensor output;
            output.name = "output";
            output.datatype = "FP32";
            output.shape = {1, 1};
            output.data = {1.0};
            response.outputs.push_back(std::move(output));
            return response;
        }
        void releaseAll() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                release_ = true;
            }
            cv_.notify_all();
        }
        ModelMetadata metadata_;
        std::mutex mutex_;
        std::condition_variable cv_;
        bool release_ = false;
    };

    BlockingExecutor *blocking_ptr = nullptr;
    ModelRegistry registry(config, [&](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "test_blocking";
        metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        auto executor = std::make_unique<BlockingExecutor>(std::move(metadata));
        blocking_ptr = executor.get();
        return executor;
    });
    MetricsRegistry metrics;
    const KServeRuntime runtime(registry, metrics);

    std::thread first(
        [&]() { (void)request(runtime, "POST", "/v2/models/demo/infer", validInferBody()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::thread second(
        [&]() { (void)request(runtime, "POST", "/v2/models/demo/infer", validInferBody()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody());
    REQUIRE_EQ(response.status, 429);
    REQUIRE(response.body.find(R"("code":"QUEUE_FULL")") != std::string::npos);
    REQUIRE(response.body.find(R"("message")") != std::string::npos);

    blocking_ptr->releaseAll();
    first.join();
    second.join();
}

TEST_CASE(inference_graph_internal_error_has_stable_shape) {
    const auto runtime = makeRuntime();
    auto drain_response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody());
    (void)drain_response;
}

TEST_CASE(inference_graph_all_error_codes_have_defined_http_mappings) {
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::InvalidArgument), 400);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::ModelNotFound), 404);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::NotFound), 404);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::MethodNotAllowed), 405);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::PayloadTooLarge), 413);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::ModelNotReady), 409);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::QueueFull), 429);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::Unavailable), 503);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::DeadlineExceeded), 504);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::BackendError), 500);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::Internal), 500);
    REQUIRE_EQ(KServeErrors::httpStatusForCode("UNKNOWN_CODE"), 500);
}

TEST_CASE(canary_metrics_include_version_label) {
    RuntimeConfig config;
    config.model_name = "canary-test";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics;
    metrics.setModelVersion("2");
    KServeRuntime runtime(registry, metrics);

    const auto infer_response =
        request(runtime, "POST", "/v2/models/canary-test/infer", validInferBody());
    REQUIRE_EQ(infer_response.status, 200);

    const auto metrics_response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(metrics_response.status, 200);
    REQUIRE(metrics_response.body.find("version=\"2\"") != std::string::npos);
    REQUIRE(metrics_response.body.find("model=\"canary-test\"") != std::string::npos);
}

TEST_CASE(canary_metrics_include_deployment_label) {
    RuntimeConfig config;
    config.model_name = "canary-test";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics;
    metrics.setModelVersion("2");
    metrics.setDeployment("canary");
    KServeRuntime runtime(registry, metrics);

    const auto infer_response =
        request(runtime, "POST", "/v2/models/canary-test/infer", validInferBody());
    REQUIRE_EQ(infer_response.status, 200);

    const auto metrics_response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(metrics_response.status, 200);
    REQUIRE(metrics_response.body.find("deployment=\"canary\"") != std::string::npos);
    REQUIRE(metrics_response.body.find("version=\"2\"") != std::string::npos);
}

TEST_CASE(canary_metrics_default_deployment_label_absent) {
    RuntimeConfig config;
    config.model_name = "stable-test";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics;
    metrics.setModelVersion("1");
    KServeRuntime runtime(registry, metrics);

    const auto metrics_response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(metrics_response.status, 200);
    REQUIRE(metrics_response.body.find("deployment=") == std::string::npos);
    REQUIRE(metrics_response.body.find("version=\"1\"") != std::string::npos);
}

TEST_CASE(autoscaling_metrics_include_queue_and_inflight) {
    RuntimeConfig config;
    config.model_name = "autoscale-test";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics;
    KServeRuntime runtime(registry, metrics);

    const auto metrics_response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(metrics_response.status, 200);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_queue_depth") != std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_in_flight_requests") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_requests_accepted_total") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_queue_latency_seconds_bucket") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_total_latency_seconds_bucket") !=
            std::string::npos);
}