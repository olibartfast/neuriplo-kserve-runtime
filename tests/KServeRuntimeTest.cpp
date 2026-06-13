#include "KServeRuntime.hpp"
#include "Executor.hpp"
#include "MetricsRegistry.hpp"
#include "ModelHandle.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace {

KServeRuntime makeRuntime() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    static MetricsRegistry metrics;
    static ModelRegistry registry(config);
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

std::string validInferBody(std::string id = "") {
    std::string body =
        "{\"inputs\":[{\"name\":\"input\",\"shape\":[1,3,224,224],\"datatype\":\"FP32\","
        "\"data\":[]}]";
    if (!id.empty()) {
        body += R"(,"id":")" + id + '"';
    }
    body += '}';
    return body;
}

} // namespace

TEST_CASE(kserve_runtime_reports_live_and_ready) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/live").status, 200);
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/ready").status, 200);
}

TEST_CASE(kserve_runtime_returns_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"demo")") != std::string::npos);
    REQUIRE(response.body.find(R"("versions":["1"])") != std::string::npos);
    REQUIRE(response.body.find(R"("platform":"neuriplo_stub")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_model) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/missing").status, 404);
}

TEST_CASE(kserve_runtime_handles_placeholder_infer) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody("abc"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_version":"1")") != std::string::npos);
    REQUIRE(response.body.find(R"("id":"abc")") != std::string::npos);
    REQUIRE(response.body.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_json) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", "{");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_shape) {
    const auto runtime = makeRuntime();
    const auto response =
        request(runtime, "POST", "/v2/models/demo/infer",
                R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[]}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("invalid shape") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_datatype) {
    const auto runtime = makeRuntime();
    const auto response = request(
        runtime, "POST", "/v2/models/demo/infer",
        R"({"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"INT32","data":[]}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("unsupported datatype") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_output_request) {
    const auto runtime = makeRuntime();
    const auto response = request(
        runtime, "POST", "/v2/models/demo/infer",
        R"({"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}],"outputs":[{"name":"missing"}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("unknown output") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_server_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"neuriplo-kserve-runtime")") != std::string::npos);
    REQUIRE(response.body.find(R"("version":"0.2.0")") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/ready");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_versioned_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/versions/1");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("versions":["1"])") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_versioned_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/versions/1/ready");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(kserve_runtime_handles_versioned_infer) {
    const auto runtime = makeRuntime();
    const auto response =
        request(runtime, "POST", "/v2/models/demo/versions/1/infer", validInferBody());
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_version":"1")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_version) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/demo/versions/2").status, 404);
    REQUIRE_EQ(
        request(runtime, "POST", "/v2/models/demo/versions/2/infer", validInferBody()).status, 404);
}

TEST_CASE(kserve_runtime_rejects_empty_method) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "", "/v2").status, 400);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/ready");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_infer) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/infer");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}

TEST_CASE(kserve_runtime_unversioned_infer_uses_default_executor_version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"42"};
        metadata.platform = "test_version";
        metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        struct VersionedExecutor final : Executor {
            explicit VersionedExecutor(ModelMetadata model_metadata)
                : model_metadata_(std::move(model_metadata)) {}
            const ModelMetadata &metadata() const override {
                return model_metadata_;
            }
            ExecutionResponse infer(const ExecutionRequest &request) override {
                (void)request;
                ExecutionResponse response;
                OutputTensor output;
                output.name = "output";
                output.datatype = "FP32";
                output.shape = {1, 1};
                output.bytes = tensorBytesFromDoubles(output.datatype, {7.0});
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<VersionedExecutor>(std::move(metadata));
    });
    MetricsRegistry metrics4a;
    const KServeRuntime runtime(registry, metrics4a);
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody());
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_version":"42")") != std::string::npos);
    REQUIRE(response.body.find(R"("data":[7.0])") != std::string::npos);
}

TEST_CASE(kserve_runtime_reports_not_ready_when_model_load_failed) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &, std::string &error) {
        error = "load failed";
        return nullptr;
    });
    MetricsRegistry metrics4b;
    const KServeRuntime runtime(registry, metrics4b);
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/ready").status, 503);
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/demo/ready").status, 503);
}

TEST_CASE(kserve_runtime_uses_injected_executor_without_route_changes) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "test_marker";
        metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        struct MarkerExecutor final : Executor {
            explicit MarkerExecutor(ModelMetadata model_metadata)
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
                output.bytes = tensorBytesFromDoubles(output.datatype, {42.0});
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<MarkerExecutor>(std::move(metadata));
    });
    MetricsRegistry metrics4c;
    const KServeRuntime runtime(registry, metrics4c);
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody());
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("data":[42.0])") != std::string::npos);
}

TEST_CASE(kserve_runtime_passes_input_tensors_to_executor) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "test_input_capture";
        metadata.inputs.push_back({"input", "FP32", {1, 3}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        struct InputEchoExecutor final : Executor {
            explicit InputEchoExecutor(ModelMetadata model_metadata)
                : model_metadata_(std::move(model_metadata)) {}
            const ModelMetadata &metadata() const override {
                return model_metadata_;
            }
            ExecutionResponse infer(const ExecutionRequest &request) override {
                ExecutionResponse response;
                OutputTensor output;
                output.name = "output";
                output.datatype = "FP32";
                output.shape = {1, 1};
                output.bytes = tensorBytesFromDoubles(
                    output.datatype, {tensorValuesAsDoubles(request.inputs.at(0).datatype,
                                                            request.inputs.at(0).bytes)
                                          .at(1)});
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<InputEchoExecutor>(std::move(metadata));
    });
    MetricsRegistry metrics4d;
    const KServeRuntime runtime(registry, metrics4d);
    const auto response = request(
        runtime, "POST", "/v2/models/demo/infer",
        R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[4.0,5.5,6.0]}]})");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("data":[5.5])") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_overload_when_queue_is_full) {
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
        ExecutionResponse infer(const ExecutionRequest &request) override {
            (void)request;
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return release_; });
            ExecutionResponse response;
            OutputTensor output;
            output.name = "output";
            output.datatype = "FP32";
            output.shape = {1, 1};
            output.bytes = tensorBytesFromDoubles(output.datatype, {1.0});
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
    MetricsRegistry metrics4e;
    const KServeRuntime runtime(registry, metrics4e);

    std::thread first([&]() {
        (void)request(runtime, "POST", "/v2/models/demo/infer", validInferBody("first"));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::thread second([&]() {
        (void)request(runtime, "POST", "/v2/models/demo/infer", validInferBody("second"));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    const auto response =
        request(runtime, "POST", "/v2/models/demo/infer", validInferBody("third"));
    REQUIRE_EQ(response.status, 429);
    REQUIRE(response.body.find(R"("code":"QUEUE_FULL")") != std::string::npos);

    blocking_ptr->releaseAll();
    first.join();
    second.join();
}

TEST_CASE(kserve_runtime_reports_not_ready_while_draining) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config);
    REQUIRE(registry.beginDrain("demo"));
    MetricsRegistry metrics4f;
    const KServeRuntime runtime(registry, metrics4f);
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/ready").status, 503);
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/demo/ready").status, 503);
}

TEST_CASE(kserve_runtime_exposes_metrics_endpoint) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.content_type.find("text/plain") != std::string::npos);
    REQUIRE(response.body.find("neuriplo_scheduler_queue_depth") != std::string::npos);
    REQUIRE(response.body.find("neuriplo_http_infer_requests_total") != std::string::npos);
    REQUIRE(response.body.find("neuriplo_process_resident_memory_bytes") != std::string::npos);
}

TEST_CASE(kserve_runtime_observability_details) {
    RuntimeConfig config;
    config.model_name = "observability-demo";
    config.backend = "stub";
    config.log_payloads = true;

    ModelRegistry registry(config);
    MetricsRegistry metrics;
    KServeRuntime runtime(registry, metrics);

    // Call infer once to populate metrics and trace logs
    const auto response = request(runtime, "POST", "/v2/models/observability-demo/infer",
                                  validInferBody("test-request-123"));
    REQUIRE_EQ(response.status, 200);

    // 1. Verify metrics content
    const auto metrics_response = request(runtime, "GET", "/metrics");
    REQUIRE_EQ(metrics_response.status, 200);

    // Check dynamic model labeling
    REQUIRE(metrics_response.body.find("model=\"observability-demo\"") != std::string::npos);

    // Check HTTP infer request counters
    REQUIRE(metrics_response.body.find("neuriplo_http_infer_requests_total") != std::string::npos);
    REQUIRE(metrics_response.body.find("model=\"observability-demo\"") != std::string::npos);
    REQUIRE(metrics_response.body.find("version=\"1\"") != std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_http_infer_requests_success_total") !=
            std::string::npos);

    // Check detailed requests counter by status and model
    REQUIRE(metrics_response.body.find("status=\"200\"") != std::string::npos);

    // Check histograms
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_queue_latency_seconds_bucket") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_infer_latency_seconds_bucket") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_total_latency_seconds_bucket") !=
            std::string::npos);
    REQUIRE(metrics_response.body.find("neuriplo_scheduler_batch_size_bucket") !=
            std::string::npos);
}

TEST_CASE(kserve_runtime_handles_openai_completions) {
    RuntimeConfig config;
    config.model_name = "llm-model";
    config.backend = "llamacpp";
    config.scheduler_strategy = "llm";
    config.max_tokens = 128;
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "neuriplo_llamacpp";
        metadata.inputs.push_back({"prompt", "BYTES", {1}});
        metadata.outputs.push_back({"text", "BYTES", {1}});
        struct LlmExecutor final : Executor {
            explicit LlmExecutor(ModelMetadata model_metadata)
                : model_metadata_(std::move(model_metadata)) {}
            const ModelMetadata &metadata() const override {
                return model_metadata_;
            }
            ExecutionResponse infer(const ExecutionRequest &request) override {
                (void)request;
                ExecutionResponse response;
                OutputTensor output;
                output.name = "text";
                output.datatype = "BYTES";
                output.shape = {1};
                output.string_data = {"This is a stub response."};
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<LlmExecutor>(std::move(metadata));
    });
    MetricsRegistry metrics4e;
    const KServeRuntime runtime(registry, metrics4e);

    const std::string completions_body = R"({
        "model": "llm-model",
        "prompt": "Hello world",
        "max_tokens": 64,
        "temperature": 0.8
    })";

    const auto response = request(runtime, "POST", "/v1/completions", completions_body);
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("object":"text_completion")") != std::string::npos);
    REQUIRE(response.body.find(R"("model":"llm-model")") != std::string::npos);
    REQUIRE(response.body.find(R"("text":"This is a stub response.")") != std::string::npos);
    REQUIRE(response.body.find(R"("finish_reason":"stop")") != std::string::npos);
    REQUIRE(response.body.find(R"("usage")") != std::string::npos);
    REQUIRE(response.body.find(R"("prompt_tokens")") != std::string::npos);
    REQUIRE(response.body.find(R"("completion_tokens")") != std::string::npos);
}

TEST_CASE(kserve_runtime_completions_rejects_missing_model) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics4f;
    const KServeRuntime runtime(registry, metrics4f);

    const std::string body = R"({"prompt": "Hello"})";
    const auto response = request(runtime, "POST", "/v1/completions", body);
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(kserve_runtime_completions_rejects_missing_prompt) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    ModelRegistry registry(config);
    MetricsRegistry metrics4g;
    const KServeRuntime runtime(registry, metrics4g);

    const std::string body = R"({"model": "demo"})";
    const auto response = request(runtime, "POST", "/v1/completions", body);
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}
