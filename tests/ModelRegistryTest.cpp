#include "ModelRegistry.hpp"
#include "Executor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <memory>
#include <string>

namespace {

RuntimeConfig demoConfig() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return config;
}

class MarkerExecutor final : public Executor {
  public:
    explicit MarkerExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        (void)request;
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {1, 1};
        output.data = {42.0};
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

} // namespace

TEST_CASE(model_registry_loads_stub_executor) {
    const ModelRegistry registry(demoConfig());
    REQUIRE(registry.allReady());
    REQUIRE(registry.ready("demo"));
    const auto metadata = registry.find("demo");
    REQUIRE(metadata.has_value());
    REQUIRE_EQ(metadata->platform, "neuriplo_stub");

    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE_EQ(handle->versions, metadata->versions);
    REQUIRE_EQ(registry.defaultVersion("demo"), "1");
}

TEST_CASE(model_registry_reports_not_ready_on_failed_load) {
    const RuntimeConfig config = demoConfig();
    const ModelRegistry registry(config, [](const RuntimeConfig &, std::string &error) {
        error = "injected load failure";
        return nullptr;
    });
    REQUIRE(!registry.allReady());
    REQUIRE(!registry.ready("demo"));
}

TEST_CASE(model_registry_resolves_version_from_executor_metadata) {
    const RuntimeConfig config = demoConfig();
    const ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
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
            ExecutionResponse infer(const ExecutionRequest &) override {
                return {};
            }
            ModelMetadata model_metadata_;
        };
        return std::make_unique<VersionedExecutor>(std::move(metadata));
    });

    REQUIRE(registry.findVersion("demo", "42").has_value());
    REQUIRE(!registry.findVersion("demo", "1").has_value());
    REQUIRE_EQ(registry.defaultVersion("demo"), "42");
    REQUIRE(registry.findHandleVersion("demo", "42") != nullptr);
}

TEST_CASE(model_registry_uses_injected_executor) {
    const RuntimeConfig config = demoConfig();
    const ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "test_marker";
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        return std::make_unique<MarkerExecutor>(std::move(metadata));
    });

    const auto *handle = registry.findHandleVersion("demo", "1");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->scheduler != nullptr);

    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto result = handle->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(result.response.outputs[0].data[0], 42.0);
}

TEST_CASE(model_registry_rejects_unknown_backend) {
    RuntimeConfig config = demoConfig();
    config.backend = "does_not_exist";

    const ModelRegistry registry(config);
    REQUIRE(!registry.allReady());
    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->load_error.has_value());
    REQUIRE(handle->load_error->find("unsupported backend") != std::string::npos);
}

TEST_CASE(model_registry_uses_llm_scheduler_for_llm_strategy) {
    RuntimeConfig config = demoConfig();
    config.scheduler_strategy = "llm";
    const ModelRegistry registry(config);
    REQUIRE(registry.allReady());

    ExecutionRequest request;
    InputTensor prompt;
    prompt.name = "prompt";
    prompt.datatype = "BYTES";
    prompt.shape = {1};
    prompt.string_data = {"hello"};
    request.inputs.push_back(std::move(prompt));
    request.requested_outputs = {"text"};
    LlmGenerationParams params;
    params.max_tokens = 16;
    request.llm_params = params;

    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    const auto result = handle->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(result.response.outputs[0].string_data[0].substr(0, 6), "stub: ");
}

#ifndef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
TEST_CASE(
    model_registry_maps_supported_real_backend_to_neuriplo_executor_when_real_support_disabled) {
    RuntimeConfig config = demoConfig();
    config.backend = "onnx_runtime";
    config.model_path = "/tmp/model.onnx";

    const ModelRegistry registry(config);
    REQUIRE(!registry.allReady());
    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->load_error.has_value());
    REQUIRE(handle->load_error->find("real neuriplo support is not enabled") != std::string::npos);
}
#endif
