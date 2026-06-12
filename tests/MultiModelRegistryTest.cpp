#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <memory>
#include <string>

namespace {

RuntimeConfig demoConfig(const std::string &name) {
    RuntimeConfig config;
    config.model_name = name;
    config.backend = "stub";
    return config;
}

ModelRegistry::ExecutorFactory markerFactory(double marker_value,
                                             const std::string &version = "1") {
    return [marker_value, version](const RuntimeConfig &cfg,
                                   std::string &error) -> std::unique_ptr<Executor> {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {version};
        metadata.platform = "test_marker";
        metadata.inputs.push_back({"input", "FP32", {1, 1}});
        metadata.outputs.push_back({"output", "FP32", {1, 1}});
        struct MarkerExecutor final : Executor {
            explicit MarkerExecutor(ModelMetadata model_metadata, double marker)
                : model_metadata_(std::move(model_metadata)), marker_(marker) {}
            const ModelMetadata &metadata() const override {
                return model_metadata_;
            }
            ExecutionResponse infer(const ExecutionRequest &) override {
                ExecutionResponse response;
                OutputTensor output;
                output.name = "output";
                output.datatype = "FP32";
                output.shape = {1, 1};
                output.bytes = tensorBytesFromDoubles(output.datatype, {marker_});
                response.outputs.push_back(std::move(output));
                return response;
            }
            ModelMetadata model_metadata_;
            double marker_;
        };
        return std::make_unique<MarkerExecutor>(std::move(metadata), marker_value);
    };
}

} // namespace

TEST_CASE(multi_model_registry_loads_and_lists_models) {
    RuntimeConfig primary = demoConfig("alpha");
    ModelRegistry registry(primary);

    RuntimeConfig secondary = demoConfig("beta");
    REQUIRE(registry.loadModel(secondary));

    const auto models = registry.listModels();
    REQUIRE_EQ(models.size(), 2);
    REQUIRE(registry.ready("alpha"));
    REQUIRE(registry.ready("beta"));
}

TEST_CASE(multi_model_registry_unloads_without_affecting_other_models) {
    RuntimeConfig primary = demoConfig("alpha");
    ModelRegistry registry(primary);

    RuntimeConfig secondary = demoConfig("beta");
    REQUIRE(registry.loadModel(secondary));
    REQUIRE(registry.unloadModel("beta"));

    REQUIRE(registry.ready("alpha"));
    REQUIRE(!registry.ready("beta"));
    REQUIRE_EQ(registry.listModels().size(), 1);
}

TEST_CASE(multi_model_registry_reload_retires_old_scheduler_in_background) {
    RuntimeConfig config = demoConfig("alpha");
    ModelRegistry registry(config, markerFactory(1.0));

    REQUIRE(registry.reload("alpha", config, markerFactory(2.0)));

    auto snapshot = registry.findHandle("alpha");
    REQUIRE(snapshot != nullptr);
    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto result = snapshot->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(tensorScalarAt<float>(result.response.outputs[0].bytes, 0), 2.0f);
}

TEST_CASE(multi_model_registry_switch_version_keeps_old_version_snapshot) {
    RuntimeConfig config = demoConfig("alpha");
    ModelRegistry registry(config, markerFactory(1.0, "1"));

    RuntimeConfig version_two = demoConfig("alpha");
    version_two.model_version = "2";
    REQUIRE(registry.switchVersion("alpha", "2", version_two, markerFactory(2.0, "2")));

    auto active = registry.findHandle("alpha");
    REQUIRE(active != nullptr);
    REQUIRE_EQ(registry.defaultVersion("alpha"), "2");

    auto old_version = registry.findHandleVersion("alpha", "1");
    REQUIRE(old_version != nullptr);
    REQUIRE(old_version->scheduler.get() != active->scheduler.get());

    ExecutionRequest new_request;
    new_request.requested_outputs = {"output"};
    const auto new_result = active->scheduler->submit(std::move(new_request));
    REQUIRE(new_result.ok);
    REQUIRE_EQ(tensorScalarAt<float>(new_result.response.outputs[0].bytes, 0), 2.0f);
}
