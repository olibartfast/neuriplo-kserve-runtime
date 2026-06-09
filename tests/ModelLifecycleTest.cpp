#include "ModelLifecycle.hpp"
#include "Executor.hpp"
#include "ModelHandle.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "StubExecutor.hpp"
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

ModelLifecycle::ExecutorFactory stubFactory() {
    return [](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
        (void)cfg;
        return makeStubExecutor(cfg, error);
    };
}

ModelLifecycle::ExecutorFactory failingFactory() {
    return [](const RuntimeConfig &, std::string &error) -> std::unique_ptr<Executor> {
        error = "injected load failure";
        return nullptr;
    };
}

ModelLifecycle::ExecutorFactory markerFactory(double marker_value) {
    return
        [marker_value](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
            (void)error;
            ModelMetadata metadata;
            metadata.name = cfg.model_name;
            metadata.versions = {"1"};
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
                    output.data = {marker_};
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

TEST_CASE(model_lifecycle_loads_ready_handle) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    const auto config = demoConfig();

    lifecycle.load(handle, config, stubFactory());

    REQUIRE_EQ(handle.state.current(), ModelState::Ready);
    REQUIRE(handle.isReady());
    REQUIRE_EQ(handle.name, "demo");
    REQUIRE_EQ(handle.metadata.platform, "neuriplo_stub");
    REQUIRE(handle.scheduler != nullptr);
}

TEST_CASE(model_lifecycle_marks_failed_on_load_error) {
    ModelHandle handle;
    ModelLifecycle lifecycle;

    lifecycle.load(handle, demoConfig(), failingFactory());

    REQUIRE_EQ(handle.state.current(), ModelState::Failed);
    REQUIRE(!handle.isReady());
    REQUIRE(handle.load_error.has_value());
    REQUIRE(handle.scheduler == nullptr);
}

TEST_CASE(model_lifecycle_rejects_load_from_non_unloaded_state) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    lifecycle.load(handle, demoConfig(), stubFactory());

    lifecycle.load(handle, demoConfig(), markerFactory(99.0));

    REQUIRE_EQ(handle.state.current(), ModelState::Ready);
    REQUIRE_EQ(handle.metadata.platform, "neuriplo_stub");
}

TEST_CASE(model_lifecycle_begin_drain_transitions_to_unloading) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    lifecycle.load(handle, demoConfig(), stubFactory());

    REQUIRE(lifecycle.beginDrain(handle));

    REQUIRE_EQ(handle.state.current(), ModelState::Unloading);
    REQUIRE(handle.scheduler->isDraining());
    REQUIRE(!handle.isReady());
}

TEST_CASE(model_lifecycle_complete_unload_returns_to_unloaded) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    lifecycle.load(handle, demoConfig(), stubFactory());
    lifecycle.beginDrain(handle);

    REQUIRE(lifecycle.completeUnload(handle));

    REQUIRE_EQ(handle.state.current(), ModelState::Unloaded);
    REQUIRE(handle.scheduler == nullptr);
    REQUIRE(handle.metadata.name.empty());
}

TEST_CASE(model_lifecycle_reload_replaces_executor_after_drain) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    lifecycle.load(handle, demoConfig(), markerFactory(1.0));

    ExecutionRequest request;
    request.requested_outputs = {"output"};
    auto first = handle.scheduler->submit(std::move(request));
    REQUIRE(first.ok);
    REQUIRE_EQ(first.response.outputs[0].data[0], 1.0);

    REQUIRE(lifecycle.reload(handle, demoConfig(), markerFactory(2.0)));
    REQUIRE(handle.isReady());

    ExecutionRequest second_request;
    second_request.requested_outputs = {"output"};
    auto second = handle.scheduler->submit(std::move(second_request));
    REQUIRE(second.ok);
    REQUIRE_EQ(second.response.outputs[0].data[0], 2.0);
}

TEST_CASE(model_lifecycle_reload_from_failed_state) {
    ModelHandle handle;
    ModelLifecycle lifecycle;
    lifecycle.load(handle, demoConfig(), failingFactory());
    REQUIRE_EQ(handle.state.current(), ModelState::Failed);

    REQUIRE(lifecycle.reload(handle, demoConfig(), stubFactory()));
    REQUIRE(handle.isReady());
}

TEST_CASE(model_registry_reload_delegates_to_lifecycle) {
    RuntimeConfig config = demoConfig();
    ModelRegistry registry(config, markerFactory(10.0));

    REQUIRE(registry.reload(config, markerFactory(20.0)));

    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto result = handle->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(result.response.outputs[0].data[0], 20.0);
}

TEST_CASE(model_registry_complete_unload_delegates_to_lifecycle) {
    RuntimeConfig config = demoConfig();
    ModelRegistry registry(config);
    REQUIRE(registry.beginDrain("demo"));

    REQUIRE(registry.completeUnload("demo"));
    REQUIRE(!registry.allReady());

    const auto *handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE_EQ(handle->state.current(), ModelState::Unloaded);
}
