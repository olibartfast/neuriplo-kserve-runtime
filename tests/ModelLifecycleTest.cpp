#include "ModelLifecycle.hpp"
#include "Executor.hpp"
#include "ModelHandle.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

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
    REQUIRE_EQ(tensorScalarAt<float>(first.response.outputs[0].bytes, 0), 1.0f);

    REQUIRE(lifecycle.reload(handle, demoConfig(), markerFactory(2.0)));
    REQUIRE(handle.isReady());

    ExecutionRequest second_request;
    second_request.requested_outputs = {"output"};
    auto second = handle.scheduler->submit(std::move(second_request));
    REQUIRE(second.ok);
    REQUIRE_EQ(tensorScalarAt<float>(second.response.outputs[0].bytes, 0), 2.0f);
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

    const auto handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto result = handle->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(tensorScalarAt<float>(result.response.outputs[0].bytes, 0), 20.0f);
}

TEST_CASE(infer_snapshot_keeps_old_scheduler_alive_during_reload) {
    struct SlowState {
        std::mutex mutex;
        std::condition_variable cv;
        bool infer_started = false;
        bool release_infer = false;
    };
    auto slow_state = std::make_shared<SlowState>();

    RuntimeConfig config = demoConfig();
    config.instances = 1;
    ModelRegistry registry(
        config,
        [slow_state](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
            (void)error;
            ModelMetadata metadata;
            metadata.name = cfg.model_name;
            metadata.versions = {"1"};
            metadata.platform = "test_slow";
            metadata.inputs.push_back({"input", "FP32", {1, 1}});
            metadata.outputs.push_back({"output", "FP32", {1, 1}});
            struct SlowExecutor final : Executor {
                explicit SlowExecutor(ModelMetadata model_metadata,
                                      std::shared_ptr<SlowState> state)
                    : model_metadata_(std::move(model_metadata)), state_(std::move(state)) {}
                const ModelMetadata &metadata() const override {
                    return model_metadata_;
                }
                ExecutionResponse infer(const ExecutionRequest &) override {
                    {
                        std::lock_guard<std::mutex> lock(state_->mutex);
                        state_->infer_started = true;
                    }
                    state_->cv.notify_all();

                    std::unique_lock<std::mutex> lock(state_->mutex);
                    state_->cv.wait(lock, [&]() { return state_->release_infer; });

                    ExecutionResponse response;
                    OutputTensor output;
                    output.name = "output";
                    output.datatype = "FP32";
                    output.shape = {1, 1};
                    output.bytes = tensorBytesFromDoubles(output.datatype, {1.0});
                    response.outputs.push_back(std::move(output));
                    return response;
                }
                ModelMetadata model_metadata_;
                std::shared_ptr<SlowState> state_;
            };
            return std::make_unique<SlowExecutor>(std::move(metadata), slow_state);
        });

    auto snapshot = registry.findHandle("demo");
    REQUIRE(snapshot != nullptr);

    std::atomic<bool> infer_done{false};
    double infer_marker = 0.0;
    std::thread infer_thread([&]() {
        ExecutionRequest request;
        request.requested_outputs = {"output"};
        const auto result = snapshot->scheduler->submit(std::move(request));
        REQUIRE(result.ok);
        infer_marker =
            static_cast<double>(tensorScalarAt<float>(result.response.outputs[0].bytes, 0));
        infer_done.store(true, std::memory_order_release);
    });

    {
        std::unique_lock<std::mutex> lock(slow_state->mutex);
        REQUIRE(slow_state->cv.wait_for(lock, std::chrono::seconds(2),
                                        [&]() { return slow_state->infer_started; }));
    }

    REQUIRE(registry.reload(config, markerFactory(2.0)));

    auto new_snapshot = registry.findHandle("demo");
    REQUIRE(new_snapshot != nullptr);
    REQUIRE(new_snapshot->scheduler.get() != snapshot->scheduler.get());

    ExecutionRequest new_request;
    new_request.requested_outputs = {"output"};
    const auto new_result = new_snapshot->scheduler->submit(std::move(new_request));
    REQUIRE(new_result.ok);
    REQUIRE_EQ(tensorScalarAt<float>(new_result.response.outputs[0].bytes, 0), 2.0f);

    {
        std::lock_guard<std::mutex> lock(slow_state->mutex);
        slow_state->release_infer = true;
    }
    slow_state->cv.notify_all();
    infer_thread.join();

    REQUIRE(infer_done.load(std::memory_order_acquire));
    REQUIRE_EQ(infer_marker, 1.0);
}

TEST_CASE(model_registry_complete_unload_delegates_to_lifecycle) {
    RuntimeConfig config = demoConfig();
    ModelRegistry registry(config);
    REQUIRE(registry.beginDrain("demo"));

    REQUIRE(registry.completeUnload("demo"));
    REQUIRE(!registry.allReady());
    REQUIRE(registry.findHandle("demo") == nullptr);
    REQUIRE(registry.listModels().empty());
}
