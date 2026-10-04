#include "ModelRegistry.hpp"
#include "Executor.hpp"
#include "RuntimeConfig.hpp"
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
        output.bytes = tensorBytesFromDoubles(output.datatype, {42.0});
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

    const auto handle = registry.findHandle("demo");
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

    const auto handle = registry.findHandleVersion("demo", "1");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->scheduler != nullptr);

    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto result = handle->scheduler->submit(std::move(request));
    REQUIRE(result.ok);
    REQUIRE_EQ(tensorScalarAt<float>(result.response.outputs[0].bytes, 0), 42.0f);
}

TEST_CASE(model_registry_rejects_unknown_backend) {
    RuntimeConfig config = demoConfig();
    config.backend = "does_not_exist";

    const ModelRegistry registry(config);
    REQUIRE(!registry.allReady());
    const auto handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->load_error.has_value());
    REQUIRE(handle->load_error->find("unsupported backend") != std::string::npos);
}

namespace {
// A rendezvous point for the gated factories below: the factory call signals
// build_started once inside, then blocks until the test releases it -- a
// stand-in for a slow executor build (e.g. an engine deserialize).
struct BuildGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool build_started = false;
    bool release = false;
};

ModelMetadata markerMetadata(const std::string &name, const std::string &version) {
    ModelMetadata metadata;
    metadata.name = name;
    metadata.versions = {version};
    metadata.platform = "test_marker";
    metadata.outputs.push_back({"output", "FP32", {1, 1}});
    return metadata;
}

ModelRegistry::ExecutorFactory gatedFactory(std::shared_ptr<BuildGate> gate,
                                            std::string version = "1") {
    return
        [gate, version](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
            (void)error;
            {
                std::lock_guard<std::mutex> lock(gate->mutex);
                gate->build_started = true;
            }
            gate->cv.notify_all();
            std::unique_lock<std::mutex> lock(gate->mutex);
            gate->cv.wait(lock, [&] { return gate->release; });
            return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, version));
        };
}

void waitForBuildStarted(const std::shared_ptr<BuildGate> &gate) {
    std::unique_lock<std::mutex> lock(gate->mutex);
    REQUIRE(gate->cv.wait_for(lock, std::chrono::seconds(2), [&] { return gate->build_started; }));
}

// Polls loadWaitersForTesting() until `expected` callers are parked on
// load_cv_, with a bounded timeout. Used before releasing a gate a first
// caller's build is blocked on, so a second caller is deterministically
// known to have reached its own wait (joining the placeholder or an
// in-flight reload) rather than racing a sleep against it -- without this,
// releasing the gate too early can let the second call see a settled slot
// instead of a Loading one and take the "already loaded" path instead of
// the wait path, flaking the test.
void waitForLoadWaiters(const ModelRegistry &registry, std::size_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (registry.loadWaitersForTesting() != expected) {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// Runs `read_once` on its own thread while a gate-guarded slow build is in
// flight elsewhere, with an independent watchdog that force-releases the
// gate after a generous bound. This makes a regression (the reader actually
// blocked on the build) fail this assertion after ~10s instead of hanging
// the test suite forever on a gate only the test itself would otherwise
// release.
template <typename ReadFn>
void requireReaderFinishesBeforeWatchdog(const std::shared_ptr<BuildGate> &gate, ReadFn read_once) {
    std::atomic<bool> watchdog_released{false};
    std::thread watchdog([gate, &watchdog_released] {
        std::unique_lock<std::mutex> lock(gate->mutex);
        if (gate->cv.wait_for(lock, std::chrono::seconds(10), [&] { return gate->release; })) {
            return; // the test's own release (below) got there first
        }
        gate->release = true;
        watchdog_released.store(true);
        lock.unlock();
        gate->cv.notify_all();
    });

    std::thread reader(read_once);
    reader.join();

    // If this is false, the reader only finished because the watchdog gave
    // up on it and force-released the gate -- i.e. it really was blocked.
    REQUIRE(!watchdog_released.load());

    {
        std::lock_guard<std::mutex> lock(gate->mutex);
        gate->release = true;
    }
    gate->cv.notify_all();
    watchdog.join();
}
} // namespace

// B-8: building an executor must not happen under the registry-wide
// exclusive lock. Proof: while a second model's factory is blocked mid-
// build, findHandle() and a real inference submit() on the *first*,
// already-loaded model must still complete promptly instead of waiting for
// the blocked build to release the lock.
TEST_CASE(model_registry_slow_load_does_not_block_other_model_inference) {
    ModelRegistry registry(demoConfig(), [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    });
    REQUIRE(registry.ready("demo"));

    auto gate = std::make_shared<BuildGate>();
    RuntimeConfig slow_config = demoConfig();
    slow_config.model_name = "slow";

    std::thread loader(
        [&registry, slow_config, gate] { registry.loadModel(slow_config, gatedFactory(gate)); });
    waitForBuildStarted(gate);

    // While the build is in flight: readiness and the index must not block
    // on it, and must report the slow model as still coming up rather than
    // either hanging or showing nothing for it (this is also the readiness
    // half of the B-8 fix: allReady() must not count a model that has never
    // been Ready as blocking every other, already-loaded model).
    REQUIRE(registry.allReady());
    const auto slow_snapshot = registry.findHandle("slow");
    REQUIRE(slow_snapshot != nullptr);
    REQUIRE_EQ(slow_snapshot->state, ModelState::Loading);
    REQUIRE(!slow_snapshot->isReady());

    requireReaderFinishesBeforeWatchdog(gate, [&] {
        const auto handle = registry.findHandle("demo");
        REQUIRE(handle != nullptr);
        REQUIRE(registry.ready("demo"));
        ExecutionRequest request;
        request.requested_outputs = {"output"};
        const auto result = handle->scheduler->submit(std::move(request));
        REQUIRE(result.ok);
    });

    loader.join();
    REQUIRE(registry.ready("slow"));
    REQUIRE(registry.allReady());
}

// Same proof as above, for a slow *reload* of an already-loaded model: it
// must not block inference/readiness on an unrelated, already-loaded model
// either.
TEST_CASE(model_registry_slow_reload_does_not_block_other_model_inference) {
    RuntimeConfig primary = demoConfig();
    ModelRegistry registry(primary, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    });
    RuntimeConfig other_config = demoConfig();
    other_config.model_name = "other";
    REQUIRE(registry.loadModel(other_config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    }));
    REQUIRE(registry.ready("other"));

    auto gate = std::make_shared<BuildGate>();
    std::thread reloader(
        [&registry, primary, gate] { registry.reload("demo", primary, gatedFactory(gate)); });
    waitForBuildStarted(gate);

    // The model being reloaded must also stay untouched and Ready while its
    // replacement builds (B-3), not just the unrelated model.
    REQUIRE(registry.ready("demo"));

    requireReaderFinishesBeforeWatchdog(gate, [&] {
        const auto handle = registry.findHandle("other");
        REQUIRE(handle != nullptr);
        REQUIRE(registry.ready("other"));
        ExecutionRequest request;
        request.requested_outputs = {"output"};
        const auto result = handle->scheduler->submit(std::move(request));
        REQUIRE(result.ok);
    });

    reloader.join();
    REQUIRE(registry.ready("demo"));
}

// Same proof again, for a slow switchVersion.
TEST_CASE(model_registry_slow_switch_version_does_not_block_other_model_inference) {
    RuntimeConfig primary = demoConfig();
    ModelRegistry registry(primary, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    });
    RuntimeConfig other_config = demoConfig();
    other_config.model_name = "other";
    REQUIRE(registry.loadModel(other_config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    }));
    REQUIRE(registry.ready("other"));

    auto gate = std::make_shared<BuildGate>();
    RuntimeConfig version_two = demoConfig();
    version_two.model_version = "2";
    std::thread switcher([&registry, version_two, gate] {
        registry.switchVersion("demo", "2", version_two, gatedFactory(gate, "2"));
    });
    waitForBuildStarted(gate);

    REQUIRE(registry.ready("demo"));

    requireReaderFinishesBeforeWatchdog(gate, [&] {
        const auto handle = registry.findHandle("other");
        REQUIRE(handle != nullptr);
        REQUIRE(registry.ready("other"));
        ExecutionRequest request;
        request.requested_outputs = {"output"};
        const auto result = handle->scheduler->submit(std::move(request));
        REQUIRE(result.ok);
    });

    switcher.join();
    REQUIRE_EQ(registry.defaultVersion("demo"), "2");
}

// B-17: two concurrent loadModel calls for the same brand-new name must not
// both build an executor, and the one that loses the race must not report
// failure if the model ends up loaded -- it is not a 409, it is a model
// someone else already brought up. The second call's own build attempt
// (a factory that must never run) is the deterministic proof that it took
// the wait path instead of racing a build of its own.
TEST_CASE(model_registry_concurrent_loads_of_same_new_model_both_succeed) {
    ModelRegistry registry(demoConfig());

    auto gate = std::make_shared<BuildGate>();
    RuntimeConfig shared_config = demoConfig();
    shared_config.model_name = "shared";

    bool first_result = false;
    std::thread first([&registry, &first_result, shared_config, gate] {
        first_result = registry.loadModel(shared_config, gatedFactory(gate));
    });
    waitForBuildStarted(gate);

    bool second_result = false;
    std::thread second([&registry, &second_result, shared_config] {
        second_result =
            registry.loadModel(shared_config, [](const RuntimeConfig &cfg, std::string &error) {
                (void)error;
                // Must never run: the first call already owns building "shared".
                return std::make_unique<MarkerExecutor>(
                    markerMetadata(cfg.model_name, "should-not-run"));
            });
    });
    // Without this, releasing the gate races `second` reaching its own
    // load_cv_.wait(): if the release (and the first build it unblocks)
    // wins, `second` instead finds an already-settled Ready slot and takes
    // the "already loaded" path (false, per restored B-4 semantics) instead
    // of the wait-and-join path (true) -- flaking second_result. Waiting
    // for the wait itself removes that race.
    waitForLoadWaiters(registry, 1);

    {
        std::lock_guard<std::mutex> lock(gate->mutex);
        gate->release = true;
    }
    gate->cv.notify_all();

    first.join();
    second.join();

    REQUIRE(first_result);
    REQUIRE(second_result);
    REQUIRE(registry.ready("shared"));
    REQUIRE_EQ(registry.defaultVersion("shared"), "1");
}

// B-4 (restored semantics) + the building block the repository handler's
// narrow-race 200 depends on: loading an already-loaded name is not
// success -- it returns false so the caller 409s with "use reload" -- but
// it must not disturb the model, which is exactly the signal
// (loadModel() == false, ready() == true) that lets a caller distinguish
// "that name is just already loaded" from a real failure.
TEST_CASE(model_registry_load_of_already_ready_model_returns_false_but_leaves_it_ready) {
    ModelRegistry registry(demoConfig());
    REQUIRE(registry.ready("demo"));

    REQUIRE(!registry.loadModel(demoConfig()));
    REQUIRE(registry.ready("demo"));
}

// Exception safety: a factory that throws must not leave the slot stuck
// Loading forever (which would hang every later caller that waits on it via
// load_cv_). ModelLifecycle::load() catches everything and reports Failed
// instead, so this call itself returns normally rather than propagating.
TEST_CASE(model_registry_throwing_factory_fails_cleanly_and_does_not_hang_later_callers) {
    ModelRegistry registry(demoConfig());
    RuntimeConfig bad_config = demoConfig();
    bad_config.model_name = "boom";

    const bool first_result =
        registry.loadModel(bad_config, [](const RuntimeConfig &, std::string &) {
            throw std::runtime_error("factory exploded");
            return std::unique_ptr<Executor>(); // unreachable
        });
    REQUIRE(first_result);
    REQUIRE(!registry.ready("boom"));
    REQUIRE(!registry.loadOrReloadInProgress("boom"));
    const auto handle = registry.findHandle("boom");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->load_error.has_value());
    REQUIRE(handle->load_error->find("factory exploded") != std::string::npos);

    // The slot is Failed, not stuck Loading, so an unload -- and a fresh
    // load after it -- resolve promptly rather than waiting forever on
    // load_cv_ for a Loading state that will never clear.
    bool unload_done = false;
    std::thread unloader([&] { unload_done = registry.unloadModel("boom"); });
    unloader.join();
    REQUIRE(unload_done);

    bool second_result = false;
    std::thread second([&] {
        second_result =
            registry.loadModel(bad_config, [](const RuntimeConfig &cfg, std::string &error) {
                (void)error;
                return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
            });
    });
    second.join();
    REQUIRE(second_result);
    REQUIRE(registry.ready("boom"));
}

// A model whose reload is mid-build (reload_pending) must refuse an unload
// rather than let it tear the slot down underneath the build that is about
// to publish into it.
TEST_CASE(model_registry_unload_during_reload_in_progress_is_refused) {
    RuntimeConfig primary = demoConfig();
    ModelRegistry registry(primary, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        return std::make_unique<MarkerExecutor>(markerMetadata(cfg.model_name, "1"));
    });

    auto gate = std::make_shared<BuildGate>();
    std::thread reloader(
        [&registry, primary, gate] { registry.reload("demo", primary, gatedFactory(gate)); });
    waitForBuildStarted(gate);

    REQUIRE(!registry.unloadModel("demo"));
    REQUIRE(registry.loadOrReloadInProgress("demo"));

    {
        std::lock_guard<std::mutex> lock(gate->mutex);
        gate->release = true;
    }
    gate->cv.notify_all();
    reloader.join();
    REQUIRE(registry.ready("demo"));
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

    const auto handle = registry.findHandle("demo");
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
    const auto handle = registry.findHandle("demo");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->load_error.has_value());
    REQUIRE(handle->load_error->find("real neuriplo support is not enabled") != std::string::npos);
}
#endif
