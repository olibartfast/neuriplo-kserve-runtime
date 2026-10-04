#pragma once

#include "InferSnapshot.hpp"
#include "ModelHandle.hpp"
#include "ModelLifecycle.hpp"
#include "ModelMetadata.hpp"
#include "RuntimeConfig.hpp"
#include "SchedulerRetireQueue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelSlot {
    ModelHandle handle;
    // Config the model was last (re)loaded with; admin reloads start from it
    // so an empty reload body keeps backend/model_path/plugin_dir intact.
    RuntimeConfig config;
    std::shared_ptr<const InferSnapshot> active_snapshot;
    std::unordered_map<std::string, std::shared_ptr<const InferSnapshot>> version_snapshots;
    // True from a successful beginReload() until the build that follows it
    // is resolved (success, failure, or an exception), i.e. for exactly the
    // window in which a replacement handle is being built with
    // models_mutex_ released. Guards against a second reload/switch-version
    // racing the same build, and against an unload tearing the slot down
    // out from under a build that is about to publish.
    bool reload_pending = false;
};

class ModelRegistry {
  public:
    using ExecutorFactory = ModelLifecycle::ExecutorFactory;

    explicit ModelRegistry(const RuntimeConfig &config);
    ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory);
    // Repository mode: every model discovered in the tree is loaded. A failed
    // model registers as Failed rather than aborting the server, so one bad
    // entry cannot take the whole repository down -- and, unlike single-model
    // mode, a Failed slot does not make allReady() report the server not
    // ready either: it is reported through /v2/repository/index (state
    // UNAVAILABLE with its reason) instead. `explicit_control_mode` is true
    // only for --model-control-mode explicit, where the registry
    // deliberately starts (and may return to) empty; allReady() then reports
    // ready on an empty registry instead of forever not-ready. `defaults`
    // supplies log_payloads/tokens_per_char when `configs` is empty (explicit
    // mode at startup, or a repository scan that found nothing), since there
    // is then no config in the vector to read them from.
    explicit ModelRegistry(const std::vector<RuntimeConfig> &configs,
                           bool explicit_control_mode = false,
                           const RuntimeConfig &defaults = RuntimeConfig{});

    // Returns true iff, after this call, `config.model_name` is a slot this
    // call itself resolved (built, successfully or not) or a concurrent
    // load for the exact same name that this call waited out resolved to
    // Ready -- i.e. "this call's load attempt, or the one it joined, is
    // done and the model ended up loaded". It returns false both for a
    // build this call owned that failed (ready()/findHandle()->load_error
    // tells the caller why) and for a name that already had a settled slot
    // before this call even started (loading an already-loaded name is not
    // success -- callers that want that use reload()).
    bool loadModel(const RuntimeConfig &config);
    bool loadModel(const RuntimeConfig &config, ExecutorFactory factory);
    bool unloadModel(const std::string &model_name);
    // Returns true iff the reload (or, after waiting out a same-name
    // placeholder/in-flight reload, the attempt it joined) finished with the
    // model Ready. A failed reload of an already-loaded model returns false
    // but leaves that model untouched and still Ready -- see
    // ModelLifecycle::finishReload. false also simply means "no such slot".
    bool reload(const std::string &model_name, const RuntimeConfig &config);
    bool reload(const std::string &model_name, const RuntimeConfig &config,
                ExecutorFactory factory);
    bool reload(const RuntimeConfig &config);
    bool reload(const RuntimeConfig &config, ExecutorFactory factory);
    bool switchVersion(const std::string &model_name, const std::string &version,
                       const RuntimeConfig &config);
    bool switchVersion(const std::string &model_name, const std::string &version,
                       const RuntimeConfig &config, ExecutorFactory factory);
    bool completeUnload(const std::string &model_name);

    // Catalog of models the repository offers, which is not the same set as the
    // models currently loaded. In explicit control mode the catalog is what a
    // client's load request resolves a bare model name against.
    void setRepositoryCatalog(std::vector<RuntimeConfig> configs);
    std::vector<std::string> catalogModels() const;
    std::optional<RuntimeConfig> catalogConfig(const std::string &model_name) const;

    std::vector<std::string> listModels() const;
    std::optional<ModelMetadata> find(const std::string &model_name) const;
    std::optional<ModelMetadata> findVersion(const std::string &model_name,
                                             const std::string &version) const;
    std::shared_ptr<const InferSnapshot> findHandle(const std::string &model_name) const;
    std::shared_ptr<const InferSnapshot> findHandleVersion(const std::string &model_name,
                                                           const std::string &version) const;
    bool ready(const std::string &model_name) const;
    bool readyVersion(const std::string &model_name, const std::string &version) const;
    bool allReady() const;
    // True for a name with a slot whose load is still building (state ==
    // Loading) or whose reload/switch-version build is in flight
    // (reload_pending): a caller deciding between 404 (no such model) and
    // 409 (busy, try again) for an unload/drain that this returns false for
    // the other way checks this first.
    bool loadOrReloadInProgress(const std::string &model_name) const;
    // Test-only visibility into how many calls are currently parked on
    // load_cv_ (joining a same-name load placeholder or waiting out an
    // in-flight reload/switch-version). Lets a concurrency test poll until a
    // second caller has actually reached its wait before the test releases
    // the gate the first caller's build is blocked on, instead of racing a
    // sleep against it.
    std::size_t loadWaitersForTesting() const;
    std::optional<std::string> defaultVersion(const std::string &model_name) const;
    std::optional<RuntimeConfig> modelConfig(const std::string &model_name) const;
    bool beginDrain(const std::string &model_name);
    SchedulerMetricsSnapshot schedulerMetrics(const std::string &model_name) const;

    std::string modelName() const;
    bool logPayloads() const;
    double tokensPerChar() const;
    size_t retiredSchedulerCount() const;
    // True for the repository-mode constructors (both "none" and
    // "explicit" --model-control-mode), false for single-model mode. Lets a
    // caller with exactly one model loaded (e.g. a one-model repository, or
    // explicit mode with one model loaded) tell that apart from genuine
    // single-model mode, where "the" global model version IS that model's
    // version.
    bool repositoryMode() const;

  private:
    static bool isPipelineConfig(const RuntimeConfig &config);
    // Builds pipeline executors up front, outside the registry lock, and
    // returns a factory that hands them to the lifecycle. Must be called
    // without models_mutex_ held.
    ExecutorFactory makePipelineFactory(const RuntimeConfig &config);
    ModelSlot *findSlotMutable(const std::string &model_name);
    const ModelSlot *findSlot(const std::string &model_name) const;
    void publishSnapshot(ModelSlot &slot);
    void retireSnapshot(const std::shared_ptr<const InferSnapshot> &snapshot);
    std::string activeVersionFor(const ModelSlot &slot) const;

    // RAII backstop for loadModel/reload/switchVersion: those build a
    // replacement handle with models_mutex_ released, between setting a
    // slot's state to Loading (fresh load) or reload_pending = true
    // (reload/switch-version) and the follow-up lock that resolves it.
    // ModelLifecycle::load() itself never throws -- it catches everything
    // and reports Failed instead -- but this guard is the backstop for
    // anything else in that window (e.g. an allocation in this class) that
    // still might: without it, an exception would leave the slot stuck in
    // Loading/reload_pending forever, hanging every later caller that
    // load_cv_.wait()s on this name. commit() disarms it once the build
    // call returns normally (by any outcome); if it is never called, the
    // destructor marks the slot Failed (fresh load) or just clears
    // reload_pending (reload/switch-version, leaving a Ready handle Ready)
    // and notifies load_cv_ so no waiter is left hanging.
    class PendingLoadGuard {
      public:
        PendingLoadGuard(ModelRegistry &registry, std::string model_name);
        ~PendingLoadGuard();
        PendingLoadGuard(const PendingLoadGuard &) = delete;
        PendingLoadGuard &operator=(const PendingLoadGuard &) = delete;
        void commit();

      private:
        ModelRegistry &registry_;
        std::string model_name_;
        bool committed_ = false;
    };

    mutable std::shared_mutex models_mutex_;
    std::unordered_map<std::string, ModelSlot> models_;
    // Guarded by models_mutex_ alongside models_: a load reads the catalog and
    // writes the slot map, and both must see one consistent view.
    std::unordered_map<std::string, RuntimeConfig> catalog_;
    ModelLifecycle lifecycle_;
    SchedulerRetireQueue retire_queue_;
    // Signaled whenever a slot leaves ModelState::Loading or reload_pending
    // becomes false (success, failure, an exception via PendingLoadGuard, or
    // the slot disappearing). A concurrent loadModel/reload/switchVersion
    // call for the same name waits on this instead of racing a second
    // build, or (for reload_pending) retries once the in-flight one clears.
    std::condition_variable_any load_cv_;
    // Incremented/decremented around every load_cv_.wait() call (see
    // loadWaitersForTesting()); mutable so the const getter can read it
    // without needing models_mutex_.
    mutable std::atomic<std::size_t> load_waiters_{0};
    bool log_payloads_ = false;
    double tokens_per_char_ = 0.25;
    // Set by the repository-mode (vector) constructors only; see allReady().
    bool repository_mode_ = false;
    bool explicit_control_mode_ = false;
};
