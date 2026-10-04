#include "ModelRegistry.hpp"

#include "BackendRegistry.hpp"
#include "PipelineExecutor.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace {

std::unique_ptr<Executor> defaultExecutorFactory(const RuntimeConfig &config, std::string &error) {
    return createExecutorFor(config.backend, config, error);
}

std::string versionFromSnapshot(const std::shared_ptr<const InferSnapshot> &snapshot) {
    if (!snapshot) {
        return {};
    }
    if (!snapshot->metadata.versions.empty()) {
        return snapshot->metadata.versions.front();
    }
    if (!snapshot->versions.empty()) {
        return snapshot->versions.front();
    }
    return {};
}

// Scopes a load_cv_.wait() call for loadWaitersForTesting(): a test gates a
// slow build and needs to know a second caller has actually reached its
// wait (joining that build, or waiting out an in-flight reload) before
// releasing the gate, rather than racing a sleep against it.
struct WaiterScope {
    explicit WaiterScope(std::atomic<std::size_t> &counter) : counter_(counter) {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }
    ~WaiterScope() {
        counter_.fetch_sub(1, std::memory_order_relaxed);
    }
    WaiterScope(const WaiterScope &) = delete;
    WaiterScope &operator=(const WaiterScope &) = delete;

  private:
    std::atomic<std::size_t> &counter_;
};

} // namespace

ModelRegistry::PendingLoadGuard::PendingLoadGuard(ModelRegistry &registry, std::string model_name)
    : registry_(registry), model_name_(std::move(model_name)) {}

ModelRegistry::PendingLoadGuard::~PendingLoadGuard() {
    if (committed_) {
        return;
    }
    // Only reachable if the guarded build threw past ModelLifecycle::load's
    // own catch-everything (it shouldn't, but this is the backstop) or
    // something else in the guarded scope did. Resolve the slot so no
    // load_cv_ waiter hangs forever, and so the slot itself is not stuck
    // mid-transition.
    std::unique_lock lock(registry_.models_mutex_);
    auto *slot = registry_.findSlotMutable(model_name_);
    if (slot != nullptr) {
        if (slot->handle.state.current() == ModelState::Loading && !slot->reload_pending) {
            slot->handle.state.markFailed();
            slot->handle.load_error = "executor build threw";
            registry_.publishSnapshot(*slot);
        } else if (slot->reload_pending) {
            slot->reload_pending = false;
            slot->handle.load_error = "executor build threw";
            registry_.publishSnapshot(*slot);
        }
    }
    lock.unlock();
    registry_.load_cv_.notify_all();
}

void ModelRegistry::PendingLoadGuard::commit() {
    committed_ = true;
}

ModelRegistry::ModelRegistry(const RuntimeConfig &config)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config);
}

ModelRegistry::ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config, std::move(factory));
}

ModelRegistry::ModelRegistry(const std::vector<RuntimeConfig> &configs, bool explicit_control_mode,
                             const RuntimeConfig &defaults)
    : log_payloads_(configs.empty() ? defaults.log_payloads : configs.front().log_payloads),
      tokens_per_char_(configs.empty() ? defaults.tokens_per_char
                                       : configs.front().tokens_per_char),
      repository_mode_(true), explicit_control_mode_(explicit_control_mode) {
    // Ensembles are loaded after plain models because a pipeline's steps are
    // resolved against models already in the registry.
    for (const auto &config : configs) {
        if (!isPipelineConfig(config)) {
            loadModel(config);
        }
    }
    for (const auto &config : configs) {
        if (isPipelineConfig(config)) {
            loadModel(config);
        }
    }
}

ModelSlot *ModelRegistry::findSlotMutable(const std::string &model_name) {
    const auto it = models_.find(model_name);
    return it == models_.end() ? nullptr : &it->second;
}

const ModelSlot *ModelRegistry::findSlot(const std::string &model_name) const {
    const auto it = models_.find(model_name);
    return it == models_.end() ? nullptr : &it->second;
}

void ModelRegistry::publishSnapshot(ModelSlot &slot) {
    std::atomic_store(&slot.active_snapshot, InferSnapshot::fromHandle(slot.handle));
}

void ModelRegistry::retireSnapshot(const std::shared_ptr<const InferSnapshot> &snapshot) {
    if (snapshot && snapshot->scheduler) {
        retire_queue_.retire(snapshot->scheduler);
    }
}

std::string ModelRegistry::activeVersionFor(const ModelSlot &slot) const {
    return versionFromSnapshot(std::atomic_load(&slot.active_snapshot));
}

bool ModelRegistry::isPipelineConfig(const RuntimeConfig &config) {
    return config.backend == pipelineBackendId();
}

// Pipeline executors are built here, before the registry lock is taken, because
// building one resolves the models its graph references -- and those lookups
// take the same lock. Handing the finished executors to the lifecycle through a
// factory keeps the load path itself unchanged.
ModelRegistry::ExecutorFactory ModelRegistry::makePipelineFactory(const RuntimeConfig &config) {
    PipelineStepResolver resolver = [this](const std::string &model_name,
                                           const std::string &model_version) {
        return model_version.empty() ? findHandle(model_name)
                                     : findHandleVersion(model_name, model_version);
    };

    auto prebuilt = std::make_shared<std::vector<std::unique_ptr<Executor>>>();
    auto build_error = std::make_shared<std::string>();
    const size_t instances = std::max<size_t>(config.instances, 1);
    for (size_t instance = 0; instance < instances; ++instance) {
        auto executor = makePipelineExecutor(config, resolver, *build_error);
        if (!executor) {
            break;
        }
        prebuilt->push_back(std::move(executor));
    }

    auto next = std::make_shared<size_t>(0);
    return [prebuilt, build_error, next](const RuntimeConfig &,
                                         std::string &error) -> std::unique_ptr<Executor> {
        if (*next >= prebuilt->size()) {
            error = build_error->empty() ? "pipeline executor unavailable" : *build_error;
            return nullptr;
        }
        return std::move((*prebuilt)[(*next)++]);
    };
}

bool ModelRegistry::loadModel(const RuntimeConfig &config) {
    if (isPipelineConfig(config)) {
        return loadModel(config, makePipelineFactory(config));
    }
    return loadModel(config, defaultExecutorFactory);
}

bool ModelRegistry::loadModel(const RuntimeConfig &config, ExecutorFactory factory) {
    const std::string &model_name = config.model_name;
    // Spans from right after the placeholder is reserved through the final
    // swap below -- not just the lifecycle_.load() call -- so a throw
    // anywhere in that window (the config copy or publishSnapshot() in the
    // swap included, not only the build itself) still resolves the
    // placeholder instead of leaving it stuck Loading forever.
    std::optional<PendingLoadGuard> guard;

    {
        std::unique_lock lock(models_mutex_);
        auto *existing = findSlotMutable(model_name);
        if (existing != nullptr) {
            if (existing->handle.state.current() != ModelState::Loading) {
                // Triton-compatible semantics: calling load on a name that
                // already has a settled slot (Ready, Failed, whatever) is
                // not success -- the caller wants reload() for that. The
                // only case that *is* success without reloading anything is
                // the wait below, for a name that is still *becoming* a
                // slot via a concurrent loadModel call.
                return false;
            }
            // A concurrent request is already loading this exact name; wait
            // for it to finish instead of racing a second build. The loser
            // here is not a failure if the model ends up loaded.
            {
                WaiterScope waiter(load_waiters_);
                load_cv_.wait(lock, [&] {
                    const auto *slot = findSlot(model_name);
                    return slot == nullptr || slot->handle.state.current() != ModelState::Loading;
                });
            }
            const auto *slot = findSlot(model_name);
            return slot != nullptr && slot->handle.isReady();
        }
        ModelSlot placeholder;
        placeholder.config = config;
        placeholder.handle.name = model_name;
        placeholder.handle.state.startLoad();
        auto [inserted_it, inserted] = models_.emplace(model_name, std::move(placeholder));
        guard.emplace(*this, model_name);
        // Published immediately (state Loading, no scheduler yet): the index
        // can report LOADING instead of nothing, and allReady() treats a
        // model still coming up for the first time as "not yet counted"
        // rather than blocking on it under this same lock.
        publishSnapshot(inserted_it->second);
    }

    // The executor build (e.g. engine deserialize) happens here, with no
    // registry lock held: inference, readiness and the index on every other
    // model stay unblocked while this one loads.
    ModelHandle built;
    lifecycle_.load(built, config, std::move(factory),
                    config.model_version_explicit && !config.model_version.empty()
                        ? std::optional<std::string>(config.model_version)
                        : std::nullopt);

    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    const bool found = slot != nullptr;
    if (found) {
        slot->config = config;
        slot->handle = std::move(built);
        publishSnapshot(*slot);
    }
    lock.unlock();
    guard->commit();
    load_cv_.notify_all();
    // Whether this call's own build ran at all (the slot could have been
    // unloaded out from under it): the caller inspects ready()/load_error to
    // tell success from failure, exactly as it did before this name was ever
    // reserved.
    return found;
}

bool ModelRegistry::unloadModel(const std::string &model_name) {
    {
        // Failed loads register a slot without a scheduler, so the drain path
        // can never release them; remove the slot directly.
        std::unique_lock lock(models_mutex_);
        auto *slot = findSlotMutable(model_name);
        if (slot == nullptr) {
            return false;
        }
        if (slot->reload_pending || slot->handle.state.current() == ModelState::Loading) {
            // A load/reload is mid-build for this model with the lock
            // released; let it finish (and publish or fail) before this
            // model can be touched again. The caller distinguishes this
            // from "no such model" via loadOrReloadInProgress().
            return false;
        }
        if (slot->handle.state.current() == ModelState::Failed &&
            slot->handle.scheduler == nullptr) {
            models_.erase(model_name);
            return true;
        }
    }
    if (!beginDrain(model_name)) {
        return false;
    }
    return completeUnload(model_name);
}

bool ModelRegistry::reload(const std::string &model_name, const RuntimeConfig &config) {
    if (isPipelineConfig(config)) {
        return reload(model_name, config, makePipelineFactory(config));
    }
    return reload(model_name, config, defaultExecutorFactory);
}

bool ModelRegistry::reload(const std::string &model_name, const RuntimeConfig &config,
                           ExecutorFactory factory) {
    while (true) {
        std::unique_lock lock(models_mutex_);
        auto *slot = findSlotMutable(model_name);
        if (slot == nullptr) {
            return false;
        }
        if (slot->handle.state.current() == ModelState::Loading && !slot->reload_pending) {
            // This name is still a fresh-load placeholder -- e.g. a repeated
            // /v2/repository/models/{m}/load for a name whose first build is
            // still in flight resolves to reload() here, because
            // modelConfig() already sees the placeholder as "known". Join
            // that build's outcome instead of trying to "reload" something
            // that was never loaded in the first place.
            {
                WaiterScope waiter(load_waiters_);
                load_cv_.wait(lock, [&] {
                    const auto *s = findSlot(model_name);
                    return s == nullptr || s->handle.state.current() != ModelState::Loading;
                });
            }
            const auto *s = findSlot(model_name);
            return s != nullptr && s->handle.isReady();
        }
        if (slot->reload_pending) {
            // Another reload/switch-version is mid-build for this model.
            // Triton serializes load/reload requests on the same model
            // rather than rejecting the second one outright: wait for the
            // in-flight attempt to finish, then retry this one for real
            // (it may want a different config than the one that just
            // finished, so it does not simply inherit that result).
            {
                WaiterScope waiter(load_waiters_);
                load_cv_.wait(lock, [&] {
                    const auto *s = findSlot(model_name);
                    return s == nullptr || !s->reload_pending;
                });
            }
            continue;
        }
        if (!lifecycle_.beginReload(slot->handle)) {
            return false;
        }
        slot->reload_pending = true;
        break;
    }

    // The executor build happens here, with no registry lock held, so
    // inference and readiness on other models are never blocked by a slow
    // reload. slot->reload_pending (checked/set above, cleared below) keeps
    // a second reload/switch-version or an unload of this same model from
    // racing this build; PendingLoadGuard is the exception-safety backstop.
    ModelHandle next;
    {
        PendingLoadGuard guard(*this, model_name);
        lifecycle_.load(next, config, std::move(factory));
        guard.commit();
    }

    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    if (slot == nullptr) {
        // Unloaded while reloading: drop the result, there is nothing left
        // to swap it into.
        lock.unlock();
        load_cv_.notify_all();
        return false;
    }
    slot->reload_pending = false;

    const auto old_snapshot = std::atomic_load(&slot->active_snapshot);
    const bool success = lifecycle_.finishReload(slot->handle, std::move(next));
    if (success) {
        // Only on success: a failed reload must not overwrite the config a
        // later empty-body reload would otherwise fall back to (it would
        // "stick" the bad backend/model_path as the new default).
        slot->config = config;
    }
    publishSnapshot(*slot);
    const bool ready = slot->handle.isReady();
    lock.unlock();
    load_cv_.notify_all();
    if (!success) {
        return false;
    }
    retireSnapshot(old_snapshot);
    return ready;
}

bool ModelRegistry::reload(const RuntimeConfig &config) {
    return reload(config.model_name, config);
}

bool ModelRegistry::reload(const RuntimeConfig &config, ExecutorFactory factory) {
    return reload(config.model_name, config, std::move(factory));
}

bool ModelRegistry::switchVersion(const std::string &model_name, const std::string &version,
                                  const RuntimeConfig &config) {
    // Same reason as load and reload: the registered ensemble backend factory
    // deliberately fails, because building a pipeline needs a resolver for the
    // models its graph references. Without this branch, activating a version of
    // an ensemble always fails.
    if (isPipelineConfig(config)) {
        return switchVersion(model_name, version, config, makePipelineFactory(config));
    }
    return switchVersion(model_name, version, config, defaultExecutorFactory);
}

bool ModelRegistry::switchVersion(const std::string &model_name, const std::string &version,
                                  const RuntimeConfig &config, ExecutorFactory factory) {
    RuntimeConfig version_config = config;
    version_config.model_name = model_name;
    version_config.model_version = version;

    while (true) {
        std::unique_lock lock(models_mutex_);
        auto *slot = findSlotMutable(model_name);
        if (slot == nullptr) {
            return false;
        }

        const auto old_snapshot = std::atomic_load(&slot->active_snapshot);
        const auto old_version = versionFromSnapshot(old_snapshot);
        if (old_version == version && old_snapshot && old_snapshot->isReady()) {
            return true;
        }

        if (slot->handle.state.current() == ModelState::Loading && !slot->reload_pending) {
            {
                WaiterScope waiter(load_waiters_);
                load_cv_.wait(lock, [&] {
                    const auto *s = findSlot(model_name);
                    return s == nullptr || s->handle.state.current() != ModelState::Loading;
                });
            }
            const auto *s = findSlot(model_name);
            return s != nullptr && s->handle.isReady();
        }
        if (slot->reload_pending) {
            {
                WaiterScope waiter(load_waiters_);
                load_cv_.wait(lock, [&] {
                    const auto *s = findSlot(model_name);
                    return s == nullptr || !s->reload_pending;
                });
            }
            continue;
        }
        if (!lifecycle_.beginReload(slot->handle)) {
            return false;
        }
        slot->reload_pending = true;
        break;
    }

    // Built with no registry lock held -- same reasoning as reload() above.
    ModelHandle next;
    {
        PendingLoadGuard guard(*this, model_name);
        lifecycle_.load(next, version_config, std::move(factory), version);
        guard.commit();
    }

    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    if (slot == nullptr) {
        lock.unlock();
        load_cv_.notify_all();
        return false;
    }
    slot->reload_pending = false;

    const auto old_snapshot = std::atomic_load(&slot->active_snapshot);
    const auto old_version = versionFromSnapshot(old_snapshot);
    const bool success = lifecycle_.finishReload(slot->handle, std::move(next));
    if (!success) {
        publishSnapshot(*slot);
        lock.unlock();
        load_cv_.notify_all();
        return false;
    }

    slot->config = version_config;
    publishSnapshot(*slot);
    const auto new_snapshot = std::atomic_load(&slot->active_snapshot);
    if (old_snapshot && !old_version.empty()) {
        slot->version_snapshots[old_version] = old_snapshot;
    }
    if (new_snapshot) {
        const auto new_version = versionFromSnapshot(new_snapshot);
        if (!new_version.empty()) {
            slot->version_snapshots[new_version] = new_snapshot;
        }
    }
    const bool ready = new_snapshot != nullptr && new_snapshot->isReady();
    lock.unlock();
    load_cv_.notify_all();
    retireSnapshot(old_snapshot);
    return ready;
}

bool ModelRegistry::completeUnload(const std::string &model_name) {
    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    if (slot == nullptr) {
        return false;
    }

    std::shared_ptr<Scheduler> retired_scheduler;
    if (!lifecycle_.completeUnload(slot->handle, &retired_scheduler)) {
        return false;
    }

    publishSnapshot(*slot);
    if (retired_scheduler) {
        retire_queue_.retire(std::move(retired_scheduler));
    }
    slot->version_snapshots.clear();
    models_.erase(model_name);
    return true;
}

void ModelRegistry::setRepositoryCatalog(std::vector<RuntimeConfig> configs) {
    std::unique_lock lock(models_mutex_);
    catalog_.clear();
    for (auto &config : configs) {
        const auto name = config.model_name;
        catalog_.emplace(name, std::move(config));
    }
}

std::vector<std::string> ModelRegistry::catalogModels() const {
    std::shared_lock lock(models_mutex_);
    std::vector<std::string> names;
    names.reserve(catalog_.size());
    for (const auto &entry : catalog_) {
        names.push_back(entry.first);
    }
    return names;
}

std::optional<RuntimeConfig> ModelRegistry::catalogConfig(const std::string &model_name) const {
    std::shared_lock lock(models_mutex_);
    const auto found = catalog_.find(model_name);
    if (found == catalog_.end()) {
        return std::nullopt;
    }
    return found->second;
}

std::vector<std::string> ModelRegistry::listModels() const {
    std::shared_lock lock(models_mutex_);
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto &entry : models_) {
        names.push_back(entry.first);
    }
    return names;
}

std::optional<ModelMetadata> ModelRegistry::find(const std::string &model_name) const {
    std::shared_lock lock(models_mutex_);
    const auto *slot = findSlot(model_name);
    if (slot == nullptr) {
        return std::nullopt;
    }
    const auto snapshot = std::atomic_load(&slot->active_snapshot);
    if (!snapshot) {
        return std::nullopt;
    }
    return snapshot->metadata;
}

std::optional<ModelMetadata> ModelRegistry::findVersion(const std::string &model_name,
                                                        const std::string &version) const {
    const auto snapshot = findHandleVersion(model_name, version);
    if (!snapshot) {
        return std::nullopt;
    }
    return snapshot->metadata;
}

std::shared_ptr<const InferSnapshot>
ModelRegistry::findHandle(const std::string &model_name) const {
    std::shared_lock lock(models_mutex_);
    const auto *slot = findSlot(model_name);
    if (slot == nullptr) {
        return nullptr;
    }
    return std::atomic_load(&slot->active_snapshot);
}

std::shared_ptr<const InferSnapshot>
ModelRegistry::findHandleVersion(const std::string &model_name, const std::string &version) const {
    std::shared_lock lock(models_mutex_);
    const auto *slot = findSlot(model_name);
    if (slot == nullptr) {
        return nullptr;
    }

    const auto version_it = slot->version_snapshots.find(version);
    if (version_it != slot->version_snapshots.end()) {
        return version_it->second;
    }

    const auto snapshot = std::atomic_load(&slot->active_snapshot);
    if (!snapshot) {
        return nullptr;
    }
    const auto &known_versions =
        snapshot->metadata.versions.empty() ? snapshot->versions : snapshot->metadata.versions;
    for (const auto &known_version : known_versions) {
        if (known_version == version) {
            return snapshot;
        }
    }
    return nullptr;
}

bool ModelRegistry::ready(const std::string &model_name) const {
    const auto snapshot = findHandle(model_name);
    return snapshot != nullptr && snapshot->isReady();
}

bool ModelRegistry::readyVersion(const std::string &model_name, const std::string &version) const {
    const auto snapshot = findHandleVersion(model_name, version);
    return snapshot != nullptr && snapshot->isReady();
}

bool ModelRegistry::allReady() const {
    std::shared_lock lock(models_mutex_);
    if (models_.empty()) {
        // Explicit control mode starts (and may return to) empty by design;
        // the server is ready to accept load requests even though nothing is
        // loaded yet. Every other mode treats an empty registry as not-ready
        // -- there is nothing to serve.
        return explicit_control_mode_;
    }
    bool any_counted = false;
    for (const auto &entry : models_) {
        const auto snapshot = std::atomic_load(&entry.second.active_snapshot);
        if (snapshot && snapshot->state == ModelState::Loading) {
            // Still coming up for the first time (a load's placeholder): a
            // load in progress must not make the whole server, and every
            // other already-loaded model with it, report not-ready.
            continue;
        }
        if (repository_mode_ && snapshot && snapshot->state == ModelState::Failed) {
            // Reported through /v2/repository/index (UNAVAILABLE + reason)
            // instead: one bad model in a repository must not take the rest
            // of the server's readiness down with it. Single-model mode
            // keeps strict readiness -- it falls through to the check below.
            continue;
        }
        any_counted = true;
        if (!snapshot || !snapshot->isReady()) {
            return false;
        }
    }
    return any_counted;
}

bool ModelRegistry::loadOrReloadInProgress(const std::string &model_name) const {
    std::shared_lock lock(models_mutex_);
    const auto *slot = findSlot(model_name);
    if (slot == nullptr) {
        return false;
    }
    return slot->reload_pending || slot->handle.state.current() == ModelState::Loading;
}

std::size_t ModelRegistry::loadWaitersForTesting() const {
    return load_waiters_.load(std::memory_order_relaxed);
}

std::optional<RuntimeConfig> ModelRegistry::modelConfig(const std::string &model_name) const {
    std::shared_lock lock(models_mutex_);
    const auto *slot = findSlot(model_name);
    if (slot == nullptr) {
        return std::nullopt;
    }
    return slot->config;
}

std::optional<std::string> ModelRegistry::defaultVersion(const std::string &model_name) const {
    const auto snapshot = findHandle(model_name);
    if (!snapshot) {
        return std::nullopt;
    }
    if (!snapshot->metadata.versions.empty()) {
        return snapshot->metadata.versions.front();
    }
    if (!snapshot->versions.empty()) {
        return snapshot->versions.front();
    }
    return std::nullopt;
}

bool ModelRegistry::beginDrain(const std::string &model_name) {
    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    if (slot == nullptr || slot->reload_pending) {
        return false;
    }
    if (!lifecycle_.beginDrain(slot->handle)) {
        return false;
    }
    publishSnapshot(*slot);
    return true;
}

SchedulerMetricsSnapshot ModelRegistry::schedulerMetrics(const std::string &model_name) const {
    const auto snapshot = findHandle(model_name);
    if (!snapshot || snapshot->scheduler == nullptr) {
        return {};
    }
    return snapshot->scheduler->metrics();
}

std::string ModelRegistry::modelName() const {
    std::shared_lock lock(models_mutex_);
    if (models_.empty()) {
        return {};
    }
    return models_.begin()->first;
}

bool ModelRegistry::logPayloads() const {
    return log_payloads_;
}

double ModelRegistry::tokensPerChar() const {
    return tokens_per_char_;
}

size_t ModelRegistry::retiredSchedulerCount() const {
    return retire_queue_.pendingCount();
}

bool ModelRegistry::repositoryMode() const {
    return repository_mode_;
}
