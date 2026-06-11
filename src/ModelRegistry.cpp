#include "ModelRegistry.hpp"

#include "BackendRegistry.hpp"

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

} // namespace

ModelRegistry::ModelRegistry(const RuntimeConfig &config)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config);
}

ModelRegistry::ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config, std::move(factory));
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

bool ModelRegistry::loadModelLocked(const std::string &model_name, const RuntimeConfig &config,
                                    ExecutorFactory factory) {
    if (models_.find(model_name) != models_.end()) {
        return false;
    }

    ModelSlot slot;
    slot.config = config;
    lifecycle_.load(slot.handle, config, factory);
    publishSnapshot(slot);
    models_.emplace(model_name, std::move(slot));
    return true;
}

bool ModelRegistry::loadModel(const RuntimeConfig &config) {
    return loadModel(config, defaultExecutorFactory);
}

bool ModelRegistry::loadModel(const RuntimeConfig &config, ExecutorFactory factory) {
    std::unique_lock lock(models_mutex_);
    return loadModelLocked(config.model_name, config, std::move(factory));
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
    return reload(model_name, config, defaultExecutorFactory);
}

bool ModelRegistry::reload(const std::string &model_name, const RuntimeConfig &config,
                           ExecutorFactory factory) {
    std::unique_lock lock(models_mutex_);
    auto *slot = findSlotMutable(model_name);
    if (slot == nullptr) {
        return false;
    }

    const auto old_snapshot = std::atomic_load(&slot->active_snapshot);
    if (!lifecycle_.reload(slot->handle, config, factory)) {
        publishSnapshot(*slot);
        return false;
    }

    slot->config = config;
    publishSnapshot(*slot);
    retireSnapshot(old_snapshot);
    return slot->handle.isReady();
}

bool ModelRegistry::reload(const RuntimeConfig &config) {
    return reload(config.model_name, config);
}

bool ModelRegistry::reload(const RuntimeConfig &config, ExecutorFactory factory) {
    return reload(config.model_name, config, std::move(factory));
}

bool ModelRegistry::switchVersion(const std::string &model_name, const std::string &version,
                                  const RuntimeConfig &config) {
    return switchVersion(model_name, version, config, defaultExecutorFactory);
}

bool ModelRegistry::switchVersion(const std::string &model_name, const std::string &version,
                                  const RuntimeConfig &config, ExecutorFactory factory) {
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

    RuntimeConfig version_config = config;
    version_config.model_name = model_name;
    version_config.model_version = version;

    if (!lifecycle_.reload(slot->handle, version_config, std::move(factory), version)) {
        publishSnapshot(*slot);
        return false;
    }

    slot->config = version_config;
    publishSnapshot(*slot);
    const auto new_snapshot = std::atomic_load(&slot->active_snapshot);
    if (old_snapshot && !old_version.empty()) {
        slot->version_snapshots[old_version] = old_snapshot;
    }
    retireSnapshot(old_snapshot);

    if (new_snapshot) {
        const auto new_version = versionFromSnapshot(new_snapshot);
        if (!new_version.empty()) {
            slot->version_snapshots[new_version] = new_snapshot;
        }
    }

    return new_snapshot != nullptr && new_snapshot->isReady();
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
        return false;
    }
    for (const auto &entry : models_) {
        const auto snapshot = std::atomic_load(&entry.second.active_snapshot);
        if (!snapshot || !snapshot->isReady()) {
            return false;
        }
    }
    return true;
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
    if (slot == nullptr) {
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
