#pragma once

#include "InferSnapshot.hpp"
#include "ModelHandle.hpp"
#include "ModelLifecycle.hpp"
#include "ModelMetadata.hpp"
#include "RuntimeConfig.hpp"
#include "SchedulerRetireQueue.hpp"

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
};

class ModelRegistry {
  public:
    using ExecutorFactory = ModelLifecycle::ExecutorFactory;

    explicit ModelRegistry(const RuntimeConfig &config);
    ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory);

    bool loadModel(const RuntimeConfig &config);
    bool loadModel(const RuntimeConfig &config, ExecutorFactory factory);
    bool unloadModel(const std::string &model_name);
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
    std::optional<std::string> defaultVersion(const std::string &model_name) const;
    std::optional<RuntimeConfig> modelConfig(const std::string &model_name) const;
    bool beginDrain(const std::string &model_name);
    SchedulerMetricsSnapshot schedulerMetrics(const std::string &model_name) const;

    std::string modelName() const;
    bool logPayloads() const;
    double tokensPerChar() const;
    size_t retiredSchedulerCount() const;

  private:
    bool loadModelLocked(const std::string &model_name, const RuntimeConfig &config,
                         ExecutorFactory factory);
    ModelSlot *findSlotMutable(const std::string &model_name);
    const ModelSlot *findSlot(const std::string &model_name) const;
    void publishSnapshot(ModelSlot &slot);
    void retireSnapshot(const std::shared_ptr<const InferSnapshot> &snapshot);
    std::string activeVersionFor(const ModelSlot &slot) const;

    mutable std::shared_mutex models_mutex_;
    std::unordered_map<std::string, ModelSlot> models_;
    ModelLifecycle lifecycle_;
    SchedulerRetireQueue retire_queue_;
    bool log_payloads_ = false;
    double tokens_per_char_ = 0.25;
};
