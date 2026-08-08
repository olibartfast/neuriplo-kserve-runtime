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
    // Repository mode: every model discovered in the tree is loaded. A failed
    // model registers as Failed rather than aborting the server, so one bad
    // entry cannot take the whole repository down.
    explicit ModelRegistry(const std::vector<RuntimeConfig> &configs);

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
    std::optional<std::string> defaultVersion(const std::string &model_name) const;
    std::optional<RuntimeConfig> modelConfig(const std::string &model_name) const;
    bool beginDrain(const std::string &model_name);
    SchedulerMetricsSnapshot schedulerMetrics(const std::string &model_name) const;

    std::string modelName() const;
    bool logPayloads() const;
    double tokensPerChar() const;
    size_t retiredSchedulerCount() const;

  private:
    static bool isPipelineConfig(const RuntimeConfig &config);
    // Builds pipeline executors up front, outside the registry lock, and
    // returns a factory that hands them to the lifecycle. Must be called
    // without models_mutex_ held.
    ExecutorFactory makePipelineFactory(const RuntimeConfig &config);
    bool loadModelLocked(const std::string &model_name, const RuntimeConfig &config,
                         ExecutorFactory factory);
    ModelSlot *findSlotMutable(const std::string &model_name);
    const ModelSlot *findSlot(const std::string &model_name) const;
    void publishSnapshot(ModelSlot &slot);
    void retireSnapshot(const std::shared_ptr<const InferSnapshot> &snapshot);
    std::string activeVersionFor(const ModelSlot &slot) const;

    mutable std::shared_mutex models_mutex_;
    std::unordered_map<std::string, ModelSlot> models_;
    // Guarded by models_mutex_ alongside models_: a load reads the catalog and
    // writes the slot map, and both must see one consistent view.
    std::unordered_map<std::string, RuntimeConfig> catalog_;
    ModelLifecycle lifecycle_;
    SchedulerRetireQueue retire_queue_;
    bool log_payloads_ = false;
    double tokens_per_char_ = 0.25;
};
