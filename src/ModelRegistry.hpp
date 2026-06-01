#pragma once

#include "ModelHandle.hpp"
#include "ModelMetadata.hpp"
#include "RuntimeConfig.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>

class ModelRegistry {
  public:
    using ExecutorFactory =
        std::function<std::unique_ptr<Executor>(const RuntimeConfig &, std::string &error)>;

    explicit ModelRegistry(const RuntimeConfig &config);
    ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory);

    std::optional<ModelMetadata> find(const std::string &model_name) const;
    std::optional<ModelMetadata> findVersion(const std::string &model_name,
                                             const std::string &version) const;
    const ModelHandle *findHandle(const std::string &model_name) const;
    const ModelHandle *findHandleVersion(const std::string &model_name,
                                         const std::string &version) const;
    bool ready(const std::string &model_name) const;
    bool readyVersion(const std::string &model_name, const std::string &version) const;
    bool allReady() const;
    std::optional<std::string> defaultVersion(const std::string &model_name) const;
    bool beginDrain(const std::string &model_name);
    SchedulerMetricsSnapshot schedulerMetrics(const std::string &model_name) const;

    std::string modelName() const;
    bool logPayloads() const;
    double tokensPerChar() const;

  private:
    void loadModel(const RuntimeConfig &config, ExecutorFactory factory);

    ModelHandle handle_;
    bool log_payloads_ = false;
    double tokens_per_char_ = 0.25;
};
