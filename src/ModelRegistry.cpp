#include "ModelRegistry.hpp"

#include "BackendRegistry.hpp"

#include <utility>

namespace {

std::unique_ptr<Executor> defaultExecutorFactory(const RuntimeConfig &config, std::string &error) {
    return createExecutorFor(config.backend, config, error);
}

} // namespace

ModelRegistry::ModelRegistry(const RuntimeConfig &config)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    lifecycle_.load(handle_, config, defaultExecutorFactory);
}

ModelRegistry::ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    lifecycle_.load(handle_, config, std::move(factory));
}

bool ModelRegistry::reload(const RuntimeConfig &config) {
    return lifecycle_.reload(handle_, config, defaultExecutorFactory);
}

bool ModelRegistry::reload(const RuntimeConfig &config, ExecutorFactory factory) {
    return lifecycle_.reload(handle_, config, std::move(factory));
}

bool ModelRegistry::completeUnload(const std::string &model_name) {
    if (model_name != handle_.name) {
        return false;
    }
    return lifecycle_.completeUnload(handle_);
}

std::optional<ModelMetadata> ModelRegistry::find(const std::string &model_name) const {
    if (model_name != handle_.name) {
        return std::nullopt;
    }
    return handle_.metadata;
}

std::optional<ModelMetadata> ModelRegistry::findVersion(const std::string &model_name,
                                                        const std::string &version) const {
    const auto model = find(model_name);
    if (!model) {
        return std::nullopt;
    }
    for (const auto &known_version : model->versions) {
        if (known_version == version) {
            return model;
        }
    }
    return std::nullopt;
}

const ModelHandle *ModelRegistry::findHandle(const std::string &model_name) const {
    if (model_name != handle_.name) {
        return nullptr;
    }
    return &handle_;
}

const ModelHandle *ModelRegistry::findHandleVersion(const std::string &model_name,
                                                    const std::string &version) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr) {
        return nullptr;
    }
    const auto &known_versions =
        handle->metadata.versions.empty() ? handle->versions : handle->metadata.versions;
    for (const auto &known_version : known_versions) {
        if (known_version == version) {
            return handle;
        }
    }
    return nullptr;
}

bool ModelRegistry::ready(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    return handle != nullptr && handle->isReady();
}

bool ModelRegistry::readyVersion(const std::string &model_name, const std::string &version) const {
    const auto *handle = findHandleVersion(model_name, version);
    return handle != nullptr && handle->isReady();
}

bool ModelRegistry::allReady() const {
    return handle_.isReady();
}

std::optional<std::string> ModelRegistry::defaultVersion(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr) {
        return std::nullopt;
    }
    if (!handle->metadata.versions.empty()) {
        return handle->metadata.versions.front();
    }
    if (!handle->versions.empty()) {
        return handle->versions.front();
    }
    return std::nullopt;
}

bool ModelRegistry::beginDrain(const std::string &model_name) {
    if (model_name != handle_.name) {
        return false;
    }
    return lifecycle_.beginDrain(handle_);
}

SchedulerMetricsSnapshot ModelRegistry::schedulerMetrics(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr || handle->scheduler == nullptr) {
        return {};
    }
    return handle->scheduler->metrics();
}

std::string ModelRegistry::modelName() const {
    return handle_.name;
}

bool ModelRegistry::logPayloads() const {
    return log_payloads_;
}

double ModelRegistry::tokensPerChar() const {
    return tokens_per_char_;
}
