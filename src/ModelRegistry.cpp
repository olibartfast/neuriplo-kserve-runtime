#include "ModelRegistry.hpp"

#include "StubExecutor.hpp"

#include <stdexcept>
#include <utility>

namespace {

std::unique_ptr<Executor> defaultExecutorFactory(const RuntimeConfig &config, std::string &error) {
    if (config.backend != "stub") {
        error = "unsupported backend: " + config.backend;
        return nullptr;
    }
    return makeStubExecutor(config, error);
}

} // namespace

ModelRegistry::ModelRegistry(const RuntimeConfig &config)
    : ModelRegistry(config, defaultExecutorFactory) {}

ModelRegistry::ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory) {
    loadModel(config, std::move(factory));
}

void ModelRegistry::loadModel(const RuntimeConfig &config, ExecutorFactory factory) {
    handle_.name = config.model_name;
    handle_.versions = {"1"};
    handle_.state = ModelState::Loading;

    std::string error;
    handle_.executor = factory(config, error);
    if (!handle_.executor) {
        handle_.state = ModelState::Failed;
        handle_.load_error = error.empty() ? "failed to create executor" : error;
        handle_.metadata.name = config.model_name;
        handle_.metadata.versions = handle_.versions;
        handle_.metadata.platform = "neuriplo_" + config.backend;
        return;
    }

    handle_.metadata = handle_.executor->metadata();
    handle_.name = handle_.metadata.name;
    handle_.versions = handle_.metadata.versions;
    handle_.state = ModelState::Ready;
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
