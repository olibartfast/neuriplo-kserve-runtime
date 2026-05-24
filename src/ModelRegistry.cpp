#include "ModelRegistry.hpp"

ModelRegistry::ModelRegistry(const RuntimeConfig &config) {
    model_.name = config.model_name;
    model_.versions.push_back("1");
    model_.platform = "neuriplo_" + config.backend;
    model_.ready = true;
    model_.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    model_.outputs.push_back({"output", "FP32", {1, 1000}});
}

std::optional<ModelMetadata> ModelRegistry::find(const std::string &model_name) const {
    if (model_name != model_.name) {
        return std::nullopt;
    }
    return model_;
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

bool ModelRegistry::ready(const std::string &model_name) const {
    const auto model = find(model_name);
    return model.has_value() && model->ready;
}

bool ModelRegistry::readyVersion(const std::string &model_name, const std::string &version) const {
    const auto model = findVersion(model_name, version);
    return model.has_value() && model->ready;
}

bool ModelRegistry::allReady() const {
    return model_.ready;
}
