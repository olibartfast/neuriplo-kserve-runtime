#include "AdminCodec.hpp"

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

bool readString(const Json &json, const char *key, std::string &out) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_string()) {
        return false;
    }
    out = json[key].get<std::string>();
    return true;
}

bool readSizeT(const Json &json, const char *key, size_t &out) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_number_unsigned()) {
        return false;
    }
    out = json[key].get<size_t>();
    return true;
}

bool readInt64(const Json &json, const char *key, int64_t &out) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_number_integer()) {
        return false;
    }
    out = json[key].get<int64_t>();
    return true;
}

bool readBool(const Json &json, const char *key, bool &out) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_boolean()) {
        return false;
    }
    out = json[key].get<bool>();
    return true;
}

bool readInputSizes(const Json &json, RuntimeConfig &config) {
    if (!json.contains("input_sizes") || !json["input_sizes"].is_array()) {
        return false;
    }
    std::vector<std::vector<int64_t>> input_sizes;
    for (const auto &shape : json["input_sizes"]) {
        if (!shape.is_array()) {
            return false;
        }
        std::vector<int64_t> dims;
        for (const auto &dim : shape) {
            if (!dim.is_number_integer()) {
                return false;
            }
            dims.push_back(dim.get<int64_t>());
        }
        input_sizes.push_back(std::move(dims));
    }
    config.input_sizes = std::move(input_sizes);
    return true;
}

void applyCommonFields(const Json &json, RuntimeConfig &config) {
    readString(json, "backend", config.backend);
    readString(json, "plugin_dir", config.plugin_dir);
    readInputSizes(json, config);
    readString(json, "model_path", config.model_path);
    readString(json, "storage_uri", config.storage_uri);
    readString(json, "scheduler_strategy", config.scheduler_strategy);
    readSizeT(json, "instances", config.instances);
    readSizeT(json, "max_queue_size", config.max_queue_size);
    readInt64(json, "request_timeout_ms", config.request_timeout_ms);
    readBool(json, "dynamic_batching_enabled", config.dynamic_batching_enabled);
    readSizeT(json, "max_batch_size", config.max_batch_size);
    readInt64(json, "max_queue_delay_us", config.max_queue_delay_us);
}

AdminParseResult makeError(std::string message) {
    AdminParseResult result;
    result.error_message = std::move(message);
    return result;
}

} // namespace

AdminParseResult parseLoadModelRequest(const std::string &body, const RuntimeConfig &defaults) {
    AdminParseResult result;
    result.config = defaults;

    Json json;
    try {
        json = Json::parse(body);
    } catch (const std::exception &error) {
        return makeError(std::string("invalid JSON: ") + error.what());
    }

    if (!readString(json, "model_name", result.config.model_name) ||
        result.config.model_name.empty()) {
        return makeError("model_name is required");
    }
    readString(json, "model_version", result.config.model_version);
    applyCommonFields(json, result.config);

    result.model_name = result.config.model_name;
    result.ok = true;
    return result;
}

AdminParseResult parseReloadModelRequest(const std::string &body, const RuntimeConfig &defaults) {
    AdminParseResult result;
    result.config = defaults;

    if (body.empty()) {
        result.ok = true;
        return result;
    }

    Json json;
    try {
        json = Json::parse(body);
    } catch (const std::exception &error) {
        return makeError(std::string("invalid JSON: ") + error.what());
    }

    applyCommonFields(json, result.config);
    result.ok = true;
    return result;
}

AdminParseResult parseSwitchVersionRequest(const std::string &body, const RuntimeConfig &defaults) {
    AdminParseResult result;
    result.config = defaults;

    Json json;
    try {
        json = Json::parse(body);
    } catch (const std::exception &error) {
        return makeError(std::string("invalid JSON: ") + error.what());
    }

    if (!readString(json, "version", result.version) || result.version.empty()) {
        return makeError("version is required");
    }
    applyCommonFields(json, result.config);
    result.config.model_version = result.version;
    result.ok = true;
    return result;
}
