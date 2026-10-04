#include "AdminCodec.hpp"

#include "PipelineExecutor.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

namespace {

using Json = nlohmann::json;

// A known field present with the wrong JSON type reports *type_error
// (callers turn that into a 400 naming the field) instead of being silently
// ignored, so e.g. "instances":"2" fails loudly rather than loading with the
// default instance count. Absent fields are not an error: *type_error is
// left untouched and the function returns false so the caller's defaults
// stand.
bool readString(const Json &json, const char *key, std::string &out, std::string *type_error) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_string()) {
        *type_error = std::string(key) + " must be a string";
        return false;
    }
    out = json[key].get<std::string>();
    return true;
}

bool readSizeT(const Json &json, const char *key, size_t &out, std::string *type_error) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_number_unsigned()) {
        *type_error = std::string(key) + " must be a non-negative integer";
        return false;
    }
    out = json[key].get<size_t>();
    return true;
}

bool readInt64(const Json &json, const char *key, int64_t &out, std::string *type_error) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_number_integer()) {
        *type_error = std::string(key) + " must be an integer";
        return false;
    }
    out = json[key].get<int64_t>();
    return true;
}

bool readBool(const Json &json, const char *key, bool &out, std::string *type_error) {
    if (!json.contains(key)) {
        return false;
    }
    if (!json[key].is_boolean()) {
        *type_error = std::string(key) + " must be a boolean";
        return false;
    }
    out = json[key].get<bool>();
    return true;
}

bool readInputSizes(const Json &json, RuntimeConfig &config, std::string *type_error) {
    if (!json.contains("input_sizes")) {
        return false;
    }
    if (!json["input_sizes"].is_array()) {
        *type_error = "input_sizes must be an array of arrays of integers";
        return false;
    }
    std::vector<std::vector<int64_t>> input_sizes;
    for (const auto &shape : json["input_sizes"]) {
        if (!shape.is_array()) {
            *type_error = "input_sizes must be an array of arrays of integers";
            return false;
        }
        std::vector<int64_t> dims;
        for (const auto &dim : shape) {
            if (!dim.is_number_integer()) {
                *type_error = "input_sizes must be an array of arrays of integers";
                return false;
            }
            dims.push_back(dim.get<int64_t>());
        }
        input_sizes.push_back(std::move(dims));
    }
    config.input_sizes = std::move(input_sizes);
    return true;
}

// A sane ceiling on requested executor instances: large enough for any real
// deployment, small enough that a typo (or a hostile client) cannot ask the
// server to spin up an unbounded number of backend instances.
constexpr size_t kMaxInstances = 64;

// Applies fields common to load/reload/switch-version bodies. Returns an
// error message naming the first invalid field -- wrong JSON type, or a
// numeric field out of its allowed range -- or nullopt when every field
// present was valid. *backend_provided reports whether "backend" appeared in
// the body at all, independent of whether the rest of the body is valid,
// since a caller resolving a bare model name needs to know whether the
// request actually named a backend rather than inheriting one from defaults.
std::optional<std::string> applyCommonFields(const Json &json, RuntimeConfig &config,
                                             bool &backend_provided) {
    std::string type_error;
    backend_provided = json.contains("backend");

    if (readString(json, "backend", config.backend, &type_error)) {
        if (config.backend.empty()) {
            return std::string("backend must not be empty");
        }
    } else if (!type_error.empty()) {
        return type_error;
    }
    if (!readString(json, "plugin_dir", config.plugin_dir, &type_error) && !type_error.empty()) {
        return type_error;
    }
    if (!readInputSizes(json, config, &type_error) && !type_error.empty()) {
        return type_error;
    }
    if (!readString(json, "model_path", config.model_path, &type_error) && !type_error.empty()) {
        return type_error;
    }
    // A pipeline graph may be sent inline as an object (the natural way to
    // write it in a load body) or as a JSON string; both reach the parser as
    // text.
    if (json.contains("pipeline_graph")) {
        config.pipeline_graph = json["pipeline_graph"].is_string()
                                    ? json["pipeline_graph"].get<std::string>()
                                    : json["pipeline_graph"].dump();
    }
    if (!readString(json, "storage_uri", config.storage_uri, &type_error) && !type_error.empty()) {
        return type_error;
    }
    if (!readString(json, "scheduler_strategy", config.scheduler_strategy, &type_error) &&
        !type_error.empty()) {
        return type_error;
    }

    if (readSizeT(json, "instances", config.instances, &type_error)) {
        if (config.instances < 1 || config.instances > kMaxInstances) {
            return "instances must be between 1 and " + std::to_string(kMaxInstances);
        }
    } else if (!type_error.empty()) {
        return type_error;
    }

    if (readSizeT(json, "max_queue_size", config.max_queue_size, &type_error)) {
        if (config.max_queue_size < 1) {
            return std::string("max_queue_size must be >= 1");
        }
    } else if (!type_error.empty()) {
        return type_error;
    }

    if (readInt64(json, "request_timeout_ms", config.request_timeout_ms, &type_error)) {
        if (config.request_timeout_ms <= 0) {
            return std::string("request_timeout_ms must be greater than 0");
        }
    } else if (!type_error.empty()) {
        return type_error;
    }
    if (!readBool(json, "use_gpu", config.use_gpu, &type_error) && !type_error.empty()) {
        return type_error;
    }
    const bool dynamic_batching_provided =
        readBool(json, "dynamic_batching_enabled", config.dynamic_batching_enabled, &type_error);
    if (!dynamic_batching_provided && !type_error.empty()) {
        return type_error;
    }

    bool max_batch_size_provided = false;
    if (readSizeT(json, "max_batch_size", config.max_batch_size, &type_error)) {
        max_batch_size_provided = true;
        if (config.max_batch_size < 1) {
            return std::string("max_batch_size must be >= 1");
        }
    } else if (!type_error.empty()) {
        return type_error;
    }

    if (config.backend == pipelineBackendId() && !dynamic_batching_provided &&
        !max_batch_size_provided) {
        // max_batch_size 1 / batching off is contractual for ensembles
        // (PipelineExecutor rejects anything else), not a tuning default. A
        // server-wide --dynamic-batching-enabled/--max-batch-size default
        // inherited from `defaults` (the model's prior config, or the CLI
        // defaults) was set for the tensor models it tunes, not for this
        // ensemble; only a request that names these fields on this model
        // itself is an actual ask for batching on an ensemble, and that is
        // still rejected below/at load time.
        config.dynamic_batching_enabled = false;
        config.max_batch_size = 1;
    }

    if (readInt64(json, "max_queue_delay_us", config.max_queue_delay_us, &type_error)) {
        if (config.max_queue_delay_us < 0) {
            return std::string("max_queue_delay_us must be greater than or equal to 0");
        }
    } else if (!type_error.empty()) {
        return type_error;
    }

    // Mirrors the CLI's own validation (RuntimeConfig.cpp): checked against
    // the final merged config, not just fields this request happened to
    // touch, since a request that only flips dynamic_batching_enabled on
    // must still be rejected if the inherited max_batch_size can't support
    // it.
    if (config.dynamic_batching_enabled && config.max_batch_size < 2) {
        return std::string(
            "max_batch_size must be at least 2 when dynamic_batching_enabled is true");
    }

    return std::nullopt;
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

    std::string type_error;
    if (!readString(json, "model_name", result.config.model_name, &type_error)) {
        return makeError(type_error.empty() ? "model_name is required" : type_error);
    }
    if (result.config.model_name.empty()) {
        return makeError("model_name is required");
    }
    if (!readString(json, "model_version", result.config.model_version, &type_error) &&
        !type_error.empty()) {
        return makeError(type_error);
    }
    if (const auto field_error = applyCommonFields(json, result.config, result.backend_provided)) {
        return makeError(*field_error);
    }

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

    if (const auto field_error = applyCommonFields(json, result.config, result.backend_provided)) {
        return makeError(*field_error);
    }
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

    std::string type_error;
    if (!readString(json, "version", result.version, &type_error)) {
        return makeError(type_error.empty() ? "version is required" : type_error);
    }
    if (result.version.empty()) {
        return makeError("version is required");
    }
    if (const auto field_error = applyCommonFields(json, result.config, result.backend_provided)) {
        return makeError(*field_error);
    }
    result.config.model_version = result.version;
    result.ok = true;
    return result;
}
