#include "RuntimeConfig.hpp"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <algorithm>

namespace {

std::string requireValue(int &index, int argc, char **argv, const std::string &flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
    }
    ++index;
    return argv[index];
}

void applyStringEnvironmentDefault(std::string &target, const RuntimeEnvironment &environment,
                                   const std::string &name) {
    const auto value = environment.get(name);
    if (value.has_value() && !value->empty()) {
        target = *value;
    }
}

void applySizeEnvironmentDefault(size_t &target, const RuntimeEnvironment &environment,
                                 const std::string &name) {
    const auto value = environment.get(name);
    if (value.has_value() && !value->empty()) {
        target = static_cast<size_t>(std::stoull(*value));
    }
}

RuntimeEnvironment processEnvironment() {
    return RuntimeEnvironment{
        [](const std::string &name) -> std::optional<std::string> {
            const char *value = std::getenv(name.c_str());
            if (value == nullptr) {
                return std::nullopt;
            }
            return std::string(value);
        },
        [](const std::string &path) { return std::filesystem::exists(path); },
    };
}

bool parseBoolFlag(const std::string &value, const std::string &flag) {
    if (value == "true" || value == "1") {
        return true;
    }
    if (value == "false" || value == "0") {
        return false;
    }
    throw std::invalid_argument(flag + " must be true or false");
}

std::vector<size_t> parsePreferredBatchSizes(const std::string &value) {
    if (value.empty()) {
        return {};
    }

    std::vector<size_t> sizes;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::invalid_argument("preferred batch sizes must not contain empty entries");
        }
        const auto parsed = static_cast<size_t>(std::stoull(token));
        if (parsed == 0) {
            throw std::invalid_argument("preferred batch sizes must be greater than 0");
        }
        sizes.push_back(parsed);
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

} // namespace

RuntimeConfig parseRuntimeConfig(int argc, char **argv) {
    return parseRuntimeConfig(argc, argv, processEnvironment());
}

RuntimeConfig parseRuntimeConfig(int argc, char **argv, const RuntimeEnvironment &environment) {
    RuntimeConfig config;

    applyStringEnvironmentDefault(config.model_name, environment, "MODEL_NAME");
    applyStringEnvironmentDefault(config.model_path, environment, "MODEL_PATH");
    applyStringEnvironmentDefault(config.backend, environment, "BACKEND");
    applyStringEnvironmentDefault(config.storage_uri, environment, "STORAGE_URI");
    applySizeEnvironmentDefault(config.max_request_bytes, environment, "MAX_REQUEST_BYTES");
    if (config.model_path.empty() && environment.pathExists("/mnt/models")) {
        config.model_path = "/mnt/models";
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host") {
            config.host = requireValue(i, argc, argv, arg);
        } else if (arg == "--port") {
            config.port = std::stoi(requireValue(i, argc, argv, arg));
        } else if (arg == "--max-request-bytes") {
            config.max_request_bytes =
                static_cast<size_t>(std::stoull(requireValue(i, argc, argv, arg)));
        } else if (arg == "--model-name") {
            config.model_name = requireValue(i, argc, argv, arg);
        } else if (arg == "--model-path") {
            config.model_path = requireValue(i, argc, argv, arg);
        } else if (arg == "--backend") {
            config.backend = requireValue(i, argc, argv, arg);
        } else if (arg == "--max-queue-size") {
            config.max_queue_size =
                static_cast<size_t>(std::stoull(requireValue(i, argc, argv, arg)));
        } else if (arg == "--request-timeout-ms") {
            config.request_timeout_ms = std::stoll(requireValue(i, argc, argv, arg));
        } else if (arg == "--instances") {
            config.instances = static_cast<size_t>(std::stoull(requireValue(i, argc, argv, arg)));
        } else if (arg == "--dynamic-batching-enabled") {
            config.dynamic_batching_enabled = parseBoolFlag(requireValue(i, argc, argv, arg), arg);
        } else if (arg == "--max-batch-size") {
            config.max_batch_size =
                static_cast<size_t>(std::stoull(requireValue(i, argc, argv, arg)));
        } else if (arg == "--max-queue-delay-us") {
            config.max_queue_delay_us = std::stoll(requireValue(i, argc, argv, arg));
        } else if (arg == "--preferred-batch-sizes") {
            config.preferred_batch_sizes =
                parsePreferredBatchSizes(requireValue(i, argc, argv, arg));
        } else if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument(
                "usage: neuriplo-kserve-runtime [--host 0.0.0.0] [--port 8080] "
                "[--max-request-bytes 67108864] [--model-name demo] [--model-path path] "
                "[--backend stub] [--max-queue-size 64] [--request-timeout-ms 30000] "
                "[--instances 1] [--dynamic-batching-enabled false] [--max-batch-size 1] "
                "[--max-queue-delay-us 0] [--preferred-batch-sizes 2,4,8]");
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (config.port <= 0 || config.port > 65535) {
        throw std::invalid_argument("port must be in range 1..65535");
    }
    if (config.model_name.empty()) {
        throw std::invalid_argument("model name must not be empty");
    }
    if (config.max_request_bytes == 0 ||
        config.max_request_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("max request bytes must be greater than 0");
    }
    if (config.max_queue_size == 0) {
        throw std::invalid_argument("max queue size must be greater than 0");
    }
    if (config.request_timeout_ms <= 0) {
        throw std::invalid_argument("request timeout must be greater than 0");
    }
    if (config.instances == 0) {
        throw std::invalid_argument("instances must be greater than 0");
    }
    if (config.max_batch_size == 0) {
        throw std::invalid_argument("max batch size must be greater than 0");
    }
    if (config.max_queue_delay_us < 0) {
        throw std::invalid_argument("max queue delay must be greater than or equal to 0");
    }
    if (config.dynamic_batching_enabled && config.max_batch_size < 2) {
        throw std::invalid_argument("max batch size must be at least 2 when dynamic batching "
                                    "is enabled");
    }

    return config;
}
