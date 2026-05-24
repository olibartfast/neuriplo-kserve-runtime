#include "RuntimeConfig.hpp"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>

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
        } else if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument(
                "usage: neuriplo-kserve-runtime [--host 0.0.0.0] [--port 8080] "
                "[--max-request-bytes 67108864] [--model-name demo] [--model-path path] "
                "[--backend stub]");
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

    return config;
}
