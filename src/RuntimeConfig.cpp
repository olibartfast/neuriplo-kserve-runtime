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

void applyDoubleEnvironmentDefault(double &target, const RuntimeEnvironment &environment,
                                   const std::string &name) {
    const auto value = environment.get(name);
    if (value.has_value() && !value->empty()) {
        target = std::stod(*value);
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

void parseSchedulerStrategy(const std::string &value) {
    if (value != "tensor" && value != "llm") {
        throw std::invalid_argument("scheduler-strategy must be tensor or llm");
    }
}

double parseTemperature(const std::string &value) {
    const auto parsed = std::stod(value);
    if (parsed < 0.0) {
        throw std::invalid_argument("temperature must be greater than or equal to 0");
    }
    return parsed;
}

double parseTopP(const std::string &value) {
    const auto parsed = std::stod(value);
    if (parsed <= 0.0 || parsed > 1.0) {
        throw std::invalid_argument("top-p must be in (0, 1]");
    }
    return parsed;
}

size_t parsePositiveSize(const std::string &value, const std::string &flag) {
    const auto parsed = static_cast<size_t>(std::stoull(value));
    if (parsed == 0) {
        throw std::invalid_argument(flag + " must be greater than 0");
    }
    return parsed;
}

size_t parseNonNegativeSize(const std::string &value, const std::string &flag) {
    (void)flag;
    return static_cast<size_t>(std::stoull(value));
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

double parseTokensPerChar(const std::string &value) {
    const auto parsed = std::stod(value);
    if (parsed <= 0.0 || parsed > 1.0) {
        throw std::invalid_argument("tokens-per-char must be in (0, 1]");
    }
    return parsed;
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
    applyDoubleEnvironmentDefault(config.tokens_per_char, environment, "TOKENS_PER_CHAR");
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
        } else if (arg == "--log-payloads") {
            config.log_payloads = parseBoolFlag(requireValue(i, argc, argv, arg), arg);
        } else if (arg == "--scheduler-strategy") {
            const auto value = requireValue(i, argc, argv, arg);
            parseSchedulerStrategy(value);
            config.scheduler_strategy = value;
        } else if (arg == "--context-length") {
            config.context_length =
                parsePositiveSize(requireValue(i, argc, argv, arg), "context-length");
        } else if (arg == "--kv-cache-slots") {
            config.kv_cache_slots =
                parsePositiveSize(requireValue(i, argc, argv, arg), "kv-cache-slots");
        } else if (arg == "--max-tokens") {
            config.max_tokens = parsePositiveSize(requireValue(i, argc, argv, arg), "max-tokens");
        } else if (arg == "--temperature") {
            config.temperature = parseTemperature(requireValue(i, argc, argv, arg));
        } else if (arg == "--top-p") {
            config.top_p = parseTopP(requireValue(i, argc, argv, arg));
        } else if (arg == "--top-k") {
            config.top_k = parseNonNegativeSize(requireValue(i, argc, argv, arg), "top-k");
        } else if (arg == "--streaming-enabled") {
            config.streaming_enabled = parseBoolFlag(requireValue(i, argc, argv, arg), arg);
        } else if (arg == "--tokens-per-char") {
            config.tokens_per_char = parseTokensPerChar(requireValue(i, argc, argv, arg));
        } else if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument(
                "usage: neuriplo-kserve-runtime [--host 0.0.0.0] [--port 8080] "
                "[--max-request-bytes 67108864] [--model-name demo] [--model-path path] "
                "[--backend stub] [--max-queue-size 64] [--request-timeout-ms 30000] "
                "[--instances 1] [--dynamic-batching-enabled false] [--max-batch-size 1] "
                "[--max-queue-delay-us 0] [--preferred-batch-sizes 2,4,8] "
                "[--log-payloads false] [--scheduler-strategy tensor] "
                "[--context-length 4096] [--kv-cache-slots 1] [--max-tokens 256] "
                "[--temperature 0.8] [--top-p 0.95] [--top-k 40] "
                "[--streaming-enabled false] [--tokens-per-char 0.25]");
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
    parseSchedulerStrategy(config.scheduler_strategy);

    return config;
}
