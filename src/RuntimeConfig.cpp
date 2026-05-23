#include "RuntimeConfig.hpp"

#include <stdexcept>
#include <string>

namespace {

std::string requireValue(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
    }
    ++index;
    return argv[index];
}

} // namespace

RuntimeConfig parseRuntimeConfig(int argc, char** argv) {
    RuntimeConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--host") {
            config.host = requireValue(i, argc, argv, arg);
        } else if (arg == "--port") {
            config.port = std::stoi(requireValue(i, argc, argv, arg));
        } else if (arg == "--model-name") {
            config.model_name = requireValue(i, argc, argv, arg);
        } else if (arg == "--model-path") {
            config.model_path = requireValue(i, argc, argv, arg);
        } else if (arg == "--backend") {
            config.backend = requireValue(i, argc, argv, arg);
        } else if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument(
                "usage: neuriplo-kserve-runtime [--host 0.0.0.0] [--port 8080] "
                "[--model-name demo] [--model-path path] [--backend stub]");
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

    return config;
}

