#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace {

bool hasFlag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

void printUsage(std::ostream &out) {
    out << "usage: neuriplo-kserve-runtime [--host 0.0.0.0] [--port 8080] "
           "[--max-request-bytes 67108864] [--model-name demo] [--model-path path] "
           "[--backend stub]"
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    if (hasFlag(argc, argv, "--version")) {
        std::cout << "neuriplo-kserve-runtime 0.1.0" << '\n';
        return 0;
    }
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(std::cout);
        return 0;
    }

    try {
        const auto config = parseRuntimeConfig(argc, argv);
        ModelRegistry registry(config);
        KServeRuntime runtime(std::move(registry));
        HttpServer server(
            config.host, config.port,
            [&runtime](const HttpRequest &request) { return runtime.handle(request); },
            config.max_request_bytes);
        server.run();
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
