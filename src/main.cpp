#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "Logging.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

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
           "[--backend stub] [--max-queue-size 64] [--request-timeout-ms 30000] "
           "[--instances 1] [--dynamic-batching-enabled false] [--max-batch-size 1] "
           "[--max-queue-delay-us 0] [--preferred-batch-sizes 2,4,8] "
           "[--log-payloads false]"
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
        MetricsRegistry metrics;
        ModelRegistry registry(config);
        KServeRuntime runtime(std::move(registry), metrics);

        auto &logger = defaultLogger();
        LogEvent startup;
        startup.severity = "info";
        startup.model = config.model_name;
        startup.backend = config.backend;
        startup.message = "runtime starting";
        logger.event(startup);

        metrics.recordModelLoadSuccess(config.model_name, config.backend);

        HttpServer server(
            config.host, config.port,
            [&runtime, &metrics](const HttpRequest &request) {
                auto response = runtime.handle(request);
                return response;
            },
            config.max_request_bytes);

        server.run();
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
