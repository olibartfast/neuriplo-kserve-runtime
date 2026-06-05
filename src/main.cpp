#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "Logging.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"

#ifdef NEURIPLO_RUNTIME_WITH_GRPC
#include "GrpcServer.hpp"
#endif

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
           "[--grpc-port 9000] [--max-request-bytes 67108864] [--model-name demo] [--model-path "
           "path] "
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
        metrics.setModelVersion(config.model_version);
        if (!config.deployment.empty()) {
            metrics.setDeployment(config.deployment);
        }
        ModelRegistry registry(config);
        KServeRuntime runtime(registry, metrics);

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

#ifdef NEURIPLO_RUNTIME_WITH_GRPC
        std::unique_ptr<grpc_v2::GrpcServer> grpc_server;
        std::thread grpc_thread;
        if (config.grpc_port > 0) {
            grpc_server = std::make_unique<grpc_v2::GrpcServer>(config.host, config.grpc_port,
                                                                 registry, metrics);
            grpc_thread = std::thread([&grpc_server]() { grpc_server->run(); });
        }
#endif
        // server.run() blocks until stopped; signal handling or explicit stop
        // would unblock it in a production setup.
        (void)server;

#ifdef NEURIPLO_RUNTIME_WITH_GRPC
        if (grpc_server) {
            grpc_server->stop();
        }
        if (grpc_thread.joinable()) {
            grpc_thread.join();
        }
#else
        if (config.grpc_port > 0) {
            std::cerr
                << "warning: --grpc-port specified but gRPC support is not compiled in; ignoring"
                << '\n';
        }
#endif
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
