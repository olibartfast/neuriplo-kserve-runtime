#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "Logging.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "ModelRepository.hpp"
#include "RuntimeConfig.hpp"
#include "RuntimeVersion.hpp"

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
           "[--models path | --model-repository path] [--model-control-mode none] "
           "[--backend stub] [--use-gpu false] [--plugin-dir path] [--max-queue-size 64] "
           "[--request-timeout-ms "
           "30000] "
           "[--instances 1] [--dynamic-batching-enabled false] [--max-batch-size 1] "
           "[--max-queue-delay-us 0] [--preferred-batch-sizes 2,4,8] "
           "[--log-payloads false]"
        << '\n';
}

} // namespace

int main(int argc, char **argv) {
    if (hasFlag(argc, argv, "--version")) {
        std::cout << "neuriplo-kserve-runtime " << neuriplo_runtime::kVersion << '\n';
        return 0;
    }
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(std::cout);
        return 0;
    }

    try {
        const auto config = parseRuntimeConfig(argc, argv);
        MetricsRegistry metrics;
        if (config.model_repository.empty()) {
            // Single-model mode: config.model_version IS the one model's
            // version. In repository mode this global label would mislabel
            // every discovered model with whichever one's version happened
            // to be set last; each model's own version is attached to its
            // own metric lines instead (KServeRuntime::metricsPage() calls
            // setSchedulerMetrics(name, version, ...) per model, which
            // MetricsRegistry::renderMetrics() now reads per model).
            metrics.setModelVersion(config.model_version);
        }
        if (!config.deployment.empty()) {
            metrics.setDeployment(config.deployment);
        }
        auto &logger = defaultLogger();

        // Repository mode serves every model in the tree; single-model mode
        // stays the default so existing deployments are unaffected.
        std::optional<ModelRegistry> registry_storage;
        // Declared at this scope (not just inside the else-branch below) so
        // the per-model load metrics after the registry is built can still
        // report each discovered model's own outcome.
        std::vector<RuntimeConfig> discovered;
        if (config.model_repository.empty()) {
            registry_storage.emplace(config);
        } else {
            std::vector<std::string> warnings;
            discovered = scanModelRepository(config.model_repository, config, warnings);
            for (const auto &warning : warnings) {
                logger.warn(warning);
            }
            for (const auto &model : discovered) {
                logger.info("discovered model " + model.model_name + " version " +
                            model.model_version + " backend " + model.backend + " at " +
                            model.model_path);
            }
            if (config.model_control_mode == "explicit") {
                // Start empty and let the client choose. Nothing is loaded, so a
                // model whose backend crashes on load cannot prevent the server
                // from coming up at all. explicit_control_mode=true so an empty
                // registry still reports ready.
                registry_storage.emplace(std::vector<RuntimeConfig>{},
                                         /*explicit_control_mode=*/true, config);
                logger.info("explicit model control: " + std::to_string(discovered.size()) +
                            " model(s) available, none loaded; use POST "
                            "/v2/repository/models/<name>/load");
            } else {
                // Deliberately built from the scan even when it found nothing:
                // the server then reports not-ready instead of quietly serving
                // a stub.
                registry_storage.emplace(discovered, /*explicit_control_mode=*/false, config);
            }
            registry_storage->setRepositoryCatalog(discovered);
        }
        ModelRegistry &registry = *registry_storage;
        KServeRuntime runtime(registry, metrics);
        LogEvent startup;
        startup.severity = "info";
        if (config.model_repository.empty()) {
            // Single-model mode: there is exactly one model/backend to name.
            startup.model = config.model_name;
            startup.backend = config.backend;
        }
        // Repository mode has no single model/backend to report here -- the
        // per-model "discovered model ..." lines above (and the per-model load
        // metrics below) are the accurate record instead of a phantom
        // "demo"/"stub" entry that was never actually loaded.
        startup.message = "runtime starting";
        logger.event(startup);

        if (config.model_repository.empty()) {
            metrics.recordModelLoadSuccess(config.model_name, config.backend);
        } else if (config.model_control_mode != "explicit") {
            // Record each discovered model's own outcome and version, rather
            // than one phantom success metric for the top-level defaults
            // (which were never actually loaded in repository mode).
            for (const auto &model : discovered) {
                if (registry.ready(model.model_name)) {
                    metrics.recordModelLoadSuccess(model.model_name, model.backend);
                } else {
                    metrics.recordModelLoadFailure(model.model_name, model.backend);
                }
            }
        }
        // Explicit mode loads nothing at startup, so nothing is recorded here;
        // /v2/repository/models/<name>/load records its own outcome.

        HttpServer server(
            config.host, config.port,
            [&runtime, &metrics](const HttpRequest &request) {
                auto response = runtime.handle(request);
                return response;
            },
            config.max_request_bytes);

#ifdef NEURIPLO_RUNTIME_WITH_GRPC
        std::unique_ptr<grpc_v2::GrpcServer> grpc_server;
        std::thread grpc_thread;
        if (config.grpc_port > 0) {
            grpc_server = std::make_unique<grpc_v2::GrpcServer>(config.host, config.grpc_port,
                                                                registry, metrics);
            grpc_thread = std::thread([&grpc_server]() { grpc_server->run(); });
        }
#endif

        server.run();

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
