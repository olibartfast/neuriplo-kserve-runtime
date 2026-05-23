#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"

#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto config = parseRuntimeConfig(argc, argv);
        ModelRegistry registry(config);
        KServeRuntime runtime(std::move(registry));
        HttpServer server(config.host, config.port, [&runtime](const HttpRequest& request) {
            return runtime.handle(request);
        });
        server.run();
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

