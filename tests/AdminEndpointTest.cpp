#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

namespace {

RuntimeConfig demoConfig() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return config;
}

HttpRequest adminRequest(const std::string &method, const std::string &path,
                         const std::string &body = {}) {
    HttpRequest request;
    request.method = method;
    request.path = path;
    request.body = body;
    return request;
}

} // namespace

TEST_CASE(admin_endpoint_lists_models) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(adminRequest("GET", "/v2/admin/models"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find("\"demo\"") != std::string::npos);
}

TEST_CASE(admin_endpoint_lists_available_backends) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(adminRequest("GET", "/v2/admin/models"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find("\"backends\"") != std::string::npos);
    REQUIRE(response.body.find("\"stub\"") != std::string::npos);
}

TEST_CASE(admin_endpoint_loads_second_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(adminRequest(
        "POST", "/v2/admin/models/load", R"({"model_name":"second","backend":"stub"})"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(registry.ready("demo"));
    REQUIRE(registry.ready("second"));
    REQUIRE_EQ(registry.listModels().size(), 2);
}

TEST_CASE(admin_endpoint_failed_load_returns_409_and_drops_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(adminRequest(
        "POST", "/v2/admin/models/load", R"({"model_name":"bad","backend":"does_not_exist"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("unsupported backend") != std::string::npos);
    REQUIRE_EQ(registry.listModels().size(), 1);
    REQUIRE(registry.ready("demo"));
}

TEST_CASE(admin_endpoint_reload_with_empty_body_keeps_model_config) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    REQUIRE(
        runtime
            .handle(adminRequest(
                "POST", "/v2/admin/models/load",
                R"({"model_name":"second","backend":"stub","model_path":"/models/x","instances":2})"))
            .status == 200);

    const auto response = runtime.handle(adminRequest("POST", "/v2/admin/models/second/reload"));
    REQUIRE_EQ(response.status, 200);

    const auto config = registry.modelConfig("second");
    REQUIRE(config.has_value());
    REQUIRE_EQ(config->backend, "stub");
    REQUIRE_EQ(config->model_path, "/models/x");
    REQUIRE_EQ(config->instances, static_cast<size_t>(2));
}

TEST_CASE(admin_endpoint_unloads_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    REQUIRE(runtime
                .handle(adminRequest("POST", "/v2/admin/models/load",
                                     R"({"model_name":"second","backend":"stub"})"))
                .status == 200);

    const auto response = runtime.handle(adminRequest("DELETE", "/v2/admin/models/second"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(registry.ready("demo"));
    REQUIRE(!registry.ready("second"));
}

TEST_CASE(admin_endpoint_reloads_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/demo/reload", "{}"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(registry.ready("demo"));
}

TEST_CASE(admin_endpoint_activates_version) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/demo/versions/2/activate", R"({"version":"2"})"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE_EQ(registry.defaultVersion("demo"), "2");
}
