#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "PipelineExecutor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

namespace {

RuntimeConfig demoConfig() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return config;
}

HttpRequest repositoryRequest(const std::string &method, const std::string &path,
                              const std::string &body = {}) {
    HttpRequest request;
    request.method = method;
    request.path = path;
    request.body = body;
    return request;
}

// A one-step ensemble over the stub model the registry starts with.
std::string demoEnsembleGraph() {
    return R"({"steps": [{"kind": "model", "name": "detect", "model_name": "demo"}]})";
}

} // namespace

TEST_CASE(repository_index_returns_json_array_with_spec_fields) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE_EQ(response.status, 200);
    // The client parses the body as a top-level array; an object would throw.
    REQUIRE_EQ(response.body.front(), '[');
    REQUIRE(response.body.find("\"name\":\"demo\"") != std::string::npos);
    REQUIRE(response.body.find("\"state\":\"READY\"") != std::string::npos);
    REQUIRE(response.body.find("\"version\"") != std::string::npos);
    REQUIRE(response.body.find("\"reason\"") != std::string::npos);
}

TEST_CASE(repository_index_rejects_get) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(repositoryRequest("GET", "/v2/repository/index"));
    REQUIRE_EQ(response.status, 405);
}

TEST_CASE(repository_loads_unknown_model_with_config_in_body) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        repositoryRequest("POST", "/v2/repository/models/second/load", R"({"backend":"stub"})"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(registry.ready("second"));
    REQUIRE_EQ(registry.listModels().size(), 2);
}

// A bare-name load resolves against the repository catalog. A name that is
// neither loaded nor in the catalog is a 404, not a malformed request.
TEST_CASE(repository_load_of_model_absent_from_catalog_returns_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/absent/load"));
    REQUIRE_EQ(response.status, 404);
    REQUIRE(response.body.find("not in the repository") != std::string::npos);
    REQUIRE_EQ(registry.listModels().size(), 1);
}

// Explicit control mode: the catalog is populated but nothing is loaded, and a
// bare-name load brings the model up with no config in the request body.
TEST_CASE(repository_bare_name_load_resolves_against_the_catalog) {
    MetricsRegistry metrics;
    ModelRegistry registry(std::vector<RuntimeConfig>{});
    RuntimeConfig available;
    available.model_name = "catalogued";
    available.backend = "stub";
    available.model_version = "4";
    available.model_version_explicit = true;
    registry.setRepositoryCatalog({available});
    KServeRuntime runtime(registry, metrics);

    REQUIRE_EQ(registry.listModels().size(), 0);

    // Listed as available before it is loaded, so a client can discover it.
    const auto before = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE(before.body.find("\"name\":\"catalogued\"") != std::string::npos);
    REQUIRE(before.body.find("\"state\":\"UNAVAILABLE\"") != std::string::npos);
    REQUIRE(before.body.find("\"version\":\"4\"") != std::string::npos);

    const auto load =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/catalogued/load"));
    REQUIRE_EQ(load.status, 200);
    REQUIRE(registry.ready("catalogued"));
    REQUIRE_EQ(registry.defaultVersion("catalogued").value(), std::string("4"));

    const auto after = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE(after.body.find("\"state\":\"READY\"") != std::string::npos);
}

// Unloading returns the model to the catalog rather than erasing it: an
// explicit-mode client must be able to load it again afterwards.
TEST_CASE(repository_unloaded_model_stays_listed_and_can_be_reloaded) {
    MetricsRegistry metrics;
    ModelRegistry registry(std::vector<RuntimeConfig>{});
    RuntimeConfig available;
    available.model_name = "catalogued";
    available.backend = "stub";
    registry.setRepositoryCatalog({available});
    KServeRuntime runtime(registry, metrics);

    REQUIRE_EQ(
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/catalogued/load")).status,
        200);
    REQUIRE_EQ(
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/catalogued/unload")).status,
        200);
    REQUIRE_EQ(registry.listModels().size(), 0);

    const auto index = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE(index.body.find("\"name\":\"catalogued\"") != std::string::npos);

    REQUIRE_EQ(
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/catalogued/load")).status,
        200);
    REQUIRE(registry.ready("catalogued"));
}

TEST_CASE(repository_load_of_known_model_reloads_from_stored_config) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    // An empty body on a known model means "reload as configured", so the model
    // must survive with its backend intact rather than reset to a default.
    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo/load"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.listModels().size(), 1);
    REQUIRE_EQ(registry.modelConfig("demo").value().backend, std::string("stub"));
}

TEST_CASE(repository_failed_load_returns_409_and_drops_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(repositoryRequest("POST", "/v2/repository/models/bad/load",
                                                           R"({"backend":"does_not_exist"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE_EQ(registry.listModels().size(), 1);
}

TEST_CASE(repository_unloads_model) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo/unload"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE_EQ(registry.listModels().size(), 0);
}

TEST_CASE(repository_unload_of_unknown_model_returns_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/absent/unload"));
    REQUIRE_EQ(response.status, 404);
}

TEST_CASE(repository_unknown_action_returns_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo/frobnicate"));
    REQUIRE_EQ(response.status, 404);
}

// Ensembles are ordinary registry entries distinguished only by their backend
// id, so the repository surface must cover them with no special-casing.
TEST_CASE(repository_loads_and_indexes_an_ensemble) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const std::string body = std::string(R"({"backend":")") + pipelineBackendId() +
                             R"(","pipeline_graph":)" + demoEnsembleGraph() + "}";
    const auto load =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo_ensemble/load", body));
    REQUIRE_EQ(load.status, 200);
    REQUIRE(registry.ready("demo_ensemble"));

    const auto index = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE_EQ(index.status, 200);
    REQUIRE(index.body.find("\"name\":\"demo_ensemble\"") != std::string::npos);
    REQUIRE(index.body.find("\"state\":\"READY\"") != std::string::npos);
}

TEST_CASE(repository_unloads_an_ensemble) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const std::string body = std::string(R"({"backend":")") + pipelineBackendId() +
                             R"(","pipeline_graph":)" + demoEnsembleGraph() + "}";
    REQUIRE_EQ(
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo_ensemble/load", body))
            .status,
        200);

    const auto unload =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/demo_ensemble/unload"));
    REQUIRE_EQ(unload.status, 200);
    REQUIRE_EQ(registry.listModels().size(), 1);
    REQUIRE(registry.ready("demo"));
}
