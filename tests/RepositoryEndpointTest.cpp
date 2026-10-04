#include "BackendRegistry.hpp"
#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "PipelineExecutor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

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

// B-1: tritonclient's load() call sends "{}", not an empty string. A name
// that is neither already loaded nor in the repository catalog has nothing
// to fall back to, so it must be rejected the same way an empty body is,
// rather than silently loading against RuntimeConfig's stub defaults.
TEST_CASE(repository_load_of_unknown_model_with_empty_object_body_returns_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/typo/load", "{}"));
    REQUIRE_EQ(response.status, 404);
    REQUIRE(!registry.ready("typo"));
    REQUIRE_EQ(registry.listModels().size(), 1);
}

// B-3 + B-12: a failed reload of an already-loaded model must leave it
// untouched and Ready (Triton semantics), and the 409 must say why it
// failed rather than a generic message.
TEST_CASE(repository_failed_reload_of_known_model_keeps_it_serving) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    REQUIRE(registry.ready("demo"));

    const auto response = runtime.handle(repositoryRequest(
        "POST", "/v2/repository/models/demo/load", R"({"backend":"does_not_exist"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("unsupported backend") != std::string::npos);

    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.listModels().size(), 1);
    const auto infer = runtime.handle(repositoryRequest(
        "POST", "/v2/models/demo/infer",
        R"({"id":"t1","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})"));
    REQUIRE_EQ(infer.status, 200);
}

namespace {
// Rendezvous for the gated factory below, local to this file's concurrency
// tests (mirrors ModelRegistryTest.cpp's BuildGate, which this file does not
// link against).
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool build_started = false;
    bool release = false;
};

void waitForBuildStarted(const std::shared_ptr<Gate> &gate) {
    std::unique_lock<std::mutex> lock(gate->mutex);
    REQUIRE(gate->cv.wait_for(lock, std::chrono::seconds(2), [&] { return gate->build_started; }));
}

void releaseGate(const std::shared_ptr<Gate> &gate) {
    {
        std::lock_guard<std::mutex> lock(gate->mutex);
        gate->release = true;
    }
    gate->cv.notify_all();
}

// Polls loadWaitersForTesting() until `expected` callers are parked on
// load_cv_, with a bounded timeout. Without this, releasing the gate races
// the second HTTP request reaching its own wait: if the release (and the
// build it unblocks) wins, the second request instead finds an
// already-settled slot and falls through to an ordinary reload instead of
// deterministically hitting the placeholder-join path this test is for.
void waitForLoadWaiters(const ModelRegistry &registry, std::size_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (registry.loadWaitersForTesting() != expected) {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
} // namespace

// B-17 at the route the finding names: a second /v2/repository/.../load for
// a name whose first build is still in flight must join that build (and
// succeed once it resolves) instead of 409ing on a Loading placeholder. The
// first build here is triggered directly on the registry (so the test
// controls exactly when it unblocks); the request under test is the second
// one, through the real HTTP endpoint.
TEST_CASE(repository_concurrent_load_of_still_loading_model_joins_the_build) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    auto gate = std::make_shared<Gate>();
    RuntimeConfig slow_config = demoConfig();
    slow_config.model_name = "slow";

    std::thread loader([&registry, slow_config, gate] {
        registry.loadModel(
            slow_config,
            [gate](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
                {
                    std::lock_guard<std::mutex> lock(gate->mutex);
                    gate->build_started = true;
                }
                gate->cv.notify_all();
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->cv.wait(lock, [&] { return gate->release; });
                lock.unlock();
                return createExecutorFor(cfg.backend, cfg, error);
            });
    });
    waitForBuildStarted(gate);

    bool second_ok = false;
    std::thread second([&runtime, &second_ok] {
        const auto response =
            runtime.handle(repositoryRequest("POST", "/v2/repository/models/slow/load"));
        second_ok = (response.status == 200);
    });
    waitForLoadWaiters(registry, 1);

    releaseGate(gate);
    loader.join();
    second.join();

    REQUIRE(second_ok);
    REQUIRE(registry.ready("slow"));
}

// B-7: unloading a name whose first load is still in flight is "try again",
// not "no such model".
TEST_CASE(repository_unload_of_a_still_loading_model_returns_409_not_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    auto gate = std::make_shared<Gate>();
    RuntimeConfig slow_config = demoConfig();
    slow_config.model_name = "slow";

    std::thread loader([&registry, slow_config, gate] {
        registry.loadModel(
            slow_config,
            [gate](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
                {
                    std::lock_guard<std::mutex> lock(gate->mutex);
                    gate->build_started = true;
                }
                gate->cv.notify_all();
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->cv.wait(lock, [&] { return gate->release; });
                lock.unlock();
                return createExecutorFor(cfg.backend, cfg, error);
            });
    });
    waitForBuildStarted(gate);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/models/slow/unload"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("in progress") != std::string::npos);

    releaseGate(gate);
    loader.join();
    REQUIRE(registry.ready("slow"));
}

// P2-B2 B-4: explicit control mode starts (and may return to) empty by
// design; the server must be ready to accept load requests rather than
// forever 503.
TEST_CASE(explicit_control_mode_empty_registry_is_ready) {
    MetricsRegistry metrics;
    ModelRegistry registry(std::vector<RuntimeConfig>{}, /*explicit_control_mode=*/true);
    KServeRuntime runtime(registry, metrics);
    REQUIRE_EQ(runtime.handle(repositoryRequest("GET", "/v2/health/ready")).status, 200);
}

// P2-B2 B-5: a Failed model in repository mode is reported via
// /v2/repository/index (not readiness); a good model alongside it keeps the
// server ready.
TEST_CASE(repository_mode_failed_model_does_not_block_readiness) {
    MetricsRegistry metrics;
    RuntimeConfig bad;
    bad.model_name = "bad";
    bad.backend = "no_such_backend";
    RuntimeConfig good;
    good.model_name = "good";
    good.backend = "stub";
    ModelRegistry registry(std::vector<RuntimeConfig>{bad, good});
    KServeRuntime runtime(registry, metrics);

    REQUIRE(!registry.ready("bad"));
    REQUIRE(registry.ready("good"));
    REQUIRE_EQ(runtime.handle(repositoryRequest("GET", "/v2/health/ready")).status, 200);

    const auto index = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    const auto bad_pos = index.body.find("\"name\":\"bad\"");
    REQUIRE(bad_pos != std::string::npos);
    // The KServe model-repository extension has no FAILED state: a failed
    // load reports UNAVAILABLE, same as a catalogued-but-unloaded model --
    // the reason field is what actually explains why.
    REQUIRE(index.body.find("\"state\":\"UNAVAILABLE\"", bad_pos) != std::string::npos);
    const auto reason_key = index.body.find("\"reason\":\"", bad_pos);
    REQUIRE(reason_key != std::string::npos);
    const auto reason_value_start = reason_key + std::string("\"reason\":\"").size();
    REQUIRE(reason_value_start < index.body.size());
    REQUIRE(index.body[reason_value_start] != '"'); // non-empty reason string
}

// P2-B2 B-5: single-model mode keeps strict readiness -- there is only one
// model, so its failure is the server's failure.
TEST_CASE(single_model_mode_failed_model_blocks_readiness) {
    MetricsRegistry metrics;
    RuntimeConfig bad;
    bad.model_name = "bad";
    bad.backend = "no_such_backend";
    ModelRegistry registry(bad);
    KServeRuntime runtime(registry, metrics);
    REQUIRE_EQ(runtime.handle(repositoryRequest("GET", "/v2/health/ready")).status, 503);
}

// P2-B2 B-7: explicit mode's empty-vector construction must still honor the
// CLI's log_payloads/tokens_per_char, not silently default them.
TEST_CASE(explicit_control_mode_honors_log_payloads_and_tokens_per_char) {
    RuntimeConfig defaults;
    defaults.log_payloads = true;
    defaults.tokens_per_char = 0.5;
    ModelRegistry registry(std::vector<RuntimeConfig>{}, /*explicit_control_mode=*/true, defaults);
    REQUIRE(registry.logPayloads());
    REQUIRE_EQ(registry.tokensPerChar(), 0.5);
}

// P2-B2 B-10: {"ready":true} filters the index down to currently-serving
// models.
TEST_CASE(repository_index_ready_filter_returns_only_ready_entries) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    RuntimeConfig other = demoConfig();
    other.model_name = "other";
    registry.setRepositoryCatalog({other});
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/index", R"({"ready":true})"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find("\"name\":\"demo\"") != std::string::npos);
    REQUIRE(response.body.find("\"name\":\"other\"") == std::string::npos);
}

// P2-B2 B-10: a malformed body is a client error, not a silently-ignored
// filter.
TEST_CASE(repository_index_rejects_invalid_json_body) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    const auto response =
        runtime.handle(repositoryRequest("POST", "/v2/repository/index", "{not json"));
    REQUIRE_EQ(response.status, 400);
}

// P2-B2 B-20: the action is checked before the method -- an unknown action is
// a 404 regardless of what method was used, not a 405.
TEST_CASE(repository_unknown_action_returns_404_for_any_method) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    const auto response =
        runtime.handle(repositoryRequest("GET", "/v2/repository/models/demo/frobnicate"));
    REQUIRE_EQ(response.status, 404);
}

// P2-B2 B-20: a known action with the wrong method is METHOD_NOT_ALLOWED, not
// INVALID_ARGUMENT.
TEST_CASE(repository_load_wrong_method_uses_method_not_allowed_code) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    const auto response =
        runtime.handle(repositoryRequest("GET", "/v2/repository/models/demo/load"));
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find("METHOD_NOT_ALLOWED") != std::string::npos);
}

// C5: each model's metric lines must carry its own version label, not a
// single global one -- a repository serving models at different versions
// would otherwise mislabel every model but one.
TEST_CASE(metrics_render_uses_each_models_own_version_label) {
    MetricsRegistry metrics;
    metrics.setSchedulerMetrics("a", "1", {});
    metrics.setSchedulerMetrics("b", "2", {});
    metrics.recordInferRequest("a", "POST", 200, 0, 0, 0);
    metrics.recordInferRequest("b", "POST", 200, 0, 0, 0);

    const auto body = metrics.renderMetrics();
    // Label keys render alphabetically: model, status, version.
    REQUIRE(body.find(R"(model="a",status="200",version="1")") != std::string::npos);
    REQUIRE(body.find(R"(model="b",status="200",version="2")") != std::string::npos);
    // The bug this guards against: "a" mislabeled with "b"'s (or the global)
    // version.
    REQUIRE(body.find(R"(model="a",status="200",version="2")") == std::string::npos);
}

// C6: a model name reaching the index is not guaranteed to be valid UTF-8
// (it can come from a repository directory name or an admin load body); the
// index must not throw -- it replaces invalid sequences and still returns
// 200.
TEST_CASE(repository_index_does_not_throw_on_non_utf8_model_name) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    std::string bad_name = "bad";
    bad_name.push_back(static_cast<char>(0xFF));
    bad_name.push_back(static_cast<char>(0xFE));

    const auto load = runtime.handle(repositoryRequest(
        "POST", "/v2/repository/models/" + bad_name + "/load", R"({"backend":"stub"})"));
    REQUIRE_EQ(load.status, 200);

    const auto index = runtime.handle(repositoryRequest("POST", "/v2/repository/index"));
    REQUIRE_EQ(index.status, 200);
}

// F2: a one-model repository (or explicit mode with one model loaded) has
// models.size() == 1 just like genuine single-model mode, but the global
// metrics.setModelVersion() label is the CLI/default version, not
// necessarily this model's own -- metricsPage() must use the model's own
// version here, not fall back to the global label the way single-model mode
// correctly does.
TEST_CASE(metrics_page_one_model_repository_uses_the_models_own_version) {
    RuntimeConfig config = demoConfig();
    config.model_version = "3";
    config.model_version_explicit = true;
    ModelRegistry registry(std::vector<RuntimeConfig>{config});
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.listModels().size(), static_cast<size_t>(1));

    MetricsRegistry metrics;
    // The CLI/default version, deliberately different from the model's own
    // "3", so a mislabel is unmistakable.
    metrics.setModelVersion("1");
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(repositoryRequest("GET", "/metrics"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"(model="demo",version="3")") != std::string::npos);
    REQUIRE(response.body.find(R"(model="demo",version="1")") == std::string::npos);
}
