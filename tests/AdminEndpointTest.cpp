#include "AdminCodec.hpp"
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
#include <utility>

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

// E-4: the request body's "version" (required by parseSwitchVersionRequest)
// disagreeing with the URL's version must 400 naming both, rather than
// silently activating whichever one KServeRuntime happens to use.
TEST_CASE(admin_endpoint_activate_rejects_mismatched_body_version) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/demo/versions/2/activate", R"({"version":"3"})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("2") != std::string::npos);
    REQUIRE(response.body.find("3") != std::string::npos);
    REQUIRE(registry.ready("demo"));
    REQUIRE(registry.defaultVersion("demo") != "3");
}

// B-3 + B-12: a failed admin reload of an already-loaded model must leave it
// untouched and Ready, and the 409 must surface why rather than a generic
// message.
TEST_CASE(admin_endpoint_failed_reload_keeps_model_ready_and_surfaces_error) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    REQUIRE(registry.ready("demo"));

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/demo/reload", R"({"backend":"does_not_exist"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("unsupported backend") != std::string::npos);
    REQUIRE(registry.ready("demo"));
}

// B-3: a failed version activate of an already-loaded model must leave the
// currently-active version untouched and Ready.
TEST_CASE(admin_endpoint_failed_version_activate_keeps_current_version_ready) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    REQUIRE(registry.ready("demo"));
    const auto before_version = registry.defaultVersion("demo");

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/demo/versions/2/activate",
                                    R"({"version":"2","backend":"does_not_exist"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.defaultVersion("demo"), before_version);
}

// B-2: a known numeric field with the wrong JSON type must 400 naming the
// field instead of being silently ignored (which would load with
// instances == 1 while the caller believed they asked for 2).
TEST_CASE(admin_endpoint_load_rejects_wrong_typed_instances) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/load",
                                    R"({"model_name":"second","backend":"stub","instances":"2"})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("instances") != std::string::npos);
    REQUIRE(!registry.modelConfig("second").has_value());
}

TEST_CASE(admin_endpoint_load_rejects_wrong_typed_use_gpu) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/load",
                     R"({"model_name":"second","backend":"stub","use_gpu":"true"})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("use_gpu") != std::string::npos);
    REQUIRE(!registry.modelConfig("second").has_value());
}

// B-2: instances < 1 must 400 rather than reach ModelLifecycle, which cannot
// build a scheduler with zero executors.
TEST_CASE(admin_endpoint_load_rejects_zero_instances) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/load",
                                    R"({"model_name":"second","backend":"stub","instances":0})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("instances") != std::string::npos);
    REQUIRE(!registry.modelConfig("second").has_value());
}

// B-2: instances above the named upper bound must 400 rather than let a
// client ask the server to spin up an unbounded number of executors.
TEST_CASE(admin_endpoint_load_rejects_instances_above_upper_bound) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/load",
                                    R"({"model_name":"second","backend":"stub","instances":65})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("instances") != std::string::npos);
    REQUIRE(!registry.modelConfig("second").has_value());
}

// B-2: an explicit but empty backend is as unusable as a missing one, and
// must 400 rather than reach a backend lookup with an empty id.
TEST_CASE(admin_endpoint_load_rejects_empty_backend) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/load", R"({"model_name":"second","backend":""})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("backend") != std::string::npos);
    REQUIRE(!registry.modelConfig("second").has_value());
}

// B-2: mirrors the CLI's own validation -- dynamic batching needs room to
// batch more than one request.
TEST_CASE(admin_endpoint_load_rejects_dynamic_batching_with_batch_size_below_two) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response =
        runtime.handle(adminRequest("POST", "/v2/admin/models/load",
                                    R"({"model_name":"second","backend":"stub",)"
                                    R"("dynamic_batching_enabled":true,"max_batch_size":1})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("max_batch_size") != std::string::npos);
}

TEST_CASE(admin_endpoint_load_rejects_negative_request_timeout) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/load",
                     R"({"model_name":"second","backend":"stub","request_timeout_ms":-1})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("request_timeout_ms") != std::string::npos);
}

TEST_CASE(admin_endpoint_load_rejects_negative_max_queue_delay) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/load",
                     R"({"model_name":"second","backend":"stub","max_queue_delay_us":-1})"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("max_queue_delay_us") != std::string::npos);
}

// B-4 (restored admin semantics): loading an already-loaded name is a 409
// telling the caller to use reload, not a silent success and not a dropped
// slot.
TEST_CASE(admin_endpoint_load_of_already_loaded_model_returns_409) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);
    REQUIRE(registry.ready("demo"));

    const auto response = runtime.handle(
        adminRequest("POST", "/v2/admin/models/load", R"({"model_name":"demo","backend":"stub"})"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("already loaded") != std::string::npos);
    REQUIRE(registry.ready("demo"));
}

// B-7: an unload while a load/reload is mid-build is "try again", not
// "no such model".
TEST_CASE(admin_endpoint_delete_during_reload_in_progress_returns_409_not_404) {
    MetricsRegistry metrics;
    ModelRegistry registry(demoConfig());
    KServeRuntime runtime(registry, metrics);

    auto gate = std::make_shared<std::pair<std::mutex, std::condition_variable>>();
    bool build_started = false;
    bool release = false;
    std::thread reloader([&registry, &gate, &build_started, &release] {
        RuntimeConfig config = *registry.modelConfig("demo");
        registry.reload(
            "demo", config,
            [&](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
                (void)error;
                {
                    std::lock_guard<std::mutex> lock(gate->first);
                    build_started = true;
                }
                gate->second.notify_all();
                std::unique_lock<std::mutex> lock(gate->first);
                gate->second.wait(lock, [&] { return release; });
                return createExecutorFor(cfg.backend, cfg, error);
            });
    });

    {
        std::unique_lock<std::mutex> lock(gate->first);
        REQUIRE(
            gate->second.wait_for(lock, std::chrono::seconds(2), [&] { return build_started; }));
    }

    const auto response = runtime.handle(adminRequest("DELETE", "/v2/admin/models/demo"));
    REQUIRE_EQ(response.status, 409);
    REQUIRE(response.body.find("in progress") != std::string::npos);

    {
        std::lock_guard<std::mutex> lock(gate->first);
        release = true;
    }
    gate->second.notify_all();
    reloader.join();
    REQUIRE(registry.ready("demo"));
}

// A-7: an ensemble load body that does not itself ask for batching must not
// inherit a server-wide dynamic-batching default (CLI flags, or a prior
// model's config used as `defaults` here) -- it would otherwise fail to load
// for a reason the request never mentioned.
TEST_CASE(ensemble_load_does_not_inherit_dynamic_batching_default) {
    RuntimeConfig defaults;
    defaults.backend = "stub";
    defaults.dynamic_batching_enabled = true;
    defaults.max_batch_size = 8;

    const auto parsed =
        parseLoadModelRequest(std::string(R"({"model_name":"ens","backend":")") +
                                  pipelineBackendId() + R"(","pipeline_graph":{"steps":[]}})",
                              defaults);
    REQUIRE(parsed.ok);
    REQUIRE(!parsed.config.dynamic_batching_enabled);
    REQUIRE_EQ(parsed.config.max_batch_size, static_cast<size_t>(1));
}

// Only an explicit per-model request for batching on an ensemble is still
// rejected (downstream, by PipelineExecutor) -- the codec must not silently
// force it off once the request itself names the field.
TEST_CASE(ensemble_load_keeps_explicit_batching_request_for_rejection) {
    RuntimeConfig defaults;
    defaults.backend = "stub";

    const auto parsed = parseLoadModelRequest(
        std::string(R"({"model_name":"ens","backend":")") + pipelineBackendId() +
            R"(","pipeline_graph":{"steps":[]},"max_batch_size":4})",
        defaults);
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.config.max_batch_size, static_cast<size_t>(4));
}
