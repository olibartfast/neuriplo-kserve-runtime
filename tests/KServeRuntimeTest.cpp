#include "KServeRuntime.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <string>
#include <utility>

namespace {

KServeRuntime makeRuntime() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return KServeRuntime(ModelRegistry(config));
}

HttpResponse request(const KServeRuntime &runtime, std::string method, std::string path) {
    HttpRequest req;
    req.method = std::move(method);
    req.path = std::move(path);
    return runtime.handle(req);
}

} // namespace

TEST_CASE(kserve_runtime_reports_live_and_ready) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/live").status, 200);
    REQUIRE_EQ(request(runtime, "GET", "/v2/health/ready").status, 200);
}

TEST_CASE(kserve_runtime_returns_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"demo")") != std::string::npos);
    REQUIRE(response.body.find(R"("platform":"neuriplo_stub")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_model) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/missing").status, 404);
}

TEST_CASE(kserve_runtime_handles_placeholder_infer) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_server_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"neuriplo-kserve-runtime")") != std::string::npos);
    REQUIRE(response.body.find(R"("version":"0.1.0")") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/ready");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_empty_method) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "", "/v2").status, 400);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo");
    REQUIRE_EQ(response.status, 404);
    REQUIRE(response.body.find(R"("code":"NOT_FOUND")") != std::string::npos);
}
