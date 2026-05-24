#include "KServeRuntime.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <cstddef>
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

HttpResponse request(const KServeRuntime &runtime, std::string method, std::string path,
                     std::string body) {
    HttpRequest req;
    req.method = std::move(method);
    req.path = std::move(path);
    req.body = std::move(body);
    return runtime.handle(req);
}

std::string validInferBody(std::string id = "") {
    std::string body =
        "{\"inputs\":[{\"name\":\"input\",\"shape\":[1,3,224,224],\"datatype\":\"FP32\","
        "\"data\":[]}]";
    if (!id.empty()) {
        body += R"(,"id":")" + id + '"';
    }
    body += '}';
    return body;
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
    REQUIRE(response.body.find(R"("versions":["1"])") != std::string::npos);
    REQUIRE(response.body.find(R"("platform":"neuriplo_stub")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_model) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/missing").status, 404);
}

TEST_CASE(kserve_runtime_handles_placeholder_infer) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", validInferBody("abc"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_version":"1")") != std::string::npos);
    REQUIRE(response.body.find(R"("id":"abc")") != std::string::npos);
    REQUIRE(response.body.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_json) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/infer", "{");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_shape) {
    const auto runtime = makeRuntime();
    const auto response =
        request(runtime, "POST", "/v2/models/demo/infer",
                R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[]}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("invalid shape") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_invalid_infer_datatype) {
    const auto runtime = makeRuntime();
    const auto response = request(
        runtime, "POST", "/v2/models/demo/infer",
        R"({"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"INT32","data":[]}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("unsupported datatype") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_output_request) {
    const auto runtime = makeRuntime();
    const auto response = request(
        runtime, "POST", "/v2/models/demo/infer",
        R"({"inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}],"outputs":[{"name":"missing"}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("unknown output") != std::string::npos);
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

TEST_CASE(kserve_runtime_returns_versioned_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/versions/1");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("versions":["1"])") != std::string::npos);
}

TEST_CASE(kserve_runtime_returns_versioned_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/versions/1/ready");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(kserve_runtime_handles_versioned_infer) {
    const auto runtime = makeRuntime();
    const auto response =
        request(runtime, "POST", "/v2/models/demo/versions/1/infer", validInferBody());
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("model_version":"1")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_unknown_version) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "GET", "/v2/models/demo/versions/2").status, 404);
    REQUIRE_EQ(
        request(runtime, "POST", "/v2/models/demo/versions/2/infer", validInferBody()).status, 404);
}

TEST_CASE(kserve_runtime_rejects_empty_method) {
    const auto runtime = makeRuntime();
    REQUIRE_EQ(request(runtime, "", "/v2").status, 400);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_metadata) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_ready) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "POST", "/v2/models/demo/ready");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}

TEST_CASE(kserve_runtime_rejects_wrong_method_for_model_infer) {
    const auto runtime = makeRuntime();
    const auto response = request(runtime, "GET", "/v2/models/demo/infer");
    REQUIRE_EQ(response.status, 405);
    REQUIRE(response.body.find(R"("code":"METHOD_NOT_ALLOWED")") != std::string::npos);
}
