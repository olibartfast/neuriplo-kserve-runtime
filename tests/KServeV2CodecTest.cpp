#include "KServeV2Codec.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <cstddef>
#include <string>

namespace {

ModelMetadata metadata() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    const ModelRegistry registry(config);
    return *registry.find("demo");
}

std::string validBody(std::string extra = "") {
    std::string body =
        "{\"id\":\"request-1\",\"inputs\":[{\"name\":\"input\",\"shape\":[1,3,224,224],"
        "\"datatype\":\"FP32\",\"data\":[]}]";
    if (!extra.empty()) {
        body += ',' + extra;
    }
    body += '}';
    return body;
}

RuntimeConfig stubConfig() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return config;
}

} // namespace

TEST_CASE(kserve_v2_codec_parses_valid_inference_request) {
    const auto parsed = parseInferenceRequest(validBody(), metadata());
    REQUIRE(parsed.ok);
    REQUIRE(parsed.request.id.has_value());
    REQUIRE_EQ(*parsed.request.id, "request-1");
    REQUIRE_EQ(parsed.request.requested_outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.requested_outputs[0], "output");
}

TEST_CASE(kserve_v2_codec_parses_requested_outputs) {
    const auto parsed =
        parseInferenceRequest(validBody(R"("outputs":[{"name":"output"}])"), metadata());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.requested_outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.requested_outputs[0], "output");
}

TEST_CASE(kserve_v2_codec_rejects_malformed_json) {
    const auto parsed = parseInferenceRequest("{", metadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("invalid JSON") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_rejects_missing_inputs) {
    const auto parsed = parseInferenceRequest(R"({"id":"request-1"})", metadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("inputs") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_rejects_unknown_input) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"missing","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})",
        metadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("unknown input") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_serializes_executor_response_with_id) {
    const auto model = metadata();
    const auto parsed = parseInferenceRequest(validBody(), model);
    REQUIRE(parsed.ok);

    std::string error;
    const auto executor = makeStubExecutor(stubConfig(), error);
    REQUIRE(executor != nullptr);
    ExecutionRequest execution_request;
    execution_request.id = parsed.request.id;
    execution_request.requested_outputs = parsed.request.requested_outputs;
    const auto execution_response = executor->infer(execution_request);

    const auto response = inferenceResponseJson("demo", "1", parsed.request, execution_response);
    REQUIRE(response.find(R"("model_name":"demo")") != std::string::npos);
    REQUIRE(response.find(R"("model_version":"1")") != std::string::npos);
    REQUIRE(response.find(R"("id":"request-1")") != std::string::npos);
    REQUIRE(response.find(R"("shape":[1,1000])") != std::string::npos);
}
