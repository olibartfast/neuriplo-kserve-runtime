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

ModelMetadata smallMetadata() {
    ModelMetadata model;
    model.name = "demo";
    model.versions = {"1"};
    model.platform = "test";
    model.inputs.push_back({"input", "FP32", {1, 3}});
    model.outputs.push_back({"output", "FP32", {1, 1}});
    return model;
}

ModelMetadata multiInputMetadata() {
    ModelMetadata model;
    model.name = "demo";
    model.versions = {"1"};
    model.platform = "test";
    model.inputs.push_back({"image", "FP32", {1, 3}});
    model.inputs.push_back({"scale", "FP32", {1}});
    model.outputs.push_back({"output", "FP32", {1, 1}});
    return model;
}

ModelMetadata llmMetadata() {
    ModelMetadata model;
    model.name = "llm-demo";
    model.versions = {"1"};
    model.platform = "test";
    model.inputs.push_back({"prompt", "BYTES", {1}});
    model.outputs.push_back({"text", "BYTES", {1}});
    return model;
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

TEST_CASE(kserve_v2_codec_preserves_input_tensor_data) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[1.25,2.5,3.75]}]})",
        smallMetadata());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.inputs[0].name, "input");
    REQUIRE_EQ(parsed.request.inputs[0].datatype, "FP32");
    REQUIRE_EQ(parsed.request.inputs[0].shape.size(), static_cast<size_t>(2));
    REQUIRE_EQ(parsed.request.inputs[0].shape[0], 1);
    REQUIRE_EQ(parsed.request.inputs[0].shape[1], 3);
    REQUIRE_EQ(parsed.request.inputs[0].data.size(), static_cast<size_t>(3));
    REQUIRE_EQ(parsed.request.inputs[0].data[0], 1.25);
    REQUIRE_EQ(parsed.request.inputs[0].data[1], 2.5);
    REQUIRE_EQ(parsed.request.inputs[0].data[2], 3.75);
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

TEST_CASE(kserve_v2_codec_rejects_duplicate_input) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[1,2,3]},)"
        R"({"name":"input","shape":[1,3],"datatype":"FP32","data":[4,5,6]}]})",
        smallMetadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("duplicate input") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_rejects_missing_required_model_input) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"image","shape":[1,3],"datatype":"FP32","data":[1,2,3]}]})",
        multiInputMetadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("missing required input: scale") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_rejects_non_numeric_input_data) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","data":[1,"bad",3]}]})",
        smallMetadata());
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("input data values") != std::string::npos);
}

TEST_CASE(kserve_v2_codec_parses_bytes_prompt_input) {
    const auto parsed = parseInferenceRequest(
        R"({"inputs":[{"name":"prompt","shape":[1],"datatype":"BYTES","data":["Explain KServe briefly."]}],)"
        R"("parameters":{"max_tokens":128,"temperature":0.7}})",
        llmMetadata());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.inputs[0].datatype, "BYTES");
    REQUIRE_EQ(parsed.request.inputs[0].string_data.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.inputs[0].string_data[0], "Explain KServe briefly.");
    REQUIRE(parsed.request.llm_params.has_value());
    REQUIRE(parsed.request.llm_params->max_tokens.has_value());
    REQUIRE_EQ(*parsed.request.llm_params->max_tokens, static_cast<size_t>(128));
    REQUIRE(parsed.request.llm_params->temperature.has_value());
    REQUIRE_EQ(*parsed.request.llm_params->temperature, 0.7);
}

TEST_CASE(kserve_v2_codec_serializes_bytes_output) {
    InferenceRequest request;
    request.id = "req-bytes";
    request.requested_outputs = {"text"};

    ExecutionResponse response;
    OutputTensor output;
    output.name = "text";
    output.datatype = "BYTES";
    output.shape = {1};
    output.string_data = {"generated text"};
    response.outputs.push_back(std::move(output));

    const auto json = inferenceResponseJson("llm-demo", "1", request, response);
    REQUIRE(json.find(R"("datatype":"BYTES")") != std::string::npos);
    REQUIRE(json.find(R"("data":["generated text"])") != std::string::npos);
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
    execution_request.inputs = parsed.request.inputs;
    execution_request.requested_outputs = parsed.request.requested_outputs;
    const auto execution_response = executor->infer(execution_request);

    const auto response = inferenceResponseJson("demo", "1", parsed.request, execution_response);
    REQUIRE(response.find(R"("model_name":"demo")") != std::string::npos);
    REQUIRE(response.find(R"("model_version":"1")") != std::string::npos);
    REQUIRE(response.find(R"("id":"request-1")") != std::string::npos);
    REQUIRE(response.find(R"("shape":[1,1000])") != std::string::npos);
}
