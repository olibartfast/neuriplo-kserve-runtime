#include "KServeV2Codec.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

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

ModelMetadata dynamicImageMetadata() {
    ModelMetadata model;
    model.name = "demo";
    model.versions = {"1"};
    model.platform = "test";
    model.inputs.push_back({"IMAGE", "UINT8", {1, -1}});
    model.outputs.push_back({"output", "FP32", {1, 1}});
    return model;
}

std::string imageBody(const std::string &shape, const std::string &data) {
    return R"({"inputs":[{"name":"IMAGE","shape":)" + shape + R"(,"datatype":"UINT8","data":)" +
           data + "}]}";
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
    REQUIRE_EQ(parsed.request.inputs[0].elementCount(), static_cast<size_t>(3));
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 0), 1.25f);
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 1), 2.5f);
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 2), 3.75f);
}

TEST_CASE(kserve_v2_codec_parses_http_binary_input) {
    float values[] = {1.25f, 2.5f, 3.75f};
    std::string header =
        R"({"inputs":[{"name":"input","shape":[1,3],"datatype":"FP32","parameters":{"binary_data_size":12}}]})";
    std::string body = header;
    body.append(reinterpret_cast<const char *>(values), sizeof(values));

    const auto parsed = parseInferenceRequest(body, smallMetadata(), header.size());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.inputs[0].elementCount(), static_cast<size_t>(3));
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 0), 1.25f);
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 1), 2.5f);
    REQUIRE_EQ(tensorScalarAt<float>(parsed.request.inputs[0].bytes, 2), 3.75f);
}

TEST_CASE(kserve_v2_codec_serializes_http_binary_output) {
    InferenceRequest request;
    request.id = "req-bin";
    request.requested_outputs = {"output"};

    ExecutionResponse response;
    OutputTensor output;
    output.name = "output";
    output.datatype = "FP32";
    output.shape = {1, 2};
    output.bytes.resize(sizeof(float) * 2);
    const float values[] = {4.0f, 5.0f};
    std::memcpy(output.bytes.data(), values, sizeof(values));
    response.outputs.push_back(std::move(output));

    const auto framed = inferenceResponseBinary("demo", "1", request, response);
    REQUIRE(framed.header_length < framed.body.size());
    const auto header = framed.body.substr(0, framed.header_length);
    REQUIRE(header.find(R"("binary_data_size":8)") != std::string::npos);
    REQUIRE_EQ(framed.body.size() - framed.header_length, sizeof(values));
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

TEST_CASE(kserve_v2_codec_dynamic_dim_accepts_json_data) {
    const auto parsed =
        parseInferenceRequest(imageBody("[1,5]", "[1,2,3,4,5]"), dynamicImageMetadata());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE((parsed.request.inputs[0].shape == std::vector<int64_t>{1, 5}));
}

TEST_CASE(kserve_v2_codec_dynamic_dim_accepts_binary_extension) {
    const std::string header =
        R"({"inputs":[{"name":"IMAGE","shape":[1,5],"datatype":"UINT8","parameters":{"binary_data_size":5}}]})";
    std::string body = header;
    body.append("\x01\x02\x03\x04\x05", 5);

    const auto parsed = parseInferenceRequest(body, dynamicImageMetadata(), header.size());
    REQUIRE(parsed.ok);
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE((parsed.request.inputs[0].shape == std::vector<int64_t>{1, 5}));
    REQUIRE_EQ(parsed.request.inputs[0].bytes.size(), static_cast<size_t>(5));
}

TEST_CASE(kserve_v2_codec_dynamic_dim_is_not_bound_by_first_request) {
    const auto metadata = dynamicImageMetadata();
    const auto first = parseInferenceRequest(imageBody("[1,3]", "[1,2,3]"), metadata);
    REQUIRE(first.ok);
    REQUIRE((first.request.inputs[0].shape == std::vector<int64_t>{1, 3}));
    const auto second = parseInferenceRequest(imageBody("[1,7]", "[1,2,3,4,5,6,7]"), metadata);
    REQUIRE(second.ok);
    REQUIRE((second.request.inputs[0].shape == std::vector<int64_t>{1, 7}));
}

TEST_CASE(kserve_v2_codec_dynamic_dim_rejects_bad_requests) {
    const auto metadata = dynamicImageMetadata();
    const std::string shape_error = "invalid shape for input: IMAGE";
    const std::vector<std::pair<std::string, std::string>> rows = {
        {imageBody("[5]", "[1,2,3,4,5]"), shape_error},
        {imageBody("[1,5,1]", "[1,2,3,4,5]"), shape_error},
        {imageBody("[1,-1]", "[1,2,3,4,5]"), shape_error},
        {imageBody("[1]", "[1]"), shape_error},
        {imageBody("[1,5.5]", "[1,2,3,4,5]"), shape_error},
        {imageBody(R"([1,"5"])", "[1,2,3,4,5]"), shape_error},
        {imageBody("[2,5]", "[1,2,3,4,5,6,7,8,9,10]"), shape_error},
        {imageBody("[1,5]", "[1,2,3,4]"), "input data element count mismatch for input: IMAGE"},
    };
    for (const auto &row : rows) {
        const auto parsed = parseInferenceRequest(row.first, metadata);
        REQUIRE(!parsed.ok);
        REQUIRE(parsed.error_message.find(row.second) != std::string::npos);
    }
}

TEST_CASE(kserve_v2_codec_concrete_dims_stay_strict) {
    ModelMetadata model = dynamicImageMetadata();
    model.inputs[0].shape = {1, 3};
    const auto parsed = parseInferenceRequest(imageBody("[1,4]", "[1,2,3,4]"), model);
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("invalid shape for input: IMAGE") != std::string::npos);
    REQUIRE(parseInferenceRequest(imageBody("[1,3]", "[1,2,3]"), model).ok);
}

// A dynamic leading axis accepts any declared extent, including one so large
// that the shape's element-count product overflows size_t. The previous
// unchecked product sometimes wrapped around to exactly zero, which the
// "expected_elements != 0" bypass (there to tolerate a genuine zero-extent
// shape) then read as nothing to check at all, instead of a request to
// reject outright.
TEST_CASE(kserve_v2_codec_rejects_an_overflowing_shape_product) {
    ModelMetadata model;
    model.name = "demo";
    model.versions = {"1"};
    model.platform = "test";
    model.inputs.push_back({"images", "FP32", {-1, 3, 640, 640}});
    model.outputs.push_back({"output", "FP32", {1, 1}});

    // 2^50 * 3 * 640 * 640 overflows a 64-bit size_t.
    const std::string body = R"({"inputs":[{"name":"images","shape":[1125899906842624,3,640,640],)"
                             R"("datatype":"FP32","data":[]}]})";
    const auto parsed = parseInferenceRequest(body, model);
    REQUIRE(!parsed.ok);
    REQUIRE(parsed.error_message.find("images") != std::string::npos);
}
