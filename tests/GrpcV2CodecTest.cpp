#ifdef NEURIPLO_RUNTIME_WITH_GRPC

#include "GrpcV2Codec.hpp"
#include "Test.hpp"

#include "kserve_grpc.pb.h"

#include <cstring>
#include <string>

namespace {

inference::ModelInferRequest makeProtoRequest() {
    inference::ModelInferRequest proto;
    proto.set_model_name("demo");
    proto.set_model_version("1");
    proto.set_id("test-grpc-1");

    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(3);
    input->add_shape(224);
    input->add_shape(224);
    auto *contents = input->mutable_contents();
    contents->add_fp64_contents(0.5);
    contents->add_fp64_contents(0.25);

    auto *output = proto.add_outputs();
    output->set_name("output");

    return proto;
}

} // namespace

TEST_CASE(grpc_codec_converts_model_infer_request) {
    const auto proto = makeProtoRequest();
    auto exec_request = grpc_v2::convertInferRequest(proto);

    REQUIRE_EQ(exec_request.id.has_value(), true);
    REQUIRE_EQ(*exec_request.id, "test-grpc-1");
    REQUIRE_EQ(exec_request.inputs.size(), 1u);
    REQUIRE_EQ(exec_request.inputs[0].name, "input");
    REQUIRE_EQ(exec_request.inputs[0].datatype, "FP32");
    REQUIRE_EQ(exec_request.inputs[0].shape.size(), 4u);
    REQUIRE_EQ(exec_request.inputs[0].shape[0], 1);
    REQUIRE_EQ(exec_request.inputs[0].shape[1], 3);
    REQUIRE_EQ(exec_request.inputs[0].shape[2], 224);
    REQUIRE_EQ(exec_request.inputs[0].shape[3], 224);
    REQUIRE_EQ(exec_request.inputs[0].elementCount(), 2u);
    REQUIRE_EQ(tensorScalarAt<float>(exec_request.inputs[0].bytes, 0), 0.5f);
    REQUIRE_EQ(tensorScalarAt<float>(exec_request.inputs[0].bytes, 1), 0.25f);
    REQUIRE_EQ(exec_request.requested_outputs.size(), 1u);
    REQUIRE_EQ(exec_request.requested_outputs[0], "output");
}

TEST_CASE(grpc_codec_converts_request_without_id) {
    inference::ModelInferRequest proto;
    proto.set_model_name("test");
    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(1);

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.id.has_value(), false);
}

TEST_CASE(grpc_codec_converts_bytes_input) {
    inference::ModelInferRequest proto;
    proto.set_model_name("llm");
    auto *input = proto.add_inputs();
    input->set_name("prompt");
    input->set_datatype("BYTES");
    input->add_shape(1);
    auto *contents = input->mutable_contents();
    contents->add_bytes_contents("hello world");

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.inputs.size(), 1u);
    REQUIRE_EQ(exec_request.inputs[0].datatype, "BYTES");
    REQUIRE_EQ(exec_request.inputs[0].string_data.size(), 1u);
    REQUIRE_EQ(exec_request.inputs[0].string_data[0], "hello world");
}

TEST_CASE(grpc_codec_builds_infer_response) {
    ExecutionResponse exec_response;
    exec_response.ok = true;
    OutputTensor output;
    output.name = "output";
    output.datatype = "FP32";
    output.shape = {1, 1};
    output.bytes = tensorBytesFromDoubles(output.datatype, {42.0, 99.0});
    exec_response.outputs.push_back(output);

    const auto proto = grpc_v2::buildInferResponse(exec_response, "demo", "1",
                                                   std::make_optional(std::string("resp-1")));

    REQUIRE_EQ(proto.model_name(), "demo");
    REQUIRE_EQ(proto.model_version(), "1");
    REQUIRE_EQ(proto.id(), "resp-1");
    REQUIRE_EQ(proto.outputs_size(), 1);
    REQUIRE_EQ(proto.outputs(0).name(), "output");
    REQUIRE_EQ(proto.outputs(0).datatype(), "FP32");
    REQUIRE_EQ(proto.outputs(0).shape_size(), 2);
    REQUIRE_EQ(proto.outputs(0).shape(0), 1);
    REQUIRE_EQ(proto.outputs(0).shape(1), 1);
    REQUIRE_EQ(proto.outputs(0).contents().fp64_contents_size(), 0);
    REQUIRE_EQ(proto.raw_output_contents_size(), 1);
    const auto &raw = proto.raw_output_contents(0);
    REQUIRE_EQ(raw.size(), output.bytes.size());
    REQUIRE(std::memcmp(raw.data(), output.bytes.data(), raw.size()) == 0);
}

TEST_CASE(grpc_codec_builds_bytes_infer_response) {
    ExecutionResponse exec_response;
    exec_response.ok = true;
    OutputTensor output;
    output.name = "text";
    output.datatype = "BYTES";
    output.shape = {1};
    output.string_data = {"hello world"};
    exec_response.outputs.push_back(output);

    const auto proto = grpc_v2::buildInferResponse(exec_response, "llm", "1",
                                                   std::optional<std::string>{});

    REQUIRE_EQ(proto.outputs_size(), 1);
    REQUIRE_EQ(proto.outputs(0).datatype(), "BYTES");
    REQUIRE_EQ(proto.outputs(0).contents().bytes_contents_size(), 1);
    REQUIRE_EQ(proto.outputs(0).contents().bytes_contents(0), "hello world");
    REQUIRE_EQ(proto.raw_output_contents_size(), 0);
}

TEST_CASE(grpc_codec_builds_infer_response_without_id) {
    ExecutionResponse exec_response;
    exec_response.ok = true;
    const auto proto =
        grpc_v2::buildInferResponse(exec_response, "demo", "1", std::optional<std::string>{});

    REQUIRE(proto.id().empty());
}

TEST_CASE(grpc_codec_builds_response_with_llm_metadata) {
    ExecutionResponse exec_response;
    exec_response.ok = true;
    LlmResultMetadata llm_meta;
    llm_meta.prompt_tokens = 10;
    llm_meta.completion_tokens = 5;
    llm_meta.finish_reason = "stop";
    exec_response.llm_metadata = llm_meta;

    const auto proto =
        grpc_v2::buildInferResponse(exec_response, "llm", "1", std::optional<std::string>{});

    REQUIRE(proto.parameters().contains("prompt_tokens"));
    REQUIRE_EQ(proto.parameters().at("prompt_tokens").int64_param(), 10);
    REQUIRE(proto.parameters().contains("completion_tokens"));
    REQUIRE_EQ(proto.parameters().at("completion_tokens").int64_param(), 5);
    REQUIRE(proto.parameters().contains("finish_reason"));
    REQUIRE_EQ(proto.parameters().at("finish_reason").string_param(), "stop");
}

TEST_CASE(grpc_codec_converts_int32_input) {
    inference::ModelInferRequest proto;
    proto.set_model_name("test");
    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("INT32");
    input->add_shape(2);
    auto *contents = input->mutable_contents();
    contents->add_int_contents(10);
    contents->add_int_contents(20);

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.inputs.size(), 1u);
    REQUIRE_EQ(exec_request.inputs[0].elementCount(), 2u);
    REQUIRE_EQ(tensorScalarAt<int32_t>(exec_request.inputs[0].bytes, 0), 10);
    REQUIRE_EQ(tensorScalarAt<int32_t>(exec_request.inputs[0].bytes, 1), 20);
}

TEST_CASE(grpc_codec_converts_int64_input) {
    inference::ModelInferRequest proto;
    proto.set_model_name("test");
    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("INT64");
    input->add_shape(1);
    auto *contents = input->mutable_contents();
    contents->add_int64_contents(100);

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.inputs[0].elementCount(), 1u);
    REQUIRE_EQ(tensorScalarAt<int64_t>(exec_request.inputs[0].bytes, 0), static_cast<int64_t>(100));
}

TEST_CASE(grpc_codec_converts_fp32_input) {
    inference::ModelInferRequest proto;
    proto.set_model_name("test");
    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(1);
    auto *contents = input->mutable_contents();
    contents->add_fp32_contents(1.5f);

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.inputs[0].elementCount(), 1u);
    REQUIRE_EQ(tensorScalarAt<float>(exec_request.inputs[0].bytes, 0), 1.5f);
}

TEST_CASE(grpc_codec_prefers_raw_input_contents) {
    inference::ModelInferRequest proto;
    proto.set_model_name("test");
    auto *input = proto.add_inputs();
    input->set_name("input");
    input->set_datatype("FP32");
    input->add_shape(2);
    // Typed contents are also set; raw_input_contents must win for the input.
    input->mutable_contents()->add_fp32_contents(9.0f);
    const float values[2] = {1.5f, -2.0f};
    proto.add_raw_input_contents(
        std::string(reinterpret_cast<const char *>(values), sizeof(values)));

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.inputs[0].elementCount(), 2u);
    REQUIRE_EQ(tensorScalarAt<float>(exec_request.inputs[0].bytes, 0), 1.5f);
    REQUIRE_EQ(tensorScalarAt<float>(exec_request.inputs[0].bytes, 1), -2.0f);
}

TEST_CASE(grpc_codec_parses_llm_parameters) {
    inference::ModelInferRequest proto;
    proto.set_model_name("llm");
    auto *input = proto.add_inputs();
    input->set_name("prompt");
    input->set_datatype("BYTES");
    input->add_shape(1);

    (*proto.mutable_parameters())["max_tokens"].set_int64_param(128);
    (*proto.mutable_parameters())["temperature"].set_double_param(0.7);
    (*proto.mutable_parameters())["top_p"].set_double_param(0.9);
    (*proto.mutable_parameters())["top_k"].set_int64_param(40);
    (*proto.mutable_parameters())["stream"].set_bool_param(true);

    auto exec_request = grpc_v2::convertInferRequest(proto);
    REQUIRE_EQ(exec_request.llm_params.has_value(), true);
    REQUIRE_EQ(exec_request.llm_params->max_tokens.has_value(), true);
    REQUIRE_EQ(*exec_request.llm_params->max_tokens, 128u);
    REQUIRE_EQ(exec_request.llm_params->temperature.has_value(), true);
    REQUIRE_EQ(*exec_request.llm_params->temperature, 0.7);
    REQUIRE_EQ(exec_request.llm_params->top_p.has_value(), true);
    REQUIRE_EQ(*exec_request.llm_params->top_p, 0.9);
    REQUIRE_EQ(exec_request.llm_params->top_k.has_value(), true);
    REQUIRE_EQ(*exec_request.llm_params->top_k, 40u);
    REQUIRE_EQ(exec_request.llm_params->streaming, true);
}

#endif // NEURIPLO_RUNTIME_WITH_GRPC
