#ifdef NEURIPLO_RUNTIME_WITH_GRPC

#include "GrpcV2Codec.hpp"

namespace grpc_v2 {

inference::ModelMetadataResponse buildModelMetadataResponse(const ModelMetadata &metadata) {
    inference::ModelMetadataResponse proto;
    proto.set_name(metadata.name);
    for (const auto &version : metadata.versions) {
        proto.add_versions(version);
    }
    proto.set_platform(metadata.platform);
    for (const auto &input : metadata.inputs) {
        auto *proto_input = proto.add_inputs();
        proto_input->set_name(input.name);
        proto_input->set_datatype(input.datatype);
        for (const auto dim : input.shape) {
            proto_input->add_shape(dim);
        }
    }
    for (const auto &output : metadata.outputs) {
        auto *proto_output = proto.add_outputs();
        proto_output->set_name(output.name);
        proto_output->set_datatype(output.datatype);
        for (const auto dim : output.shape) {
            proto_output->add_shape(dim);
        }
    }
    return proto;
}

namespace {

void fillBytesContents(inference::InferTensorContents *contents,
                       const std::vector<std::string> &string_data) {
    for (const auto &s : string_data) {
        contents->add_bytes_contents(s);
    }
}

// Appends proto repeated-field values to the tensor's typed byte buffer,
// converting each value to the tensor's declared datatype.
template <typename Repeated> void appendContents(InputTensor &tensor, const Repeated &values) {
    withTensorElementType(tensor.datatype, [&](auto element) {
        using T = decltype(element);
        size_t offset = tensor.bytes.size();
        tensor.bytes.resize(offset + (static_cast<size_t>(values.size()) * sizeof(T)));
        for (const auto value : values) {
            const auto typed = static_cast<T>(value);
            std::memcpy(tensor.bytes.data() + offset, &typed, sizeof(T));
            offset += sizeof(T);
        }
    });
}

} // namespace

ExecutionRequest convertInferRequest(const inference::ModelInferRequest &proto_request) {
    ExecutionRequest request;
    if (!proto_request.id().empty()) {
        request.id = proto_request.id();
    }

    int input_index = 0;
    for (const auto &proto_input : proto_request.inputs()) {
        InputTensor tensor;
        tensor.name = proto_input.name();
        tensor.datatype = proto_input.datatype();
        tensor.shape.assign(proto_input.shape().begin(), proto_input.shape().end());

        const auto &contents = proto_input.contents();
        if (isBytesDatatype(tensor.datatype)) {
            tensor.string_data.assign(contents.bytes_contents().begin(),
                                      contents.bytes_contents().end());
        } else if (input_index < proto_request.raw_input_contents_size()) {
            // raw_input_contents carries the tensor's bytes verbatim in
            // little-endian element order: one memcpy, no per-element work.
            const auto &raw = proto_request.raw_input_contents(input_index);
            tensor.bytes.resize(raw.size());
            std::memcpy(tensor.bytes.data(), raw.data(), raw.size());
        } else if (contents.fp64_contents_size() > 0) {
            appendContents(tensor, contents.fp64_contents());
        } else if (contents.fp32_contents_size() > 0) {
            appendContents(tensor, contents.fp32_contents());
        } else if (contents.int64_contents_size() > 0) {
            appendContents(tensor, contents.int64_contents());
        } else if (contents.int_contents_size() > 0) {
            appendContents(tensor, contents.int_contents());
        } else if (contents.uint64_contents_size() > 0) {
            appendContents(tensor, contents.uint64_contents());
        } else if (contents.uint_contents_size() > 0) {
            appendContents(tensor, contents.uint_contents());
        }
        request.inputs.push_back(std::move(tensor));
        ++input_index;
    }

    for (const auto &proto_output : proto_request.outputs()) {
        request.requested_outputs.push_back(proto_output.name());
    }

    if (!proto_request.parameters().empty()) {
        LlmGenerationParams params;
        auto it = proto_request.parameters().find("max_tokens");
        if (it != proto_request.parameters().end()) {
            params.max_tokens = static_cast<size_t>(it->second.int64_param());
        }
        it = proto_request.parameters().find("temperature");
        if (it != proto_request.parameters().end()) {
            params.temperature = it->second.double_param();
        }
        it = proto_request.parameters().find("top_p");
        if (it != proto_request.parameters().end()) {
            params.top_p = it->second.double_param();
        }
        it = proto_request.parameters().find("top_k");
        if (it != proto_request.parameters().end()) {
            params.top_k = static_cast<size_t>(it->second.int64_param());
        }
        it = proto_request.parameters().find("stream");
        if (it != proto_request.parameters().end()) {
            params.streaming = it->second.bool_param();
        }
        request.llm_params = params;
    }

    return request;
}

inference::ModelInferResponse buildInferResponse(const ExecutionResponse &exec_response,
                                                 const std::string &model_name,
                                                 const std::string &model_version,
                                                 const std::optional<std::string> &request_id) {
    inference::ModelInferResponse proto;
    proto.set_model_name(model_name);
    proto.set_model_version(model_version);
    if (request_id.has_value()) {
        proto.set_id(*request_id);
    }

    for (const auto &output : exec_response.outputs) {
        auto *proto_output = proto.add_outputs();
        proto_output->set_name(output.name);
        proto_output->set_datatype(output.datatype);
        for (const auto dim : output.shape) {
            proto_output->add_shape(dim);
        }
        if (isBytesDatatype(output.datatype)) {
            fillBytesContents(proto_output->mutable_contents(), output.string_data);
        } else {
            proto.add_raw_output_contents(std::string(
                reinterpret_cast<const char *>(output.bytes.data()), output.bytes.size()));
        }
    }

    if (exec_response.llm_metadata.has_value()) {
        (*proto.mutable_parameters())["prompt_tokens"].set_int64_param(
            static_cast<int64_t>(exec_response.llm_metadata->prompt_tokens));
        (*proto.mutable_parameters())["completion_tokens"].set_int64_param(
            static_cast<int64_t>(exec_response.llm_metadata->completion_tokens));
        (*proto.mutable_parameters())["finish_reason"].set_string_param(
            exec_response.llm_metadata->finish_reason);
    }

    return proto;
}

} // namespace grpc_v2

#endif // NEURIPLO_RUNTIME_WITH_GRPC
