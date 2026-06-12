#include "KServeV2Codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;

InferenceParseResult invalid(std::string message) {
    InferenceParseResult result;
    result.error_message = std::move(message);
    return result;
}

const TensorMetadata *findTensor(const std::vector<TensorMetadata> &tensors,
                                 const std::string &name) {
    const auto found = std::find_if(tensors.begin(), tensors.end(),
                                    [&name](const auto &tensor) { return tensor.name == name; });
    if (found == tensors.end()) {
        return nullptr;
    }
    return &*found;
}

bool containsName(const std::vector<std::string> &names, const std::string &name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool shapeMatches(const Json &shape, const TensorMetadata &metadata) {
    if (!shape.is_array() || shape.size() != metadata.shape.size()) {
        return false;
    }
    for (size_t i = 0; i < metadata.shape.size(); ++i) {
        if (!shape[i].is_number_integer()) {
            return false;
        }
        const auto dimension = shape[i].get<int64_t>();
        if (dimension < 0 || dimension != metadata.shape[i]) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> parseShape(const Json &shape) {
    std::vector<int64_t> parsed;
    parsed.reserve(shape.size());
    for (const auto &dimension : shape) {
        parsed.push_back(dimension.get<int64_t>());
    }
    return parsed;
}

// Parses a JSON number array straight into the tensor's typed byte buffer.
std::optional<std::vector<std::byte>> parseData(const Json &data, const std::string &datatype) {
    std::vector<std::byte> parsed;
    bool ok = true;
    const bool known = withTensorElementType(datatype, [&](auto element) {
        using T = decltype(element);
        parsed.resize(data.size() * sizeof(T));
        size_t index = 0;
        for (const auto &value : data) {
            if (!value.is_number()) {
                ok = false;
                return;
            }
            const auto typed = static_cast<T>(value.get<double>());
            std::memcpy(parsed.data() + (index * sizeof(T)), &typed, sizeof(T));
            ++index;
        }
    });
    if (!known || !ok) {
        return std::nullopt;
    }
    return parsed;
}

std::optional<std::vector<std::string>> parseStringData(const Json &data) {
    std::vector<std::string> parsed;
    parsed.reserve(data.size());
    for (const auto &value : data) {
        if (!value.is_string()) {
            return std::nullopt;
        }
        parsed.push_back(value.get<std::string>());
    }
    return parsed;
}

size_t tensorElementCount(const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

std::optional<LlmGenerationParams> parseLlmParameters(const Json &parameters) {
    if (!parameters.is_object()) {
        return std::nullopt;
    }

    LlmGenerationParams params;
    if (parameters.contains("max_tokens")) {
        if (!parameters["max_tokens"].is_number_unsigned()) {
            return std::nullopt;
        }
        params.max_tokens = parameters["max_tokens"].get<size_t>();
    }
    if (parameters.contains("temperature")) {
        if (!parameters["temperature"].is_number()) {
            return std::nullopt;
        }
        params.temperature = parameters["temperature"].get<double>();
    }
    if (parameters.contains("top_p")) {
        if (!parameters["top_p"].is_number()) {
            return std::nullopt;
        }
        params.top_p = parameters["top_p"].get<double>();
    }
    if (parameters.contains("top_k")) {
        if (!parameters["top_k"].is_number_unsigned()) {
            return std::nullopt;
        }
        params.top_k = parameters["top_k"].get<size_t>();
    }
    if (parameters.contains("stream")) {
        if (!parameters["stream"].is_boolean()) {
            return std::nullopt;
        }
        params.streaming = parameters["stream"].get<bool>();
    }
    return params;
}

std::vector<std::string> allOutputNames(const ModelMetadata &metadata) {
    std::vector<std::string> outputs;
    outputs.reserve(metadata.outputs.size());
    for (const auto &output : metadata.outputs) {
        outputs.push_back(output.name);
    }
    return outputs;
}

Json tensorMetadataJson(const TensorMetadata &tensor) {
    Json shape = Json::array();
    for (const auto dimension : tensor.shape) {
        shape.push_back(dimension);
    }
    return Json{{"name", tensor.name}, {"datatype", tensor.datatype}, {"shape", shape}};
}

Json binaryOutputTensorJson(const OutputTensor &tensor, std::string &blob) {
    Json shape = Json::array();
    for (const auto dimension : tensor.shape) {
        shape.push_back(dimension);
    }

    Json node{{"name", tensor.name}, {"datatype", tensor.datatype}, {"shape", shape}};
    if (isBytesDatatype(tensor.datatype)) {
        Json data = Json::array();
        for (const auto &value : tensor.string_data) {
            data.push_back(value);
        }
        node["data"] = std::move(data);
        return node;
    }

    node["parameters"]["binary_data_size"] = tensor.bytes.size();
    blob.append(reinterpret_cast<const char *>(tensor.bytes.data()), tensor.bytes.size());
    return node;
}

long binaryDataSize(const Json &node) {
    if (!node.contains("parameters") || !node["parameters"].is_object()) {
        return -1;
    }
    const auto &params = node["parameters"];
    if (!params.contains("binary_data_size") || !params["binary_data_size"].is_number_integer()) {
        return -1;
    }
    return params["binary_data_size"].get<long>();
}

std::vector<std::byte> sliceBlob(const std::string &blob, size_t offset, size_t size) {
    if (offset > blob.size() || size > blob.size() - offset) {
        throw std::runtime_error("binary tensor data exceeds request body");
    }
    std::vector<std::byte> bytes(size);
    if (size != 0) {
        std::memcpy(bytes.data(), blob.data() + offset, size);
    }
    return bytes;
}

Json outputTensorJson(const OutputTensor &tensor) {
    Json shape = Json::array();
    for (const auto dimension : tensor.shape) {
        shape.push_back(dimension);
    }
    Json data = Json::array();
    if (isBytesDatatype(tensor.datatype)) {
        for (const auto &value : tensor.string_data) {
            data.push_back(value);
        }
    } else {
        withTensorElementType(tensor.datatype, [&](auto element) {
            using T = decltype(element);
            const size_t count = tensor.bytes.size() / sizeof(T);
            for (size_t i = 0; i < count; ++i) {
                data.push_back(static_cast<double>(tensorScalarAt<T>(tensor.bytes, i)));
            }
        });
    }
    return Json{
        {"name", tensor.name}, {"datatype", tensor.datatype}, {"shape", shape}, {"data", data}};
}

} // namespace

InferenceParseResult parseInferenceRequest(const std::string &body, const ModelMetadata &metadata) {
    Json root;
    try {
        root = Json::parse(body);
    } catch (const Json::parse_error &) {
        return invalid("invalid JSON request body");
    }
    return parseInferenceRequest(root.dump(), metadata, root.dump().size());
}

InferenceParseResult parseInferenceRequest(const std::string &body, const ModelMetadata &metadata,
                                           size_t header_length) {
    if (header_length > body.size()) {
        return invalid("binary inference header exceeds request body");
    }

    Json root;
    try {
        root = Json::parse(body.substr(0, header_length));
    } catch (const Json::parse_error &) {
        return invalid("invalid JSON request body");
    }
    const std::string binary_blob = body.substr(header_length);
    size_t binary_offset = 0;

    if (!root.is_object()) {
        return invalid("inference request body must be a JSON object");
    }

    InferenceRequest request;
    if (root.contains("id")) {
        if (!root["id"].is_string()) {
            return invalid("request id must be a string");
        }
        request.id = root["id"].get<std::string>();
    }

    if (!root.contains("inputs") || !root["inputs"].is_array()) {
        return invalid("inputs must be an array");
    }
    if (root["inputs"].empty()) {
        return invalid("inputs must not be empty");
    }

    std::vector<std::string> seen_input_names;
    seen_input_names.reserve(root["inputs"].size());
    for (const auto &input : root["inputs"]) {
        if (!input.is_object()) {
            return invalid("each input must be an object");
        }
        if (!input.contains("name") || !input["name"].is_string()) {
            return invalid("input name must be a string");
        }
        if (!input.contains("datatype") || !input["datatype"].is_string()) {
            return invalid("input datatype must be a string");
        }
        if (!input.contains("shape")) {
            return invalid("input shape is required");
        }
        const auto binary_size = binaryDataSize(input);
        const bool has_binary_data = binary_size >= 0;
        if (!has_binary_data && (!input.contains("data") || !input["data"].is_array())) {
            return invalid("input data must be an array");
        }
        if (has_binary_data && binary_size < 0) {
            return invalid("invalid binary data size");
        }

        const auto name = input["name"].get<std::string>();
        if (containsName(seen_input_names, name)) {
            return invalid("duplicate input: " + name);
        }
        const auto *metadata_input = findTensor(metadata.inputs, name);
        if (metadata_input == nullptr) {
            return invalid("unknown input: " + name);
        }
        seen_input_names.push_back(name);
        const auto datatype = input["datatype"].get<std::string>();
        if (datatype != metadata_input->datatype) {
            return invalid("unsupported datatype for input: " + name);
        }
        if (!shapeMatches(input["shape"], *metadata_input)) {
            return invalid("invalid shape for input: " + name);
        }

        if (!shapeMatches(input["shape"], *metadata_input)) {
            return invalid("invalid shape for input: " + name);
        }

        const auto parsed_shape = parseShape(input["shape"]);
        const auto expected_elements = tensorElementCount(parsed_shape);

        InputTensor tensor;
        tensor.name = name;
        tensor.datatype = datatype;
        tensor.shape = parsed_shape;

        if (isBytesDatatype(datatype)) {
            if (has_binary_data) {
                return invalid("binary BYTES input is not supported: " + name);
            }
            auto string_data = parseStringData(input["data"]);
            if (!string_data.has_value()) {
                return invalid("input data values must be strings for BYTES input: " + name);
            }
            if (!string_data->empty() && expected_elements != 0 &&
                string_data->size() != expected_elements) {
                return invalid("input data element count mismatch for input: " + name);
            }
            tensor.string_data = std::move(*string_data);
        } else if (has_binary_data) {
            const auto byte_count = static_cast<size_t>(binary_size);
            try {
                tensor.bytes = sliceBlob(binary_blob, binary_offset, byte_count);
            } catch (const std::runtime_error &error) {
                return invalid(error.what());
            }
            binary_offset += byte_count;
            const auto parsed_elements = tensorElementCount(datatype, tensor.bytes);
            if (expected_elements != 0 && parsed_elements != expected_elements) {
                return invalid("input data element count mismatch for input: " + name);
            }
        } else {
            auto data = parseData(input["data"], datatype);
            if (!data.has_value()) {
                return invalid("input data values must be numbers for input: " + name);
            }
            const auto parsed_elements = tensorElementCount(datatype, *data);
            if (!data->empty() && expected_elements != 0 && parsed_elements != expected_elements) {
                return invalid("input data element count mismatch for input: " + name);
            }
            tensor.bytes = std::move(*data);
        }
        request.inputs.push_back(std::move(tensor));
    }

    if (binary_offset != binary_blob.size()) {
        return invalid("unused binary tensor data in request body");
    }

    for (const auto &expected_input : metadata.inputs) {
        if (!containsName(seen_input_names, expected_input.name)) {
            return invalid("missing required input: " + expected_input.name);
        }
    }

    if (root.contains("outputs")) {
        if (!root["outputs"].is_array()) {
            return invalid("outputs must be an array");
        }
        for (const auto &output : root["outputs"]) {
            if (!output.is_object()) {
                return invalid("each output must be an object");
            }
            if (!output.contains("name") || !output["name"].is_string()) {
                return invalid("output name must be a string");
            }
            const auto name = output["name"].get<std::string>();
            if (findTensor(metadata.outputs, name) == nullptr) {
                return invalid("unknown output: " + name);
            }
            request.requested_outputs.push_back(name);
        }
    } else {
        request.requested_outputs = allOutputNames(metadata);
    }

    if (root.contains("parameters")) {
        const auto params = parseLlmParameters(root["parameters"]);
        if (!params.has_value()) {
            return invalid("invalid generation parameters");
        }
        request.llm_params = *params;
    }

    InferenceParseResult result;
    result.ok = true;
    result.request = std::move(request);
    return result;
}

std::string modelMetadataJson(const ModelMetadata &metadata) {
    Json inputs = Json::array();
    for (const auto &input : metadata.inputs) {
        inputs.push_back(tensorMetadataJson(input));
    }
    Json outputs = Json::array();
    for (const auto &output : metadata.outputs) {
        outputs.push_back(tensorMetadataJson(output));
    }
    return Json{{"name", metadata.name},
                {"versions", metadata.versions},
                {"platform", metadata.platform},
                {"inputs", inputs},
                {"outputs", outputs}}
        .dump();
}

std::string inferenceResponseJson(const std::string &model_name, const std::string &model_version,
                                  const InferenceRequest &request,
                                  const ExecutionResponse &response) {
    Json body{{"model_name", model_name}, {"model_version", model_version}};
    if (request.id.has_value()) {
        body["id"] = *request.id;
    }

    Json outputs = Json::array();
    for (const auto &output : response.outputs) {
        outputs.push_back(outputTensorJson(output));
    }
    body["outputs"] = outputs;
    return body.dump();
}

BinaryInferenceResponse inferenceResponseBinary(const std::string &model_name,
                                                const std::string &model_version,
                                                const InferenceRequest &request,
                                                const ExecutionResponse &response) {
    Json header{{"model_name", model_name}, {"model_version", model_version}};
    if (request.id.has_value()) {
        header["id"] = *request.id;
    }

    std::string blob;
    Json outputs = Json::array();
    for (const auto &output : response.outputs) {
        outputs.push_back(binaryOutputTensorJson(output, blob));
    }
    header["outputs"] = outputs;

    BinaryInferenceResponse framed;
    framed.body = header.dump();
    framed.header_length = framed.body.size();
    framed.body += blob;
    return framed;
}
