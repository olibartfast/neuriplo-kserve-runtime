#include "KServeV2Codec.hpp"

#include <algorithm>
#include <cstdint>
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

std::optional<std::vector<double>> parseData(const Json &data) {
    std::vector<double> parsed;
    parsed.reserve(data.size());
    for (const auto &value : data) {
        if (!value.is_number()) {
            return std::nullopt;
        }
        parsed.push_back(value.get<double>());
    }
    return parsed;
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

Json outputTensorJson(const OutputTensor &tensor) {
    Json shape = Json::array();
    for (const auto dimension : tensor.shape) {
        shape.push_back(dimension);
    }
    Json data = Json::array();
    for (const auto value : tensor.data) {
        data.push_back(value);
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
        if (!input.contains("data") || !input["data"].is_array()) {
            return invalid("input data must be an array");
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

        auto data = parseData(input["data"]);
        if (!data.has_value()) {
            return invalid("input data values must be numbers for input: " + name);
        }

        const auto parsed_shape = parseShape(input["shape"]);

        InputTensor tensor;
        tensor.name = name;
        tensor.datatype = datatype;
        tensor.shape = std::move(parsed_shape);
        tensor.data = std::move(*data);
        request.inputs.push_back(std::move(tensor));
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
