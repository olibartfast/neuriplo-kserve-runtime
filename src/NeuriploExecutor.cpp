#include "NeuriploExecutor.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {

bool isSupportedInputDatatype(const std::string &datatype) {
    static const std::string supported[] = {
        "BOOL",   "INT8",   "INT16",  "INT32", "INT64", "UINT8",
        "UINT16", "UINT32", "UINT64", "FP16",  "FP32",  "FP64",
    };
    for (const auto &dt : supported) {
        if (datatype == dt)
            return true;
    }
    return false;
}

std::optional<size_t> elementCount(const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) {
            return std::nullopt;
        }
        const auto size_dimension = static_cast<size_t>(dimension);
        if (size_dimension != 0 && count > std::numeric_limits<size_t>::max() / size_dimension) {
            return std::nullopt;
        }
        count *= size_dimension;
    }
    return count;
}

std::vector<std::string> selectedOutputNames(const ModelMetadata &metadata,
                                             const ExecutionRequest &request) {
    if (!request.requested_outputs.empty()) {
        return request.requested_outputs;
    }

    std::vector<std::string> names;
    names.reserve(metadata.outputs.size());
    for (const auto &output : metadata.outputs) {
        names.push_back(output.name);
    }
    return names;
}

bool containsName(const std::vector<std::string> &names, const std::string &name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

const InputTensor *findInput(const std::vector<InputTensor> &inputs, const std::string &name) {
    const auto found = std::find_if(inputs.begin(), inputs.end(),
                                    [&name](const auto &input) { return input.name == name; });
    if (found == inputs.end()) {
        return nullptr;
    }
    return &*found;
}

const TensorMetadata *findMetadataInput(const ModelMetadata &metadata, const std::string &name) {
    const auto found = std::find_if(metadata.inputs.begin(), metadata.inputs.end(),
                                    [&name](const auto &input) { return input.name == name; });
    if (found == metadata.inputs.end()) {
        return nullptr;
    }
    return &*found;
}

ExecutionResponse invalidArgument(std::string message) {
    ExecutionResponse response;
    response.ok = false;
    response.error_code = "INVALID_ARGUMENT";
    response.error_message = std::move(message);
    return response;
}

ExecutionResponse backendError(std::string message) {
    ExecutionResponse response;
    response.ok = false;
    response.error_code = "BACKEND_ERROR";
    response.error_message = std::move(message);
    return response;
}

std::optional<std::vector<InputTensor>> orderedInputs(const ModelMetadata &metadata,
                                                      const ExecutionRequest &request,
                                                      ExecutionResponse &error) {
    std::vector<std::string> seen_input_names;
    seen_input_names.reserve(request.inputs.size());
    for (const auto &input : request.inputs) {
        if (containsName(seen_input_names, input.name)) {
            error = invalidArgument("duplicate neuriplo input: " + input.name);
            return std::nullopt;
        }
        seen_input_names.push_back(input.name);

        const auto *metadata_input = findMetadataInput(metadata, input.name);
        if (metadata_input == nullptr) {
            error = invalidArgument("unknown neuriplo input: " + input.name);
            return std::nullopt;
        }
        if (input.datatype != metadata_input->datatype) {
            error = invalidArgument("unsupported datatype for neuriplo input: " + input.name);
            return std::nullopt;
        }
        if (input.shape != metadata_input->shape) {
            error = invalidArgument("invalid shape for neuriplo input: " + input.name);
            return std::nullopt;
        }
        if (!isSupportedInputDatatype(input.datatype)) {
            error = invalidArgument("unsupported datatype for neuriplo input: " + input.name);
            return std::nullopt;
        }
        const auto expected_count = elementCount(input.shape);
        if (!expected_count.has_value() || input.data.size() != *expected_count) {
            error = invalidArgument("input data length does not match shape for neuriplo input: " +
                                    input.name);
            return std::nullopt;
        }
    }

    std::vector<InputTensor> ordered;
    ordered.reserve(metadata.inputs.size());
    for (const auto &metadata_input : metadata.inputs) {
        const auto *input = findInput(request.inputs, metadata_input.name);
        if (input == nullptr) {
            error = invalidArgument("missing required neuriplo input: " + metadata_input.name);
            return std::nullopt;
        }
        ordered.push_back(*input);
    }
    return ordered;
}

} // namespace

NeuriploExecutor::NeuriploExecutor(const RuntimeConfig &config,
                                   std::unique_ptr<NeuriploAdapter> adapter)
    : adapter_(std::move(adapter)) {
    if (!adapter_) {
        throw std::invalid_argument("neuriplo adapter is required");
    }
    metadata_ = adapter_->load(config);
    if (metadata_.name.empty()) {
        metadata_.name = config.model_name;
    }
    if (metadata_.versions.empty()) {
        metadata_.versions.push_back("1");
    }
    if (metadata_.platform.empty()) {
        metadata_.platform = "neuriplo_" + config.backend;
    }
}

const ModelMetadata &NeuriploExecutor::metadata() const {
    return metadata_;
}

ExecutionResponse NeuriploExecutor::infer(const ExecutionRequest &request) {
    ExecutionResponse validation_error;
    auto inputs = orderedInputs(metadata_, request, validation_error);
    if (!inputs.has_value()) {
        return validation_error;
    }

    if (request.cancel_token && request.cancel_token->load(std::memory_order_acquire)) {
        ExecutionResponse cancelled;
        cancelled.ok = false;
        cancelled.error_code = "DEADLINE_EXCEEDED";
        cancelled.error_message = "request cancelled before inference";
        return cancelled;
    }

    NeuriploInferenceResult result;
    try {
        result = adapter_->infer(*inputs);
    } catch (const std::exception &ex) {
        return backendError(ex.what());
    }

    if (!result.ok) {
        return backendError(result.error_message.empty() ? "neuriplo inference failed"
                                                         : result.error_message);
    }

    const auto output_names = selectedOutputNames(metadata_, request);
    ExecutionResponse response;
    response.outputs.reserve(result.outputs.size());
    for (auto &output : result.outputs) {
        if (containsName(output_names, output.name)) {
            response.outputs.push_back(std::move(output));
        }
    }
    return response;
}

std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error,
                                               std::unique_ptr<NeuriploAdapter> adapter) {
    try {
        return std::make_unique<NeuriploExecutor>(config, std::move(adapter));
    } catch (const std::exception &ex) {
        error = ex.what();
        return nullptr;
    }
}

std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error) {
    try {
        return makeNeuriploExecutor(config, error, makeRealNeuriploAdapter());
    } catch (const std::exception &ex) {
        error = ex.what();
        return nullptr;
    }
}
