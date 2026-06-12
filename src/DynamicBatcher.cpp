#include "DynamicBatcher.hpp"

#include "BatchCompatibility.hpp"

#include <algorithm>
#include <numeric>

namespace {

size_t shapeElementCount(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t count = 1;
    for (const auto dimension : shape) {
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

size_t sliceElementCount(const std::vector<int64_t> &shape) {
    if (shape.size() <= 1) {
        return 1;
    }
    size_t count = 1;
    for (size_t index = 1; index < shape.size(); ++index) {
        count *= static_cast<size_t>(shape[index]);
    }
    return count;
}

InputTensor mergeInputTensors(const std::vector<InputTensor> &inputs,
                              const std::vector<size_t> &batch_sizes) {
    InputTensor merged = inputs.front();
    merged.shape = inputs.front().shape;
    merged.shape[0] = static_cast<int64_t>(
        std::accumulate(batch_sizes.begin(), batch_sizes.end(), static_cast<size_t>(0)));

    size_t total_bytes = 0;
    for (const auto &input : inputs) {
        total_bytes += input.bytes.size();
    }
    merged.bytes.clear();
    merged.bytes.reserve(total_bytes);
    for (const auto &input : inputs) {
        merged.bytes.insert(merged.bytes.end(), input.bytes.begin(), input.bytes.end());
    }
    return merged;
}

} // namespace

bool shouldDispatchBatchSize(size_t merged_batch_dimension, const DynamicBatchingConfig &config,
                             bool delay_elapsed) {
    if (merged_batch_dimension == 0) {
        return false;
    }
    if (merged_batch_dimension >= config.max_batch_size) {
        return true;
    }
    if (!config.preferred_batch_sizes.empty()) {
        for (const auto preferred : config.preferred_batch_sizes) {
            if (merged_batch_dimension == preferred) {
                return true;
            }
        }
    }
    return delay_elapsed;
}

std::optional<std::string> mergeExecutionRequests(const std::vector<ExecutionRequest> &requests,
                                                  MergedBatch &merged) {
    if (requests.empty()) {
        return "batch requires at least one request";
    }

    merged.batch_sizes.clear();
    merged.batch_sizes.reserve(requests.size());
    for (const auto &request : requests) {
        if (request.inputs.empty()) {
            return "request missing inputs";
        }
        merged.batch_sizes.push_back(static_cast<size_t>(requestBatchSize(request)));
    }

    for (size_t index = 1; index < requests.size(); ++index) {
        if (const auto error = batchCompatibilityError(requests.front(), requests[index])) {
            return error;
        }
    }

    merged.request.requested_outputs = requests.front().requested_outputs;
    merged.request.inputs.clear();
    merged.request.inputs.reserve(requests.front().inputs.size());

    for (const auto &reference_input : requests.front().inputs) {
        std::vector<InputTensor> matching_inputs;
        matching_inputs.reserve(requests.size());
        for (const auto &request : requests) {
            const InputTensor *found = nullptr;
            for (const auto &input : request.inputs) {
                if (input.name == reference_input.name) {
                    found = &input;
                    break;
                }
            }
            if (found == nullptr) {
                return "input name mismatch during merge";
            }
            matching_inputs.push_back(*found);
        }
        merged.request.inputs.push_back(mergeInputTensors(matching_inputs, merged.batch_sizes));
    }

    return std::nullopt;
}

std::vector<ExecutionResponse>
splitExecutionResponse(const ExecutionResponse &batched, const std::vector<size_t> &batch_sizes,
                       const std::vector<ExecutionRequest> &source_requests) {
    std::vector<ExecutionResponse> responses;
    responses.reserve(source_requests.size());
    if (!batched.ok) {
        for (size_t index = 0; index < source_requests.size(); ++index) {
            (void)index;
            responses.push_back(batched);
        }
        return responses;
    }

    for (size_t request_index = 0; request_index < source_requests.size(); ++request_index) {
        ExecutionResponse response;
        response.ok = true;
        response.outputs.reserve(batched.outputs.size());

        size_t batch_offset = 0;
        for (size_t prior = 0; prior < request_index; ++prior) {
            batch_offset += batch_sizes.at(prior);
        }
        const size_t request_batch = batch_sizes.at(request_index);

        for (const auto &output : batched.outputs) {
            OutputTensor split_output;
            split_output.name = output.name;
            split_output.datatype = output.datatype;
            split_output.shape = output.shape;
            if (!split_output.shape.empty()) {
                split_output.shape[0] = static_cast<int64_t>(request_batch);
            }

            const size_t expected_batch_dim =
                std::accumulate(batch_sizes.begin(), batch_sizes.end(), static_cast<size_t>(0));
            if (!output.shape.empty() &&
                static_cast<size_t>(output.shape[0]) != expected_batch_dim) {
                response.ok = false;
                response.error_code = "BACKEND_ERROR";
                response.error_message = "batched output batch dimension mismatch";
                response.outputs.clear();
                break;
            }

            const size_t element_size = tensorElementSize(output.datatype);
            const size_t expected_elements = shapeElementCount(output.shape);
            if (element_size == 0 || expected_elements * element_size != output.bytes.size()) {
                response.ok = false;
                response.error_code = "BACKEND_ERROR";
                response.error_message = "batched output data size mismatch";
                response.outputs.clear();
                break;
            }

            const size_t slice_elements = sliceElementCount(output.shape);
            if (slice_elements == 0) {
                response.ok = false;
                response.error_code = "BACKEND_ERROR";
                response.error_message = "batched output slice size mismatch";
                response.outputs.clear();
                break;
            }

            const size_t start = batch_offset * slice_elements * element_size;
            const size_t end = start + (request_batch * slice_elements * element_size);
            if (start > output.bytes.size() || end > output.bytes.size()) {
                response.ok = false;
                response.error_code = "BACKEND_ERROR";
                response.error_message = "batched output size mismatch";
                response.outputs.clear();
                break;
            }

            split_output.bytes.assign(output.bytes.begin() + static_cast<std::ptrdiff_t>(start),
                                      output.bytes.begin() + static_cast<std::ptrdiff_t>(end));
            if (shapeElementCount(split_output.shape) * element_size != split_output.bytes.size()) {
                response.ok = false;
                response.error_code = "BACKEND_ERROR";
                response.error_message = "batched output slice data size mismatch";
                response.outputs.clear();
                break;
            }
            response.outputs.push_back(std::move(split_output));
        }
        responses.push_back(std::move(response));
    }

    return responses;
}
