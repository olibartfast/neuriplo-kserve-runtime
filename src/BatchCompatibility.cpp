#include "BatchCompatibility.hpp"

#include <algorithm>

namespace {

bool shapesMatchExceptBatch(const std::vector<int64_t> &left, const std::vector<int64_t> &right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 1; index < left.size(); ++index) {
        if (left[index] != right[index]) {
            return false;
        }
    }
    return true;
}

const InputTensor *findInputByName(const ExecutionRequest &request, const std::string &name) {
    for (const auto &input : request.inputs) {
        if (input.name == name) {
            return &input;
        }
    }
    return nullptr;
}

} // namespace

int64_t requestBatchSize(const ExecutionRequest &request) {
    if (request.inputs.empty() || request.inputs.front().shape.empty()) {
        return 1;
    }
    return request.inputs.front().shape.front();
}

std::optional<std::string> batchCompatibilityError(const ExecutionRequest &left,
                                                   const ExecutionRequest &right) {
    for (const auto &input : left.inputs) {
        if (isBytesDatatype(input.datatype)) {
            return "BYTES tensors are not batch-compatible";
        }
    }
    for (const auto &input : right.inputs) {
        if (isBytesDatatype(input.datatype)) {
            return "BYTES tensors are not batch-compatible";
        }
    }

    if (left.inputs.size() != right.inputs.size()) {
        return "input count mismatch";
    }
    if (left.requested_outputs != right.requested_outputs) {
        return "requested output mismatch";
    }

    for (const auto &left_input : left.inputs) {
        const auto *right_input = findInputByName(right, left_input.name);
        if (right_input == nullptr) {
            return "input name mismatch";
        }
        if (left_input.datatype != right_input->datatype) {
            return "input datatype mismatch";
        }
        if (left_input.shape.empty() || right_input->shape.empty()) {
            return "input shape missing batch dimension";
        }
        if (!shapesMatchExceptBatch(left_input.shape, right_input->shape)) {
            return "input shape mismatch";
        }
        if (left_input.shape.size() != right_input->shape.size()) {
            return "input rank mismatch";
        }
    }

    for (const auto &right_input : right.inputs) {
        if (findInputByName(left, right_input.name) == nullptr) {
            return "input name mismatch";
        }
    }

    return std::nullopt;
}

bool areBatchCompatible(const ExecutionRequest &left, const ExecutionRequest &right) {
    return !batchCompatibilityError(left, right).has_value();
}
