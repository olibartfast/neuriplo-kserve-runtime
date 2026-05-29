#include "BatchCompatibility.hpp"
#include "Test.hpp"

#include <string>

namespace {

ExecutionRequest makeRequest(std::vector<std::string> output_names = {}) {
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 224, 224};
    request.inputs.push_back(input);
    request.requested_outputs = std::move(output_names);
    return request;
}

} // namespace

TEST_CASE(batch_compatibility_accepts_matching_requests) {
    const auto left = makeRequest();
    auto right = makeRequest();
    REQUIRE(areBatchCompatible(left, right));
    REQUIRE(!batchCompatibilityError(left, right).has_value());
}

TEST_CASE(batch_compatibility_rejects_input_count_mismatch) {
    auto left = makeRequest();
    auto right = makeRequest();
    InputTensor extra;
    extra.name = "extra";
    extra.datatype = "FP32";
    extra.shape = {1, 1};
    right.inputs.push_back(extra);
    REQUIRE(!areBatchCompatible(left, right));
}

TEST_CASE(batch_compatibility_rejects_requested_output_mismatch) {
    const auto left = makeRequest({"output"});
    const auto right = makeRequest({"other"});
    REQUIRE(!areBatchCompatible(left, right));
}

TEST_CASE(batch_compatibility_rejects_shape_mismatch) {
    auto left = makeRequest();
    auto right = makeRequest();
    right.inputs[0].shape = {1, 3, 224, 128};
    REQUIRE(!areBatchCompatible(left, right));
}

TEST_CASE(batch_compatibility_allows_different_batch_dimension) {
    auto left = makeRequest();
    auto right = makeRequest();
    right.inputs[0].shape[0] = 2;
    REQUIRE(areBatchCompatible(left, right));
    REQUIRE_EQ(requestBatchSize(left), static_cast<int64_t>(1));
    REQUIRE_EQ(requestBatchSize(right), static_cast<int64_t>(2));
}
