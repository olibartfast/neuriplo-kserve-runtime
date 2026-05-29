#include "DynamicBatcher.hpp"
#include "Test.hpp"

#include <string>
#include <vector>

namespace {

ExecutionRequest makeRequest(int64_t batch_size, double marker) {
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {batch_size, 2};
    input.data.assign(static_cast<size_t>(batch_size * 2), marker);
    request.inputs.push_back(input);
    request.requested_outputs = {"output"};
    return request;
}

ExecutionResponse makeBatchedResponse(const std::vector<double> &values) {
    ExecutionResponse response;
    OutputTensor output;
    output.name = "output";
    output.datatype = "FP32";
    output.shape = {static_cast<int64_t>(values.size()), 1};
    output.data = values;
    response.outputs.push_back(std::move(output));
    return response;
}

} // namespace

TEST_CASE(dynamic_batcher_merges_compatible_requests) {
    MergedBatch merged;
    const std::vector<ExecutionRequest> requests = {makeRequest(1, 1.0), makeRequest(2, 2.0)};
    REQUIRE(!mergeExecutionRequests(requests, merged).has_value());
    REQUIRE_EQ(merged.batch_sizes.size(), static_cast<size_t>(2));
    REQUIRE_EQ(merged.batch_sizes[0], static_cast<size_t>(1));
    REQUIRE_EQ(merged.batch_sizes[1], static_cast<size_t>(2));
    REQUIRE_EQ(merged.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(merged.request.inputs[0].shape[0], static_cast<int64_t>(3));
    REQUIRE_EQ(merged.request.inputs[0].data.size(), static_cast<size_t>(6));
}

TEST_CASE(dynamic_batcher_splits_batched_outputs) {
    const std::vector<ExecutionRequest> requests = {makeRequest(1, 1.0), makeRequest(2, 2.0)};
    const auto batched = makeBatchedResponse({10.0, 20.0, 30.0});
    const auto split = splitExecutionResponse(batched, {1, 2}, requests);
    REQUIRE_EQ(split.size(), static_cast<size_t>(2));
    REQUIRE_EQ(split[0].outputs[0].shape[0], static_cast<int64_t>(1));
    REQUIRE_EQ(split[1].outputs[0].shape[0], static_cast<int64_t>(2));
    REQUIRE_EQ(split[0].outputs[0].data[0], 10.0);
    REQUIRE_EQ(split[1].outputs[0].data[0], 20.0);
    REQUIRE_EQ(split[1].outputs[0].data[1], 30.0);
}

TEST_CASE(dynamic_batcher_honors_preferred_batch_sizes) {
    DynamicBatchingConfig config;
    config.max_batch_size = 8;
    config.preferred_batch_sizes = {2, 4, 8};
    REQUIRE(shouldDispatchBatchSize(2, config, false));
    REQUIRE(!shouldDispatchBatchSize(3, config, false));
    REQUIRE(shouldDispatchBatchSize(3, config, true));
}

TEST_CASE(dynamic_batcher_rejects_incompatible_merge) {
    MergedBatch merged;
    auto first = makeRequest(1, 1.0);
    auto second = makeRequest(1, 2.0);
    second.requested_outputs = {"missing"};
    const std::vector<ExecutionRequest> requests = {first, second};
    REQUIRE(mergeExecutionRequests(requests, merged).has_value());
}

TEST_CASE(dynamic_batcher_rejects_short_batched_output_data) {
    const std::vector<ExecutionRequest> requests = {makeRequest(1, 1.0), makeRequest(2, 2.0)};
    auto batched = makeBatchedResponse({10.0, 20.0});
    batched.outputs[0].shape = {3, 1};
    const auto split = splitExecutionResponse(batched, {1, 2}, requests);
    REQUIRE_EQ(split.size(), static_cast<size_t>(2));
    REQUIRE(!split[0].ok);
    REQUIRE_EQ(split[0].error_code, "BACKEND_ERROR");
}

TEST_CASE(dynamic_batcher_rejects_batch_dimension_shape_mismatch) {
    const std::vector<ExecutionRequest> requests = {makeRequest(1, 1.0), makeRequest(2, 2.0)};
    auto batched = makeBatchedResponse({10.0, 20.0, 30.0});
    batched.outputs[0].shape = {2, 1};
    const auto split = splitExecutionResponse(batched, {1, 2}, requests);
    REQUIRE_EQ(split.size(), static_cast<size_t>(2));
    REQUIRE(!split[0].ok);
    REQUIRE_EQ(split[0].error_code, "BACKEND_ERROR");
}
