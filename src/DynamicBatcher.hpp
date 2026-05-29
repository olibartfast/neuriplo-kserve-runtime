#pragma once

#include "Executor.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

struct DynamicBatchingConfig {
    bool enabled = false;
    size_t max_batch_size = 1;
    int64_t max_queue_delay_us = 0;
    std::vector<size_t> preferred_batch_sizes;
};

struct MergedBatch {
    ExecutionRequest request;
    std::vector<size_t> batch_sizes;
};

std::optional<std::string> mergeExecutionRequests(const std::vector<ExecutionRequest> &requests,
                                                  MergedBatch &merged);

std::vector<ExecutionResponse>
splitExecutionResponse(const ExecutionResponse &batched, const std::vector<size_t> &batch_sizes,
                       const std::vector<ExecutionRequest> &source_requests);

bool shouldDispatchBatchSize(size_t merged_batch_dimension, const DynamicBatchingConfig &config,
                             bool delay_elapsed);
