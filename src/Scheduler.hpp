#pragma once

#include "DynamicBatcher.hpp"
#include "Executor.hpp"
#include "SchedulerMetrics.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Tokenizer;

enum class SchedulerError {
    None,
    Overloaded,
    Timeout,
    Draining,
    Unavailable,
};

struct SchedulerConfig {
    size_t max_queue_size = 64;
    int64_t request_timeout_ms = 30000;
    size_t instances = 1;
    DynamicBatchingConfig dynamic_batching;
};

struct LlmSchedulerConfig {
    size_t max_queue_size = 64;
    int64_t request_timeout_ms = 30000;
    size_t instances = 1;
    size_t context_length = 4096;
    size_t kv_cache_slots = 1;
    size_t max_tokens = 256;
    double tokens_per_char = 0.25;
    size_t memory_budget_bytes = 0;
};

struct SchedulerResult {
    bool ok = true;
    SchedulerError scheduler_error = SchedulerError::None;
    std::string error_code;
    std::string error_message;
    ExecutionResponse response;
    uint64_t queue_latency_ns = 0;
    uint64_t execution_latency_ns = 0;
    uint64_t total_latency_ns = 0;
    uint64_t batch_size = 0;
};

class Scheduler {
  public:
    virtual ~Scheduler() = default;

    virtual SchedulerResult submit(ExecutionRequest request) = 0;
    virtual bool isReady() const = 0;
    virtual bool isDraining() const = 0;
    virtual void beginDrain() = 0;
    virtual SchedulerMetricsSnapshot metrics() const = 0;
};

std::unique_ptr<Scheduler> makeModelScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                              SchedulerConfig config, std::string model_name = {});

std::unique_ptr<Scheduler> makeLlmScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                            LlmSchedulerConfig config, std::string model_name = {});

std::unique_ptr<Scheduler> makeLlmScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                            LlmSchedulerConfig config, std::string model_name,
                                            std::unique_ptr<Tokenizer> tokenizer);