#pragma once

#include "DynamicBatcher.hpp"
#include "Executor.hpp"
#include "SchedulerMetrics.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

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

struct SchedulerResult {
    bool ok = true;
    SchedulerError scheduler_error = SchedulerError::None;
    std::string error_code;
    std::string error_message;
    ExecutionResponse response;
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
