#include "Scheduler.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct LatencyStats {
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
    double mean = 0;
    size_t count = 0;
};

LatencyStats computeStats(std::vector<double> &latencies_us) {
    std::sort(latencies_us.begin(), latencies_us.end());
    LatencyStats stats;
    stats.count = latencies_us.size();
    if (stats.count == 0) {
        return stats;
    }
    stats.mean = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
                 static_cast<double>(stats.count);
    stats.p50 = latencies_us[stats.count * 50 / 100];
    stats.p95 = latencies_us[stats.count * 95 / 100];
    stats.p99 = latencies_us[stats.count * 99 / 100];
    return stats;
}

RuntimeConfig benchConfig() {
    RuntimeConfig cfg;
    cfg.model_name = "bench";
    cfg.backend = "stub";
    cfg.instances = 2;
    cfg.max_queue_size = 1024;
    cfg.request_timeout_ms = 5000;
    return cfg;
}

TEST_CASE(benchmark_scheduler_latency_stub_executor) {
    const auto cfg = benchConfig();
    std::string error;
    std::vector<std::unique_ptr<Executor>> executors;
    for (size_t i = 0; i < cfg.instances; ++i) {
        (void)i;
        executors.push_back(makeStubExecutor(cfg, error));
    }

    SchedulerConfig sched_cfg;
    sched_cfg.max_queue_size = cfg.max_queue_size;
    sched_cfg.request_timeout_ms = cfg.request_timeout_ms;
    sched_cfg.instances = cfg.instances;
    auto scheduler = makeModelScheduler(std::move(executors), sched_cfg, cfg.model_name);
    REQUIRE(scheduler != nullptr);

    constexpr size_t warmup = 100;
    constexpr size_t samples = 1000;

    ExecutionRequest req;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 224, 224};
    input.bytes = tensorBytesFromDoubles(
        input.datatype, std::vector<double>(static_cast<size_t>(3) * 224 * 224, 1.0));
    req.inputs.push_back(std::move(input));
    req.requested_outputs = {"output"};

    for (size_t i = 0; i < warmup; ++i) {
        (void)scheduler->submit(req);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(samples);

    for (size_t i = 0; i < samples; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto result = scheduler->submit(req);
        const auto t1 = std::chrono::steady_clock::now();
        REQUIRE(result.ok);
        latencies_us.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));
    }

    const auto stats = computeStats(latencies_us);

    printf("\n  Stub executor scheduler latency (N=%zu):\n", stats.count);
    printf("    mean: %8.1f us\n", stats.mean);
    printf("    p50:  %8.1f us\n", stats.p50);
    printf("    p95:  %8.1f us\n", stats.p95);
    printf("    p99:  %8.1f us\n", stats.p99);

    REQUIRE(stats.p50 >= 0);
    REQUIRE(stats.p99 < 1000000.0);
}

TEST_CASE(benchmark_scheduler_latency_multi_mb_fp32) {
    const auto cfg = benchConfig();
    std::string error;
    std::vector<std::unique_ptr<Executor>> executors;
    for (size_t i = 0; i < cfg.instances; ++i) {
        (void)i;
        executors.push_back(makeStubExecutor(cfg, error));
    }

    SchedulerConfig sched_cfg;
    sched_cfg.max_queue_size = cfg.max_queue_size;
    sched_cfg.request_timeout_ms = cfg.request_timeout_ms;
    sched_cfg.instances = cfg.instances;
    auto scheduler = makeModelScheduler(std::move(executors), sched_cfg, cfg.model_name);
    REQUIRE(scheduler != nullptr);

    constexpr size_t warmup = 20;
    constexpr size_t samples = 200;

    // 1x3x1024x1024 FP32 = 12 MiB per request; dominated by payload copies
    // through the scheduler rather than executor work.
    ExecutionRequest req;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 1024, 1024};
    input.bytes = tensorBytesFromDoubles(
        input.datatype, std::vector<double>(static_cast<size_t>(3) * 1024 * 1024, 1.0));
    req.inputs.push_back(std::move(input));
    req.requested_outputs = {"output"};

    for (size_t i = 0; i < warmup; ++i) {
        (void)scheduler->submit(req);
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(samples);

    for (size_t i = 0; i < samples; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto result = scheduler->submit(req);
        const auto t1 = std::chrono::steady_clock::now();
        REQUIRE(result.ok);
        latencies_us.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()));
    }

    const auto stats = computeStats(latencies_us);

    printf("\n  Multi-MB FP32 (12 MiB/request) scheduler latency (N=%zu):\n", stats.count);
    printf("    mean: %8.1f us\n", stats.mean);
    printf("    p50:  %8.1f us\n", stats.p50);
    printf("    p95:  %8.1f us\n", stats.p95);
    printf("    p99:  %8.1f us\n", stats.p99);

    REQUIRE(stats.p50 >= 0);
    REQUIRE(stats.p99 < 1000000.0);
}

TEST_CASE(benchmark_scheduler_throughput_submit_wait) {
    const auto cfg = benchConfig();
    std::string error;
    std::vector<std::unique_ptr<Executor>> executors;
    for (size_t i = 0; i < cfg.instances; ++i) {
        (void)i;
        executors.push_back(makeStubExecutor(cfg, error));
    }

    SchedulerConfig sched_cfg;
    sched_cfg.max_queue_size = cfg.max_queue_size;
    sched_cfg.request_timeout_ms = cfg.request_timeout_ms;
    sched_cfg.instances = cfg.instances;
    auto scheduler = makeModelScheduler(std::move(executors), sched_cfg, cfg.model_name);
    REQUIRE(scheduler != nullptr);

    ExecutionRequest req;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 224, 224};
    input.bytes = tensorBytesFromDoubles(
        input.datatype, std::vector<double>(static_cast<size_t>(3) * 224 * 224, 1.0));
    req.inputs.push_back(std::move(input));
    req.requested_outputs = {"output"};

    constexpr size_t iterations = 2000;
    const auto t0 = std::chrono::steady_clock::now();
    size_t ok_count = 0;
    for (size_t i = 0; i < iterations; ++i) {
        auto result = scheduler->submit(req);
        if (result.ok) {
            ++ok_count;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const double throughput = (static_cast<double>(iterations) / elapsed_us) * 1e6;

    printf("\n  Throughput (2-instance stub): %.1f req/s (ok=%zu/%zu)\n", throughput, ok_count,
           iterations);

    REQUIRE(ok_count > iterations / 2);
    REQUIRE(throughput > 0);
}

} // namespace
