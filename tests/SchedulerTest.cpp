#include "Scheduler.hpp"
#include "Executor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

class CountingExecutor final : public Executor {
  public:
    explicit CountingExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        const auto id = request.id.has_value() ? *request.id : std::string();
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {1, 1};
        output.data = {static_cast<double>(completed_.fetch_add(1, std::memory_order_relaxed) + 1)};
        response.outputs.push_back(std::move(output));
        completed_ids_.push_back(id);
        return response;
    }

    std::atomic<int> completed_{0};
    std::vector<std::string> completed_ids_;

  private:
    ModelMetadata metadata_;
};

class SlowExecutor final : public Executor {
  public:
    explicit SlowExecutor(ModelMetadata metadata, std::chrono::milliseconds delay)
        : metadata_(std::move(metadata)), delay_(delay) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        (void)request;
        std::this_thread::sleep_for(delay_);
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {1, 1};
        output.data = {1.0};
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
    std::chrono::milliseconds delay_;
};

class ThrowingExecutor final : public Executor {
  public:
    explicit ThrowingExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        (void)request;
        throw std::runtime_error("executor boom");
    }

  private:
    ModelMetadata metadata_;
};

class BlockingExecutor final : public Executor {
  public:
    explicit BlockingExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        (void)request;
        std::unique_lock<std::mutex> lock(mutex_);
        ++inflight_;
        cv_.wait(lock, [this]() { return release_; });
        --inflight_;
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {1, 1};
        output.data = {1.0};
        response.outputs.push_back(std::move(output));
        return response;
    }

    void releaseAll() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            release_ = true;
        }
        cv_.notify_all();
    }

    int inflight() const {
        return inflight_;
    }

  private:
    ModelMetadata metadata_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool release_ = false;
    int inflight_ = 0;
};

ModelMetadata testMetadata() {
    ModelMetadata metadata;
    metadata.name = "demo";
    metadata.versions = {"1"};
    metadata.platform = "test_scheduler";
    metadata.outputs.push_back({"output", "FP32", {1, 1}});
    return metadata;
}

ExecutionRequest makeRequest(std::string id = "") {
    ExecutionRequest request;
    if (!id.empty()) {
        request.id = std::move(id);
    }
    return request;
}

std::unique_ptr<Scheduler> makeScheduler(SchedulerConfig config,
                                         std::vector<std::unique_ptr<Executor>> executors = {}) {
    if (executors.empty()) {
        executors.push_back(std::make_unique<CountingExecutor>(testMetadata()));
    }
    return makeModelScheduler(std::move(executors), config, "demo");
}

std::unique_ptr<Scheduler> makeSchedulerWithExecutor(SchedulerConfig config,
                                                     std::unique_ptr<Executor> executor) {
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(std::move(executor));
    return makeScheduler(config, std::move(executors));
}

} // namespace

TEST_CASE(scheduler_executes_single_request) {
    auto scheduler = makeScheduler(SchedulerConfig{});
    const auto result = scheduler->submit(makeRequest("one"));
    REQUIRE(result.ok);
    REQUIRE_EQ(result.response.outputs.size(), static_cast<size_t>(1));
}

TEST_CASE(scheduler_rejects_when_queue_is_full) {
    SchedulerConfig config;
    config.max_queue_size = 1;
    config.instances = 1;
    config.request_timeout_ms = 5000;

    auto blocking = std::make_unique<BlockingExecutor>(testMetadata());
    BlockingExecutor *blocking_ptr = blocking.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(blocking));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::thread second([&]() { (void)scheduler->submit(makeRequest("second")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    const auto rejected = scheduler->submit(makeRequest("third"));
    REQUIRE(!rejected.ok);
    REQUIRE(rejected.scheduler_error == SchedulerError::Overloaded);
    REQUIRE_EQ(rejected.error_code, "OVERLOADED");

    blocking_ptr->releaseAll();
    first.join();
    second.join();
}

TEST_CASE(scheduler_preserves_fifo_order) {
    SchedulerConfig config;
    config.max_queue_size = 8;
    config.instances = 1;
    config.request_timeout_ms = 5000;

    auto counting = std::make_unique<CountingExecutor>(testMetadata());
    CountingExecutor *counting_ptr = counting.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(counting));

    for (int index = 0; index < 4; ++index) {
        const auto result = scheduler->submit(makeRequest(std::to_string(index)));
        REQUIRE(result.ok);
    }

    REQUIRE_EQ(counting_ptr->completed_.load(), 4);
    REQUIRE_EQ(counting_ptr->completed_ids_.size(), static_cast<size_t>(4));
    REQUIRE_EQ(counting_ptr->completed_ids_.front(), "0");
    REQUIRE_EQ(counting_ptr->completed_ids_.back(), "3");
}

TEST_CASE(scheduler_times_out_queued_request) {
    SchedulerConfig config;
    config.max_queue_size = 4;
    config.instances = 1;
    config.request_timeout_ms = 50;

    auto blocking = std::make_unique<BlockingExecutor>(testMetadata());
    BlockingExecutor *blocking_ptr = blocking.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(blocking));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    const auto timed_out = scheduler->submit(makeRequest("second"));
    REQUIRE(!timed_out.ok);
    REQUIRE(timed_out.scheduler_error == SchedulerError::Timeout);
    REQUIRE_EQ(timed_out.error_code, "DEADLINE_EXCEEDED");

    blocking_ptr->releaseAll();
    first.join();
}

TEST_CASE(scheduler_times_out_slow_execution) {
    SchedulerConfig config;
    config.max_queue_size = 4;
    config.instances = 2;
    config.request_timeout_ms = 50;

    auto slow = std::make_unique<SlowExecutor>(testMetadata(), std::chrono::milliseconds(200));
    auto scheduler = makeSchedulerWithExecutor(config, std::move(slow));

    const auto timed_out = scheduler->submit(makeRequest("slow"));
    REQUIRE(!timed_out.ok);
    REQUIRE(timed_out.scheduler_error == SchedulerError::Timeout);
}

TEST_CASE(scheduler_draining_marks_not_ready_and_rejects_new_requests) {
    auto scheduler = makeScheduler(SchedulerConfig{});
    scheduler->beginDrain();
    REQUIRE(!scheduler->isReady());
    REQUIRE(scheduler->isDraining());

    const auto rejected = scheduler->submit(makeRequest("late"));
    REQUIRE(!rejected.ok);
    REQUIRE(rejected.scheduler_error == SchedulerError::Draining);
}

TEST_CASE(scheduler_updates_metrics) {
    SchedulerConfig config;
    config.max_queue_size = 1;
    config.instances = 1;
    config.request_timeout_ms = 5000;

    auto blocking = std::make_unique<BlockingExecutor>(testMetadata());
    BlockingExecutor *blocking_ptr = blocking.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(blocking));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::thread second([&]() { (void)scheduler->submit(makeRequest("second")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    (void)scheduler->submit(makeRequest("rejected"));

    blocking_ptr->releaseAll();
    first.join();
    second.join();

    const auto metrics = scheduler->metrics();
    REQUIRE_EQ(metrics.requests_accepted, static_cast<uint64_t>(2));
    REQUIRE_EQ(metrics.requests_rejected, static_cast<uint64_t>(1));
    REQUIRE_EQ(metrics.completed_requests, static_cast<uint64_t>(2));
    REQUIRE(metrics.total_ns_total > 0);
}

TEST_CASE(scheduler_maps_executor_exception_to_backend_error) {
    auto scheduler = makeSchedulerWithExecutor(SchedulerConfig{},
                                               std::make_unique<ThrowingExecutor>(testMetadata()));

    const auto result = scheduler->submit(makeRequest("boom"));
    REQUIRE(!result.ok);
    REQUIRE(result.scheduler_error == SchedulerError::None);
    REQUIRE_EQ(result.response.error_code, "BACKEND_ERROR");
    REQUIRE(result.response.error_message.find("executor boom") != std::string::npos);
}

TEST_CASE(scheduler_destructor_joins_workers) {
    auto scheduler = makeScheduler(SchedulerConfig{});
    scheduler->submit(makeRequest("one"));
    scheduler.reset();
}
