#include "Scheduler.hpp"
#include "Executor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
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

class BatchTrackingExecutor final : public Executor {
  public:
    explicit BatchTrackingExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        infer_calls_.fetch_add(1, std::memory_order_relaxed);
        int64_t batch_size = 1;
        if (!request.inputs.empty() && !request.inputs.front().shape.empty()) {
            batch_size = request.inputs.front().shape.front();
        }

        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {batch_size, 1};
        output.data.assign(static_cast<size_t>(batch_size), 0.0);
        response.outputs.push_back(std::move(output));
        return response;
    }

    std::atomic<int> infer_calls_{0};

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
    metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    metadata.outputs.push_back({"output", "FP32", {1, 1}});
    return metadata;
}

ExecutionRequest makeRequest(std::string id = "") {
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 224, 224};
    request.inputs.push_back(input);
    if (!id.empty()) {
        request.id = std::move(id);
    }
    return request;
}

ExecutionRequest makeRequestWithBatch(std::string id, int64_t batch_dimension) {
    auto request = makeRequest(std::move(id));
    request.inputs.front().shape[0] = batch_dimension;
    return request;
}

class RecordingBatchExecutor final : public Executor {
  public:
    explicit RecordingBatchExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        int64_t batch_dimension = 1;
        if (!request.inputs.empty() && !request.inputs.front().shape.empty()) {
            batch_dimension = request.inputs.front().shape.front();
        }
        batch_dimensions_.push_back(batch_dimension);

        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {batch_dimension, 1};
        output.data.assign(static_cast<size_t>(batch_dimension), 0.0);
        response.outputs.push_back(std::move(output));
        return response;
    }

    std::vector<int64_t> batch_dimensions_;

  private:
    ModelMetadata metadata_;
};

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

TEST_CASE(scheduler_batches_compatible_requests) {
    SchedulerConfig config;
    config.instances = 1;
    config.request_timeout_ms = 5000;
    config.dynamic_batching.enabled = true;
    config.dynamic_batching.max_batch_size = 4;
    config.dynamic_batching.max_queue_delay_us = 100000;
    config.dynamic_batching.preferred_batch_sizes = {2};

    auto tracking = std::make_unique<BatchTrackingExecutor>(testMetadata());
    BatchTrackingExecutor *tracking_ptr = tracking.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(tracking));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::thread second([&]() { (void)scheduler->submit(makeRequest("second")); });
    first.join();
    second.join();

    REQUIRE_EQ(tracking_ptr->infer_calls_.load(), 1);
    const auto metrics = scheduler->metrics();
    REQUIRE_EQ(metrics.batches_formed, static_cast<uint64_t>(1));
    REQUIRE_EQ(metrics.batched_requests_total, static_cast<uint64_t>(2));
}

TEST_CASE(scheduler_batched_outputs_match_single_request_shape) {
    SchedulerConfig single_config;
    single_config.instances = 1;
    auto single_scheduler = makeSchedulerWithExecutor(
        single_config, std::make_unique<BatchTrackingExecutor>(testMetadata()));
    const auto single = single_scheduler->submit(makeRequest("single"));
    REQUIRE(single.ok);
    REQUIRE_EQ(single.response.outputs.size(), static_cast<size_t>(1));
    const auto &single_output = single.response.outputs.front();
    REQUIRE_EQ(single_output.name, "output");
    REQUIRE_EQ(single_output.datatype, "FP32");
    REQUIRE_EQ(single_output.shape, (std::vector<int64_t>{1, 1}));
    REQUIRE_EQ(single_output.data.size(), static_cast<size_t>(1));

    SchedulerConfig batch_config;
    batch_config.instances = 1;
    batch_config.request_timeout_ms = 5000;
    batch_config.dynamic_batching.enabled = true;
    batch_config.dynamic_batching.max_batch_size = 4;
    batch_config.dynamic_batching.max_queue_delay_us = 100000;
    batch_config.dynamic_batching.preferred_batch_sizes = {2};
    auto batch_scheduler = makeSchedulerWithExecutor(
        batch_config, std::make_unique<BatchTrackingExecutor>(testMetadata()));

    std::vector<SchedulerResult> batched_results(2);
    std::thread first(
        [&]() { batched_results[0] = batch_scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::thread second(
        [&]() { batched_results[1] = batch_scheduler->submit(makeRequest("second")); });
    first.join();
    second.join();

    for (const auto &batched : batched_results) {
        REQUIRE(batched.ok);
        REQUIRE_EQ(batched.response.outputs.size(), static_cast<size_t>(1));
        const auto &batched_output = batched.response.outputs.front();
        REQUIRE_EQ(batched_output.name, single_output.name);
        REQUIRE_EQ(batched_output.datatype, single_output.datatype);
        REQUIRE_EQ(batched_output.shape, single_output.shape);
        REQUIRE_EQ(batched_output.data.size(), single_output.data.size());
    }
}

TEST_CASE(scheduler_destructor_joins_workers) {
    auto scheduler = makeScheduler(SchedulerConfig{});
    scheduler->submit(makeRequest("one"));
    scheduler.reset();
}

TEST_CASE(scheduler_skips_incompatible_queue_neighbors_during_batch_formation) {
    SchedulerConfig config;
    config.instances = 1;
    config.request_timeout_ms = 5000;
    config.dynamic_batching.enabled = true;
    config.dynamic_batching.max_batch_size = 4;
    config.dynamic_batching.max_queue_delay_us = 100000;
    config.dynamic_batching.preferred_batch_sizes = {2};

    auto recording = std::make_unique<RecordingBatchExecutor>(testMetadata());
    RecordingBatchExecutor *recording_ptr = recording.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(recording));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    auto blocker = makeRequest("blocker");
    blocker.requested_outputs = {"missing"};
    std::thread blocker_thread([&]() { (void)scheduler->submit(std::move(blocker)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    std::thread second([&]() { (void)scheduler->submit(makeRequest("second")); });
    first.join();
    blocker_thread.join();
    second.join();

    REQUIRE_EQ(recording_ptr->batch_dimensions_.size(), static_cast<size_t>(2));
    REQUIRE_EQ(recording_ptr->batch_dimensions_[0], static_cast<int64_t>(2));
    REQUIRE_EQ(recording_ptr->batch_dimensions_[1], static_cast<int64_t>(1));
}

TEST_CASE(scheduler_respects_merged_batch_dimension_limit) {
    SchedulerConfig config;
    config.instances = 1;
    config.request_timeout_ms = 5000;
    config.dynamic_batching.enabled = true;
    config.dynamic_batching.max_batch_size = 4;
    config.dynamic_batching.max_queue_delay_us = 100000;
    config.dynamic_batching.preferred_batch_sizes = {4};

    auto recording = std::make_unique<RecordingBatchExecutor>(testMetadata());
    RecordingBatchExecutor *recording_ptr = recording.get();
    auto scheduler = makeSchedulerWithExecutor(config, std::move(recording));

    std::thread first([&]() { (void)scheduler->submit(makeRequestWithBatch("first", 3)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::thread second([&]() { (void)scheduler->submit(makeRequestWithBatch("second", 3)); });
    first.join();
    second.join();

    REQUIRE_EQ(recording_ptr->batch_dimensions_.size(), static_cast<size_t>(2));
    REQUIRE_EQ(recording_ptr->batch_dimensions_[0], static_cast<int64_t>(3));
    REQUIRE_EQ(recording_ptr->batch_dimensions_[1], static_cast<int64_t>(3));
}

TEST_CASE(scheduler_form_batch_fulfills_expired_queue_entries) {
    SchedulerConfig config;
    config.instances = 1;
    config.request_timeout_ms = 100;
    config.dynamic_batching.enabled = true;
    config.dynamic_batching.max_batch_size = 4;
    config.dynamic_batching.max_queue_delay_us = 200000;

    auto scheduler =
        makeSchedulerWithExecutor(config, std::make_unique<RecordingBatchExecutor>(testMetadata()));

    std::thread first([&]() { (void)scheduler->submit(makeRequest("first")); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto expired = makeRequest("expired");
    expired.requested_outputs = {"missing"};
    auto expired_future =
        std::async(std::launch::async, [&]() { return scheduler->submit(std::move(expired)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(120));

    std::thread second([&]() { (void)scheduler->submit(makeRequest("second")); });
    first.join();
    second.join();

    const auto expired_result = expired_future.get();
    REQUIRE(!expired_result.ok);
    REQUIRE(expired_result.scheduler_error == SchedulerError::Timeout);
}
