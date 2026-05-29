#include "Scheduler.hpp"

#include "BatchCompatibility.hpp"
#include "DynamicBatcher.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct PendingRequest {
    ExecutionRequest request;
    TimePoint enqueued_at;
    TimePoint deadline;
    std::shared_ptr<std::promise<SchedulerResult>> promise;
    std::atomic<bool> cancelled{false};
};

SchedulerResult makeSchedulerError(SchedulerError error, std::string code, std::string message) {
    SchedulerResult result;
    result.ok = false;
    result.scheduler_error = error;
    result.error_code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

SchedulerResult makeTimeoutResult() {
    return makeSchedulerError(SchedulerError::Timeout, "DEADLINE_EXCEEDED",
                              "request exceeded deadline");
}

SchedulerResult makeDrainingResult() {
    return makeSchedulerError(SchedulerError::Draining, "UNAVAILABLE", "model is draining");
}

SchedulerResult makeOverloadedResult() {
    return makeSchedulerError(SchedulerError::Overloaded, "OVERLOADED", "request queue is full");
}

SchedulerResult makeExecutorFailureResult(std::string message) {
    SchedulerResult result;
    result.ok = false;
    result.response.ok = false;
    result.response.error_code = "BACKEND_ERROR";
    result.response.error_message =
        message.empty() ? "executor inference failed" : std::move(message);
    return result;
}

uint64_t elapsedNs(const TimePoint &start, const TimePoint &end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

class ModelScheduler final : public Scheduler {
  public:
    ModelScheduler(std::vector<std::unique_ptr<Executor>> executors, SchedulerConfig config,
                   std::string model_name)
        : executors_(std::move(executors)), config_(config), model_name_(std::move(model_name)) {
        if (executors_.empty()) {
            throw std::invalid_argument("scheduler requires at least one executor");
        }
        inflight_.assign(executors_.size(), 0);
        workers_.reserve(config_.instances);
        for (size_t worker_index = 0; worker_index < config_.instances; ++worker_index) {
            (void)worker_index;
            workers_.emplace_back([this]() { workerLoop(); });
        }
    }

    ~ModelScheduler() override {
        beginDrain();
    }

    SchedulerResult submit(ExecutionRequest request) override {
        if (isDraining()) {
            return makeDrainingResult();
        }

        const auto enqueued_at = Clock::now();
        const auto deadline = enqueued_at + std::chrono::milliseconds(config_.request_timeout_ms);
        auto promise = std::make_shared<std::promise<SchedulerResult>>();
        auto future = promise->get_future();

        auto pending = std::make_shared<PendingRequest>();
        pending->request = std::move(request);
        pending->enqueued_at = enqueued_at;
        pending->deadline = deadline;
        pending->promise = std::move(promise);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (draining_) {
                return makeDrainingResult();
            }
            if (queue_.size() >= config_.max_queue_size) {
                incrementMetric(&SchedulerMetricsSnapshot::requests_rejected);
                return makeOverloadedResult();
            }
            queue_.push_back(pending);
            incrementMetric(&SchedulerMetricsSnapshot::requests_accepted);
        }
        queue_cv_.notify_one();

        if (future.wait_until(deadline) == std::future_status::ready) {
            return future.get();
        }

        pending->cancelled.store(true, std::memory_order_release);
        incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
        return makeTimeoutResult();
    }

    bool isReady() const override {
        return !draining_.load(std::memory_order_acquire);
    }

    bool isDraining() const override {
        return draining_.load(std::memory_order_acquire);
    }

    void beginDrain() override {
        if (draining_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }

        queue_cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
        rejectQueuedRequests();
    }

    SchedulerMetricsSnapshot metrics() const override {
        const size_t queue_depth = currentQueueDepth();
        const size_t in_flight = currentInFlight();
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        SchedulerMetricsSnapshot snapshot = metrics_;
        snapshot.queue_depth = queue_depth;
        snapshot.in_flight = in_flight;
        return snapshot;
    }

  private:
    size_t currentQueueDepth() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    size_t currentInFlight() const {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        size_t total = 0;
        for (const auto count : inflight_) {
            total += count;
        }
        return total;
    }

    void rejectQueuedRequests() {
        std::deque<std::shared_ptr<PendingRequest>> pending;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending.swap(queue_);
        }
        for (const auto &request : pending) {
            request->promise->set_value(makeDrainingResult());
        }
    }

    size_t selectExecutorIndex() {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        size_t best_index = 0;
        size_t best_inflight = inflight_[0];
        for (size_t index = 1; index < inflight_.size(); ++index) {
            if (inflight_[index] < best_inflight) {
                best_index = index;
                best_inflight = inflight_[index];
            }
        }
        ++inflight_[best_index];
        return best_index;
    }

    void decrementInflight(size_t executor_index) {
        std::lock_guard<std::mutex> lock(inflight_mutex_);
        if (inflight_[executor_index] > 0) {
            --inflight_[executor_index];
        }
    }

    void incrementMetric(uint64_t SchedulerMetricsSnapshot::*field) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.*field += 1;
    }

    void recordBatchMetrics(size_t batch_size, uint64_t formation_ns, uint64_t execution_ns) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        ++metrics_.batches_formed;
        metrics_.batched_requests_total += batch_size;
        metrics_.batch_formation_ns_total += formation_ns;
        metrics_.batch_execution_ns_total += execution_ns;
    }

    void recordLatencies(const TimePoint &enqueued_at, const TimePoint &execution_started,
                         const TimePoint &execution_finished) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.queue_wait_ns_total += elapsedNs(enqueued_at, execution_started);
        metrics_.execution_ns_total += elapsedNs(execution_started, execution_finished);
        metrics_.total_ns_total += elapsedNs(enqueued_at, execution_finished);
        ++metrics_.completed_requests;
    }

    std::shared_ptr<PendingRequest> popNextPending() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() {
            return draining_.load(std::memory_order_acquire) || !queue_.empty();
        });
        if (draining_.load(std::memory_order_acquire) && queue_.empty()) {
            return nullptr;
        }
        auto pending = queue_.front();
        queue_.pop_front();
        return pending;
    }

    size_t mergedBatchDimension(const std::vector<std::shared_ptr<PendingRequest>> &batch) const {
        size_t total = 0;
        for (const auto &pending : batch) {
            total += static_cast<size_t>(requestBatchSize(pending->request));
        }
        return total;
    }

    void fulfillExpiredRequest(const std::shared_ptr<PendingRequest> &pending) {
        incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
        fulfillResult(pending, makeTimeoutResult());
    }

    std::vector<std::shared_ptr<PendingRequest>>
    formBatch(const std::shared_ptr<PendingRequest> &first) {
        std::vector<std::shared_ptr<PendingRequest>> batch;
        batch.push_back(first);

        const auto &batching = config_.dynamic_batching;
        if (!batching.enabled || batching.max_batch_size <= 1) {
            return batch;
        }

        const auto formation_started = Clock::now();
        const auto formation_deadline =
            formation_started + std::chrono::microseconds(batching.max_queue_delay_us);
        size_t merged_batch_dimension = mergedBatchDimension(batch);

        while (true) {
            const auto now = Clock::now();
            const bool delay_elapsed = now >= formation_deadline;
            if (shouldDispatchBatchSize(merged_batch_dimension, batching, delay_elapsed) &&
                batch.size() > 1) {
                break;
            }
            if (merged_batch_dimension >= batching.max_batch_size) {
                break;
            }

            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (draining_.load(std::memory_order_acquire)) {
                break;
            }

            bool made_progress = false;
            for (auto iterator = queue_.begin(); iterator != queue_.end();) {
                auto next = *iterator;
                if (next->cancelled.load(std::memory_order_acquire)) {
                    iterator = queue_.erase(iterator);
                    made_progress = true;
                    continue;
                }
                if (Clock::now() >= next->deadline) {
                    iterator = queue_.erase(iterator);
                    lock.unlock();
                    queue_cv_.notify_one();
                    fulfillExpiredRequest(next);
                    lock.lock();
                    made_progress = true;
                    continue;
                }
                if (!areBatchCompatible(batch.front()->request, next->request)) {
                    ++iterator;
                    continue;
                }

                const size_t next_batch_dimension =
                    static_cast<size_t>(requestBatchSize(next->request));
                if (merged_batch_dimension + next_batch_dimension > batching.max_batch_size) {
                    ++iterator;
                    continue;
                }

                batch.push_back(next);
                merged_batch_dimension += next_batch_dimension;
                iterator = queue_.erase(iterator);
                lock.unlock();
                queue_cv_.notify_one();
                made_progress = true;

                if (shouldDispatchBatchSize(merged_batch_dimension, batching,
                                            Clock::now() >= formation_deadline) &&
                    batch.size() > 1) {
                    const auto formation_finished = Clock::now();
                    recordBatchMetrics(batch.size(),
                                       elapsedNs(formation_started, formation_finished), 0);
                    return batch;
                }

                lock.lock();
                break;
            }

            if (!made_progress) {
                if (delay_elapsed) {
                    break;
                }
                auto wait_deadline = formation_deadline;
                for (const auto &queued : queue_) {
                    if (!queued->cancelled.load(std::memory_order_acquire)) {
                        wait_deadline = std::min(wait_deadline, queued->deadline);
                    }
                }
                queue_cv_.wait_until(lock, wait_deadline);
            }
        }

        const auto formation_finished = Clock::now();
        if (batch.size() > 1) {
            recordBatchMetrics(batch.size(), elapsedNs(formation_started, formation_finished), 0);
        }
        return batch;
    }

    bool runInference(Executor &executor, const ExecutionRequest &request,
                      const TimePoint &deadline, ExecutionResponse &response) {
        try {
            if (config_.instances == 1) {
                response = executor.infer(request);
                return true;
            }

            auto infer_future = std::async(
                std::launch::async, [&executor, request]() { return executor.infer(request); });
            if (infer_future.wait_until(deadline) == std::future_status::timeout) {
                return false;
            }
            response = infer_future.get();
            return true;
        } catch (const std::exception &error) {
            response.ok = false;
            response.error_code = "BACKEND_ERROR";
            response.error_message = error.what();
            return true;
        } catch (...) {
            response.ok = false;
            response.error_code = "BACKEND_ERROR";
            response.error_message = "executor inference failed";
            return true;
        }
    }

    TimePoint batchDeadline(const std::vector<std::shared_ptr<PendingRequest>> &batch) const {
        auto deadline = batch.front()->deadline;
        for (const auto &pending : batch) {
            deadline = std::min(deadline, pending->deadline);
        }
        return deadline;
    }

    void fulfillResult(const std::shared_ptr<PendingRequest> &pending, SchedulerResult result) {
        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }
        pending->promise->set_value(std::move(result));
    }

    void processBatch(std::vector<std::shared_ptr<PendingRequest>> batch) {
        const auto now = Clock::now();
        for (const auto &pending : batch) {
            if (pending->cancelled.load(std::memory_order_acquire)) {
                continue;
            }
            if (now >= pending->deadline) {
                incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
                fulfillResult(pending, makeTimeoutResult());
            }
        }

        std::vector<std::shared_ptr<PendingRequest>> active_batch;
        active_batch.reserve(batch.size());
        for (const auto &pending : batch) {
            if (pending->cancelled.load(std::memory_order_acquire)) {
                continue;
            }
            if (Clock::now() >= pending->deadline) {
                continue;
            }
            active_batch.push_back(pending);
        }
        if (active_batch.empty()) {
            return;
        }

        if (active_batch.size() == 1) {
            processSingle(active_batch.front());
            return;
        }

        std::vector<ExecutionRequest> source_requests;
        source_requests.reserve(active_batch.size());
        for (const auto &pending : active_batch) {
            source_requests.push_back(pending->request);
        }

        MergedBatch merged;
        if (const auto error = mergeExecutionRequests(source_requests, merged)) {
            for (const auto &pending : active_batch) {
                fulfillResult(pending, makeExecutorFailureResult(error.value()));
            }
            return;
        }

        const auto executor_index = selectExecutorIndex();
        auto &executor = executors_.at(executor_index);
        const auto execution_started = Clock::now();
        const auto deadline = batchDeadline(active_batch);

        ExecutionResponse batched_response;
        const bool completed = runInference(*executor, merged.request, deadline, batched_response);
        const auto execution_finished = Clock::now();
        decrementInflight(executor_index);

        if (!completed) {
            for (const auto &pending : active_batch) {
                incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
                fulfillResult(pending, makeTimeoutResult());
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.batch_execution_ns_total += elapsedNs(execution_started, execution_finished);
        }

        const auto split_responses =
            splitExecutionResponse(batched_response, merged.batch_sizes, source_requests);
        for (size_t index = 0; index < active_batch.size(); ++index) {
            const auto &pending = active_batch[index];
            if (execution_finished >= pending->deadline) {
                incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
                fulfillResult(pending, makeTimeoutResult());
                continue;
            }

            recordLatencies(pending->enqueued_at, execution_started, execution_finished);
            SchedulerResult result;
            result.response = split_responses.at(index);
            if (!result.response.ok) {
                result.ok = false;
            }
            fulfillResult(pending, std::move(result));
        }
    }

    void processSingle(const std::shared_ptr<PendingRequest> &pending) {
        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }

        const auto now = Clock::now();
        if (now >= pending->deadline) {
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        const auto executor_index = selectExecutorIndex();
        auto &executor = executors_.at(executor_index);
        const auto execution_started = Clock::now();

        ExecutionResponse response;
        const bool completed =
            runInference(*executor, pending->request, pending->deadline, response);
        const auto execution_finished = Clock::now();
        decrementInflight(executor_index);

        if (!completed) {
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }
        if (execution_finished >= pending->deadline) {
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        recordLatencies(pending->enqueued_at, execution_started, execution_finished);
        SchedulerResult result;
        result.response = std::move(response);
        if (!result.response.ok) {
            result.ok = false;
        }
        fulfillResult(pending, std::move(result));
    }

    void workerLoop() {
        while (true) {
            const auto pending = popNextPending();
            if (pending == nullptr) {
                return;
            }

            const auto batch = formBatch(pending);
            processBatch(std::move(batch));
        }
    }

    std::vector<std::unique_ptr<Executor>> executors_;
    SchedulerConfig config_;
    std::string model_name_;

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<PendingRequest>> queue_;
    std::atomic<bool> draining_{false};

    std::vector<std::thread> workers_;
    mutable std::mutex inflight_mutex_;
    std::vector<size_t> inflight_;

    mutable std::mutex metrics_mutex_;
    SchedulerMetricsSnapshot metrics_;
};

} // namespace

std::unique_ptr<Scheduler> makeModelScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                              SchedulerConfig config, std::string model_name) {
    return std::make_unique<ModelScheduler>(std::move(executors), config, std::move(model_name));
}
