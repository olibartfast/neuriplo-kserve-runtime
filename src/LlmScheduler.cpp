#include "Scheduler.hpp"
#include "Tokenizer.hpp"

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
    return makeSchedulerError(SchedulerError::Overloaded, "QUEUE_FULL", "request queue is full");
}

SchedulerResult makeInvalidArgumentResult(std::string message) {
    SchedulerResult result;
    result.ok = false;
    result.error_code = "INVALID_ARGUMENT";
    result.error_message = std::move(message);
    return result;
}

SchedulerResult makeMemoryPressureResult() {
    return makeSchedulerError(SchedulerError::Overloaded, "QUEUE_FULL",
                              "request rejected due to KV cache memory pressure");
}

uint64_t elapsedNs(const TimePoint &start, const TimePoint &end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

const InputTensor *findBytesPrompt(const ExecutionRequest &request) {
    for (const auto &input : request.inputs) {
        if (isBytesDatatype(input.datatype)) {
            return &input;
        }
    }
    return nullptr;
}

size_t effectiveMaxTokens(const ExecutionRequest &request, const LlmSchedulerConfig &config) {
    if (request.llm_params && request.llm_params->max_tokens) {
        return *request.llm_params->max_tokens;
    }
    return config.max_tokens;
}

std::optional<std::string> validateLlmRequest(const ExecutionRequest &request,
                                              const LlmSchedulerConfig &config,
                                              size_t estimated_tokens) {
    const auto *prompt = findBytesPrompt(request);
    if (prompt == nullptr) {
        return "LLM request requires a BYTES prompt input";
    }
    if (prompt->string_data.empty()) {
        return "LLM prompt must not be empty";
    }

    if (estimated_tokens > config.context_length) {
        return "prompt exceeds context length (estimated " + std::to_string(estimated_tokens) +
               " tokens)";
    }

    const auto max_tokens = effectiveMaxTokens(request, config);
    if (max_tokens == 0 || max_tokens > config.max_tokens) {
        return "max_tokens exceeds configured limit";
    }

    if (estimated_tokens + max_tokens > config.context_length) {
        return "prompt + max_tokens exceeds context length";
    }

    if (request.llm_params) {
        if (request.llm_params->temperature && *request.llm_params->temperature < 0.0) {
            return "temperature must be greater than or equal to 0";
        }
        if (request.llm_params->top_p &&
            (*request.llm_params->top_p <= 0.0 || *request.llm_params->top_p > 1.0)) {
            return "top_p must be in (0, 1]";
        }
    }

    return std::nullopt;
}

class LlmScheduler final : public Scheduler {
  public:
    LlmScheduler(std::vector<std::unique_ptr<Executor>> executors, LlmSchedulerConfig config,
                 std::string model_name, std::unique_ptr<Tokenizer> tokenizer)
        : executors_(std::move(executors)), config_(config), model_name_(std::move(model_name)),
          tokenizer_(std::move(tokenizer)) {
        if (executors_.empty()) {
            throw std::invalid_argument("scheduler requires at least one executor");
        }
        if (config_.kv_cache_slots == 0) {
            throw std::invalid_argument("kv cache slots must be greater than 0");
        }
        inflight_.assign(executors_.size(), 0);
        workers_.reserve(config_.kv_cache_slots);
        for (size_t worker_index = 0; worker_index < config_.kv_cache_slots; ++worker_index) {
            (void)worker_index;
            workers_.emplace_back([this]() { workerLoop(); });
        }
    }

    ~LlmScheduler() override {
        beginDrain();
    }

    SchedulerResult submit(ExecutionRequest request) override {
        std::string prompt_text;
        const auto *prompt = findBytesPrompt(request);
        if (prompt && !prompt->string_data.empty()) {
            prompt_text = prompt->string_data.front();
        }

        const size_t estimated_tokens = tokenizer_->countTokens(prompt_text);

        if (const auto validation_error = validateLlmRequest(request, config_, estimated_tokens)) {
            return makeInvalidArgumentResult(*validation_error);
        }

        if (isDraining()) {
            return makeDrainingResult();
        }

        if (config_.memory_budget_bytes > 0) {
            const size_t estimated_context_bytes = estimated_tokens * memory_per_token_bytes;
            const size_t active_context_bytes =
                active_decodes_.load(std::memory_order_acquire) * estimated_context_bytes;
            if (active_context_bytes + estimated_context_bytes > config_.memory_budget_bytes) {
                incrementMetric(&SchedulerMetricsSnapshot::requests_memory_pressure_rejected);
                return makeMemoryPressureResult();
            }
        }

        const auto enqueued_at = Clock::now();
        const auto deadline = enqueued_at + std::chrono::milliseconds(config_.request_timeout_ms);
        auto promise = std::make_shared<std::promise<SchedulerResult>>();
        auto future = promise->get_future();

        auto cancel_token = request.cancel_token ? request.cancel_token : makeCancelToken();
        request.cancel_token = cancel_token;

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

        cancel_token->store(true, std::memory_order_release);
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
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (draining_.exchange(true, std::memory_order_acq_rel)) {
                return;
            }
        }

        queue_cv_.notify_all();
        slots_cv_.notify_all();
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
        snapshot.kv_cache_slots_total = config_.kv_cache_slots;
        snapshot.kv_cache_slots_active = in_flight;
        return snapshot;
    }

  private:
    static constexpr size_t memory_per_token_bytes = 2;

    size_t currentQueueDepth() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    size_t currentInFlight() const {
        return active_decodes_.load(std::memory_order_acquire);
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

    bool acquireDecodeSlot(const TimePoint &deadline) {
        std::unique_lock<std::mutex> lock(slots_mutex_);
        const bool woke = slots_cv_.wait_until(lock, deadline, [this]() {
            return draining_.load(std::memory_order_acquire) ||
                   active_decodes_.load(std::memory_order_acquire) < config_.kv_cache_slots;
        });
        if (!woke) {
            return false;
        }
        if (draining_.load(std::memory_order_acquire)) {
            return false;
        }
        if (active_decodes_.load(std::memory_order_acquire) >= config_.kv_cache_slots) {
            return false;
        }
        active_decodes_.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }

    void releaseDecodeSlot() {
        active_decodes_.fetch_sub(1, std::memory_order_acq_rel);
        slots_cv_.notify_one();
        queue_cv_.notify_one();
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

    bool runInference(Executor &executor, const ExecutionRequest &request,
                      const TimePoint &deadline, ExecutionResponse &response) {
        try {
            if (request.llm_params && request.llm_params->streaming) {
                response = executor.inferStreaming(
                    request, [](const std::string &) { /* no-op for blocking path */ });
            } else {
                response = executor.infer(request);
            }

            if (request.cancel_token && request.cancel_token->load(std::memory_order_acquire)) {
                return false;
            }
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

    void fulfillResult(const std::shared_ptr<PendingRequest> &pending, SchedulerResult result) {
        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }
        pending->promise->set_value(std::move(result));
    }

    void processSingle(const std::shared_ptr<PendingRequest> &pending) {
        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }

        const auto now = Clock::now();
        if (now >= pending->deadline) {
            if (pending->request.cancel_token) {
                pending->request.cancel_token->store(true, std::memory_order_release);
            }
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        if (!acquireDecodeSlot(pending->deadline)) {
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }
        if (draining_.load(std::memory_order_acquire)) {
            releaseDecodeSlot();
            fulfillResult(pending, makeDrainingResult());
            return;
        }
        if (Clock::now() >= pending->deadline) {
            if (pending->request.cancel_token) {
                pending->request.cancel_token->store(true, std::memory_order_release);
            }
            releaseDecodeSlot();
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
        releaseDecodeSlot();

        if (!completed) {
            if (pending->request.cancel_token) {
                pending->request.cancel_token->store(true, std::memory_order_release);
            }
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        if (pending->cancelled.load(std::memory_order_acquire)) {
            return;
        }
        if (execution_finished >= pending->deadline) {
            if (pending->request.cancel_token) {
                pending->request.cancel_token->store(true, std::memory_order_release);
            }
            incrementMetric(&SchedulerMetricsSnapshot::requests_timed_out);
            fulfillResult(pending, makeTimeoutResult());
            return;
        }

        recordLatencies(pending->enqueued_at, execution_started, execution_finished);
        SchedulerResult result;
        result.queue_latency_ns = elapsedNs(pending->enqueued_at, execution_started);
        result.execution_latency_ns = elapsedNs(execution_started, execution_finished);
        result.total_latency_ns = elapsedNs(pending->enqueued_at, execution_finished);
        result.batch_size = 1;
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
            processSingle(pending);
        }
    }

    std::vector<std::unique_ptr<Executor>> executors_;
    LlmSchedulerConfig config_;
    std::string model_name_;
    std::unique_ptr<Tokenizer> tokenizer_;

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<PendingRequest>> queue_;
    std::atomic<bool> draining_{false};

    std::vector<std::thread> workers_;

    mutable std::mutex slots_mutex_;
    std::condition_variable slots_cv_;
    std::atomic<size_t> active_decodes_{0};

    mutable std::mutex inflight_mutex_;
    std::vector<size_t> inflight_;

    mutable std::mutex metrics_mutex_;
    SchedulerMetricsSnapshot metrics_;
};

} // namespace

std::unique_ptr<Scheduler> makeLlmScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                            LlmSchedulerConfig config, std::string model_name) {
    auto tokenizer = std::make_unique<CharRatioTokenizer>(config.tokens_per_char);
    return std::make_unique<LlmScheduler>(std::move(executors), std::move(config),
                                          std::move(model_name), std::move(tokenizer));
}

std::unique_ptr<Scheduler> makeLlmScheduler(std::vector<std::unique_ptr<Executor>> executors,
                                            LlmSchedulerConfig config, std::string model_name,
                                            std::unique_ptr<Tokenizer> tokenizer) {
    return std::make_unique<LlmScheduler>(std::move(executors), std::move(config),
                                          std::move(model_name), std::move(tokenizer));
}