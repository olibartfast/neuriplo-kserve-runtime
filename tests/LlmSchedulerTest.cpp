#include "Executor.hpp"
#include "RuntimeConfig.hpp"
#include "Scheduler.hpp"
#include "Test.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

ModelMetadata llmMetadata() {
    ModelMetadata metadata;
    metadata.name = "llm-demo";
    metadata.versions = {"1"};
    metadata.platform = "test";
    metadata.inputs.push_back({"prompt", "BYTES", {1}});
    metadata.outputs.push_back({"text", "BYTES", {1}});
    return metadata;
}

ExecutionRequest llmRequest(std::string prompt, size_t max_tokens = 32) {
    ExecutionRequest request;
    request.id = "req-1";
    InputTensor input;
    input.name = "prompt";
    input.datatype = "BYTES";
    input.shape = {1};
    input.string_data = {std::move(prompt)};
    request.inputs.push_back(std::move(input));
    request.requested_outputs = {"text"};
    LlmGenerationParams params;
    params.max_tokens = max_tokens;
    request.llm_params = params;
    return request;
}

class LlmStubExecutor final : public Executor {
  public:
    explicit LlmStubExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        ExecutionResponse response;
        OutputTensor output;
        output.name = "text";
        output.datatype = "BYTES";
        output.shape = {1};
        output.string_data = {request.inputs.front().string_data.front()};
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

class SlowLlmExecutor final : public Executor {
  public:
    explicit SlowLlmExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        ExecutionResponse response;
        OutputTensor output;
        output.name = "text";
        output.datatype = "BYTES";
        output.shape = {1};
        output.string_data = {request.inputs.front().string_data.front()};
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

LlmSchedulerConfig testConfig() {
    LlmSchedulerConfig config;
    config.max_queue_size = 8;
    config.request_timeout_ms = 5000;
    config.instances = 1;
    config.context_length = 128;
    config.kv_cache_slots = 1;
    config.max_tokens = 64;
    return config;
}

} // namespace

TEST_CASE(llm_scheduler_rejects_missing_bytes_prompt) {
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(std::make_unique<LlmStubExecutor>(llmMetadata()));
    const auto scheduler = makeLlmScheduler(std::move(executors), testConfig(), "llm-demo");

    ExecutionRequest request;
    request.requested_outputs = {"text"};
    const auto result = scheduler->submit(std::move(request));
    REQUIRE(!result.ok);
    REQUIRE_EQ(result.error_code, "INVALID_ARGUMENT");
}

TEST_CASE(llm_scheduler_rejects_prompt_over_context_length) {
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(std::make_unique<LlmStubExecutor>(llmMetadata()));
    auto config = testConfig();
    config.context_length = 8;
    const auto scheduler = makeLlmScheduler(std::move(executors), config, "llm-demo");

    const auto result = scheduler->submit(llmRequest(std::string(64, 'x')));
    REQUIRE(!result.ok);
    REQUIRE_EQ(result.error_code, "INVALID_ARGUMENT");
}

TEST_CASE(llm_scheduler_runs_bytes_prompt) {
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(std::make_unique<LlmStubExecutor>(llmMetadata()));
    const auto scheduler = makeLlmScheduler(std::move(executors), testConfig(), "llm-demo");

    const auto result = scheduler->submit(llmRequest("Explain KServe briefly."));
    REQUIRE(result.ok);
    REQUIRE_EQ(result.response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(result.response.outputs[0].datatype, "BYTES");
    REQUIRE_EQ(result.response.outputs[0].string_data[0], "Explain KServe briefly.");
}

TEST_CASE(llm_scheduler_limits_kv_cache_slots) {
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(std::make_unique<SlowLlmExecutor>(llmMetadata()));
    auto config = testConfig();
    config.kv_cache_slots = 1;
    config.request_timeout_ms = 500;
    const auto scheduler = makeLlmScheduler(std::move(executors), config, "llm-demo");

    auto first = scheduler->submit(llmRequest("first"));
    auto second = scheduler->submit(llmRequest("second"));

    const bool first_ok = first.ok;
    const bool second_ok = second.ok;
    REQUIRE(first_ok || second_ok);
    if (!first_ok) {
        REQUIRE_EQ(first.error_code, "DEADLINE_EXCEEDED");
    }
    if (!second_ok) {
        REQUIRE_EQ(second.error_code, "DEADLINE_EXCEEDED");
    }
}
