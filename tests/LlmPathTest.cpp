#include "Executor.hpp"
#include "KServeRuntime.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Scheduler.hpp"
#include "Test.hpp"

#include <memory>
#include <string>

namespace {

class LlmEchoExecutor final : public Executor {
  public:
    explicit LlmEchoExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        ExecutionResponse response;
        OutputTensor output;
        output.name = "text";
        output.datatype = "BYTES";
        output.shape = {1};

        std::string prompt;
        for (const auto &input : request.inputs) {
            if (input.datatype == "BYTES" && !input.string_data.empty()) {
                prompt = input.string_data.front();
                break;
            }
        }
        output.string_data = {"echo: " + prompt};
        response.outputs.push_back(std::move(output));

        LlmResultMetadata llm_meta;
        llm_meta.prompt_tokens = prompt.size() / 4;
        llm_meta.completion_tokens = 3;
        llm_meta.finish_reason = "stop";
        response.llm_metadata = llm_meta;
        return response;
    }

  private:
    ModelMetadata metadata_;
};

KServeRuntime makeLlmRuntime() {
    RuntimeConfig config;
    config.model_name = "llm-model";
    config.backend = "llamacpp";
    config.scheduler_strategy = "llm";
    config.max_tokens = 128;
    static MetricsRegistry metrics;
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "neuriplo_llamacpp";
        metadata.inputs.push_back({"prompt", "BYTES", {1}});
        metadata.outputs.push_back({"text", "BYTES", {1}});
        return std::make_unique<LlmEchoExecutor>(std::move(metadata));
    });
    return KServeRuntime(std::move(registry), metrics);
}

HttpResponse request(const KServeRuntime &runtime, std::string method, std::string path,
                     std::string body = "") {
    HttpRequest req;
    req.method = std::move(method);
    req.path = std::move(path);
    req.body = std::move(body);
    return runtime.handle(req);
}

} // namespace

TEST_CASE(llm_path_completions_returns_response) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(runtime, "POST", "/v1/completions",
                                  R"({"model":"llm-model","prompt":"Hello","max_tokens":32})");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("object":"text_completion")") != std::string::npos);
    REQUIRE(response.body.find(R"("model":"llm-model")") != std::string::npos);
    REQUIRE(response.body.find(R"("echo: Hello")") != std::string::npos);
    REQUIRE(response.body.find(R"("prompt_tokens")") != std::string::npos);
    REQUIRE(response.body.find(R"("completion_tokens")") != std::string::npos);
}

TEST_CASE(llm_path_chat_completions_returns_response) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(
        runtime, "POST", "/v1/chat/completions",
        R"({"model":"llm-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32})");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("object":"chat.completion")") != std::string::npos);
    REQUIRE(response.body.find(R"("model":"llm-model")") != std::string::npos);
    REQUIRE(response.body.find(R"("role":"assistant")") != std::string::npos);
    REQUIRE(response.body.find(R"("prompt_tokens")") != std::string::npos);
}

TEST_CASE(llm_path_embeddings_returns_response) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(runtime, "POST", "/v1/embeddings",
                                  R"({"model":"llm-model","input":"Hello world"})");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("object":"list")") != std::string::npos);
    REQUIRE(response.body.find(R"("embedding")") != std::string::npos);
    REQUIRE(response.body.find(R"("prompt_tokens")") != std::string::npos);
}

TEST_CASE(llm_path_completions_missing_model) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(runtime, "POST", "/v1/completions", R"({"prompt":"Hello"})");
    REQUIRE_EQ(response.status, 400);
}

TEST_CASE(llm_path_chat_completions_missing_messages) {
    const auto runtime = makeLlmRuntime();
    const auto response =
        request(runtime, "POST", "/v1/chat/completions", R"({"model":"llm-model"})");
    REQUIRE_EQ(response.status, 400);
}

TEST_CASE(llm_path_embeddings_missing_input) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(runtime, "POST", "/v1/embeddings", R"({"model":"llm-model"})");
    REQUIRE_EQ(response.status, 400);
}

TEST_CASE(llm_path_completions_unknown_model) {
    const auto runtime = makeLlmRuntime();
    const auto response =
        request(runtime, "POST", "/v1/completions", R"({"model":"nonexistent","prompt":"Hello"})");
    REQUIRE_EQ(response.status, 404);
}

TEST_CASE(llm_path_chat_completions_unknown_model) {
    const auto runtime = makeLlmRuntime();
    const auto response =
        request(runtime, "POST", "/v1/chat/completions",
                R"({"model":"nonexistent","messages":[{"role":"user","content":"Hi"}]})");
    REQUIRE_EQ(response.status, 404);
}

TEST_CASE(llm_path_embeddings_unknown_model) {
    const auto runtime = makeLlmRuntime();
    const auto response =
        request(runtime, "POST", "/v1/embeddings", R"({"model":"nonexistent","input":"Hello"})");
    REQUIRE_EQ(response.status, 404);
}

TEST_CASE(llm_path_kserve_infer_returns_metadata) {
    const auto runtime = makeLlmRuntime();
    const auto response = request(
        runtime, "POST", "/v2/models/llm-model/infer",
        R"({"id":"test-1","inputs":[{"name":"prompt","shape":[1],"datatype":"BYTES","data":["Hello"]}],"parameters":{"max_tokens":32}})");
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"text")") != std::string::npos);
    REQUIRE(response.body.find(R"("id":"test-1")") != std::string::npos);
}

TEST_CASE(llm_path_kserve_infer_rejects_over_context) {
    RuntimeConfig config;
    config.model_name = "tiny-llm";
    config.backend = "llamacpp";
    config.scheduler_strategy = "llm";
    config.context_length = 8;
    config.max_tokens = 4;
    static MetricsRegistry metrics;
    ModelRegistry registry(config, [](const RuntimeConfig &cfg, std::string &error) {
        (void)error;
        ModelMetadata metadata;
        metadata.name = cfg.model_name;
        metadata.versions = {"1"};
        metadata.platform = "neuriplo_llamacpp";
        metadata.inputs.push_back({"prompt", "BYTES", {1}});
        metadata.outputs.push_back({"text", "BYTES", {1}});
        return std::make_unique<LlmEchoExecutor>(std::move(metadata));
    });
    const KServeRuntime runtime(std::move(registry), metrics);

    const auto response = request(
        runtime, "POST", "/v2/models/tiny-llm/infer",
        R"({"id":"over-context","inputs":[{"name":"prompt","shape":[1],"datatype":"BYTES","data":["This is a very long prompt that exceeds the tiny context length"]}]})");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find("context length") != std::string::npos);
}