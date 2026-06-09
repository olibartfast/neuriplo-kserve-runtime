#include "KServeRuntime.hpp"

#include "BackendRegistry.hpp"
#include "KServeErrors.hpp"
#include "KServeV2Codec.hpp"
#include "Logging.hpp"
#include "OpenAiCodec.hpp"
#include "RequestPipeline.hpp"
#include "Scheduler.hpp"
#include "Tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;

HttpResponse json(int status, std::string body) {
    HttpResponse response;
    response.status = status;
    response.content_type = "application/json";
    response.body = std::move(body);
    return response;
}

HttpResponse error(int status, const std::string &code, const std::string &message) {
    return json(status, Json{{"error", Json{{"code", code}, {"message", message}}}}.dump());
}

bool startsWith(const std::string &value, const std::string &prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string extractModelRouteTail(const std::string &path) {
    constexpr auto prefix = "/v2/models/";
    if (!startsWith(path, prefix)) {
        return "";
    }
    return path.substr(std::string(prefix).size());
}

struct VersionedRoute {
    bool matched = false;
    std::string version;
    std::string suffix;
};

VersionedRoute parseVersionedRoute(const std::string &suffix) {
    constexpr auto prefix = "/versions/";
    if (!startsWith(suffix, prefix)) {
        return {};
    }
    const auto rest = suffix.substr(std::string(prefix).size());
    const auto slash = rest.find('/');
    VersionedRoute route;
    route.matched = true;
    route.version = rest.substr(0, slash);
    route.suffix = slash == std::string::npos ? "" : rest.substr(slash);
    return route;
}

HttpResponse modelStateError(const ModelHandle &handle) {
    switch (handle.state.current()) {
    case ModelState::Failed: {
        const auto message =
            handle.load_error.has_value() ? *handle.load_error : "model failed to load";
        return error(503, KServeErrors::Unavailable, message);
    }
    case ModelState::Unloaded:
        return error(503, KServeErrors::Unavailable, "model has been unloaded");
    case ModelState::Loading:
        return error(503, KServeErrors::Unavailable, "model is still loading");
    case ModelState::Unavailable:
        return error(503, KServeErrors::Unavailable, "model is unavailable");
    default:
        break;
    }
    return error(409, KServeErrors::ModelNotReady, "model is not ready: " + handle.name);
}

std::string errorCodeFor(SchedulerError err) {
    switch (err) {
    case SchedulerError::Overloaded:
        return KServeErrors::QueueFull;
    case SchedulerError::Timeout:
        return KServeErrors::DeadlineExceeded;
    case SchedulerError::Draining:
    case SchedulerError::Unavailable:
        return KServeErrors::Unavailable;
    case SchedulerError::None:
        break;
    }
    return KServeErrors::Internal;
}

std::string errorMessageFor(SchedulerError err) {
    switch (err) {
    case SchedulerError::Overloaded:
        return "request queue is full";
    case SchedulerError::Timeout:
        return "request exceeded deadline";
    case SchedulerError::Draining:
    case SchedulerError::Unavailable:
        return "model is unavailable";
    case SchedulerError::None:
        break;
    }
    return "internal error";
}

std::string generateRequestId() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static thread_local std::uniform_int_distribution<uint8_t> dist(0, 255);

    uint8_t bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = dist(rng);
    }

    bytes[6] = (bytes[6] & 0x0fu) | 0x40u;
    bytes[8] = (bytes[8] & 0x3fu) | 0x80u;

    char buf[37];
    snprintf(buf, sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", bytes[0],
             bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8],
             bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
    return buf;
}

bool modelIsLlm(const ModelHandle &handle, const std::string &backend) {
    return usesLlmScheduler(handle.metadata.platform.find("llamacpp") != std::string::npos ||
                                    handle.metadata.platform.find("cactus") != std::string::npos ||
                                    handle.metadata.platform.find("ggml") != std::string::npos
                                ? "llm"
                                : "tensor",
                            backend);
}

bool backendIsLlm(const std::string &backend) {
    const auto cap = findBackendCapability(backend);
    return cap.has_value() && cap->kind == BackendKind::Llm;
}

size_t estimateTokens(const std::string &text, double tokens_per_char) {
    return static_cast<size_t>(static_cast<double>(text.size()) * tokens_per_char + 0.5);
}

} // namespace

KServeRuntime::KServeRuntime(ModelRegistry &registry, MetricsRegistry &metrics)
    : registry_(registry), metrics_(metrics) {}

HttpResponse KServeRuntime::handle(const HttpRequest &request) const {
    if (request.method.empty()) {
        return error(400, KServeErrors::InvalidArgument, "malformed HTTP request");
    }

    if (request.method == "GET" && request.path == "/v2") {
        return serverMetadata();
    }
    if (request.method == "GET" && request.path == "/v2/health/live") {
        return live();
    }
    if (request.method == "GET" && request.path == "/v2/health/ready") {
        return ready();
    }
    if (request.method == "GET" && request.path == "/metrics") {
        return metricsPage();
    }
    if (request.method == "POST" && request.path == "/v1/completions") {
        return completions(request);
    }
    if (request.method == "POST" && request.path == "/v1/chat/completions") {
        return chatCompletions(request);
    }
    if (request.method == "POST" && request.path == "/v1/embeddings") {
        return embeddings(request);
    }

    const auto route_tail = extractModelRouteTail(request.path);
    if (!route_tail.empty()) {
        const auto slash = route_tail.find('/');
        const auto model_name = route_tail.substr(0, slash);
        const auto suffix = slash == std::string::npos ? "" : route_tail.substr(slash);

        if (request.method == "GET" && suffix.empty()) {
            return modelMetadata(model_name);
        }
        if (request.method == "GET" && suffix == "/ready") {
            return modelReady(model_name);
        }
        if (request.method == "POST" && suffix == "/infer") {
            const auto version = registry_.defaultVersion(model_name);
            if (!version) {
                return error(404, KServeErrors::ModelNotFound, "model not found: " + model_name);
            }
            return infer(model_name, *version, request);
        }

        const auto versioned_route = parseVersionedRoute(suffix);
        if (versioned_route.matched) {
            if (request.method == "GET" && versioned_route.suffix.empty()) {
                return modelMetadata(model_name, versioned_route.version);
            }
            if (request.method == "GET" && versioned_route.suffix == "/ready") {
                return modelReady(model_name, versioned_route.version);
            }
            if (request.method == "POST" && versioned_route.suffix == "/infer") {
                return infer(model_name, versioned_route.version, request);
            }
            if (versioned_route.suffix.empty() || versioned_route.suffix == "/ready" ||
                versioned_route.suffix == "/infer") {
                return error(405, KServeErrors::MethodNotAllowed, "method not allowed for route");
            }
        }
        if (suffix.empty() || suffix == "/ready" || suffix == "/infer") {
            return error(405, KServeErrors::MethodNotAllowed, "method not allowed for route");
        }
    }

    return error(404, KServeErrors::NotFound, "route not found");
}

HttpResponse KServeRuntime::serverMetadata() const {
    return json(200, R"({"name":"neuriplo-kserve-runtime","version":"0.1.0","extensions":[]})");
}

HttpResponse KServeRuntime::live() const {
    return json(200, R"({"live":true})");
}

HttpResponse KServeRuntime::ready() const {
    if (!registry_.allReady()) {
        return error(503, KServeErrors::Unavailable, "runtime is not ready");
    }
    return json(200, R"({"ready":true})");
}

HttpResponse KServeRuntime::metricsPage() const {
    const auto name = registry_.modelName();
    metrics_.setSchedulerMetrics(registry_.schedulerMetrics(name));
    auto body = metrics_.renderMetrics();
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/plain; version=0.0.4";
    response.body = std::move(body);
    return response;
}

HttpResponse KServeRuntime::modelMetadata(const std::string &model_name,
                                          const std::string &model_version) const {
    const auto model = model_version.empty() ? registry_.find(model_name)
                                             : registry_.findVersion(model_name, model_version);
    if (!model) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + model_name);
    }
    return json(200, modelMetadataJson(*model));
}

HttpResponse KServeRuntime::modelReady(const std::string &model_name,
                                       const std::string &model_version) const {
    const auto *handle = model_version.empty()
                             ? registry_.findHandle(model_name)
                             : registry_.findHandleVersion(model_name, model_version);
    if (handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + model_name);
    }
    if (!handle->isReady()) {
        if (handle->scheduler != nullptr && handle->scheduler->isDraining()) {
            return error(503, KServeErrors::Unavailable, "model is draining");
        }
        return modelStateError(*handle);
    }
    return json(200, R"({"ready":true})");
}

HttpResponse KServeRuntime::infer(const std::string &model_name, const std::string &model_version,
                                  const HttpRequest &request) const {
    InferContext ctx;
    ctx.request = &request;
    ctx.model_name = model_name;
    ctx.model_version = model_version;

    RequestPipeline pipeline;
    pipeline
        .addStage([this](InferContext &ctx) {
            auto result = validateInferModel(ctx.model_name, ctx.model_version, ctx.handle);
            if (result) {
                return result;
            }
            return std::optional<HttpResponse>{};
        })
        .addStage([this](InferContext &ctx) {
            auto result = decodeAndAdmit(*ctx.request, ctx.model_name, ctx.model_version,
                                         *ctx.handle, ctx.parsed);
            if (result) {
                return result;
            }
            return std::optional<HttpResponse>{};
        })
        .addStage([this](InferContext &ctx) {
            ctx.scheduled =
                scheduleInfer(ctx.model_name, ctx.model_version, ctx.parsed, *ctx.handle);
            return std::optional<HttpResponse>{};
        })
        .addStage([this](InferContext &ctx) {
            return std::optional(encodeInferResponse(ctx.model_name, ctx.model_version,
                                                     *ctx.request, ctx.parsed, ctx.scheduled));
        });

    return pipeline.run(ctx);
}

std::optional<HttpResponse>
KServeRuntime::validateInferModel(const std::string &model_name, const std::string &model_version,
                                  const ModelHandle *&out_handle) const {
    out_handle = registry_.findHandleVersion(model_name, model_version);
    if (out_handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + model_name);
    }
    if (!out_handle->isReady()) {
        if (out_handle->scheduler != nullptr && out_handle->scheduler->isDraining()) {
            return error(503, KServeErrors::Unavailable, "model is draining");
        }
        return modelStateError(*out_handle);
    }
    if (out_handle->scheduler == nullptr) {
        return error(503, KServeErrors::Unavailable, "model scheduler is unavailable");
    }
    return std::nullopt;
}

std::optional<HttpResponse> KServeRuntime::decodeAndAdmit(const HttpRequest &request,
                                                          const std::string &model_name,
                                                          const std::string &model_version,
                                                          const ModelHandle &handle,
                                                          InferenceParseResult &out_parsed) const {
    out_parsed = parseInferenceRequest(request.body, handle.metadata);
    if (!out_parsed.ok) {
        return error(400, KServeErrors::InvalidArgument, out_parsed.error_message);
    }

    const std::string request_id =
        out_parsed.request.id.has_value() ? *out_parsed.request.id : generateRequestId();

    {
        LogEvent span;
        span.severity = "info";
        span.message =
            "span: admit" + (registry_.logPayloads() ? " - payload: " + request.body : "");
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        span.request_bytes = request.body.size();
        defaultLogger().event(span);
    }

    return std::nullopt;
}

SchedulerResult KServeRuntime::scheduleInfer(const std::string &model_name,
                                             const std::string &model_version,
                                             const InferenceParseResult &parsed,
                                             const ModelHandle &handle) const {
    const std::string request_id =
        parsed.request.id.has_value() ? *parsed.request.id : generateRequestId();

    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: queue";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        defaultLogger().event(span);
    }

    ExecutionRequest execution_request;
    execution_request.id = parsed.request.id;
    execution_request.inputs = parsed.request.inputs;
    execution_request.requested_outputs = parsed.request.requested_outputs;
    execution_request.llm_params = parsed.request.llm_params;

    auto scheduled = handle.scheduler->submit(std::move(execution_request));

    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: batch form";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.batch_size = scheduled.batch_size;
        span.queue_latency_ns = scheduled.queue_latency_ns;
        defaultLogger().event(span);
    }

    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: infer";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.batch_size = scheduled.batch_size;
        span.infer_latency_ns = scheduled.execution_latency_ns;
        defaultLogger().event(span);
    }

    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: split";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.batch_size = scheduled.batch_size;
        defaultLogger().event(span);
    }

    return scheduled;
}

HttpResponse KServeRuntime::encodeInferResponse(const std::string &model_name,
                                                const std::string &model_version,
                                                const HttpRequest &request,
                                                const InferenceParseResult &parsed,
                                                const SchedulerResult &scheduled) const {
    const std::string request_id =
        parsed.request.id.has_value() ? *parsed.request.id : generateRequestId();

    if (!scheduled.ok) {
        const auto code = scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                                       : scheduled.error_code;
        const auto msg = scheduled.error_message.empty()
                             ? errorMessageFor(scheduled.scheduler_error)
                             : scheduled.error_message;
        const int status = KServeErrors::httpStatusForCode(code);

        metrics_.recordInferRequest(model_name, request.method, status, 0, 0, 0);

        HttpResponse err_res = error(status, code, msg);

        {
            LogEvent span;
            span.severity = "warn";
            span.message = "span: respond (failed) - infer request " + code +
                           (registry_.logPayloads() ? " - payload: " + err_res.body : "");
            span.request_id = request_id;
            span.model = model_name;
            span.model_version = model_version;
            span.route = request.path;
            span.status = status;
            span.error_code = code;
            span.request_bytes = request.body.size();
            span.response_bytes = err_res.body.size();
            defaultLogger().event(span);
        }

        return err_res;
    }

    const auto &execution_response = scheduled.response;
    if (!execution_response.ok) {
        const auto status =
            execution_response.error_code == KServeErrors::InvalidArgument ? 400 : 500;
        const auto code = execution_response.error_code.empty() ? KServeErrors::BackendError
                                                                : execution_response.error_code;
        const auto message = execution_response.error_message.empty()
                                 ? "backend inference failed"
                                 : execution_response.error_message;

        metrics_.recordInferRequest(model_name, request.method, status, scheduled.queue_latency_ns,
                                    scheduled.execution_latency_ns, scheduled.total_latency_ns,
                                    scheduled.batch_size);

        HttpResponse err_res = error(status, code, message);

        {
            LogEvent span;
            span.severity = "error";
            span.message = "span: respond (failed) - infer request failed" +
                           (registry_.logPayloads() ? " - payload: " + err_res.body : "");
            span.request_id = request_id;
            span.model = model_name;
            span.model_version = model_version;
            span.route = request.path;
            span.status = status;
            span.queue_latency_ns = scheduled.queue_latency_ns;
            span.infer_latency_ns = scheduled.execution_latency_ns;
            span.total_latency_ns = scheduled.total_latency_ns;
            span.batch_size = scheduled.batch_size;
            span.error_code = code;
            span.request_bytes = request.body.size();
            span.response_bytes = err_res.body.size();
            defaultLogger().event(span);
        }

        return err_res;
    }

    metrics_.recordInferRequest(model_name, request.method, 200, scheduled.queue_latency_ns,
                                scheduled.execution_latency_ns, scheduled.total_latency_ns,
                                scheduled.batch_size);

    HttpResponse success_res = json(
        200, inferenceResponseJson(model_name, model_version, parsed.request, execution_response));

    {
        LogEvent span;
        span.severity = "info";
        span.message =
            "span: respond" + (registry_.logPayloads() ? " - payload: " + success_res.body : "");
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        span.status = 200;
        span.queue_latency_ns = scheduled.queue_latency_ns;
        span.infer_latency_ns = scheduled.execution_latency_ns;
        span.total_latency_ns = scheduled.total_latency_ns;
        span.batch_size = scheduled.batch_size;
        span.request_bytes = request.body.size();
        span.response_bytes = success_res.body.size();
        defaultLogger().event(span);
    }

    return success_res;
}

HttpResponse KServeRuntime::completions(const HttpRequest &request) const {
    OpenAiCompletionRequest comp_req;
    const auto parse_result = parseCompletionRequest(request.body, comp_req);
    if (!parse_result.ok) {
        return error(400, KServeErrors::InvalidArgument, parse_result.error_message);
    }

    const auto *handle = registry_.findHandle(comp_req.model);
    if (handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + comp_req.model);
    }
    if (!handle->isReady()) {
        return error(503, KServeErrors::Unavailable, "model is not ready");
    }
    if (handle->scheduler == nullptr) {
        return error(503, KServeErrors::Unavailable, "model scheduler is unavailable");
    }

    if (comp_req.stream) {
        return completionsStreaming(request, comp_req, *handle);
    }

    InferenceRequest inf_req;
    inf_req.id = generateRequestId();
    inf_req.llm_params = LlmGenerationParams{};
    if (comp_req.max_tokens) {
        inf_req.llm_params->max_tokens = comp_req.max_tokens;
    }
    if (comp_req.temperature) {
        inf_req.llm_params->temperature = comp_req.temperature;
    }
    if (comp_req.top_p) {
        inf_req.llm_params->top_p = comp_req.top_p;
    }

    InputTensor prompt_tensor;
    prompt_tensor.name = "prompt";
    prompt_tensor.datatype = "BYTES";
    prompt_tensor.shape = {1};
    prompt_tensor.string_data = {comp_req.prompt};
    inf_req.inputs.push_back(std::move(prompt_tensor));

    ExecutionRequest exec_req;
    exec_req.id = inf_req.id;
    exec_req.inputs = inf_req.inputs;
    exec_req.llm_params = inf_req.llm_params;

    auto scheduled = handle->scheduler->submit(std::move(exec_req));

    if (!scheduled.ok) {
        const int status = KServeErrors::httpStatusForCode(
            scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                         : scheduled.error_code);
        const auto code = scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                                       : scheduled.error_code;
        return error(status, code, scheduled.error_message);
    }

    if (!scheduled.response.ok) {
        return error(500, KServeErrors::BackendError, scheduled.response.error_message);
    }

    std::string generated_text;
    for (const auto &output : scheduled.response.outputs) {
        if (output.datatype == "BYTES" && !output.string_data.empty()) {
            generated_text += output.string_data.front();
        }
    }

    const auto now = std::chrono::system_clock::now();
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    size_t prompt_tokens = estimateTokens(comp_req.prompt, registry_.tokensPerChar());
    size_t completion_tokens = estimateTokens(generated_text, registry_.tokensPerChar());
    if (scheduled.response.llm_metadata) {
        prompt_tokens = scheduled.response.llm_metadata->prompt_tokens;
        completion_tokens = scheduled.response.llm_metadata->completion_tokens;
    }

    OpenAiCompletionResponse resp;
    resp.id = "cmpl-" + (inf_req.id.has_value() ? *inf_req.id : "");
    resp.object = "text_completion";
    resp.created = created;
    resp.model = comp_req.model;
    resp.text = generated_text;
    resp.finish_reason =
        scheduled.response.llm_metadata ? scheduled.response.llm_metadata->finish_reason : "stop";
    resp.prompt_tokens = prompt_tokens;
    resp.completion_tokens = completion_tokens;

    return json(200, completionResponseJson(resp));
}

HttpResponse KServeRuntime::completionsStreaming(const HttpRequest &request,
                                                 const OpenAiCompletionRequest &comp_req,
                                                 const ModelHandle &handle) const {
    const std::string request_id = "cmpl-" + generateRequestId();
    const auto now = std::chrono::system_clock::now();
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    HttpResponse response;
    response.streaming = true;
    response.stream_callback = [this, &handle, comp_req, request_id,
                                created](StreamWriter &writer) {
        ExecutionRequest exec_req;
        exec_req.id = request_id;
        InputTensor prompt_tensor;
        prompt_tensor.name = "prompt";
        prompt_tensor.datatype = "BYTES";
        prompt_tensor.shape = {1};
        prompt_tensor.string_data = {comp_req.prompt};
        exec_req.inputs.push_back(std::move(prompt_tensor));

        LlmGenerationParams params;
        if (comp_req.max_tokens) {
            params.max_tokens = comp_req.max_tokens;
        }
        if (comp_req.temperature) {
            params.temperature = comp_req.temperature;
        }
        if (comp_req.top_p) {
            params.top_p = comp_req.top_p;
        }
        params.streaming = true;
        exec_req.llm_params = params;

        exec_req.cancel_token = makeCancelToken();

        exec_req.streaming_callback = [&writer, &request_id, &comp_req,
                                       created](const std::string &token) {
            const auto chunk = streamingChunkJson(request_id, "text_completion", comp_req.model,
                                                  created, token, "");
            writer.write(chunk);
        };

        auto scheduled = handle.scheduler->submit(std::move(exec_req));
        if (!scheduled.ok) {
            const auto error_json = Json{{"error", scheduled.error_message}}.dump();
            writer.write("data: " + error_json + "\n\n");
            return;
        }

        const auto finish_chunk = streamingChunkJson(request_id, "text_completion",
                                                     comp_req.model, created, "", "stop");
        writer.write(finish_chunk);
    };

    return response;
}

HttpResponse KServeRuntime::chatCompletions(const HttpRequest &request) const {
    OpenAiChatRequest chat_req;
    const auto parse_result = parseChatRequest(request.body, chat_req);
    if (!parse_result.ok) {
        return error(400, KServeErrors::InvalidArgument, parse_result.error_message);
    }

    const auto *handle = registry_.findHandle(chat_req.model);
    if (handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + chat_req.model);
    }
    if (!handle->isReady()) {
        return error(503, KServeErrors::Unavailable, "model is not ready");
    }
    if (handle->scheduler == nullptr) {
        return error(503, KServeErrors::Unavailable, "model scheduler is unavailable");
    }

    if (chat_req.stream) {
        return chatCompletionsStreaming(request, chat_req, *handle);
    }

    std::string prompt_text;
    for (const auto &msg : chat_req.messages) {
        prompt_text += msg.role + ": " + msg.content + "\n";
    }
    prompt_text += "assistant: ";

    InferenceRequest inf_req;
    inf_req.id = generateRequestId();
    inf_req.llm_params = LlmGenerationParams{};
    if (chat_req.max_tokens) {
        inf_req.llm_params->max_tokens = chat_req.max_tokens;
    }
    if (chat_req.temperature) {
        inf_req.llm_params->temperature = chat_req.temperature;
    }
    if (chat_req.top_p) {
        inf_req.llm_params->top_p = chat_req.top_p;
    }

    InputTensor prompt_tensor;
    prompt_tensor.name = "prompt";
    prompt_tensor.datatype = "BYTES";
    prompt_tensor.shape = {1};
    prompt_tensor.string_data = {std::move(prompt_text)};
    inf_req.inputs.push_back(std::move(prompt_tensor));

    ExecutionRequest exec_req;
    exec_req.id = inf_req.id;
    exec_req.inputs = inf_req.inputs;
    exec_req.llm_params = inf_req.llm_params;

    auto scheduled = handle->scheduler->submit(std::move(exec_req));

    if (!scheduled.ok) {
        const int status = KServeErrors::httpStatusForCode(
            scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                         : scheduled.error_code);
        const auto code = scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                                       : scheduled.error_code;
        return error(status, code, scheduled.error_message);
    }

    if (!scheduled.response.ok) {
        return error(500, KServeErrors::BackendError, scheduled.response.error_message);
    }

    std::string generated_text;
    for (const auto &output : scheduled.response.outputs) {
        if (output.datatype == "BYTES" && !output.string_data.empty()) {
            generated_text += output.string_data.front();
        }
    }

    const auto now = std::chrono::system_clock::now();
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    size_t prompt_tokens =
        estimateTokens(chat_req.messages.begin()->content, registry_.tokensPerChar());
    size_t completion_tokens = estimateTokens(generated_text, registry_.tokensPerChar());
    if (scheduled.response.llm_metadata) {
        prompt_tokens = scheduled.response.llm_metadata->prompt_tokens;
        completion_tokens = scheduled.response.llm_metadata->completion_tokens;
    }

    OpenAiChatResponse resp;
    resp.id = "chatcmpl-" + (inf_req.id.has_value() ? *inf_req.id : "");
    resp.object = "chat.completion";
    resp.created = created;
    resp.model = chat_req.model;
    resp.message.role = "assistant";
    resp.message.content = generated_text;
    resp.finish_reason =
        scheduled.response.llm_metadata ? scheduled.response.llm_metadata->finish_reason : "stop";
    resp.prompt_tokens = prompt_tokens;
    resp.completion_tokens = completion_tokens;

    return json(200, chatResponseJson(resp));
}

HttpResponse KServeRuntime::chatCompletionsStreaming(const HttpRequest &request,
                                                     const OpenAiChatRequest &chat_req,
                                                     const ModelHandle &handle) const {
    const std::string request_id = "chatcmpl-" + generateRequestId();
    const auto now = std::chrono::system_clock::now();
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    HttpResponse response;
    response.streaming = true;
    response.stream_callback = [this, &handle, chat_req, request_id,
                                created](StreamWriter &writer) {
        std::string prompt_text;
        for (const auto &msg : chat_req.messages) {
            prompt_text += msg.role + ": " + msg.content + "\n";
        }
        prompt_text += "assistant: ";

        ExecutionRequest exec_req;
        exec_req.id = request_id;
        InputTensor prompt_tensor;
        prompt_tensor.name = "prompt";
        prompt_tensor.datatype = "BYTES";
        prompt_tensor.shape = {1};
        prompt_tensor.string_data = {prompt_text};
        exec_req.inputs.push_back(std::move(prompt_tensor));

        LlmGenerationParams params;
        if (chat_req.max_tokens) {
            params.max_tokens = chat_req.max_tokens;
        }
        if (chat_req.temperature) {
            params.temperature = chat_req.temperature;
        }
        if (chat_req.top_p) {
            params.top_p = chat_req.top_p;
        }
        params.streaming = true;
        exec_req.llm_params = params;
        exec_req.cancel_token = makeCancelToken();

        exec_req.streaming_callback = [&writer, &request_id, &chat_req,
                                       created](const std::string &token) {
            const auto chunk = streamingChunkJson(request_id, "chat.completion.chunk",
                                                  chat_req.model, created, token, "");
            writer.write(chunk);
        };

        auto scheduled = handle.scheduler->submit(std::move(exec_req));
        if (!scheduled.ok) {
            const auto error_json = Json{{"error", scheduled.error_message}}.dump();
            writer.write("data: " + error_json + "\n\n");
            return;
        }

        const auto finish_chunk = streamingChunkJson(request_id, "chat.completion.chunk",
                                                     chat_req.model, created, "", "stop");
        writer.write(finish_chunk);
    };

    return response;
}

HttpResponse KServeRuntime::embeddings(const HttpRequest &request) const {
    OpenAiEmbeddingRequest emb_req;
    const auto parse_result = parseEmbeddingRequest(request.body, emb_req);
    if (!parse_result.ok) {
        return error(400, KServeErrors::InvalidArgument, parse_result.error_message);
    }

    const auto *handle = registry_.findHandle(emb_req.model);
    if (handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + emb_req.model);
    }
    if (!handle->isReady()) {
        return error(503, KServeErrors::Unavailable, "model is not ready");
    }
    if (handle->scheduler == nullptr) {
        return error(503, KServeErrors::Unavailable, "model scheduler is unavailable");
    }

    InputTensor input_tensor;
    input_tensor.name = "input";
    input_tensor.datatype = "BYTES";
    input_tensor.shape = {1};
    input_tensor.string_data = {emb_req.input};

    ExecutionRequest exec_req;
    exec_req.id = generateRequestId();
    exec_req.inputs.push_back(std::move(input_tensor));

    auto scheduled = handle->scheduler->submit(std::move(exec_req));

    if (!scheduled.ok) {
        const int status = KServeErrors::httpStatusForCode(
            scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                         : scheduled.error_code);
        const auto code = scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                                       : scheduled.error_code;
        return error(status, code, scheduled.error_message);
    }

    if (!scheduled.response.ok) {
        return error(500, KServeErrors::BackendError, scheduled.response.error_message);
    }

    std::vector<double> embedding;
    for (const auto &output : scheduled.response.outputs) {
        if (output.datatype != "BYTES") {
            embedding = output.data;
            break;
        }
    }

    size_t prompt_tokens = estimateTokens(emb_req.input, registry_.tokensPerChar());

    OpenAiEmbeddingResponse resp;
    resp.model = emb_req.model;
    resp.embedding = std::move(embedding);
    resp.prompt_tokens = prompt_tokens;

    return json(200, embeddingResponseJson(resp));
}