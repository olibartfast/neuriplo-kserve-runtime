#include "KServeRuntime.hpp"

#include "KServeErrors.hpp"
#include "KServeV2Codec.hpp"
#include "Logging.hpp"
#include "Scheduler.hpp"

#include <chrono>
#include <nlohmann/json.hpp>
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
    if (handle.state == ModelState::Failed) {
        const auto message =
            handle.load_error.has_value() ? *handle.load_error : "model failed to load";
        return error(503, KServeErrors::Unavailable, message);
    }
    if (handle.state == ModelState::Unavailable) {
        return error(503, KServeErrors::Unavailable, "model is unavailable");
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

} // namespace

KServeRuntime::KServeRuntime(ModelRegistry registry, MetricsRegistry &metrics)
    : registry_(std::move(registry)), metrics_(metrics) {}

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
    const auto *handle = registry_.findHandleVersion(model_name, model_version);
    if (handle == nullptr) {
        return error(404, KServeErrors::ModelNotFound, "model not found: " + model_name);
    }
    if (!handle->isReady()) {
        if (handle->scheduler != nullptr && handle->scheduler->isDraining()) {
            return error(503, KServeErrors::Unavailable, "model is draining");
        }
        return modelStateError(*handle);
    }
    if (handle->scheduler == nullptr) {
        return error(503, KServeErrors::Unavailable, "model scheduler is unavailable");
    }

    const auto parsed = parseInferenceRequest(request.body, handle->metadata);
    if (!parsed.ok) {
        return error(400, KServeErrors::InvalidArgument, parsed.error_message);
    }

    const std::string request_id = parsed.request.id.has_value() ? *parsed.request.id : "";

    // Trace Span: admit
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

    // Trace Span: queue (enqueue start)
    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: queue";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        defaultLogger().event(span);
    }

    ExecutionRequest execution_request;
    execution_request.id = parsed.request.id;
    execution_request.inputs = parsed.request.inputs;
    execution_request.requested_outputs = parsed.request.requested_outputs;

    auto scheduled = handle->scheduler->submit(std::move(execution_request));

    if (!scheduled.ok) {
        const auto code = scheduled.error_code.empty() ? errorCodeFor(scheduled.scheduler_error)
                                                       : scheduled.error_code;
        const auto msg = scheduled.error_message.empty()
                             ? errorMessageFor(scheduled.scheduler_error)
                             : scheduled.error_message;
        const int status = KServeErrors::httpStatusForCode(code);

        metrics_.recordInferRequest(model_name, request.method, status, 0, 0, 0);

        HttpResponse err_res = error(status, code, msg);

        // Trace Span: respond (on queue failure)
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

    // Trace Span: batch form
    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: batch form";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        span.batch_size = scheduled.batch_size;
        span.queue_latency_ns = scheduled.queue_latency_ns;
        defaultLogger().event(span);
    }

    // Trace Span: infer (execution starts/completes)
    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: infer";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        span.batch_size = scheduled.batch_size;
        span.infer_latency_ns = scheduled.execution_latency_ns;
        defaultLogger().event(span);
    }

    // Trace Span: split (results split from batch)
    {
        LogEvent span;
        span.severity = "info";
        span.message = "span: split";
        span.request_id = request_id;
        span.model = model_name;
        span.model_version = model_version;
        span.route = request.path;
        span.batch_size = scheduled.batch_size;
        defaultLogger().event(span);
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

        // Trace Span: respond (on execution failure)
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

    // Trace Span: respond (on success)
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
