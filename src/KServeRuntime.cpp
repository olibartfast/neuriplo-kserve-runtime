#include "KServeRuntime.hpp"

#include "KServeV2Codec.hpp"

#include <nlohmann/json.hpp>
#include <sstream>

namespace {

using Json = nlohmann::json;

HttpResponse json(int status, std::string body) {
    HttpResponse response;
    response.status = status;
    response.body = std::move(body);
    return response;
}

HttpResponse error(int status, const std::string &code, const std::string &message) {
    return json(status, Json{{"error", Json{{"code", code}, {"message", message}}}}.dump());
}

bool startsWith(const std::string &value, const std::string &prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string shapeJson(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::string tensorMetadataJson(const TensorMetadata &tensor) {
    return R"({"name":")" + tensor.name + R"(","datatype":")" + tensor.datatype + R"(","shape":)" +
           shapeJson(tensor.shape) + '}';
}

std::string tensorsJson(const std::vector<TensorMetadata> &tensors) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << tensorMetadataJson(tensors[i]);
    }
    out << ']';
    return out.str();
}

std::string versionsJson(const std::vector<std::string> &versions) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < versions.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << '"' << versions[i] << '"';
    }
    out << ']';
    return out.str();
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

} // namespace

KServeRuntime::KServeRuntime(ModelRegistry registry) : registry_(std::move(registry)) {}

HttpResponse KServeRuntime::handle(const HttpRequest &request) const {
    if (request.method.empty()) {
        return error(400, "INVALID_ARGUMENT", "malformed HTTP request");
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
            return infer(model_name, "1", request);
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
                return error(405, "METHOD_NOT_ALLOWED", "method not allowed for route");
            }
        }
        if (suffix.empty() || suffix == "/ready" || suffix == "/infer") {
            return error(405, "METHOD_NOT_ALLOWED", "method not allowed for route");
        }
    }

    return error(404, "NOT_FOUND", "route not found");
}

HttpResponse KServeRuntime::serverMetadata() const {
    return json(200, R"({"name":"neuriplo-kserve-runtime","version":"0.1.0","extensions":[]})");
}

HttpResponse KServeRuntime::live() const {
    return json(200, R"({"live":true})");
}

HttpResponse KServeRuntime::ready() const {
    if (!registry_.allReady()) {
        return error(503, "UNAVAILABLE", "runtime is not ready");
    }
    return json(200, R"({"ready":true})");
}

HttpResponse KServeRuntime::modelMetadata(const std::string &model_name,
                                          const std::string &model_version) const {
    const auto model = model_version.empty() ? registry_.find(model_name)
                                             : registry_.findVersion(model_name, model_version);
    if (!model) {
        return error(404, "MODEL_NOT_FOUND", "model not found: " + model_name);
    }

    const auto body = R"({"name":")" + model->name + R"(","versions":)" +
                      versionsJson(model->versions) + R"(,"platform":")" + model->platform +
                      R"(","inputs":)" + tensorsJson(model->inputs) + R"(,"outputs":)" +
                      tensorsJson(model->outputs) + '}';
    return json(200, body);
}

HttpResponse KServeRuntime::modelReady(const std::string &model_name,
                                       const std::string &model_version) const {
    const auto model = model_version.empty() ? registry_.find(model_name)
                                             : registry_.findVersion(model_name, model_version);
    if (!model) {
        return error(404, "MODEL_NOT_FOUND", "model not found: " + model_name);
    }
    const auto is_ready = model_version.empty() ? registry_.ready(model_name)
                                                : registry_.readyVersion(model_name, model_version);
    if (!is_ready) {
        return error(409, "MODEL_NOT_READY", "model is not ready: " + model_name);
    }
    return json(200, R"({"ready":true})");
}

HttpResponse KServeRuntime::infer(const std::string &model_name, const std::string &model_version,
                                  const HttpRequest &request) const {
    const auto model = registry_.findVersion(model_name, model_version);
    if (!model) {
        return error(404, "MODEL_NOT_FOUND", "model not found: " + model_name);
    }
    if (!registry_.readyVersion(model_name, model_version)) {
        return error(409, "MODEL_NOT_READY", "model is not ready: " + model_name);
    }

    const auto parsed = parseInferenceRequest(request.body, *model);
    if (!parsed.ok) {
        return error(400, "INVALID_ARGUMENT", parsed.error_message);
    }
    return json(200, inferenceResponseJson(model_name, model_version, parsed.request, *model));
}
