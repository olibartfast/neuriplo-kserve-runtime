#include "BackendRegistry.hpp"
#include "DynamicBatcher.hpp"
#include "Executor.hpp"
#include "HttpServer.hpp"
#include "KServeErrors.hpp"
#include "KServeRuntime.hpp"
#include "KServeV2Codec.hpp"
#include "MetricsRegistry.hpp"
#include "ModelHandle.hpp"
#include "ModelMetadata.hpp"
#include "ModelRegistry.hpp"
#include "ModelState.hpp"
#include "ModelStateMachine.hpp"
#include "RequestPipeline.hpp"
#include "RuntimeConfig.hpp"
#include "Scheduler.hpp"
#include "SchedulerMetrics.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <future>
#include <mutex>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
#include "NeuriploAdapter.hpp"
#include <filesystem>
#endif

namespace {

int findFreePort() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        throw std::runtime_error("socket failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error("bind failed");
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) < 0) {
        ::close(fd);
        throw std::runtime_error("getsockname failed");
    }
    const auto port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

struct RawHttpResult {
    int status = 0;
    std::string body;
    std::map<std::string, std::string> headers;
};

RawHttpResult sendRawHttp(int port, const std::string &request_str) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        throw std::runtime_error("socket failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        throw std::runtime_error("connect failed");
    }
    ::send(fd, request_str.data(), request_str.size(), 0);

    std::string raw;
    char buffer[8192];
    while (true) {
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0)
            break;
        raw.append(buffer, static_cast<size_t>(n));
    }
    ::close(fd);

    RawHttpResult result;
    const auto sp1 = raw.find(' ');
    const auto sp2 = raw.find(' ', sp1 + 1);
    if (sp1 != std::string::npos && sp2 != std::string::npos) {
        result.status = std::stoi(raw.substr(sp1 + 1, sp2 - sp1 - 1));
    }
    const auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end != std::string::npos) {
        result.body = raw.substr(hdr_end + 4);

        std::istringstream hdr_stream(raw.substr(0, hdr_end));
        std::string line;
        while (std::getline(hdr_stream, line, '\r')) {
            if (!line.empty() && line.front() == '\n') {
                line.erase(line.begin());
            }
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                auto key = line.substr(0, colon);
                auto val = line.substr(colon + 1);
                while (!val.empty() && val.front() == ' ')
                    val.erase(val.begin());
                result.headers[key] = val;
            }
        }
    }
    return result;
}

std::string httpGet(int /*port*/, const std::string &path) {
    return "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
}

class MetadataStubExecutor final : public Executor {
  public:
    explicit MetadataStubExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        ExecutionResponse response;
        for (const auto &output : metadata_.outputs) {
            if (!request.requested_outputs.empty() &&
                std::find(request.requested_outputs.begin(), request.requested_outputs.end(),
                          output.name) == request.requested_outputs.end()) {
                continue;
            }
            OutputTensor tensor;
            tensor.name = output.name;
            tensor.datatype = output.datatype;
            tensor.shape = output.shape;
            size_t element_count = 1;
            for (const auto dim : output.shape) {
                element_count *= dim > 0 ? static_cast<size_t>(dim) : 1;
            }
            tensor.bytes =
                tensorBytesFromDoubles(tensor.datatype, std::vector<double>(element_count, 0.0));
            response.outputs.push_back(std::move(tensor));
        }
        return response;
    }

  private:
    ModelMetadata metadata_;
};

std::string httpPost(int port, const std::string &path, const std::string &body) {
    return "POST " + path +
           " HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " +
           std::to_string(body.size()) + "\r\n\r\n" + body;
}

class StubServer {
  public:
    explicit StubServer(size_t max_request_bytes = 2048, RuntimeConfig config = RuntimeConfig{})
        : config_(config), port_(findFreePort()), registry_(config_), metrics_(),
          runtime_(registry_, metrics_),
          server_(
              "127.0.0.1", port_, [this](const HttpRequest &req) { return runtime_.handle(req); },
              max_request_bytes) {
        thread_ = std::thread([this] { server_.run(); });
        waitReady();
    }

    ~StubServer() {
        server_.stop();
        if (thread_.joinable())
            thread_.join();
    }

    int port() const {
        return port_;
    }
    const ModelRegistry &registry() const {
        return registry_;
    }
    const MetricsRegistry &metrics() const {
        return metrics_;
    }
    const KServeRuntime &runtime() const {
        return runtime_;
    }

  private:
    void waitReady() const {
        for (int i = 0; i < 100; ++i) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd >= 0) {
                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_port = htons(static_cast<uint16_t>(port_));
                ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
                if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                    ::close(fd);
                    return;
                }
                ::close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("test server did not start");
    }

    RuntimeConfig config_;
    int port_;
    ModelRegistry registry_;
    MetricsRegistry metrics_;
    KServeRuntime runtime_;
    HttpServer server_;
    std::thread thread_;
};

class CustomExecutorServer {
  public:
    CustomExecutorServer(
        std::function<std::unique_ptr<Executor>(const RuntimeConfig &, std::string &)> factory,
        RuntimeConfig config = RuntimeConfig{})
        : config_(config), port_(findFreePort()), registry_(config_, std::move(factory)),
          metrics_(), runtime_(registry_, metrics_),
          server_(
              "127.0.0.1", port_, [this](const HttpRequest &req) { return runtime_.handle(req); },
              4096) {
        thread_ = std::thread([this] { server_.run(); });
        waitReady();
    }

    ~CustomExecutorServer() {
        server_.stop();
        if (thread_.joinable())
            thread_.join();
    }

    int port() const {
        return port_;
    }
    ModelRegistry &registry() {
        return registry_;
    }
    MetricsRegistry &metrics() {
        return metrics_;
    }

  private:
    void waitReady() const {
        for (int i = 0; i < 100; ++i) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd >= 0) {
                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_port = htons(static_cast<uint16_t>(port_));
                ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
                if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                    ::close(fd);
                    return;
                }
                ::close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("test server did not start");
    }

    RuntimeConfig config_;
    int port_;
    ModelRegistry registry_;
    MetricsRegistry metrics_;
    KServeRuntime runtime_;
    HttpServer server_;
    std::thread thread_;
};

std::string validInferBody(const std::string &model_name = "demo",
                           const std::string &input_name = "input",
                           const std::vector<int64_t> &shape = {1, 3, 224, 224}) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            shape_str += ",";
        shape_str += std::to_string(shape[i]);
    }
    shape_str += "]";

    return R"({"id":"e2e-1","inputs":[{"name":")" + input_name + R"(","shape":)" + shape_str +
           R"(,"datatype":"FP32","data":[]}],"outputs":[{"name":"output"}]})";
}

void requireNear(double a, double b, double eps = 1e-6) {
    REQUIRE(std::fabs(a - b) < eps);
}

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO

using Bytes = std::vector<uint8_t>;

void appendVarint(Bytes &bytes, uint64_t value) {
    while (value >= 0x80) {
        bytes.push_back(static_cast<uint8_t>(value | 0x80));
        value >>= 7;
    }
    bytes.push_back(static_cast<uint8_t>(value));
}

void appendKey(Bytes &bytes, int field_number, int wire_type) {
    appendVarint(bytes,
                 (static_cast<uint64_t>(field_number) << 3U) | static_cast<uint64_t>(wire_type));
}

void appendString(Bytes &bytes, int field_number, const std::string &value) {
    appendKey(bytes, field_number, 2);
    appendVarint(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendMessage(Bytes &bytes, int field_number, const Bytes &message) {
    appendKey(bytes, field_number, 2);
    appendVarint(bytes, message.size());
    bytes.insert(bytes.end(), message.begin(), message.end());
}

void appendInt64(Bytes &bytes, int field_number, int64_t value) {
    appendKey(bytes, field_number, 0);
    appendVarint(bytes, static_cast<uint64_t>(value));
}

void appendInt32(Bytes &bytes, int field_number, int32_t value) {
    appendKey(bytes, field_number, 0);
    appendVarint(bytes, static_cast<uint64_t>(value));
}

Bytes tensorShapeDimension(int64_t value) {
    Bytes b;
    appendInt64(b, 1, value);
    return b;
}

Bytes tensorShape(const std::vector<int64_t> &shape) {
    Bytes b;
    for (auto d : shape)
        appendMessage(b, 1, tensorShapeDimension(d));
    return b;
}

Bytes tensorType(const std::vector<int64_t> &shape) {
    Bytes b;
    appendInt32(b, 1, 1);
    appendMessage(b, 2, tensorShape(shape));
    return b;
}

Bytes typeProto(const std::vector<int64_t> &shape) {
    Bytes b;
    appendMessage(b, 1, tensorType(shape));
    return b;
}

Bytes valueInfo(const std::string &name, const std::vector<int64_t> &shape) {
    Bytes b;
    appendString(b, 1, name);
    appendMessage(b, 2, typeProto(shape));
    return b;
}

Bytes identityNode() {
    Bytes b;
    appendString(b, 1, "input");
    appendString(b, 2, "output");
    appendString(b, 3, "identity");
    appendString(b, 4, "Identity");
    return b;
}

Bytes graphProto(const std::vector<int64_t> &shape) {
    Bytes b;
    appendMessage(b, 1, identityNode());
    appendString(b, 2, "identity_graph");
    appendMessage(b, 11, valueInfo("input", shape));
    appendMessage(b, 12, valueInfo("output", shape));
    return b;
}

Bytes opsetImport() {
    Bytes b;
    appendInt64(b, 2, 13);
    return b;
}

Bytes identityOnnxModel(const std::vector<int64_t> &shape = {1, 3, 4, 4}) {
    Bytes b;
    appendInt64(b, 1, 7);
    appendString(b, 2, "neuriplo-kserve-runtime");
    appendMessage(b, 7, graphProto(shape));
    appendMessage(b, 8, opsetImport());
    return b;
}

std::string writeOnnxModel(const Bytes &model_bytes, const std::string &filename) {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to open ONNX model for writing");
    out.write(reinterpret_cast<const char *>(model_bytes.data()),
              static_cast<std::streamsize>(model_bytes.size()));
    return path.string();
}

#endif

} // namespace

TEST_CASE(e2e_health_live_returns_200) {
    const StubServer server;
    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/v2/health/live"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("live":true)") != std::string::npos);
}

TEST_CASE(e2e_health_ready_returns_200_when_model_loaded) {
    const StubServer server;
    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/v2/health/ready"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(e2e_server_metadata) {
    const StubServer server;
    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/v2"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("name":"neuriplo-kserve-runtime")") != std::string::npos);
    REQUIRE(resp.body.find(R"("version":"0.2.0")") != std::string::npos);
    REQUIRE(resp.body.find(R"("extensions")") != std::string::npos);
}

TEST_CASE(e2e_model_metadata_returns_input_output_shapes) {
    const StubServer server;
    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/v2/models/demo"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("name":"demo")") != std::string::npos);
    REQUIRE(resp.body.find(R"("inputs")") != std::string::npos);
    REQUIRE(resp.body.find(R"("outputs")") != std::string::npos);
    REQUIRE(resp.body.find(R"("platform")") != std::string::npos);
}

TEST_CASE(e2e_model_ready_returns_true) {
    const StubServer server;
    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/v2/models/demo/ready"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("ready":true)") != std::string::npos);
}

TEST_CASE(e2e_infer_returns_output_tensor) {
    const StubServer server;
    const auto body = validInferBody();
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", body));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("model_name":"demo")") != std::string::npos);
    REQUIRE(resp.body.find(R"("outputs")") != std::string::npos);
    REQUIRE(resp.body.find(R"("id":"e2e-1")") != std::string::npos);
}

TEST_CASE(e2e_infer_id_echo) {
    const StubServer server;
    const auto body =
        R"({"id":"echo-test-42","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})";
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", body));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("id":"echo-test-42")") != std::string::npos);
}

TEST_CASE(e2e_infer_unknown_model_returns_404) {
    const StubServer server;
    const auto body = validInferBody("nonexistent");
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/nonexistent/infer", body));
    REQUIRE_EQ(resp.status, 404);
    REQUIRE(resp.body.find(R"("MODEL_NOT_FOUND")") != std::string::npos);
}

TEST_CASE(e2e_infer_malformed_json_returns_400) {
    const StubServer server;
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", "{bad"));
    REQUIRE_EQ(resp.status, 400);
    REQUIRE(resp.body.find(R"("INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(e2e_infer_oversized_request_returns_413) {
    const StubServer server(2048);
    const auto big = std::string(4096, 'x');
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", big));
    REQUIRE_EQ(resp.status, 413);
    REQUIRE(resp.body.find(R"("PAYLOAD_TOO_LARGE")") != std::string::npos);
}

TEST_CASE(e2e_wrong_method_returns_405) {
    const StubServer server;
    REQUIRE_EQ(sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo", "")).status,
               405);
    REQUIRE_EQ(sendRawHttp(server.port(), httpGet(server.port(), "/v2/models/demo/infer")).status,
               405);
}

TEST_CASE(e2e_metrics_endpoint_exposes_prometheus) {
    const StubServer server;
    const auto infer_body = validInferBody();
    sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", infer_body));

    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/metrics"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.headers.count("Content-Type") > 0);
    REQUIRE(resp.headers.at("Content-Type").find("text/plain") != std::string::npos);

    REQUIRE(resp.body.find("neuriplo_http_infer_requests_total") != std::string::npos);
    REQUIRE(resp.body.find("neuriplo_scheduler_queue_depth") != std::string::npos);
    REQUIRE(resp.body.find("neuriplo_scheduler_in_flight_requests") != std::string::npos);
    REQUIRE(resp.body.find("model=\"demo\"") != std::string::npos);
    REQUIRE(resp.body.find("neuriplo_scheduler_queue_latency_seconds_bucket") != std::string::npos);
    REQUIRE(resp.body.find("neuriplo_scheduler_total_latency_seconds_bucket") != std::string::npos);
}

TEST_CASE(e2e_metrics_include_scheduler_metrics_after_infer) {
    const StubServer server;
    const auto infer_body = validInferBody();
    sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", infer_body));

    const auto resp = sendRawHttp(server.port(), httpGet(server.port(), "/metrics"));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find("neuriplo_scheduler_requests_accepted_total") != std::string::npos);
    REQUIRE(resp.body.find("neuriplo_scheduler_batch_size") != std::string::npos);
}

TEST_CASE(e2e_model_state_machine_transitions) {
    ModelStateMachine sm;
    REQUIRE_EQ(sm.current(), ModelState::Unloaded);

    REQUIRE(sm.startLoad());
    REQUIRE_EQ(sm.current(), ModelState::Loading);

    REQUIRE(sm.markReady());
    REQUIRE_EQ(sm.current(), ModelState::Ready);

    REQUIRE(sm.beginUnload());
    REQUIRE_EQ(sm.current(), ModelState::Unloading);

    REQUIRE(sm.completeUnload());
    REQUIRE_EQ(sm.current(), ModelState::Unloaded);
}

TEST_CASE(e2e_model_state_machine_failed_path) {
    ModelStateMachine sm;
    sm.startLoad();
    REQUIRE(sm.markFailed());
    REQUIRE_EQ(sm.current(), ModelState::Failed);

    REQUIRE(sm.reset());
    REQUIRE_EQ(sm.current(), ModelState::Unloaded);
}

TEST_CASE(e2e_model_state_machine_invalid_transitions_rejected) {
    ModelStateMachine sm;
    REQUIRE(!sm.markReady());
    REQUIRE_EQ(sm.current(), ModelState::Unloaded);

    REQUIRE(!sm.beginUnload());
    REQUIRE_EQ(sm.current(), ModelState::Unloaded);
}

TEST_CASE(e2e_model_state_machine_observer_fires) {
    ModelStateMachine sm;
    ModelState observed_from = ModelState::Unloaded;
    ModelState observed_to = ModelState::Unloaded;
    int call_count = 0;

    sm.onTransition([&](ModelState from, ModelState to) {
        observed_from = from;
        observed_to = to;
        call_count++;
    });

    sm.startLoad();
    REQUIRE_EQ(call_count, 1);
    REQUIRE_EQ(observed_from, ModelState::Unloaded);
    REQUIRE_EQ(observed_to, ModelState::Loading);

    sm.markReady();
    REQUIRE_EQ(call_count, 2);
    REQUIRE_EQ(observed_from, ModelState::Loading);
    REQUIRE_EQ(observed_to, ModelState::Ready);
}

TEST_CASE(e2e_model_registry_failed_model_not_ready) {
    RuntimeConfig config;
    config.model_name = "fail-model";
    config.backend = "stub";

    ModelRegistry registry(
        config, [](const RuntimeConfig &cfg, std::string &error) -> std::unique_ptr<Executor> {
            error = "simulated load failure";
            return nullptr;
        });

    REQUIRE(!registry.allReady());
    REQUIRE(!registry.ready("fail-model"));
}

TEST_CASE(e2e_model_registry_drain_changes_readiness) {
    RuntimeConfig config;
    config.model_name = "drain-test";
    config.backend = "stub";
    ModelRegistry registry(config);

    REQUIRE(registry.allReady());
    REQUIRE(registry.ready("drain-test"));

    registry.beginDrain("drain-test");

    const auto handle = registry.findHandle("drain-test");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->scheduler->isDraining());
}

TEST_CASE(e2e_scheduler_submit_and_complete) {
    RuntimeConfig config;
    config.model_name = "sched-test";
    config.backend = "stub";
    ModelRegistry registry(config);

    const auto handle = registry.findHandle("sched-test");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->scheduler != nullptr);
    REQUIRE(handle->scheduler->isReady());

    ExecutionRequest req;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 224, 224};
    req.inputs.push_back(input);

    auto result = handle->scheduler->submit(std::move(req));
    REQUIRE(result.ok);
    REQUIRE(result.response.ok);
    REQUIRE(!result.response.outputs.empty());
    REQUIRE(result.total_latency_ns > 0);
}

TEST_CASE(e2e_scheduler_queue_full_returns_overloaded) {
    RuntimeConfig config;
    config.model_name = "queue-full-test";
    config.backend = "stub";
    config.max_queue_size = 1;
    config.instances = 1;
    config.request_timeout_ms = 5000;

    struct BlockingExec final : Executor {
        explicit BlockingExec(ModelMetadata m) : metadata_(std::move(m)) {}
        const ModelMetadata &metadata() const override {
            return metadata_;
        }
        ExecutionResponse infer(const ExecutionRequest &) override {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [this] { return released_; });
            ExecutionResponse resp;
            OutputTensor out;
            out.name = "output";
            out.datatype = "FP32";
            out.shape = {1, 1};
            out.bytes = tensorBytesFromDoubles(out.datatype, {1.0});
            resp.outputs.push_back(std::move(out));
            return resp;
        }
        void release() {
            std::lock_guard<std::mutex> lk(mutex_);
            released_ = true;
            cv_.notify_all();
        }
        ModelMetadata metadata_;
        std::mutex mutex_;
        std::condition_variable cv_;
        bool released_ = false;
    };

    BlockingExec *blocking = nullptr;
    ModelRegistry registry(config, [&](const RuntimeConfig &cfg, std::string &) {
        ModelMetadata m;
        m.name = cfg.model_name;
        m.versions = {"1"};
        m.platform = "test_blocking";
        m.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
        m.outputs.push_back({"output", "FP32", {1, 1}});
        auto exec = std::make_unique<BlockingExec>(std::move(m));
        blocking = exec.get();
        return exec;
    });

    const auto handle = registry.findHandle("queue-full-test");
    REQUIRE(handle != nullptr);

    std::thread t1([&] {
        ExecutionRequest req;
        InputTensor in;
        in.name = "input";
        in.datatype = "FP32";
        in.shape = {1, 3, 224, 224};
        req.inputs.push_back(in);
        (void)handle->scheduler->submit(std::move(req));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    std::thread t2([&] {
        ExecutionRequest req;
        InputTensor in;
        in.name = "input";
        in.datatype = "FP32";
        in.shape = {1, 3, 224, 224};
        req.inputs.push_back(in);
        (void)handle->scheduler->submit(std::move(req));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    ExecutionRequest overflow;
    InputTensor in;
    in.name = "input";
    in.datatype = "FP32";
    in.shape = {1, 3, 224, 224};
    overflow.inputs.push_back(in);
    auto result = handle->scheduler->submit(std::move(overflow));
    REQUIRE(!result.ok);
    REQUIRE_EQ(result.scheduler_error, SchedulerError::Overloaded);

    blocking->release();
    t1.join();
    t2.join();
}

TEST_CASE(e2e_dynamic_batching_merge_and_split) {
    ExecutionRequest req1;
    req1.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    req1.inputs[0].bytes = tensorBytesFromDoubles("FP32", {1.0, 2.0, 3.0, 4.0});
    req1.requested_outputs = {"output"};

    ExecutionRequest req2;
    req2.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    req2.inputs[0].bytes = tensorBytesFromDoubles("FP32", {5.0, 6.0, 7.0, 8.0});
    req2.requested_outputs = {"output"};

    MergedBatch merged;
    const auto err = mergeExecutionRequests({req1, req2}, merged);
    REQUIRE(!err.has_value());
    REQUIRE_EQ(merged.batch_sizes.size(), static_cast<size_t>(2));
    REQUIRE_EQ(merged.batch_sizes[0], static_cast<size_t>(1));
    REQUIRE_EQ(merged.batch_sizes[1], static_cast<size_t>(1));
    REQUIRE_EQ(merged.request.inputs[0].shape[0], static_cast<int64_t>(2));
    REQUIRE_EQ(merged.request.inputs[0].elementCount(), static_cast<size_t>(8));

    ExecutionResponse batch_resp;
    batch_resp.ok = true;
    OutputTensor out;
    out.name = "output";
    out.datatype = "FP32";
    out.shape = {2, 4};
    out.bytes =
        tensorBytesFromDoubles(out.datatype, {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0});
    batch_resp.outputs.push_back(std::move(out));

    auto split = splitExecutionResponse(batch_resp, merged.batch_sizes, {req1, req2});
    REQUIRE_EQ(split.size(), static_cast<size_t>(2));
    REQUIRE_EQ(split[0].outputs[0].elementCount(), static_cast<size_t>(4));
    REQUIRE_EQ(split[1].outputs[0].elementCount(), static_cast<size_t>(4));
}

TEST_CASE(e2e_kserve_v2_codec_parse_and_serialize) {
    ModelMetadata metadata;
    metadata.name = "codec-test";
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_stub";
    metadata.inputs.push_back({"input", "FP32", {1, 2}});
    metadata.outputs.push_back({"output", "FP32", {1, 1000}});

    const auto body =
        R"({"id":"codec-1","inputs":[{"name":"input","shape":[1,2],"datatype":"FP32","data":[1.0,2.0]}],"outputs":[{"name":"output"}]})";
    const auto parsed = parseInferenceRequest(body, metadata);
    REQUIRE(parsed.ok);
    REQUIRE(parsed.request.id.has_value());
    REQUIRE_EQ(*parsed.request.id, "codec-1");
    REQUIRE_EQ(parsed.request.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(parsed.request.inputs[0].name, "input");
    REQUIRE_EQ(parsed.request.inputs[0].elementCount(), static_cast<size_t>(2));

    ExecutionResponse exec_resp;
    exec_resp.ok = true;
    OutputTensor out;
    out.name = "output";
    out.datatype = "FP32";
    out.shape = {1, 1000};
    out.bytes = tensorBytesFromDoubles(out.datatype, std::vector<double>(1000, 0.5));
    exec_resp.outputs.push_back(std::move(out));

    const auto json_str = inferenceResponseJson("codec-test", "1", parsed.request, exec_resp);
    REQUIRE(json_str.find(R"("model_name":"codec-test")") != std::string::npos);
    REQUIRE(json_str.find(R"("id":"codec-1")") != std::string::npos);
    REQUIRE(json_str.find(R"("outputs")") != std::string::npos);
}

TEST_CASE(e2e_kserve_v2_codec_rejects_unknown_input) {
    ModelMetadata metadata;
    metadata.name = "codec-err";
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_stub";
    metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    metadata.outputs.push_back({"output", "FP32", {1, 1000}});

    const auto body =
        R"({"inputs":[{"name":"wrong_name","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})";
    const auto parsed = parseInferenceRequest(body, metadata);
    REQUIRE(!parsed.ok);
}

TEST_CASE(e2e_model_metadata_json_serialization) {
    ModelMetadata metadata;
    metadata.name = "yolo";
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_onnx_runtime";
    metadata.inputs.push_back({"images", "FP32", {1, 3, 640, 640}});
    metadata.outputs.push_back({"output0", "FP32", {1, 300, 6}});

    const auto json_str = modelMetadataJson(metadata);
    REQUIRE(json_str.find(R"("name":"yolo")") != std::string::npos);
    REQUIRE(json_str.find(R"("name":"images")") != std::string::npos);
    REQUIRE(json_str.find(R"("name":"output0")") != std::string::npos);
    REQUIRE(json_str.find(R"("platform":"neuriplo_onnx_runtime")") != std::string::npos);
}

TEST_CASE(e2e_request_pipeline_stages_execute_in_order) {
    std::vector<std::string> order;
    RequestPipeline pipeline;
    pipeline
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("validate");
            return std::nullopt;
        })
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("decode");
            return std::nullopt;
        })
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("schedule");
            return std::nullopt;
        })
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("encode");
            HttpResponse r;
            r.status = 200;
            r.body = R"({"ok":true})";
            return r;
        });

    InferContext ctx;
    const auto result = pipeline.run(ctx);
    REQUIRE_EQ(result.status, 200);
    REQUIRE_EQ(order.size(), static_cast<size_t>(4));
    REQUIRE_EQ(order[0], "validate");
    REQUIRE_EQ(order[1], "decode");
    REQUIRE_EQ(order[2], "schedule");
    REQUIRE_EQ(order[3], "encode");
}

TEST_CASE(e2e_request_pipeline_short_circuits_on_error) {
    std::vector<std::string> order;
    RequestPipeline pipeline;
    pipeline
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("validate");
            return std::nullopt;
        })
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("error");
            HttpResponse r;
            r.status = 400;
            r.body = R"({"error":{"code":"INVALID_ARGUMENT"}})";
            return r;
        })
        .addStage([&](InferContext &ctx) -> std::optional<HttpResponse> {
            order.push_back("should_not_run");
            return std::nullopt;
        });

    InferContext ctx;
    const auto result = pipeline.run(ctx);
    REQUIRE_EQ(result.status, 400);
    REQUIRE_EQ(order.size(), static_cast<size_t>(2));
    REQUIRE_EQ(order[0], "validate");
    REQUIRE_EQ(order[1], "error");
}

TEST_CASE(e2e_error_code_http_mappings_complete) {
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::InvalidArgument), 400);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::ModelNotFound), 404);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::NotFound), 404);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::MethodNotAllowed), 405);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::PayloadTooLarge), 413);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::ModelNotReady), 409);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::QueueFull), 429);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::Unavailable), 503);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::DeadlineExceeded), 504);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::BackendError), 500);
    REQUIRE_EQ(KServeErrors::httpStatusForCode(KServeErrors::Internal), 500);
}

TEST_CASE(e2e_versioned_model_routes) {
    RuntimeConfig config;
    config.model_name = "versioned";
    config.backend = "stub";

    ModelRegistry registry(
        config, [](const RuntimeConfig &cfg, std::string &) -> std::unique_ptr<Executor> {
            ModelMetadata m;
            m.name = cfg.model_name;
            m.versions = {"2"};
            m.platform = "test_versioned";
            m.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
            m.outputs.push_back({"output", "FP32", {1, 1}});
            return std::make_unique<MetadataStubExecutor>(std::move(m));
        });

    MetricsRegistry metrics;
    KServeRuntime runtime(registry, metrics);

    HttpRequest req;
    req.method = "GET";
    req.path = "/v2/models/versioned/versions/2";
    auto resp = runtime.handle(req);
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("versions":["2"])") != std::string::npos);

    req.method = "GET";
    req.path = "/v2/models/versioned/versions/2/ready";
    resp = runtime.handle(req);
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("ready":true)") != std::string::npos);

    req.method = "POST";
    req.path = "/v2/models/versioned/versions/2/infer";
    req.body = validInferBody("versioned");
    resp = runtime.handle(req);
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("model_version":"2")") != std::string::npos);
}

TEST_CASE(e2e_concurrent_infer_requests) {
    const StubServer server;
    const auto body = validInferBody();

    std::vector<std::future<RawHttpResult>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(std::async(std::launch::async, [&] {
            return sendRawHttp(server.port(),
                               httpPost(server.port(), "/v2/models/demo/infer", body));
        }));
    }

    for (auto &f : futures) {
        const auto resp = f.get();
        REQUIRE_EQ(resp.status, 200);
        REQUIRE(resp.body.find(R"("outputs")") != std::string::npos);
    }
}

TEST_CASE(e2e_scheduler_metrics_snapshot) {
    RuntimeConfig config;
    config.model_name = "metrics-test";
    config.backend = "stub";
    ModelRegistry registry(config);

    const auto handle = registry.findHandle("metrics-test");
    REQUIRE(handle != nullptr);

    ExecutionRequest req;
    InputTensor in;
    in.name = "input";
    in.datatype = "FP32";
    in.shape = {1, 3, 224, 224};
    req.inputs.push_back(in);
    auto result = handle->scheduler->submit(std::move(req));
    REQUIRE(result.ok);

    const auto snap = handle->scheduler->metrics();
    REQUIRE_EQ(snap.requests_accepted, static_cast<uint64_t>(1));
    REQUIRE(snap.completed_requests >= static_cast<uint64_t>(1));
}

TEST_CASE(e2e_batch_compatibility_check) {
    DynamicBatchingConfig config;
    config.enabled = true;
    config.max_batch_size = 8;
    config.preferred_batch_sizes = {2, 4, 8};

    REQUIRE(shouldDispatchBatchSize(2, config, true));
    REQUIRE(shouldDispatchBatchSize(4, config, true));
    REQUIRE(shouldDispatchBatchSize(8, config, true));
    REQUIRE(shouldDispatchBatchSize(1, config, true));
}

TEST_CASE(e2e_stub_executor_returns_identity_output) {
    ModelMetadata m;
    m.name = "stub-test";
    m.versions = {"1"};
    m.platform = "neuriplo_stub";
    m.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    m.outputs.push_back({"output", "FP32", {1, 1000}});

    MetadataStubExecutor exec(m);
    REQUIRE_EQ(exec.metadata().name, "stub-test");

    ExecutionRequest req;
    InputTensor in;
    in.name = "input";
    in.datatype = "FP32";
    in.shape = {1, 3, 224, 224};
    in.bytes = tensorBytesFromDoubles(in.datatype, {1.0, 2.0, 3.0});
    req.inputs.push_back(in);

    auto resp = exec.infer(req);
    REQUIRE(resp.ok);
    REQUIRE_EQ(resp.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(resp.outputs[0].name, "output");
}

TEST_CASE(e2e_custom_executor_full_http_flow) {
    struct ScaledExecutor final : Executor {
        explicit ScaledExecutor(ModelMetadata m) : metadata_(std::move(m)) {}
        const ModelMetadata &metadata() const override {
            return metadata_;
        }
        ExecutionResponse infer(const ExecutionRequest &req) override {
            ExecutionResponse resp;
            for (const auto &input : req.inputs) {
                OutputTensor out;
                out.name = input.name + "_scaled";
                out.datatype = input.datatype;
                out.shape = input.shape;
                auto values = tensorValuesAsDoubles(input.datatype, input.bytes);
                for (auto &v : values)
                    v *= 2.0;
                out.bytes = tensorBytesFromDoubles(out.datatype, values);
                resp.outputs.push_back(std::move(out));
            }
            return resp;
        }
        ModelMetadata metadata_;
    };

    CustomExecutorServer server(
        [](const RuntimeConfig &cfg, std::string &) -> std::unique_ptr<Executor> {
            ModelMetadata m;
            m.name = cfg.model_name;
            m.versions = {"1"};
            m.platform = "test_scaled";
            m.inputs.push_back({"input", "FP32", {1, 2}});
            m.outputs.push_back({"input_scaled", "FP32", {1, 2}});
            return std::make_unique<ScaledExecutor>(std::move(m));
        });

    const auto body =
        R"({"id":"scaled-1","inputs":[{"name":"input","shape":[1,2],"datatype":"FP32","data":[1.5,2.5]}]})";
    const auto resp =
        sendRawHttp(server.port(), httpPost(server.port(), "/v2/models/demo/infer", body));
    REQUIRE_EQ(resp.status, 200);
    REQUIRE(resp.body.find(R"("input_scaled")") != std::string::npos);
}

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO

TEST_CASE(e2e_real_neuriplo_identity_model_full_http_flow) {
    const auto model_path = writeOnnxModel(identityOnnxModel(), "e2e-identity.onnx");

    RuntimeConfig config;
    config.model_name = "identity";
    config.model_path = model_path;
    config.backend = "onnx_runtime";

    ModelRegistry registry(config);
    REQUIRE(registry.allReady());

    MetricsRegistry metrics;
    KServeRuntime runtime(registry, metrics);

    HttpRequest req;
    req.method = "GET";
    req.path = "/v2/models/identity";
    auto meta_resp = runtime.handle(req);
    REQUIRE_EQ(meta_resp.status, 200);
    REQUIRE(meta_resp.body.find(R"("name":"identity")") != std::string::npos);
    REQUIRE(meta_resp.body.find(R"("input")") != std::string::npos);
    REQUIRE(meta_resp.body.find(R"("output")") != std::string::npos);
    REQUIRE(meta_resp.body.find(R"("neuriplo_onnx_runtime")") != std::string::npos);

    req.method = "GET";
    req.path = "/v2/models/identity/ready";
    auto ready_resp = runtime.handle(req);
    REQUIRE_EQ(ready_resp.status, 200);

    req.method = "POST";
    req.path = "/v2/models/identity/infer";
    req.body =
        R"({"id":"real-e2e-1","inputs":[{"name":"input","shape":[1,3,4,4],"datatype":"FP32","data":[)";
    for (int i = 0; i < 48; ++i) {
        if (i > 0)
            req.body += ",";
        req.body += std::to_string(static_cast<double>(i) / 10.0);
    }
    req.body += R"(]}],"outputs":[{"name":"output"}]})";

    auto infer_resp = runtime.handle(req);
    REQUIRE_EQ(infer_resp.status, 200);
    REQUIRE(infer_resp.body.find(R"("model_name":"identity")") != std::string::npos);
    REQUIRE(infer_resp.body.find(R"("id":"real-e2e-1")") != std::string::npos);
    REQUIRE(infer_resp.body.find(R"("outputs")") != std::string::npos);
    REQUIRE(infer_resp.body.find(R"("output")") != std::string::npos);
}

TEST_CASE(e2e_real_neuriplo_scheduler_latency_measured) {
    const auto model_path = writeOnnxModel(identityOnnxModel(), "e2e-latency.onnx");

    RuntimeConfig config;
    config.model_name = "identity";
    config.model_path = model_path;
    config.backend = "onnx_runtime";

    ModelRegistry registry(config);
    const auto handle = registry.findHandle("identity");
    REQUIRE(handle != nullptr);

    ExecutionRequest req;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 4, 4};
    std::vector<double> input_values;
    input_values.reserve(48);
    for (int i = 0; i < 48; ++i) {
        input_values.push_back(static_cast<double>(i) / 10.0);
    }
    input.bytes = tensorBytesFromDoubles(input.datatype, input_values);
    req.inputs.push_back(input);

    auto result = handle->scheduler->submit(std::move(req));
    REQUIRE(result.ok);
    REQUIRE(result.total_latency_ns > 0);
    REQUIRE(result.execution_latency_ns > 0);
}

#endif
