#include "HttpServer.hpp"
#include "KServeRuntime.hpp"
#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

struct HttpResult {
    int status = 0;
    std::string body;
};

int findFreePort() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
        ::close(fd);
        throw std::runtime_error("bind failed");
    }

    socklen_t length = sizeof(address);
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) < 0) {
        ::close(fd);
        throw std::runtime_error("getsockname failed");
    }
    const auto port = ntohs(address.sin_port);
    ::close(fd);
    return port;
}

class TestServer {
  public:
    TestServer()
        : port(findFreePort()), registry(config), runtime(std::move(registry)),
          server(
              "127.0.0.1", port,
              [this](const HttpRequest &request) { return runtime.handle(request); }, 1024) {
        thread = std::thread([this] { server.run(); });
        waitUntilReady();
    }

    ~TestServer() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }

    int port;

  private:
    void waitUntilReady() const {
        for (int attempt = 0; attempt < 100; ++attempt) {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd >= 0) {
                sockaddr_in address{};
                address.sin_family = AF_INET;
                address.sin_port = htons(static_cast<uint16_t>(port));
                ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
                if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
                    ::close(fd);
                    return;
                }
                ::close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("test HTTP server did not start");
    }

    RuntimeConfig config;
    ModelRegistry registry;
    KServeRuntime runtime;
    HttpServer server;
    std::thread thread;
};

HttpResult sendRaw(int port, const std::string &request) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        ::close(fd);
        throw std::runtime_error("connect failed");
    }

    ::send(fd, request.data(), request.size(), 0);

    std::string raw;
    char buffer[4096];
    while (true) {
        const auto received = ::recv(fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        raw.append(buffer, static_cast<size_t>(received));
    }
    ::close(fd);

    HttpResult result;
    const auto first_space = raw.find(' ');
    const auto second_space = raw.find(' ', first_space + 1);
    result.status = std::stoi(raw.substr(first_space + 1, second_space - first_space - 1));
    const auto body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        result.body = raw.substr(body_start + 4);
    }
    return result;
}

std::string httpRequest(std::string method, std::string path, std::string body = "") {
    std::string request = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n";
    if (!body.empty()) {
        request +=
            "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
            "\r\n";
    }
    request += "\r\n";
    request += body;
    return request;
}

std::string validInferBody() {
    return R"({"id":"integration-1","inputs":[{"name":"input","shape":[1,3,224,224],"datatype":"FP32","data":[]}]})";
}

} // namespace

TEST_CASE(http_integration_returns_server_metadata) {
    const TestServer server;
    const auto response = sendRaw(server.port, httpRequest("GET", "/v2"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("name":"neuriplo-kserve-runtime")") != std::string::npos);
}

TEST_CASE(http_integration_returns_model_versions) {
    const TestServer server;
    const auto response = sendRaw(server.port, httpRequest("GET", "/v2/models/demo"));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("versions":["1"])") != std::string::npos);
}

TEST_CASE(http_integration_rejects_wrong_methods) {
    const TestServer server;
    REQUIRE_EQ(sendRaw(server.port, httpRequest("POST", "/v2/models/demo")).status, 405);
    REQUIRE_EQ(sendRaw(server.port, httpRequest("GET", "/v2/models/demo/infer")).status, 405);
}

TEST_CASE(http_integration_handles_infer_id_echo) {
    const TestServer server;
    const auto response =
        sendRaw(server.port, httpRequest("POST", "/v2/models/demo/infer", validInferBody()));
    REQUIRE_EQ(response.status, 200);
    REQUIRE(response.body.find(R"("id":"integration-1")") != std::string::npos);
    REQUIRE(response.body.find(R"("model_version":"1")") != std::string::npos);
}

TEST_CASE(http_integration_rejects_malformed_json) {
    const TestServer server;
    const auto response = sendRaw(server.port, httpRequest("POST", "/v2/models/demo/infer", "{"));
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(http_integration_rejects_unknown_model) {
    const TestServer server;
    const auto response =
        sendRaw(server.port, httpRequest("POST", "/v2/models/missing/infer", validInferBody()));
    REQUIRE_EQ(response.status, 404);
    REQUIRE(response.body.find(R"("code":"MODEL_NOT_FOUND")") != std::string::npos);
}

TEST_CASE(http_integration_rejects_oversized_request) {
    const TestServer server;
    const auto response =
        sendRaw(server.port, httpRequest("POST", "/v2/models/demo/infer", std::string(2048, 'x')));
    REQUIRE_EQ(response.status, 413);
    REQUIRE(response.body.find(R"("code":"PAYLOAD_TOO_LARGE")") != std::string::npos);
}

TEST_CASE(http_integration_rejects_invalid_content_length) {
    const TestServer server;
    const auto response = sendRaw(
        server.port,
        "POST /v2/models/demo/infer HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: abc\r\n\r\n");
    REQUIRE_EQ(response.status, 400);
    REQUIRE(response.body.find(R"("code":"INVALID_ARGUMENT")") != std::string::npos);
}

TEST_CASE(http_integration_rejects_declared_content_length_over_limit) {
    const TestServer server;
    const auto response =
        sendRaw(server.port, "POST /v2/models/demo/infer HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                             "Content-Type: application/json\r\nContent-Length: 4096\r\n\r\n{}");
    REQUIRE_EQ(response.status, 413);
    REQUIRE(response.body.find(R"("code":"PAYLOAD_TOO_LARGE")") != std::string::npos);
}
