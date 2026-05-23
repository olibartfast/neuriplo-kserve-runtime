#pragma once

#include "HttpTypes.hpp"

#include <atomic>
#include <functional>
#include <string>

class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;

    HttpServer(std::string host, int port, Handler handler);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run();
    void stop();

private:
    void handleClient(int client_fd) const;

    std::string host_;
    int port_;
    Handler handler_;
    std::atomic<bool> running_{false};
    int server_fd_ = -1;
};

