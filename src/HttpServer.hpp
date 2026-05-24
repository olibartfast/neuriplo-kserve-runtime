#pragma once

#include "HttpTypes.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class HttpServer {
  public:
    using Handler = std::function<HttpResponse(const HttpRequest &)>;

    HttpServer(std::string host, int port, Handler handler, size_t max_request_bytes = 67108864);
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer &operator=(const HttpServer &) = delete;

    void run();
    void stop();

  private:
    void handleClient(int client_fd) const;
    void joinClientThreads();

    std::string host_;
    int port_;
    Handler handler_;
    size_t max_request_bytes_;
    std::atomic<bool> running_{false};
    std::atomic<int> server_fd_{-1};
    std::mutex client_threads_mutex_;
    std::vector<std::thread> client_threads_;
};
