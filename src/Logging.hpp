#pragma once

#include <chrono>
#include <ostream>
#include <string>

struct LogEvent {
    std::string severity = "info";
    std::string message;
    std::string request_id;
    std::string model;
    std::string model_version;
    std::string backend;
    std::string route;
    int status = 0;
    uint64_t queue_latency_ns = 0;
    uint64_t infer_latency_ns = 0;
    uint64_t total_latency_ns = 0;
    uint64_t batch_size = 0;
    std::string error_code;
    size_t request_bytes = 0;
    size_t response_bytes = 0;
};

class Logger {
  public:
    explicit Logger(std::ostream &out);

    void event(const LogEvent &event);
    void info(const std::string &msg);
    void warn(const std::string &msg);
    void error(const std::string &msg);

  private:
    void writeEvent(const LogEvent &event);

    std::ostream &out_;
};

Logger &defaultLogger();
