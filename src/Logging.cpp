#include "Logging.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

namespace {

using Json = nlohmann::json;

std::string iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;
    std::ostringstream oss;
    std::tm tm_buf;
    gmtime_r(&time_t_now, &tm_buf);
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms << 'Z';
    return oss.str();
}

} // namespace

Logger::Logger(std::ostream &out) : out_(out) {}

void Logger::event(const LogEvent &event) {
    writeEvent(event);
}

void Logger::info(const std::string &msg) {
    LogEvent evt;
    evt.severity = "info";
    evt.message = msg;
    writeEvent(evt);
}

void Logger::warn(const std::string &msg) {
    LogEvent evt;
    evt.severity = "warn";
    evt.message = msg;
    writeEvent(evt);
}

void Logger::error(const std::string &msg) {
    LogEvent evt;
    evt.severity = "error";
    evt.message = msg;
    writeEvent(evt);
}

void Logger::writeEvent(const LogEvent &event) {
    Json j;
    j["timestamp"] = iso8601();
    j["severity"] = event.severity;
    j["message"] = event.message;
    if (!event.request_id.empty()) {
        j["request_id"] = event.request_id;
    }
    if (!event.model.empty()) {
        j["model"] = event.model;
    }
    if (!event.model_version.empty()) {
        j["model_version"] = event.model_version;
    }
    if (!event.backend.empty()) {
        j["backend"] = event.backend;
    }
    if (!event.route.empty()) {
        j["route"] = event.route;
    }
    if (event.status != 0) {
        j["status"] = event.status;
    }
    if (event.queue_latency_ns > 0) {
        j["queue_latency_ns"] = event.queue_latency_ns;
    }
    if (event.infer_latency_ns > 0) {
        j["infer_latency_ns"] = event.infer_latency_ns;
    }
    if (event.total_latency_ns > 0) {
        j["total_latency_ns"] = event.total_latency_ns;
    }
    if (event.batch_size > 0) {
        j["batch_size"] = event.batch_size;
    }
    if (!event.error_code.empty()) {
        j["error_code"] = event.error_code;
    }
    if (event.request_bytes > 0) {
        j["request_bytes"] = event.request_bytes;
    }
    if (event.response_bytes > 0) {
        j["response_bytes"] = event.response_bytes;
    }
    out_ << j.dump() << '\n';
}

static Logger default_logger(std::cerr);

Logger &defaultLogger() {
    return default_logger;
}
