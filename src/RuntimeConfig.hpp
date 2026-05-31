#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct RuntimeConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    size_t max_request_bytes = 67108864;
    std::string model_name = "demo";
    std::string model_path;
    std::string backend = "stub";
    std::string storage_uri;
    size_t max_queue_size = 64;
    int64_t request_timeout_ms = 30000;
    size_t instances = 1;
    bool dynamic_batching_enabled = false;
    size_t max_batch_size = 1;
    int64_t max_queue_delay_us = 0;
    std::vector<size_t> preferred_batch_sizes;
    bool log_payloads = false;
};

struct RuntimeEnvironment {
    std::function<std::optional<std::string>(const std::string &)> get;
    std::function<bool(const std::string &)> pathExists;
};

RuntimeConfig parseRuntimeConfig(int argc, char **argv);
RuntimeConfig parseRuntimeConfig(int argc, char **argv, const RuntimeEnvironment &environment);
