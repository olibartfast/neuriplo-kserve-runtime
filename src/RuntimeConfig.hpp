#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct RuntimeConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int grpc_port = 0;
    size_t max_request_bytes = 67108864;
    std::string model_name = "demo";
    std::string model_version = "1";
    // True when the version was actually asked for (--model-version, the
    // MODEL_VERSION env, or a repository tree's version directory) rather than
    // left at the default. Only then does it override the version a backend
    // reports for itself, which is otherwise authoritative.
    bool model_version_explicit = false;
    std::string model_path;
    // Root of a Triton-style model repository tree
    // (<root>/<model>/<version>/<file>). When set, every model found under it is
    // served and model_name/model_path/backend act only as defaults, since the
    // tree supplies them per model.
    std::string model_repository;
    // "none"     - load every model in the repository at startup (default)
    // "explicit" - load nothing at startup; the client loads and unloads
    //              through /v2/repository/models/<name>/{load,unload}
    // Explicit mode also contains damage: a model whose backend crashes on load
    // cannot take the server down with it before it has even been asked for.
    std::string model_control_mode = "none";
    std::string backend = "stub";
    // Run inference on the GPU when the selected backend supports it. Without
    // this the runtime could only ever serve on the CPU, whatever the backend
    // build was capable of.
    bool use_gpu = false;
    std::string plugin_dir;
    // Per-input tensor shapes (without batch dimension) for backends that
    // cannot introspect them from the model file (e.g. opencv_dnn).
    std::vector<std::vector<int64_t>> input_sizes;
    // Pipeline (ensemble) graph as inline JSON. Empty means read the graph from
    // model_path instead. Only used when backend == "ensemble".
    std::string pipeline_graph;
    std::string storage_uri;
    std::string deployment;
    size_t max_queue_size = 64;
    int64_t request_timeout_ms = 30000;
    size_t instances = 1;
    bool dynamic_batching_enabled = false;
    size_t max_batch_size = 1;
    int64_t max_queue_delay_us = 0;
    std::vector<size_t> preferred_batch_sizes;
    bool log_payloads = false;
    std::string scheduler_strategy = "tensor";
    size_t context_length = 4096;
    size_t kv_cache_slots = 1;
    size_t max_tokens = 256;
    double tokens_per_char = 0.25;
    double temperature = 0.8;
    double top_p = 0.95;
    size_t top_k = 40;
    bool streaming_enabled = false;
    size_t memory_budget_bytes = 0;
};

struct RuntimeEnvironment {
    std::function<std::optional<std::string>(const std::string &)> get;
    std::function<bool(const std::string &)> pathExists;
};

RuntimeConfig parseRuntimeConfig(int argc, char **argv);
RuntimeConfig parseRuntimeConfig(int argc, char **argv, const RuntimeEnvironment &environment);
