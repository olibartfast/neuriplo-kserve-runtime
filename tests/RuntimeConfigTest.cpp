#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

RuntimeEnvironment testEnvironment(std::unordered_map<std::string, std::string> values = {},
                                   bool mntModelsExists = false) {
    return RuntimeEnvironment{
        [values = std::move(values)](const std::string &name) -> std::optional<std::string> {
            const auto found = values.find(name);
            if (found == values.end()) {
                return std::nullopt;
            }
            return found->second;
        },
        [mntModelsExists](const std::string &path) {
            return path == "/mnt/models" && mntModelsExists;
        },
    };
}

RuntimeConfig parse(const std::vector<const char *> &args,
                    RuntimeEnvironment environment = testEnvironment()) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const char *arg : args) {
        argv.push_back(const_cast<char *>(arg));
    }
    return parseRuntimeConfig(static_cast<int>(argv.size()), argv.data(), environment);
}

} // namespace

TEST_CASE(parse_runtime_config_uses_defaults) {
    const auto config = parse({"neuriplo-kserve-runtime"});
    REQUIRE_EQ(config.host, "0.0.0.0");
    REQUIRE_EQ(config.port, 8080);
    REQUIRE_EQ(config.max_request_bytes, static_cast<size_t>(67108864));
    REQUIRE_EQ(config.model_name, "demo");
    REQUIRE_EQ(config.model_version, "1");
    REQUIRE_EQ(config.backend, "stub");
    REQUIRE_EQ(config.model_path, "");
    REQUIRE_EQ(config.storage_uri, "");
    REQUIRE_EQ(config.deployment, "");
    REQUIRE_EQ(config.max_queue_size, static_cast<size_t>(64));
    REQUIRE_EQ(config.request_timeout_ms, static_cast<int64_t>(30000));
    REQUIRE_EQ(config.instances, static_cast<size_t>(1));
}

TEST_CASE(parse_runtime_config_uses_environment_defaults) {
    const auto config = parse({"neuriplo-kserve-runtime"},
                              testEnvironment({{"MODEL_NAME", "image-model"},
                                               {"MODEL_PATH", "/models/from-env"},
                                               {"BACKEND", "neuriplo_backend"},
                                               {"STORAGE_URI", "pvc://models/image-model"},
                                               {"MAX_REQUEST_BYTES", "1024"}}));
    REQUIRE_EQ(config.model_name, "image-model");
    REQUIRE_EQ(config.model_path, "/models/from-env");
    REQUIRE_EQ(config.backend, "neuriplo_backend");
    REQUIRE_EQ(config.storage_uri, "pvc://models/image-model");
    REQUIRE_EQ(config.max_request_bytes, static_cast<size_t>(1024));
}

TEST_CASE(parse_runtime_config_cli_overrides_environment_defaults) {
    const auto config = parse({"neuriplo-kserve-runtime", "--model-name", "cli-model",
                               "--model-path", "/models/from-cli", "--backend", "cli_backend"},
                              testEnvironment({{"MODEL_NAME", "env-model"},
                                               {"MODEL_PATH", "/models/from-env"},
                                               {"BACKEND", "env_backend"}}));
    REQUIRE_EQ(config.model_name, "cli-model");
    REQUIRE_EQ(config.model_path, "/models/from-cli");
    REQUIRE_EQ(config.backend, "cli_backend");
}

TEST_CASE(parse_runtime_config_plugin_dir_env_and_cli) {
    const auto env_config = parse({"neuriplo-kserve-runtime"},
                                  testEnvironment({{"NEURIPLO_PLUGIN_DIR", "/opt/plugins"}}));
    REQUIRE_EQ(env_config.plugin_dir, "/opt/plugins");

    const auto cli_config = parse({"neuriplo-kserve-runtime", "--plugin-dir", "/cli/plugins"},
                                  testEnvironment({{"NEURIPLO_PLUGIN_DIR", "/opt/plugins"}}));
    REQUIRE_EQ(cli_config.plugin_dir, "/cli/plugins");
}

TEST_CASE(parse_runtime_config_defaults_to_mnt_models_when_present) {
    const auto config = parse({"neuriplo-kserve-runtime"}, testEnvironment({}, true));
    REQUIRE_EQ(config.model_path, "/mnt/models");
}

TEST_CASE(parse_runtime_config_model_path_env_overrides_mnt_models_default) {
    const auto config = parse({"neuriplo-kserve-runtime"},
                              testEnvironment({{"MODEL_PATH", "/models/from-env"}}, true));
    REQUIRE_EQ(config.model_path, "/models/from-env");
}

TEST_CASE(parse_runtime_config_accepts_overrides) {
    const auto config = parse({"neuriplo-kserve-runtime", "--host", "127.0.0.1", "--port", "9090",
                               "--max-request-bytes", "2048", "--model-name", "classifier",
                               "--model-path", "/models/model.onnx", "--backend", "onnx_runtime"});
    REQUIRE_EQ(config.host, "127.0.0.1");
    REQUIRE_EQ(config.port, 9090);
    REQUIRE_EQ(config.max_request_bytes, static_cast<size_t>(2048));
    REQUIRE_EQ(config.model_name, "classifier");
    REQUIRE_EQ(config.model_path, "/models/model.onnx");
    REQUIRE_EQ(config.backend, "onnx_runtime");
}

TEST_CASE(parse_runtime_config_rejects_invalid_port) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--port", "70000"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_zero_max_request_bytes_cli) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--max-request-bytes", "0"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_zero_max_request_bytes_env) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime"}, testEnvironment({{"MAX_REQUEST_BYTES", "0"}}));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_accepts_scheduler_overrides) {
    const auto config = parse({"neuriplo-kserve-runtime", "--max-queue-size", "128",
                               "--request-timeout-ms", "1500", "--instances", "4"});
    REQUIRE_EQ(config.max_queue_size, static_cast<size_t>(128));
    REQUIRE_EQ(config.request_timeout_ms, static_cast<int64_t>(1500));
    REQUIRE_EQ(config.instances, static_cast<size_t>(4));
}

TEST_CASE(parse_runtime_config_rejects_zero_max_queue_size) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--max-queue-size", "0"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_zero_request_timeout) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--request-timeout-ms", "0"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_zero_instances) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--instances", "0"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_excessive_max_request_bytes) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--max-request-bytes", "9223372036854775808"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_accepts_llm_overrides) {
    const auto config =
        parse({"neuriplo-kserve-runtime", "--scheduler-strategy", "llm", "--context-length", "8192",
               "--kv-cache-slots", "2", "--max-tokens", "512", "--temperature", "0.7", "--top-p",
               "0.9", "--top-k", "0", "--streaming-enabled", "true"});
    REQUIRE_EQ(config.scheduler_strategy, "llm");
    REQUIRE_EQ(config.context_length, static_cast<size_t>(8192));
    REQUIRE_EQ(config.kv_cache_slots, static_cast<size_t>(2));
    REQUIRE_EQ(config.max_tokens, static_cast<size_t>(512));
    REQUIRE_EQ(config.temperature, 0.7);
    REQUIRE_EQ(config.top_p, 0.9);
    REQUIRE_EQ(config.top_k, static_cast<size_t>(0));
    REQUIRE(config.streaming_enabled);
}

TEST_CASE(parse_runtime_config_rejects_invalid_scheduler_strategy) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--scheduler-strategy", "batch"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_invalid_temperature) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--temperature", "-0.1"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_rejects_invalid_top_p) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--top-p", "0"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(parse_runtime_config_accepts_model_version_and_deployment) {
    const auto config =
        parse({"neuriplo-kserve-runtime", "--model-version", "v2", "--deployment", "canary"});
    REQUIRE_EQ(config.model_version, "v2");
    REQUIRE_EQ(config.deployment, "canary");
}

TEST_CASE(parse_runtime_config_model_version_env) {
    const auto config = parse({"neuriplo-kserve-runtime"},
                              testEnvironment({{"MODEL_VERSION", "3"}, {"DEPLOYMENT", "prod"}}));
    REQUIRE_EQ(config.model_version, "3");
    REQUIRE_EQ(config.deployment, "prod");
}

// P2-B2 B-16/C-8: an explicit single-model flag alongside --models must fail
// fast naming both, not silently pick one or the other.
TEST_CASE(parse_runtime_config_rejects_model_path_with_models) {
    bool threw = false;
    std::string message;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--models", "/repo", "--model-path", "/one.onnx"});
    } catch (const std::invalid_argument &error) {
        threw = true;
        message = error.what();
    }
    REQUIRE(threw);
    REQUIRE(message.find("--model-path") != std::string::npos);
    REQUIRE(message.find("--models") != std::string::npos ||
            message.find("--model-repository") != std::string::npos);
}

TEST_CASE(parse_runtime_config_rejects_backend_and_model_name_with_models) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--models", "/repo", "--backend", "stub",
                     "--model-name", "x"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

// The conflict must be caught the same way when --models comes from the
// MODEL_REPOSITORY env var an image sets and --model-path is given on the
// command line -- neither side should silently win.
TEST_CASE(parse_runtime_config_rejects_model_path_cli_with_model_repository_env) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--model-path", "/one.onnx"},
                    testEnvironment({{"MODEL_REPOSITORY", "/repo"}}));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(threw);
}

// --model-version alone, without a repository, remains valid (regression
// guard for the conflict check above).
TEST_CASE(parse_runtime_config_model_version_without_repository_still_valid) {
    const auto config = parse({"neuriplo-kserve-runtime", "--model-version", "2"});
    REQUIRE_EQ(config.model_version, "2");
}

// C2 regression: a deployment manifest commonly sets MODEL_REPOSITORY and
// MODEL_VERSION in the same env block (e.g. deploy/k3d/runtime-trt.yaml) --
// the latter is not an operator asking for a conflicting single-model
// version, so this combination must not crash-loop the pod.
TEST_CASE(parse_runtime_config_env_model_repository_and_model_version_does_not_throw) {
    bool threw = false;
    try {
        (void)parse({"neuriplo-kserve-runtime"},
                    testEnvironment({{"MODEL_REPOSITORY", "/repo"}, {"MODEL_VERSION", "3"}}));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    REQUIRE(!threw);
}

// C2: an operator who actually types --model-version on the command line
// next to --models/--model-repository still gets the fail-fast conflict.
TEST_CASE(parse_runtime_config_rejects_cli_model_version_with_models) {
    bool threw = false;
    std::string message;
    try {
        (void)parse({"neuriplo-kserve-runtime", "--models", "/repo", "--model-version", "3"});
    } catch (const std::invalid_argument &error) {
        threw = true;
        message = error.what();
    }
    REQUIRE(threw);
    REQUIRE(message.find("--model-version") != std::string::npos);
}
