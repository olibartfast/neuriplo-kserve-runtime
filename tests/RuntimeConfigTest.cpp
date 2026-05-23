#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <stdexcept>
#include <vector>

namespace {

RuntimeConfig parse(const std::vector<const char *> &args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const char *arg : args) {
        argv.push_back(const_cast<char *>(arg));
    }
    return parseRuntimeConfig(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST_CASE(parse_runtime_config_uses_defaults) {
    const auto config = parse({"neuriplo-kserve-runtime"});
    REQUIRE_EQ(config.host, "0.0.0.0");
    REQUIRE_EQ(config.port, 8080);
    REQUIRE_EQ(config.model_name, "demo");
    REQUIRE_EQ(config.backend, "stub");
}

TEST_CASE(parse_runtime_config_accepts_overrides) {
    const auto config =
        parse({"neuriplo-kserve-runtime", "--host", "127.0.0.1", "--port", "9090", "--model-name",
               "classifier", "--model-path", "/models/model.onnx", "--backend", "onnx_runtime"});
    REQUIRE_EQ(config.host, "127.0.0.1");
    REQUIRE_EQ(config.port, 9090);
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
