#include "StubExecutor.hpp"
#include "Executor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <cstddef>
#include <string>

namespace {

RuntimeConfig stubConfig() {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    return config;
}

} // namespace

TEST_CASE(stub_executor_exposes_metadata) {
    std::string error;
    const auto executor = makeStubExecutor(stubConfig(), error);
    REQUIRE(executor != nullptr);
    REQUIRE_EQ(executor->metadata().name, "demo");
    REQUIRE_EQ(executor->metadata().versions.size(), static_cast<size_t>(1));
    REQUIRE_EQ(executor->metadata().versions[0], "1");
    REQUIRE_EQ(executor->metadata().outputs[0].shape.size(), static_cast<size_t>(2));
    REQUIRE_EQ(executor->metadata().outputs[0].shape[0], 1);
    REQUIRE_EQ(executor->metadata().outputs[0].shape[1], 1000);
}

TEST_CASE(stub_executor_returns_deterministic_output) {
    std::string error;
    const auto executor = makeStubExecutor(stubConfig(), error);
    REQUIRE(executor != nullptr);

    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto response = executor->infer(request);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].name, "output");
    REQUIRE_EQ(response.outputs[0].shape.size(), static_cast<size_t>(2));
    REQUIRE_EQ(response.outputs[0].shape[1], 1000);
    REQUIRE_EQ(response.outputs[0].elementCount(), static_cast<size_t>(1000));
    REQUIRE_EQ(tensorScalarAt<float>(response.outputs[0].bytes, 0), 0.0f);
}

TEST_CASE(stub_executor_honors_requested_outputs) {
    std::string error;
    const auto executor = makeStubExecutor(stubConfig(), error);
    REQUIRE(executor != nullptr);

    ExecutionRequest request;
    request.requested_outputs = {"output"};
    const auto response = executor->infer(request);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].name, "output");
}
