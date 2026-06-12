#include "BackendRegistry.hpp"
#include "NeuriploAdapter.hpp"
#include "Test.hpp"

#include <algorithm>
#include <string>

TEST_CASE(backend_registry_classifies_tensor_backends) {
    const auto onnx = findBackendCapability("onnx_runtime");
    REQUIRE(onnx.has_value());
    REQUIRE_EQ(onnx->kind, BackendKind::Tensor);
    REQUIRE(onnx->uses_neuriplo);
}

TEST_CASE(backend_registry_classifies_llm_backends) {
    const auto llamacpp = findBackendCapability("llamacpp");
    REQUIRE(llamacpp.has_value());
    REQUIRE_EQ(llamacpp->kind, BackendKind::Llm);
    REQUIRE(llamacpp->uses_neuriplo);

    const auto cactus = findBackendCapability("cactus");
    REQUIRE(cactus.has_value());
    REQUIRE_EQ(cactus->kind, BackendKind::Llm);

    const auto ggml = findBackendCapability("ggml");
    REQUIRE(ggml.has_value());
    REQUIRE_EQ(ggml->kind, BackendKind::Llm);
}

TEST_CASE(backend_registry_keeps_stub_for_tests) {
    const auto stub = findBackendCapability("stub");
    REQUIRE(stub.has_value());
    REQUIRE_EQ(stub->kind, BackendKind::Tensor);
    REQUIRE(!stub->uses_neuriplo);
}

TEST_CASE(backend_registry_rejects_unknown_backend) {
    REQUIRE(!isKnownBackend("does_not_exist"));
    REQUIRE(!findBackendCapability("does_not_exist").has_value());
}

TEST_CASE(backend_registry_allows_custom_registration) {
    const bool registered = registerBackend({"custom_tensor_backend", BackendKind::Tensor, false});
    REQUIRE(registered);
    const auto capability = findBackendCapability("custom_tensor_backend");
    REQUIRE(capability.has_value());
    REQUIRE_EQ(capability->kind, BackendKind::Tensor);
}

TEST_CASE(backend_registry_selects_llm_scheduler_from_strategy) {
    REQUIRE(usesLlmScheduler("llm", "stub"));
    REQUIRE(!usesLlmScheduler("tensor", "stub"));
}

TEST_CASE(backend_registry_auto_detects_llm_backends) {
    REQUIRE(usesLlmScheduler("tensor", "llamacpp"));
    REQUIRE(usesLlmScheduler("tensor", "cactus"));
    REQUIRE(!usesLlmScheduler("tensor", "onnx_runtime"));
}

TEST_CASE(backend_registry_available_ids_are_sorted_and_include_stub) {
    const auto ids = availableBackendIds();
    REQUIRE(!ids.empty());
    REQUIRE(std::is_sorted(ids.begin(), ids.end()));
    REQUIRE(std::find(ids.begin(), ids.end(), "stub") != ids.end());
    if (!realNeuriploSupportEnabled()) {
        // Without real neuriplo support no neuriplo backend can be served.
        REQUIRE(std::find(ids.begin(), ids.end(), "onnx_runtime") == ids.end());
    }
}

TEST_CASE(backend_registry_stub_build_keeps_legacy_neuriplo_error) {
    if (realNeuriploSupportEnabled()) {
        return; // real builds cover this through RealNeuriploSmokeTest
    }
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "onnx_runtime";
    std::string error;
    const auto executor = createExecutorFor("onnx_runtime", config, error);
    REQUIRE(executor == nullptr);
    REQUIRE(error.find("real neuriplo support is not enabled") != std::string::npos);
}
