#include "BackendRegistry.hpp"
#include "Test.hpp"

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
