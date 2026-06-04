#include "NeuriploAdapter.hpp"

LlmInferenceResult NeuriploAdapter::llmInfer(const LlmInferenceParams & /*params*/) {
    NeuriploInferenceResult tensor_result;
    auto result = infer({});

    LlmInferenceResult llm_result;
    llm_result.ok = result.ok;
    if (!result.ok) {
        llm_result.error_message = result.error_message;
        return llm_result;
    }
    llm_result.outputs = std::move(result.outputs);
    llm_result.finish_reason = "stop";
    return llm_result;
}