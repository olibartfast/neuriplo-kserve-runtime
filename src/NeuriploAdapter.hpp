#pragma once

#include "Executor.hpp"
#include "ModelMetadata.hpp"
#include "RuntimeConfig.hpp"

#include <memory>
#include <string>
#include <vector>

struct NeuriploInferenceResult {
    bool ok = true;
    std::string error_message;
    std::vector<OutputTensor> outputs;
};

struct LlmInferenceParams {
    std::string prompt;
    size_t max_tokens = 256;
    double temperature = 0.8;
    double top_p = 0.95;
    size_t top_k = 40;
    CancelToken cancel_token;
    StreamingTokenCallback streaming_callback;
};

struct LlmInferenceResult {
    bool ok = true;
    std::string error_message;
    std::vector<OutputTensor> outputs;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
    std::string finish_reason;
};

class NeuriploAdapter {
  public:
    virtual ~NeuriploAdapter() = default;
    virtual ModelMetadata load(const RuntimeConfig &config) = 0;
    virtual NeuriploInferenceResult infer(const std::vector<InputTensor> &inputs) = 0;
    virtual LlmInferenceResult llmInfer(const LlmInferenceParams &params);
};

using NeuriploAdapterFactory = std::unique_ptr<NeuriploAdapter> (*)();

std::unique_ptr<NeuriploAdapter> makeRealNeuriploAdapter();
