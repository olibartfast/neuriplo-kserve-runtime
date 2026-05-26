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

class NeuriploAdapter {
  public:
    virtual ~NeuriploAdapter() = default;
    virtual ModelMetadata load(const RuntimeConfig &config) = 0;
    virtual NeuriploInferenceResult infer(const std::vector<InputTensor> &inputs) = 0;
};

using NeuriploAdapterFactory = std::unique_ptr<NeuriploAdapter> (*)();

std::unique_ptr<NeuriploAdapter> makeRealNeuriploAdapter();
