#pragma once

#include "Executor.hpp"

#include "NeuriploAdapter.hpp"
#include "RuntimeConfig.hpp"
#include <optional>

#include <memory>
#include <string>

class NeuriploExecutor final : public Executor {
  public:
    NeuriploExecutor(const RuntimeConfig &config, std::unique_ptr<NeuriploAdapter> adapter);

    const ModelMetadata &metadata() const override;
    ExecutionResponse infer(const ExecutionRequest &request) override;
    ExecutionResponse inferStreaming(const ExecutionRequest &request,
                                     StreamingTokenCallback callback) override;

  private:
    ModelMetadata metadata_;
    std::unique_ptr<NeuriploAdapter> adapter_;
    std::string extractPrompt(const ExecutionRequest &request) const;
};

// Validates and orders a request's inputs against model metadata. Exposed so
// the validation rules -- notably dynamic-axis acceptance -- can be tested
// without standing up an adapter.
std::optional<std::vector<InputTensor>> neuriploOrderedInputs(const ModelMetadata &metadata,
                                                              const ExecutionRequest &request,
                                                              ExecutionResponse &error);

std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error);
std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error,
                                               std::unique_ptr<NeuriploAdapter> adapter);
