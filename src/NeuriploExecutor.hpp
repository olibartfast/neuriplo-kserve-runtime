#pragma once

#include "Executor.hpp"
#include "NeuriploAdapter.hpp"
#include "RuntimeConfig.hpp"

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

std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error);
std::unique_ptr<Executor> makeNeuriploExecutor(const RuntimeConfig &config, std::string &error,
                                               std::unique_ptr<NeuriploAdapter> adapter);
