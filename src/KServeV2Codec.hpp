#pragma once

#include "Executor.hpp"
#include "ModelMetadata.hpp"

#include <optional>
#include <string>
#include <vector>

struct InferenceRequest {
    std::optional<std::string> id;
    std::vector<InputTensor> inputs;
    std::vector<std::string> requested_outputs;
};

struct InferenceParseResult {
    bool ok = false;
    InferenceRequest request;
    std::string error_message;
};

InferenceParseResult parseInferenceRequest(const std::string &body, const ModelMetadata &metadata);

std::string modelMetadataJson(const ModelMetadata &metadata);

std::string inferenceResponseJson(const std::string &model_name, const std::string &model_version,
                                  const InferenceRequest &request,
                                  const ExecutionResponse &response);
