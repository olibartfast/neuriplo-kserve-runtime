#pragma once

#include "ModelRegistry.hpp"

#include <optional>
#include <string>
#include <vector>

struct RequestedOutput {
    std::string name;
};

struct InferenceRequest {
    std::optional<std::string> id;
    std::vector<RequestedOutput> requested_outputs;
};

struct InferenceParseResult {
    bool ok = false;
    InferenceRequest request;
    std::string error_message;
};

InferenceParseResult parseInferenceRequest(const std::string &body, const ModelMetadata &metadata);

std::string inferenceResponseJson(const std::string &model_name, const std::string &model_version,
                                  const InferenceRequest &request, const ModelMetadata &metadata);
