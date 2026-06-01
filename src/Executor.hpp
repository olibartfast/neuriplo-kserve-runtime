#pragma once

#include "ModelMetadata.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using CancelToken = std::shared_ptr<std::atomic<bool>>;

inline CancelToken makeCancelToken() {
    return std::make_shared<std::atomic<bool>>(false);
}

inline bool isBytesDatatype(const std::string &datatype) {
    return datatype == "BYTES";
}

struct OutputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<double> data;
    std::vector<std::string> string_data;
};

struct InputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<double> data;
    std::vector<std::string> string_data;
};

struct LlmGenerationParams {
    std::optional<size_t> max_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<size_t> top_k;
    bool streaming = false;
};

struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<InputTensor> inputs;
    std::vector<std::string> requested_outputs;
    std::optional<LlmGenerationParams> llm_params;
    CancelToken cancel_token;
};

struct ExecutionResponse {
    bool ok = true;
    std::string error_code;
    std::string error_message;
    std::vector<OutputTensor> outputs;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual const ModelMetadata &metadata() const = 0;
    virtual ExecutionResponse infer(const ExecutionRequest &request) = 0;
};
