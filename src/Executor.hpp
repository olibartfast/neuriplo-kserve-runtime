#pragma once

#include "ModelMetadata.hpp"

#include <optional>
#include <string>
#include <vector>

struct OutputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<double> data;
};

struct InputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<double> data;
};

struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<InputTensor> inputs;
    std::vector<std::string> requested_outputs;
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
