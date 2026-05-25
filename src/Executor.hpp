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

struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<std::string> requested_outputs;
};

struct ExecutionResponse {
    std::vector<OutputTensor> outputs;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual const ModelMetadata &metadata() const = 0;
    virtual ExecutionResponse infer(const ExecutionRequest &request) = 0;
};
