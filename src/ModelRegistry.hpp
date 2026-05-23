#pragma once

#include "RuntimeConfig.hpp"

#include <optional>
#include <string>
#include <vector>

struct TensorMetadata {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
};

struct ModelMetadata {
    std::string name;
    std::string platform;
    std::vector<TensorMetadata> inputs;
    std::vector<TensorMetadata> outputs;
    bool ready = false;
};

class ModelRegistry {
  public:
    explicit ModelRegistry(const RuntimeConfig &config);

    std::optional<ModelMetadata> find(const std::string &model_name) const;
    bool ready(const std::string &model_name) const;
    bool allReady() const;

  private:
    ModelMetadata model_;
};
