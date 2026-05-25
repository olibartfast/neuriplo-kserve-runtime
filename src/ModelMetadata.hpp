#pragma once

#include <string>
#include <vector>

struct TensorMetadata {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
};

struct ModelMetadata {
    std::string name;
    std::vector<std::string> versions;
    std::string platform;
    std::vector<TensorMetadata> inputs;
    std::vector<TensorMetadata> outputs;
};
