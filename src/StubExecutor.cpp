#include "StubExecutor.hpp"

#include <algorithm>
#include <utility>

namespace {

class StubExecutor final : public Executor {
  public:
    explicit StubExecutor(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        std::vector<std::string> output_names;
        if (request.requested_outputs.empty()) {
            output_names.reserve(metadata_.outputs.size());
            for (const auto &output : metadata_.outputs) {
                output_names.push_back(output.name);
            }
        } else {
            output_names = request.requested_outputs;
        }

        ExecutionResponse response;
        response.outputs.reserve(output_names.size());
        for (const auto &name : output_names) {
            const auto *tensor_metadata = findOutput(name);
            if (tensor_metadata == nullptr) {
                continue;
            }
            response.outputs.push_back(makeOutput(*tensor_metadata));
        }
        return response;
    }

  private:
    const TensorMetadata *findOutput(const std::string &name) const {
        const auto found =
            std::find_if(metadata_.outputs.begin(), metadata_.outputs.end(),
                         [&name](const auto &tensor) { return tensor.name == name; });
        if (found == metadata_.outputs.end()) {
            return nullptr;
        }
        return &*found;
    }

    static OutputTensor makeOutput(const TensorMetadata &metadata) {
        OutputTensor output;
        output.name = metadata.name;
        output.datatype = metadata.datatype;
        output.shape = metadata.shape;

        size_t element_count = 1;
        for (const auto dimension : metadata.shape) {
            element_count *= static_cast<size_t>(dimension);
        }
        output.data.assign(element_count, 0.0);
        return output;
    }

    ModelMetadata metadata_;
};

ModelMetadata stubMetadata(const RuntimeConfig &config) {
    ModelMetadata metadata;
    metadata.name = config.model_name;
    metadata.versions.push_back("1");
    metadata.platform = "neuriplo_" + config.backend;
    metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    metadata.outputs.push_back({"output", "FP32", {1, 1000}});
    return metadata;
}

} // namespace

std::unique_ptr<Executor> makeStubExecutor(const RuntimeConfig &config, std::string &error) {
    (void)error;
    return std::make_unique<StubExecutor>(stubMetadata(config));
}
