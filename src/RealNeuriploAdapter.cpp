#include "NeuriploAdapter.hpp"

#include <stdexcept>

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
#include "InferenceBackendSetup.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <variant>

namespace {

std::vector<uint8_t> fp32Bytes(const InputTensor &tensor) {
    std::vector<uint8_t> bytes(tensor.data.size() * sizeof(float));
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        const auto value = static_cast<float>(tensor.data[i]);
        std::memcpy(bytes.data() + (i * sizeof(float)), &value, sizeof(float));
    }
    return bytes;
}

std::vector<TensorMetadata> convertLayers(const std::vector<LayerInfo> &layers,
                                          const std::string &datatype) {
    std::vector<TensorMetadata> tensors;
    tensors.reserve(layers.size());
    for (const auto &layer : layers) {
        tensors.push_back({layer.name, datatype, layer.shape});
    }
    return tensors;
}

double tensorElementToDouble(const TensorElement &element) {
    return std::visit([](const auto value) { return static_cast<double>(value); }, element);
}

class RealNeuriploAdapter final : public NeuriploAdapter {
  public:
    ModelMetadata load(const RuntimeConfig &config) override {
        engine_ = setup_inference_engine(config.model_path);
        if (!engine_) {
            throw std::runtime_error("setup_inference_engine returned null");
        }

        const auto backend_metadata = engine_->get_inference_metadata();
        ModelMetadata metadata;
        metadata.name = config.model_name;
        metadata.versions = {"1"};
        metadata.platform = "neuriplo_" + config.backend;
        metadata.inputs = convertLayers(backend_metadata.getInputs(), "FP32");
        metadata.outputs = convertLayers(backend_metadata.getOutputs(), "FP32");
        output_names_.clear();
        output_names_.reserve(metadata.outputs.size());
        for (const auto &output : metadata.outputs) {
            output_names_.push_back(output.name);
        }
        return metadata;
    }

    NeuriploInferenceResult infer(const std::vector<InputTensor> &inputs) override {
        if (!engine_) {
            throw std::runtime_error("neuriplo engine is not loaded");
        }

        std::vector<std::vector<uint8_t>> backend_inputs;
        backend_inputs.reserve(inputs.size());
        for (const auto &input : inputs) {
            if (input.datatype != "FP32") {
                throw std::runtime_error("only FP32 inputs are supported");
            }
            backend_inputs.push_back(fp32Bytes(input));
        }

        auto [values, shapes] = engine_->get_infer_results(backend_inputs);

        NeuriploInferenceResult result;
        result.outputs.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            OutputTensor output;
            output.name =
                i < output_names_.size() ? output_names_[i] : "output_" + std::to_string(i);
            output.datatype = "FP32";
            output.shape = i < shapes.size() ? shapes[i] : std::vector<int64_t>{};
            output.data.reserve(values[i].size());
            for (const auto &value : values[i]) {
                output.data.push_back(tensorElementToDouble(value));
            }
            result.outputs.push_back(std::move(output));
        }
        return result;
    }

  private:
    std::unique_ptr<InferenceInterface> engine_;
    std::vector<std::string> output_names_;
};

} // namespace

std::unique_ptr<NeuriploAdapter> makeRealNeuriploAdapter() {
    return std::make_unique<RealNeuriploAdapter>();
}
#else
std::unique_ptr<NeuriploAdapter> makeRealNeuriploAdapter() {
    throw std::runtime_error("real neuriplo support is not enabled; rebuild with "
                             "NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON");
}
#endif
