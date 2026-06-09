#include "NeuriploAdapter.hpp"

#include <stdexcept>

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
#include "BackendRegistry.hpp"
#include "InferenceBackendSetup.hpp"
#include "Tokenizer.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <variant>

namespace {

template <typename T> std::vector<uint8_t> numericBytes(const InputTensor &tensor) {
    std::vector<uint8_t> bytes(tensor.data.size() * sizeof(T));
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        const auto value = static_cast<T>(tensor.data[i]);
        std::memcpy(bytes.data() + (i * sizeof(T)), &value, sizeof(T));
    }
    return bytes;
}

std::vector<uint8_t> tensorToBytes(const InputTensor &tensor) {
    const auto &dt = tensor.datatype;
    if (dt == "BYTES") {
        std::vector<uint8_t> bytes;
        for (const auto &element : tensor.string_data) {
            bytes.insert(bytes.end(), element.begin(), element.end());
        }
        return bytes;
    }
    if (dt == "BOOL" || dt == "INT8")
        return numericBytes<int8_t>(tensor);
    if (dt == "INT16")
        return numericBytes<int16_t>(tensor);
    if (dt == "INT32")
        return numericBytes<int32_t>(tensor);
    if (dt == "INT64")
        return numericBytes<int64_t>(tensor);
    if (dt == "UINT8")
        return numericBytes<uint8_t>(tensor);
    if (dt == "UINT16")
        return numericBytes<uint16_t>(tensor);
    if (dt == "UINT32")
        return numericBytes<uint32_t>(tensor);
    if (dt == "UINT64")
        return numericBytes<uint64_t>(tensor);
    if (dt == "FP16") {
        std::vector<uint8_t> bytes(tensor.data.size() * sizeof(float));
        for (size_t i = 0; i < tensor.data.size(); ++i) {
            const auto value = static_cast<float>(tensor.data[i]);
            std::memcpy(bytes.data() + (i * sizeof(float)), &value, sizeof(float));
        }
        return bytes;
    }
    if (dt == "FP32")
        return numericBytes<float>(tensor);
    if (dt == "FP64")
        return numericBytes<double>(tensor);
    throw std::runtime_error("unsupported input datatype: " + dt);
}

template <typename T> std::vector<double> numericFromBytes(const std::vector<uint8_t> &bytes) {
    const size_t count = bytes.size() / sizeof(T);
    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        T value;
        std::memcpy(&value, bytes.data() + (i * sizeof(T)), sizeof(T));
        data[i] = static_cast<double>(value);
    }
    return data;
}

std::vector<double> bytesToTensor(const std::vector<uint8_t> &bytes, const std::string &datatype) {
    if (bytes.empty()) {
        return {};
    }
    const auto &dt = datatype;
    if (dt == "BOOL" || dt == "INT8")
        return numericFromBytes<int8_t>(bytes);
    if (dt == "INT16")
        return numericFromBytes<int16_t>(bytes);
    if (dt == "INT32")
        return numericFromBytes<int32_t>(bytes);
    if (dt == "INT64")
        return numericFromBytes<int64_t>(bytes);
    if (dt == "UINT8")
        return numericFromBytes<uint8_t>(bytes);
    if (dt == "UINT16")
        return numericFromBytes<uint16_t>(bytes);
    if (dt == "UINT32")
        return numericFromBytes<uint32_t>(bytes);
    if (dt == "UINT64")
        return numericFromBytes<uint64_t>(bytes);
    if (dt == "FP32")
        return numericFromBytes<float>(bytes);
    if (dt == "FP64")
        return numericFromBytes<double>(bytes);
    throw std::runtime_error("unsupported output datatype: " + dt);
}

std::vector<std::string> extractDatatypes(const std::vector<LayerInfo> &layers) {
    std::vector<std::string> datatypes;
    datatypes.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        datatypes.push_back("FP32");
    }
    return datatypes;
}

std::vector<TensorMetadata> convertLayers(const std::vector<LayerInfo> &layers,
                                          const std::string &default_datatype) {
    std::vector<TensorMetadata> tensors;
    tensors.reserve(layers.size());
    for (const auto &layer : layers) {
        tensors.push_back({layer.name, default_datatype, layer.shape});
    }
    return tensors;
}

std::vector<TensorMetadata> convertLayers(const std::vector<LayerInfo> &layers,
                                          const std::vector<std::string> &datatypes) {
    std::vector<TensorMetadata> tensors;
    tensors.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto &dt = i < datatypes.size() ? datatypes[i] : "FP32";
        tensors.push_back({layers[i].name, dt, layers[i].shape});
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
        if (backendKind(config.backend) == BackendKind::Llm) {
            // LLM backends report raw layer metadata (e.g. "prompt1"/FP32), but
            // the KServe LLM convention this runtime serves is a BYTES prompt
            // in and a BYTES text tensor out, matching the stub LLM executor.
            metadata.inputs = {{"prompt", "BYTES", {1}}};
            metadata.outputs = {{"text", "BYTES", {1}}};
        } else {
            metadata.inputs = convertLayers(backend_metadata.getInputs(),
                                            extractDatatypes(backend_metadata.getInputs()));
            metadata.outputs = convertLayers(backend_metadata.getOutputs(),
                                             extractDatatypes(backend_metadata.getOutputs()));
        }
        output_metadata_.clear();
        output_metadata_.reserve(metadata.outputs.size());
        for (const auto &output : metadata.outputs) {
            output_metadata_.push_back(output);
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
            backend_inputs.push_back(tensorToBytes(input));
        }

        auto [values, shapes] = engine_->get_infer_results(backend_inputs);

        NeuriploInferenceResult result;
        result.outputs.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            OutputTensor output;
            if (i < output_metadata_.size()) {
                output.name = output_metadata_[i].name;
                output.datatype = output_metadata_[i].datatype;
            } else {
                output.name = "output_" + std::to_string(i);
                output.datatype = "FP32";
            }
            output.shape = i < shapes.size() ? shapes[i] : std::vector<int64_t>{};
            output.data.reserve(values[i].size());
            for (const auto &value : values[i]) {
                output.data.push_back(tensorElementToDouble(value));
            }
            result.outputs.push_back(std::move(output));
        }
        return result;
    }

    // neuriplo LLM backends (llama.cpp, Cactus) take raw prompt bytes as the
    // first input tensor and return generated text as a tensor of per-character
    // codes. InferenceInterface::get_infer_results has no generation-parameter,
    // cancellation, or streaming hooks, so max_tokens/temperature/top_p/top_k
    // and mid-decode cancellation cannot propagate until neuriplo exposes them.
    LlmInferenceResult llmInfer(const LlmInferenceParams &params) override {
        LlmInferenceResult result;
        if (!engine_) {
            result.ok = false;
            result.error_message = "neuriplo engine is not loaded";
            return result;
        }
        if (params.cancel_token && params.cancel_token->load(std::memory_order_acquire)) {
            result.ok = false;
            result.error_message = "request cancelled before LLM inference";
            return result;
        }

        std::vector<std::vector<uint8_t>> backend_inputs;
        backend_inputs.emplace_back(params.prompt.begin(), params.prompt.end());

        std::vector<std::vector<TensorElement>> values;
        try {
            auto [out_values, out_shapes] = engine_->get_infer_results(backend_inputs);
            values = std::move(out_values);
        } catch (const std::exception &ex) {
            result.ok = false;
            result.error_message = ex.what();
            return result;
        }

        std::string generated_text;
        if (!values.empty()) {
            generated_text.reserve(values[0].size());
            for (const auto &element : values[0]) {
                generated_text.push_back(
                    static_cast<char>(static_cast<unsigned char>(tensorElementToDouble(element))));
            }
        }

        if (params.streaming_callback && !generated_text.empty()) {
            params.streaming_callback(generated_text);
        }

        OutputTensor text_output;
        text_output.name = "text";
        text_output.datatype = "BYTES";
        text_output.shape = {1};
        text_output.string_data = {generated_text};
        result.outputs.push_back(std::move(text_output));

        const WhitespaceTokenizer tokenizer;
        result.prompt_tokens = tokenizer.countTokens(params.prompt);
        result.completion_tokens = tokenizer.countTokens(generated_text);
        result.finish_reason = "stop";
        return result;
    }

  private:
    std::unique_ptr<InferenceInterface> engine_;
    std::vector<TensorMetadata> output_metadata_;
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