#include "NeuriploAdapter.hpp"

#include <stdexcept>

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
#include "BackendRegistry.hpp"
#include "InferenceBackendSetup.hpp"
#include "Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <variant>

namespace {

// Runtime backend ids are lowercase ("onnx_runtime"); neuriplo registry ids
// are the same words uppercased ("ONNX_RUNTIME").
std::string toNeuriploBackendId(const std::string &backend) {
    std::string id = backend;
    std::transform(id.begin(), id.end(), id.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return id;
}

std::string toRuntimeBackendId(const std::string &backend) {
    std::string id = backend;
    std::transform(id.begin(), id.end(), id.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return id;
}

// Canonical tensors already hold typed contiguous bytes (FP16 widened to
// float, matching what this adapter always handed to backends), so the
// backend payload is a straight byte copy.
std::vector<uint8_t> tensorToBytes(const InputTensor &tensor) {
    if (tensor.datatype == "BYTES") {
        std::vector<uint8_t> bytes;
        for (const auto &element : tensor.string_data) {
            bytes.insert(bytes.end(), element.begin(), element.end());
        }
        return bytes;
    }
    if (tensorElementSize(tensor.datatype) == 0) {
        throw std::runtime_error("unsupported input datatype: " + tensor.datatype);
    }
    const auto *begin = reinterpret_cast<const uint8_t *>(tensor.bytes.data());
    return {begin, begin + tensor.bytes.size()};
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
        EngineOptions options;
        options.model_path = config.model_path;
        options.backend_id = toNeuriploBackendId(config.backend);
        options.plugin_dir = config.plugin_dir;
        options.input_sizes = config.input_sizes;
        engine_ = setup_inference_engine(options);
        if (!engine_) {
            std::string available;
            for (const auto &id : realNeuriploAvailableBackends(config.plugin_dir)) {
                if (!available.empty()) {
                    available += ", ";
                }
                available += id;
            }
            throw std::runtime_error("failed to load backend '" + config.backend +
                                     "' (available in this process: " + available + ")");
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
            // Outputs are declared FP32 (extractDatatypes); convert each
            // TensorElement variant to the declared element type once here.
            withTensorElementType(output.datatype, [&](auto element) {
                using T = decltype(element);
                output.bytes.resize(values[i].size() * sizeof(T));
                for (size_t j = 0; j < values[i].size(); ++j) {
                    const auto typed = static_cast<T>(tensorElementToDouble(values[i][j]));
                    std::memcpy(output.bytes.data() + (j * sizeof(T)), &typed, sizeof(T));
                }
            });
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

bool realNeuriploSupportEnabled() noexcept {
    return true;
}

std::vector<std::string> realNeuriploAvailableBackends(const std::string &plugin_dir) {
    std::vector<std::string> ids;
    for (const auto &id : available_backend_ids(plugin_dir)) {
        ids.push_back(toRuntimeBackendId(id));
    }
    return ids;
}
#else
std::unique_ptr<NeuriploAdapter> makeRealNeuriploAdapter() {
    throw std::runtime_error("real neuriplo support is not enabled; rebuild with "
                             "NEURIPLO_RUNTIME_ENABLE_REAL_NEURIPLO=ON");
}

bool realNeuriploSupportEnabled() noexcept {
    return false;
}

std::vector<std::string> realNeuriploAvailableBackends(const std::string & /*plugin_dir*/) {
    return {};
}
#endif