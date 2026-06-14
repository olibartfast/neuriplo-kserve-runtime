#include "NeuriploAdapter.hpp"

#include <stdexcept>

#ifdef NEURIPLO_RUNTIME_WITH_REAL_NEURIPLO
#include "BackendRegistry.hpp"
#include "InferenceBackendSetup.hpp"
#include "Logging.hpp"
#include "Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
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

std::string tensorDataTypeToKserve(TensorDataType dtype) {
    switch (dtype) {
    case TensorDataType::Float32:
        return "FP32";
    case TensorDataType::Int32:
        return "INT32";
    case TensorDataType::Int64:
        return "INT64";
    case TensorDataType::UInt8:
        return "UINT8";
    case TensorDataType::Int8:
        return "INT8";
    case TensorDataType::Bool:
        return "BOOL";
    }
    throw std::runtime_error("unsupported neuriplo tensor datatype");
}

std::vector<std::string> extractDatatypes(const std::vector<LayerInfo> &layers) {
    std::vector<std::string> datatypes;
    datatypes.reserve(layers.size());
    for (const auto &layer : layers) {
        // Carry the real per-layer datatype reported by the backend (e.g. INT64
        // for EdgeCrafter's orig_target_sizes input and labels output) instead
        // of assuming every tensor is FP32.
        datatypes.push_back(tensorDataTypeToKserve(layer.datatype));
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

// --- config.pbtxt I/O overlay -------------------------------------------------
// The .pte ExecuTorch format carries no tensor names, so a name-less backend
// advertises positional input_0/output_0 metadata. Triton-style model configs
// already declare the authoritative I/O contract, so when a config.pbtxt sits
// beside the model we overlay its names (and datatypes) onto backend metadata
// by index. Backends that introspect real names (ONNX, TensorRT, OpenVINO) are
// unaffected because the config matches the model. This keeps the backends
// fully model-agnostic instead of hardcoding any single model's tensor names.
struct PbtxtTensor {
    std::string name;
    std::string datatype; // KServe datatype string; empty when unmapped
};

std::string pbtxtTypeToKserve(const std::string &type) {
    if (type == "TYPE_FP32")
        return "FP32";
    if (type == "TYPE_FP16")
        return "FP16";
    if (type == "TYPE_INT64")
        return "INT64";
    if (type == "TYPE_INT32")
        return "INT32";
    if (type == "TYPE_INT8")
        return "INT8";
    if (type == "TYPE_UINT8")
        return "UINT8";
    if (type == "TYPE_BOOL")
        return "BOOL";
    return {};
}

std::string stripPbtxtComments(const std::string &content) {
    std::string out;
    out.reserve(content.size());
    bool in_string = false;
    for (size_t i = 0; i < content.size(); ++i) {
        const char c = content[i];
        if (c == '"') {
            in_string = !in_string;
        }
        if (c == '#' && !in_string) {
            while (i < content.size() && content[i] != '\n') {
                ++i;
            }
            if (i < content.size()) {
                out.push_back('\n');
            }
            continue;
        }
        out.push_back(c);
    }
    return out;
}

// Returns the balanced [ ... ] body that follows a top-level key (dims use
// nested [ ] so bracket depth, not the first ']', delimits the section).
std::optional<std::string> extractBracketSection(const std::string &content,
                                                 const std::string &key) {
    const std::regex header(key + R"(\s*\[)");
    std::smatch match;
    if (!std::regex_search(content, match, header)) {
        return std::nullopt;
    }
    size_t pos = static_cast<size_t>(match.position(0)) + static_cast<size_t>(match.length(0));
    const size_t start = pos;
    int depth = 1;
    for (; pos < content.size() && depth > 0; ++pos) {
        if (content[pos] == '[') {
            ++depth;
        } else if (content[pos] == ']') {
            --depth;
        }
    }
    if (depth != 0) {
        return std::nullopt;
    }
    return content.substr(start, pos - 1 - start);
}

std::vector<PbtxtTensor> parsePbtxtTensors(const std::string &section) {
    std::vector<PbtxtTensor> tensors;
    const std::regex name_re(R"re(name\s*:\s*"([^"]*)")re");
    const std::regex dtype_re(R"re(data_type\s*:\s*(TYPE_[A-Z0-9]+))re");
    for (size_t i = 0; i < section.size();) {
        if (section[i] != '{') {
            ++i;
            continue;
        }
        const size_t start = ++i;
        int depth = 1;
        for (; i < section.size() && depth > 0; ++i) {
            if (section[i] == '{') {
                ++depth;
            } else if (section[i] == '}') {
                --depth;
            }
        }
        const std::string object = section.substr(start, (i - 1) - start);
        PbtxtTensor tensor;
        std::smatch m;
        if (std::regex_search(object, m, name_re)) {
            tensor.name = m[1];
        }
        if (std::regex_search(object, m, dtype_re)) {
            tensor.datatype = pbtxtTypeToKserve(m[1]);
        }
        if (!tensor.name.empty()) {
            tensors.push_back(std::move(tensor));
        }
    }
    return tensors;
}

std::optional<std::string> findConfigPbtxt(const std::string &model_path) {
    namespace fs = std::filesystem;
    const fs::path path(model_path);
    std::vector<fs::path> candidates;
    if (path.has_parent_path()) {
        // <model>/<version>/model.ext -> <model>/config.pbtxt
        candidates.push_back(path.parent_path().parent_path() / "config.pbtxt");
        candidates.push_back(path.parent_path() / "config.pbtxt");
    }
    candidates.push_back(path / "config.pbtxt");
    for (const auto &candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec)) {
            return candidate.string();
        }
    }
    return std::nullopt;
}

void overlayPbtxt(std::vector<TensorMetadata> &tensors, const std::vector<PbtxtTensor> &config,
                  const char *kind) {
    if (config.empty()) {
        return;
    }
    if (config.size() != tensors.size()) {
        defaultLogger().warn(std::string("config.pbtxt ") + kind + " count (" +
                             std::to_string(config.size()) + ") does not match backend metadata (" +
                             std::to_string(tensors.size()) + "); keeping backend " + kind +
                             " metadata");
        return;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensors[i].name = config[i].name;
        if (!config[i].datatype.empty()) {
            tensors[i].datatype = config[i].datatype;
        }
    }
}

void applyConfigPbtxt(const std::string &model_path, ModelMetadata &metadata) {
    const auto config_path = findConfigPbtxt(model_path);
    if (!config_path) {
        return;
    }
    std::ifstream in(*config_path);
    if (!in) {
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    content = stripPbtxtComments(content);
    const auto inputs = extractBracketSection(content, "input");
    const auto outputs = extractBracketSection(content, "output");
    if (inputs) {
        overlayPbtxt(metadata.inputs, parsePbtxtTensors(*inputs), "input");
    }
    if (outputs) {
        overlayPbtxt(metadata.outputs, parsePbtxtTensors(*outputs), "output");
    }
    defaultLogger().info("applied config.pbtxt I/O metadata from " + *config_path);
}

double tensorElementToDouble(const TensorElement &element) {
    return std::visit([](const auto value) { return static_cast<double>(value); }, element);
}

std::string tensorDtypeToKserve(TensorDtype dtype) {
    switch (dtype) {
    case TensorDtype::FP32:
        return "FP32";
    case TensorDtype::INT32:
        return "INT32";
    case TensorDtype::INT64:
        return "INT64";
    case TensorDtype::UINT8:
        return "UINT8";
    }
    throw std::runtime_error("unsupported neuriplo output dtype");
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
            // Triton-style config.pbtxt (when present) is authoritative for I/O
            // names/datatypes; this supplies names for backends whose model
            // format carries none (ExecuTorch .pte) without backend hardcoding.
            applyConfigPbtxt(config.model_path, metadata);
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

        const auto raw_outputs = engine_->get_infer_results_raw(backend_inputs);

        NeuriploInferenceResult result;
        result.outputs.reserve(raw_outputs.size());
        for (size_t i = 0; i < raw_outputs.size(); ++i) {
            const auto &raw = raw_outputs[i];
            OutputTensor output;
            if (i < output_metadata_.size()) {
                output.name = output_metadata_[i].name;
            } else {
                output.name = "output_" + std::to_string(i);
            }
            output.datatype = tensorDtypeToKserve(raw.dtype);
            output.shape = raw.shape;
            output.bytes.resize(raw.bytes.size());
            if (!raw.bytes.empty()) {
                std::memcpy(output.bytes.data(), raw.bytes.data(), raw.bytes.size());
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