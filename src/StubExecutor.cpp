#include "StubExecutor.hpp"

#include "BackendRegistry.hpp"

#include <algorithm>
#include <utility>

namespace {

class StubExecutor final : public Executor {
  public:
    explicit StubExecutor(ModelMetadata metadata, bool llm_mode, size_t default_max_tokens)
        : metadata_(std::move(metadata)), llm_mode_(llm_mode),
          default_max_tokens_(default_max_tokens) {}

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        if (request.cancel_token && request.cancel_token->load(std::memory_order_acquire)) {
            ExecutionResponse response;
            response.ok = false;
            response.error_code = "DEADLINE_EXCEEDED";
            response.error_message = "request cancelled before inference";
            return response;
        }

        if (llm_mode_) {
            return inferLlm(request);
        }

        std::vector<std::string> output_names;
        if (request.requested_outputs.empty()) {
            output_names.reserve(metadata_.outputs.size());
            for (const auto &output : metadata_.outputs) {
                output_names.push_back(output.name);
            }
        } else {
            output_names = request.requested_outputs;
        }

        int64_t batch_size = 1;
        if (!request.inputs.empty() && !request.inputs.front().shape.empty()) {
            batch_size = request.inputs.front().shape.front();
        }

        ExecutionResponse response;
        response.outputs.reserve(output_names.size());
        for (const auto &name : output_names) {
            const auto *tensor_metadata = findOutput(name);
            if (tensor_metadata == nullptr) {
                continue;
            }
            response.outputs.push_back(makeOutput(*tensor_metadata, batch_size));
        }
        return response;
    }

    ExecutionResponse inferStreaming(const ExecutionRequest &request,
                                     StreamingTokenCallback callback) override {
        ExecutionResponse response = infer(request);
        if (response.ok && llm_mode_ && callback) {
            for (const auto &output : response.outputs) {
                if (output.datatype == "BYTES" && !output.string_data.empty()) {
                    const auto &text = output.string_data.front();
                    const size_t chunk_size = 2;
                    for (size_t i = 0; i < text.size(); i += chunk_size) {
                        callback(text.substr(i, chunk_size));
                    }
                }
            }
        }
        return response;
    }

  private:
    ExecutionResponse inferLlm(const ExecutionRequest &request) {
        const InputTensor *prompt = nullptr;
        for (const auto &input : request.inputs) {
            if (isBytesDatatype(input.datatype)) {
                prompt = &input;
                break;
            }
        }
        if (prompt == nullptr || prompt->string_data.empty()) {
            ExecutionResponse response;
            response.ok = false;
            response.error_code = "INVALID_ARGUMENT";
            response.error_message = "LLM request requires a BYTES prompt input";
            return response;
        }

        size_t max_tokens = default_max_tokens_;
        if (request.llm_params && request.llm_params->max_tokens) {
            max_tokens = *request.llm_params->max_tokens;
        }

        std::string generated = "stub: ";
        const auto &prompt_text = prompt->string_data.front();
        const size_t append_chars = std::min(max_tokens, prompt_text.size());
        generated.append(prompt_text.data(), append_chars);

        size_t prompt_token_estimate = prompt_text.size() / 4;
        if (prompt_token_estimate == 0)
            prompt_token_estimate = 1;
        size_t completion_token_estimate = append_chars / 4;
        if (completion_token_estimate == 0)
            completion_token_estimate = 1;

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
            OutputTensor output;
            output.name = tensor_metadata->name;
            output.datatype = tensor_metadata->datatype;
            output.shape = tensor_metadata->shape;
            output.string_data = {generated};
            response.outputs.push_back(std::move(output));
        }

        LlmResultMetadata llm_meta;
        llm_meta.prompt_tokens = prompt_token_estimate;
        llm_meta.completion_tokens = completion_token_estimate;
        llm_meta.finish_reason = "stop";
        response.llm_metadata = llm_meta;

        return response;
    }

    const TensorMetadata *findOutput(const std::string &name) const {
        const auto found =
            std::find_if(metadata_.outputs.begin(), metadata_.outputs.end(),
                         [&name](const auto &tensor) { return tensor.name == name; });
        if (found == metadata_.outputs.end()) {
            return nullptr;
        }
        return &*found;
    }

    static OutputTensor makeOutput(const TensorMetadata &metadata, int64_t batch_size) {
        OutputTensor output;
        output.name = metadata.name;
        output.datatype = metadata.datatype;
        output.shape = metadata.shape;
        if (!output.shape.empty()) {
            output.shape[0] = batch_size;
        }

        size_t element_count = 1;
        for (const auto dimension : output.shape) {
            element_count *= static_cast<size_t>(dimension);
        }
        // All-zero bytes decode to zero for every fixed-size datatype.
        output.bytes.assign(element_count * tensorElementSize(output.datatype), std::byte{0});
        return output;
    }

    ModelMetadata metadata_;
    bool llm_mode_ = false;
    size_t default_max_tokens_ = 256;
};

ModelMetadata tensorStubMetadata(const RuntimeConfig &config) {
    ModelMetadata metadata;
    metadata.name = config.model_name;
    metadata.versions.push_back("1");
    metadata.platform = "neuriplo_" + config.backend;
    metadata.inputs.push_back({"input", "FP32", {1, 3, 224, 224}});
    metadata.outputs.push_back({"output", "FP32", {1, 1000}});
    return metadata;
}

ModelMetadata llmStubMetadata(const RuntimeConfig &config) {
    ModelMetadata metadata;
    metadata.name = config.model_name;
    metadata.versions.push_back("1");
    metadata.platform = "neuriplo_" + config.backend;
    metadata.inputs.push_back({"prompt", "BYTES", {1}});
    metadata.outputs.push_back({"text", "BYTES", {1}});
    return metadata;
}

} // namespace

std::unique_ptr<Executor> makeStubExecutor(const RuntimeConfig &config, std::string &error) {
    (void)error;
    const bool llm_mode = usesLlmScheduler(config.scheduler_strategy, config.backend);
    const auto metadata = llm_mode ? llmStubMetadata(config) : tensorStubMetadata(config);
    return std::make_unique<StubExecutor>(std::move(metadata), llm_mode, config.max_tokens);
}
