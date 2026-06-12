#pragma once

#include "ModelMetadata.hpp"
#include "TensorBytes.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using CancelToken = std::shared_ptr<std::atomic<bool>>;

inline CancelToken makeCancelToken() {
    return std::make_shared<std::atomic<bool>>(false);
}

inline bool isBytesDatatype(const std::string &datatype) {
    return datatype == "BYTES";
}

// Tensor payloads are native-endian typed bytes (see TensorBytes.hpp); BYTES
// tensors use string_data instead. `bytes` holds elementCount() elements of
// tensorElementSize(datatype) bytes each.
struct OutputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<std::byte> bytes;
    std::vector<std::string> string_data;

    size_t elementCount() const {
        return tensorElementCount(datatype, bytes);
    }
};

struct InputTensor {
    std::string name;
    std::string datatype;
    std::vector<int64_t> shape;
    std::vector<std::byte> bytes;
    std::vector<std::string> string_data;

    size_t elementCount() const {
        return tensorElementCount(datatype, bytes);
    }
};

struct LlmGenerationParams {
    std::optional<size_t> max_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<size_t> top_k;
    bool streaming = false;
};

struct LlmResultMetadata {
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
    std::string finish_reason;
};

using StreamingTokenCallback = std::function<void(const std::string &token)>;

struct ExecutionRequest {
    std::optional<std::string> id;
    std::vector<InputTensor> inputs;
    std::vector<std::string> requested_outputs;
    std::optional<LlmGenerationParams> llm_params;
    CancelToken cancel_token;
    StreamingTokenCallback streaming_callback;
};

struct ExecutionResponse {
    bool ok = true;
    std::string error_code;
    std::string error_message;
    std::vector<OutputTensor> outputs;
    std::optional<LlmResultMetadata> llm_metadata;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual const ModelMetadata &metadata() const = 0;
    virtual ExecutionResponse infer(const ExecutionRequest &request) = 0;
    virtual ExecutionResponse inferStreaming(const ExecutionRequest &request,
                                             StreamingTokenCallback callback);
};