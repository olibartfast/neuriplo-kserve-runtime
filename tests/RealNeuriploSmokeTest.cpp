#include "BackendRegistry.hpp"
#include "ModelRegistry.hpp"
#include "NeuriploAdapter.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<uint8_t>;

void appendVarint(Bytes &bytes, uint64_t value) {
    while (value >= 0x80) {
        bytes.push_back(static_cast<uint8_t>(value | 0x80));
        value >>= 7;
    }
    bytes.push_back(static_cast<uint8_t>(value));
}

void appendKey(Bytes &bytes, int field_number, int wire_type) {
    appendVarint(bytes,
                 (static_cast<uint64_t>(field_number) << 3U) | static_cast<uint64_t>(wire_type));
}

void appendInt64(Bytes &bytes, int field_number, int64_t value) {
    appendKey(bytes, field_number, 0);
    appendVarint(bytes, static_cast<uint64_t>(value));
}

void appendInt32(Bytes &bytes, int field_number, int32_t value) {
    appendKey(bytes, field_number, 0);
    appendVarint(bytes, static_cast<uint64_t>(value));
}

void appendString(Bytes &bytes, int field_number, const std::string &value) {
    appendKey(bytes, field_number, 2);
    appendVarint(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendMessage(Bytes &bytes, int field_number, const Bytes &message) {
    appendKey(bytes, field_number, 2);
    appendVarint(bytes, message.size());
    bytes.insert(bytes.end(), message.begin(), message.end());
}

Bytes tensorShapeDimension(int64_t value) {
    Bytes bytes;
    appendInt64(bytes, 1, value);
    return bytes;
}

Bytes tensorShape(const std::vector<int64_t> &shape) {
    Bytes bytes;
    for (const auto dimension : shape) {
        appendMessage(bytes, 1, tensorShapeDimension(dimension));
    }
    return bytes;
}

// elem_type is an onnx TensorProto.DataType (FLOAT = 1, INT64 = 7, ...).
Bytes tensorType(const std::vector<int64_t> &shape, int elem_type) {
    Bytes bytes;
    appendInt32(bytes, 1, elem_type);
    appendMessage(bytes, 2, tensorShape(shape));
    return bytes;
}

Bytes typeProto(const std::vector<int64_t> &shape, int elem_type) {
    Bytes bytes;
    appendMessage(bytes, 1, tensorType(shape, elem_type));
    return bytes;
}

Bytes valueInfo(const std::string &name, const std::vector<int64_t> &shape, int elem_type) {
    Bytes bytes;
    appendString(bytes, 1, name);
    appendMessage(bytes, 2, typeProto(shape, elem_type));
    return bytes;
}

Bytes identityNode() {
    Bytes bytes;
    appendString(bytes, 1, "input");
    appendString(bytes, 2, "output");
    appendString(bytes, 3, "identity");
    appendString(bytes, 4, "Identity");
    return bytes;
}

Bytes graphProto(int elem_type) {
    constexpr int kGraphInputField = 11;
    constexpr int kGraphOutputField = 12;
    const std::vector<int64_t> shape{1, 3, 4, 4};

    Bytes bytes;
    appendMessage(bytes, 1, identityNode());
    appendString(bytes, 2, "identity_graph");
    appendMessage(bytes, kGraphInputField, valueInfo("input", shape, elem_type));
    appendMessage(bytes, kGraphOutputField, valueInfo("output", shape, elem_type));
    return bytes;
}

Bytes opsetImport() {
    Bytes bytes;
    appendInt64(bytes, 2, 13);
    return bytes;
}

// onnx_elem_type defaults to FLOAT (1); the IR version field below (value 7) is
// unrelated to the tensor element type.
Bytes identityOnnxModel(int onnx_elem_type) {
    Bytes bytes;
    appendInt64(bytes, 1, 7);
    appendString(bytes, 2, "neuriplo-kserve-runtime");
    appendMessage(bytes, 7, graphProto(onnx_elem_type));
    appendMessage(bytes, 8, opsetImport());
    return bytes;
}

std::string writeIdentityModel(int onnx_elem_type = 1) {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("neuriplo-kserve-runtime-identity-" + std::to_string(onnx_elem_type) + ".onnx");
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open ONNX smoke model for writing");
    }
    const auto model = identityOnnxModel(onnx_elem_type);
    output.write(reinterpret_cast<const char *>(model.data()),
                 static_cast<std::streamsize>(model.size()));
    return path.string();
}

RuntimeConfig realOnnxConfig(const std::string &model_path) {
    RuntimeConfig config;
    config.model_name = "identity";
    config.model_path = model_path;
    config.backend = "onnx_runtime";
    return config;
}

void requireNear(double actual, double expected) {
    REQUIRE(std::fabs(actual - expected) < 1.0e-6);
}

} // namespace

TEST_CASE(real_neuriplo_onnx_identity_golden_comparison) {
    const auto model_path = writeIdentityModel();
    const ModelRegistry registry(realOnnxConfig(model_path));
    REQUIRE(registry.allReady());

    const auto handle = registry.findHandle("identity");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->scheduler != nullptr);
    REQUIRE_EQ(handle->metadata.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(handle->metadata.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(handle->metadata.inputs[0].name, "input");
    REQUIRE_EQ(handle->metadata.outputs[0].name, "output");

    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3, 4, 4};
    std::vector<double> input_values;
    input_values.reserve(48);
    for (int i = 0; i < 48; ++i) {
        input_values.push_back(static_cast<double>(i) / 10.0);
    }
    input.bytes = tensorBytesFromDoubles(input.datatype, input_values);
    request.inputs.push_back(input);
    request.requested_outputs = {"output"};

    const auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(scheduled.ok);
    const auto &response = scheduled.response;
    REQUIRE(response.ok);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].name, "output");
    REQUIRE_EQ(response.outputs[0].shape.size(), static_cast<size_t>(4));
    const auto output_values =
        tensorValuesAsDoubles(response.outputs[0].datatype, response.outputs[0].bytes);
    REQUIRE_EQ(output_values.size(), input_values.size());
    for (size_t i = 0; i < input_values.size(); ++i) {
        requireNear(output_values[i], input_values[i]);
    }
}

// Milestone proof: one process serves the same model through two different
// neuriplo backends (built-in and/or dlopen plugin, depending on the preset).
TEST_CASE(real_neuriplo_serves_two_backends_in_one_process) {
    const auto available = realNeuriploAvailableBackends();
    const auto has = [&available](const char *id) {
        return std::find(available.begin(), available.end(), id) != available.end();
    };
    if (!has("opencv_dnn") || !has("onnx_runtime")) {
        return; // single-backend build
    }

    const auto model_path = writeIdentityModel();
    ModelRegistry registry(realOnnxConfig(model_path));
    REQUIRE(registry.allReady());

    auto cv_config = realOnnxConfig(model_path);
    cv_config.model_name = "identity_cv";
    cv_config.backend = "opencv_dnn";
    // opencv_dnn cannot introspect input shapes from the model file; the
    // backend prepends the batch dimension itself.
    cv_config.input_sizes = {{3, 4, 4}};
    REQUIRE(registry.loadModel(cv_config));
    REQUIRE(registry.ready("identity"));
    REQUIRE(registry.ready("identity_cv"));

    for (const auto *model_name : {"identity", "identity_cv"}) {
        const auto handle = registry.findHandle(model_name);
        REQUIRE(handle != nullptr);
        REQUIRE(handle->scheduler != nullptr);
        REQUIRE(!handle->metadata.inputs.empty());

        ExecutionRequest request;
        InputTensor input;
        input.name = handle->metadata.inputs[0].name;
        input.datatype = "FP32";
        input.shape = handle->metadata.inputs[0].shape;
        size_t element_count = 1;
        for (const auto dim : input.shape) {
            element_count *= static_cast<size_t>(dim);
        }
        REQUIRE_EQ(element_count, static_cast<size_t>(48));
        std::vector<double> input_values;
        for (size_t i = 0; i < element_count; ++i) {
            input_values.push_back(static_cast<double>(i) / 10.0);
        }
        input.bytes = tensorBytesFromDoubles(input.datatype, input_values);
        request.inputs.push_back(input);

        const auto scheduled = handle->scheduler->submit(std::move(request));
        REQUIRE(scheduled.ok);
        REQUIRE(scheduled.response.ok);
        REQUIRE_EQ(scheduled.response.outputs.size(), static_cast<size_t>(1));
        const auto output_values = tensorValuesAsDoubles(scheduled.response.outputs[0].datatype,
                                                         scheduled.response.outputs[0].bytes);
        REQUIRE_EQ(output_values.size(), static_cast<size_t>(48));
        for (size_t i = 0; i < 48; ++i) {
            requireNear(output_values[i], static_cast<double>(i) / 10.0);
        }
    }
}

// Regression guard for the dtype-propagation bug: the runtime once hardcoded
// every backend tensor datatype to "FP32", which corrupted non-float tensors
// (e.g. EdgeCrafter's INT64 orig_target_sizes input and labels output) crossing
// the serving metadata boundary. Build an INT64 identity model and require the
// real adapter to advertise the true element type, not FP32.
TEST_CASE(real_neuriplo_reports_non_fp32_metadata_datatype) {
    constexpr int kOnnxInt64 = 7; // onnx TensorProto.DataType.INT64
    const auto model_path = writeIdentityModel(kOnnxInt64);
    auto config = realOnnxConfig(model_path);
    config.model_name = "identity_i64";
    const ModelRegistry registry(config);
    REQUIRE(registry.allReady());

    const auto handle = registry.findHandle("identity_i64");
    REQUIRE(handle != nullptr);
    REQUIRE_EQ(handle->metadata.inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(handle->metadata.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(handle->metadata.inputs[0].datatype, "INT64");
    REQUIRE_EQ(handle->metadata.outputs[0].datatype, "INT64");
}

TEST_CASE(real_neuriplo_reports_available_backends) {
    REQUIRE(realNeuriploSupportEnabled());
    const auto available = realNeuriploAvailableBackends();
    REQUIRE(!available.empty());
    REQUIRE(std::find(available.begin(), available.end(), "onnx_runtime") != available.end());
}

TEST_CASE(real_neuriplo_unavailable_backend_lists_alternatives) {
    const auto available = realNeuriploAvailableBackends();
    if (std::find(available.begin(), available.end(), "tensorrt") != available.end()) {
        return; // TensorRT is genuinely available in this build
    }

    RuntimeConfig config;
    config.model_name = "raft";
    config.model_path = "/nonexistent/model.plan";
    config.backend = "tensorrt";
    std::string error;
    const auto executor = createExecutorFor("tensorrt", config, error);
    REQUIRE(executor == nullptr);
    REQUIRE(error.find("'tensorrt' is not available") != std::string::npos);
    REQUIRE(error.find("onnx_runtime") != std::string::npos);
}
