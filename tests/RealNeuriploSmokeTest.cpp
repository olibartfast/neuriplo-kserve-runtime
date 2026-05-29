#include "ModelRegistry.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

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

Bytes tensorType(const std::vector<int64_t> &shape) {
    Bytes bytes;
    appendInt32(bytes, 1, 1); // FLOAT
    appendMessage(bytes, 2, tensorShape(shape));
    return bytes;
}

Bytes typeProto(const std::vector<int64_t> &shape) {
    Bytes bytes;
    appendMessage(bytes, 1, tensorType(shape));
    return bytes;
}

Bytes valueInfo(const std::string &name, const std::vector<int64_t> &shape) {
    Bytes bytes;
    appendString(bytes, 1, name);
    appendMessage(bytes, 2, typeProto(shape));
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

Bytes graphProto() {
    constexpr int kGraphInputField = 11;
    constexpr int kGraphOutputField = 12;
    const std::vector<int64_t> shape{1, 3, 4, 4};

    Bytes bytes;
    appendMessage(bytes, 1, identityNode());
    appendString(bytes, 2, "identity_graph");
    appendMessage(bytes, kGraphInputField, valueInfo("input", shape));
    appendMessage(bytes, kGraphOutputField, valueInfo("output", shape));
    return bytes;
}

Bytes opsetImport() {
    Bytes bytes;
    appendInt64(bytes, 2, 13);
    return bytes;
}

Bytes identityOnnxModel() {
    Bytes bytes;
    appendInt64(bytes, 1, 7);
    appendString(bytes, 2, "neuriplo-kserve-runtime");
    appendMessage(bytes, 7, graphProto());
    appendMessage(bytes, 8, opsetImport());
    return bytes;
}

std::string writeIdentityModel() {
    const auto path =
        std::filesystem::temp_directory_path() / "neuriplo-kserve-runtime-identity.onnx";
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open ONNX smoke model for writing");
    }
    const auto model = identityOnnxModel();
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

    const auto *handle = registry.findHandle("identity");
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
    input.data.reserve(48);
    for (int i = 0; i < 48; ++i) {
        input.data.push_back(static_cast<double>(i) / 10.0);
    }
    request.inputs.push_back(input);
    request.requested_outputs = {"output"};

    const auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(scheduled.ok);
    const auto &response = scheduled.response;
    REQUIRE(response.ok);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].name, "output");
    REQUIRE_EQ(response.outputs[0].shape.size(), static_cast<size_t>(4));
    REQUIRE_EQ(response.outputs[0].data.size(), input.data.size());
    for (size_t i = 0; i < input.data.size(); ++i) {
        requireNear(response.outputs[0].data[i], input.data[i]);
    }
}
