#include "NeuriploExecutor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

RuntimeConfig config(std::string backend = "onnx_runtime") {
    RuntimeConfig cfg;
    cfg.model_name = "demo";
    cfg.model_path = "/tmp/model.onnx";
    cfg.backend = std::move(backend);
    return cfg;
}

ModelMetadata adapterMetadata(const RuntimeConfig &cfg) {
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"7"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"input", "FP32", {1, 3}});
    metadata.outputs.push_back({"scores", "FP32", {1, 2}});
    metadata.outputs.push_back({"labels", "FP32", {1, 1}});
    return metadata;
}

ModelMetadata multiInputAdapterMetadata(const RuntimeConfig &cfg) {
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"7"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"first", "FP32", {1}});
    metadata.inputs.push_back({"second", "FP32", {1}});
    metadata.outputs.push_back({"scores", "FP32", {1, 2}});
    return metadata;
}

class FakeNeuriploAdapter final : public NeuriploAdapter {
  public:
    explicit FakeNeuriploAdapter(ModelMetadata metadata) : metadata_(std::move(metadata)) {}

    ModelMetadata load(const RuntimeConfig &) override {
        if (!load_error_.empty()) {
            throw std::runtime_error(load_error_);
        }
        return metadata_;
    }

    NeuriploInferenceResult infer(const std::vector<InputTensor> &inputs) override {
        last_inputs = inputs;
        if (!infer_error.empty()) {
            NeuriploInferenceResult result;
            result.ok = false;
            result.error_message = infer_error;
            return result;
        }

        NeuriploInferenceResult result;
        OutputTensor scores;
        scores.name = "scores";
        scores.datatype = "FP32";
        scores.shape = {1, 2};
        if (inputs.size() == 1) {
            scores.data = {inputs.at(0).data.at(0), inputs.at(0).data.at(2)};
        } else {
            scores.data = {inputs.at(0).data.at(0), inputs.at(1).data.at(0)};
        }
        result.outputs.push_back(std::move(scores));

        OutputTensor labels;
        labels.name = "labels";
        labels.datatype = "FP32";
        labels.shape = {1, 1};
        labels.data = {9.0};
        result.outputs.push_back(std::move(labels));
        return result;
    }

    std::vector<InputTensor> last_inputs;
    std::string infer_error;
    std::string load_error_;

  private:
    ModelMetadata metadata_;
};

ExecutionRequest request(std::vector<std::string> outputs = {}) {
    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"input", "FP32", {1, 3}, {1.0, 2.0, 3.0}});
    execution_request.requested_outputs = std::move(outputs);
    return execution_request;
}

} // namespace

TEST_CASE(neuriplo_executor_loads_metadata_from_adapter) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error, std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config())));
    REQUIRE(executor != nullptr);
    REQUIRE_EQ(executor->metadata().name, "demo");
    REQUIRE_EQ(executor->metadata().versions[0], "7");
    REQUIRE_EQ(executor->metadata().platform, "neuriplo_onnx_runtime");
    REQUIRE_EQ(executor->metadata().outputs.size(), static_cast<size_t>(2));
}

TEST_CASE(neuriplo_executor_reports_load_failure) {
    auto adapter = std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config()));
    adapter->load_error_ = "load failed";

    std::string error;
    const auto executor = makeNeuriploExecutor(config(), error, std::move(adapter));
    REQUIRE(executor == nullptr);
    REQUIRE(error.find("load failed") != std::string::npos);
}

TEST_CASE(neuriplo_executor_runs_inference_and_filters_outputs) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error, std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config())));
    REQUIRE(executor != nullptr);

    const auto response = executor->infer(request({"scores"}));
    REQUIRE(response.ok);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].name, "scores");
    REQUIRE_EQ(response.outputs[0].data[0], 1.0);
    REQUIRE_EQ(response.outputs[0].data[1], 3.0);
}

TEST_CASE(neuriplo_executor_orders_inputs_by_metadata) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error,
        std::make_unique<FakeNeuriploAdapter>(multiInputAdapterMetadata(config())));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"second", "FP32", {1}, {2.0}});
    execution_request.inputs.push_back({"first", "FP32", {1}, {1.0}});
    execution_request.requested_outputs = {"scores"};

    const auto response = executor->infer(execution_request);
    REQUIRE(response.ok);
    REQUIRE_EQ(response.outputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(response.outputs[0].data[0], 1.0);
    REQUIRE_EQ(response.outputs[0].data[1], 2.0);
}

TEST_CASE(neuriplo_executor_rejects_missing_required_input) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error,
        std::make_unique<FakeNeuriploAdapter>(multiInputAdapterMetadata(config())));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"first", "FP32", {1}, {1.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(!response.ok);
    REQUIRE_EQ(response.error_code, "INVALID_ARGUMENT");
    REQUIRE(response.error_message.find("missing required") != std::string::npos);
}

TEST_CASE(neuriplo_executor_rejects_duplicate_input) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error, std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config())));
    REQUIRE(executor != nullptr);

    auto execution_request = request();
    execution_request.inputs.push_back({"input", "FP32", {1, 3}, {4.0, 5.0, 6.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(!response.ok);
    REQUIRE_EQ(response.error_code, "INVALID_ARGUMENT");
    REQUIRE(response.error_message.find("duplicate") != std::string::npos);
}

TEST_CASE(neuriplo_executor_rejects_unsupported_datatype) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error, std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config())));
    REQUIRE(executor != nullptr);

    auto execution_request = request();
    execution_request.inputs[0].datatype = "INT64";

    const auto response = executor->infer(execution_request);
    REQUIRE(!response.ok);
    REQUIRE_EQ(response.error_code, "INVALID_ARGUMENT");
}

TEST_CASE(neuriplo_executor_rejects_data_length_mismatch) {
    std::string error;
    const auto executor = makeNeuriploExecutor(
        config(), error, std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config())));
    REQUIRE(executor != nullptr);

    auto execution_request = request();
    execution_request.inputs[0].data = {1.0, 2.0};

    const auto response = executor->infer(execution_request);
    REQUIRE(!response.ok);
    REQUIRE_EQ(response.error_code, "INVALID_ARGUMENT");
    REQUIRE(response.error_message.find("data length") != std::string::npos);
}

TEST_CASE(neuriplo_executor_supports_integer_datatypes) {
    RuntimeConfig cfg = config();
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"input", "INT32", {1, 3}});
    metadata.outputs.push_back({"scores", "INT32", {1, 2}});

    auto adapter = std::make_unique<FakeNeuriploAdapter>(std::move(metadata));
    adapter->infer_error = "";
    auto real_adapter = adapter.get();

    std::string error;
    const auto executor = makeNeuriploExecutor(cfg, error, std::move(adapter));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"input", "INT32", {1, 3}, {1.0, 2.0, 3.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(response.ok);
    REQUIRE_EQ(real_adapter->last_inputs.size(), static_cast<size_t>(1));
    REQUIRE_EQ(real_adapter->last_inputs[0].datatype, "INT32");
    REQUIRE_EQ(real_adapter->last_inputs[0].data.size(), static_cast<size_t>(3));
}

TEST_CASE(neuriplo_executor_supports_fp16_datatype) {
    RuntimeConfig cfg = config();
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"input", "FP16", {1, 3}});
    metadata.outputs.push_back({"scores", "FP16", {1, 2}});

    auto adapter = std::make_unique<FakeNeuriploAdapter>(std::move(metadata));

    std::string error;
    const auto executor = makeNeuriploExecutor(cfg, error, std::move(adapter));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"input", "FP16", {1, 3}, {1.0, 2.0, 3.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(response.ok);
}

TEST_CASE(neuriplo_executor_supports_fp64_datatype) {
    RuntimeConfig cfg = config();
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"input", "FP64", {1, 3}});
    metadata.outputs.push_back({"scores", "FP64", {1, 2}});

    auto adapter = std::make_unique<FakeNeuriploAdapter>(std::move(metadata));

    std::string error;
    const auto executor = makeNeuriploExecutor(cfg, error, std::move(adapter));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"input", "FP64", {1, 3}, {1.0, 2.0, 3.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(response.ok);
}

TEST_CASE(neuriplo_executor_supports_uint8_datatype) {
    RuntimeConfig cfg = config();
    ModelMetadata metadata;
    metadata.name = cfg.model_name;
    metadata.versions = {"1"};
    metadata.platform = "neuriplo_" + cfg.backend;
    metadata.inputs.push_back({"input", "UINT8", {1, 3}});
    metadata.outputs.push_back({"scores", "UINT8", {1, 2}});

    auto adapter = std::make_unique<FakeNeuriploAdapter>(std::move(metadata));

    std::string error;
    const auto executor = makeNeuriploExecutor(cfg, error, std::move(adapter));
    REQUIRE(executor != nullptr);

    ExecutionRequest execution_request;
    execution_request.inputs.push_back({"input", "UINT8", {1, 3}, {1.0, 2.0, 3.0}});

    const auto response = executor->infer(execution_request);
    REQUIRE(response.ok);
}

TEST_CASE(neuriplo_executor_maps_adapter_inference_failure) {
    auto adapter = std::make_unique<FakeNeuriploAdapter>(adapterMetadata(config()));
    adapter->infer_error = "backend failed";

    std::string error;
    const auto executor = makeNeuriploExecutor(config(), error, std::move(adapter));
    REQUIRE(executor != nullptr);

    const auto response = executor->infer(request());
    REQUIRE(!response.ok);
    REQUIRE_EQ(response.error_code, "BACKEND_ERROR");
    REQUIRE(response.error_message.find("backend failed") != std::string::npos);
}
