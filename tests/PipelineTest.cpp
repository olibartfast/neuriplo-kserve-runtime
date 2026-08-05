#include "ModelRegistry.hpp"
#include "NeuriploExecutor.hpp"
#include "PipelineConfig.hpp"
#include "PipelineExecutor.hpp"
#include "PipelineSteps.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace {

std::string twoStepGraph() {
    return R"({
        "steps": [
            {"kind": "model", "name": "detect", "model_name": "yolo"},
            {"kind": "model", "name": "refine", "model_name": "yolo",
             "input_map": {"input": "output"},
             "output_map": {"output": "refined"}}
        ]
    })";
}

RuntimeConfig pipelineConfig(const std::string &graph) {
    RuntimeConfig config;
    config.model_name = "yolo_ensemble";
    config.backend = pipelineBackendId();
    config.pipeline_graph = graph;
    return config;
}

// A model whose metadata and behaviour we control, standing in for a served
// model a pipeline step references.
class EchoExecutor final : public Executor {
  public:
    EchoExecutor() {
        metadata_.name = "yolo";
        metadata_.versions.push_back("1");
        metadata_.platform = "neuriplo_stub";
        metadata_.inputs.push_back({"input", "FP32", {1, 3}});
        metadata_.outputs.push_back({"output", "FP32", {1, 3}});
    }

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {1, 3};
        // Echo the input so a test can tell which tensor reached the step.
        if (!request.inputs.empty()) {
            output.bytes = request.inputs.front().bytes;
        }
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

ModelRegistry registryWithEchoModel() {
    RuntimeConfig config;
    config.model_name = "yolo";
    config.backend = "stub";
    return ModelRegistry(config, [](const RuntimeConfig &, std::string &) {
        return std::make_unique<EchoExecutor>();
    });
}

ExecutionRequest requestWith(const std::string &name, std::vector<double> values) {
    ExecutionRequest request;
    InputTensor input;
    input.name = name;
    input.datatype = "FP32";
    input.shape = {1, static_cast<int64_t>(values.size())};
    input.bytes = tensorBytesFromDoubles(input.datatype, values);
    request.inputs.push_back(std::move(input));
    return request;
}

} // namespace

TEST_CASE(pipeline_config_parses_ordered_steps) {
    PipelineConfig config;
    std::string error;
    REQUIRE(parsePipelineConfig(twoStepGraph(), config, error));
    REQUIRE_EQ(config.steps.size(), 2u);
    REQUIRE_EQ(config.steps[0].name, "detect");
    REQUIRE(config.steps[0].kind == PipelineStepKind::Model);
    REQUIRE_EQ(config.steps[1].input_map.at("input"), "output");
    REQUIRE_EQ(config.steps[1].output_map.at("output"), "refined");
}

TEST_CASE(pipeline_config_rejects_unknown_step_kind) {
    PipelineConfig config;
    std::string error;
    const std::string graph = R"({"steps": [{"kind": "sorcery", "name": "a"}]})";
    REQUIRE(!parsePipelineConfig(graph, config, error));
    REQUIRE(error.find("sorcery") != std::string::npos);
}

TEST_CASE(pipeline_config_rejects_duplicate_step_names) {
    PipelineConfig config;
    std::string error;
    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "a", "model_name": "m"},
        {"kind": "model", "name": "a", "model_name": "m"}
    ]})";
    REQUIRE(!parsePipelineConfig(graph, config, error));
    REQUIRE(error.find("duplicate") != std::string::npos);
}

TEST_CASE(pipeline_config_rejects_model_step_without_model_name) {
    PipelineConfig config;
    std::string error;
    const std::string graph = R"({"steps": [{"kind": "model", "name": "a"}]})";
    REQUIRE(!parsePipelineConfig(graph, config, error));
    REQUIRE(error.find("model_name") != std::string::npos);
}

// A step consuming a tensor nothing produces is a wiring mistake that must be
// caught when the graph loads, not on the first request.
TEST_CASE(pipeline_config_rejects_unproduced_input_tensor) {
    PipelineConfig config;
    std::string error;
    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "a", "model_name": "m",
         "input_map": {"input": "nowhere"}}
    ]})";
    REQUIRE(!parsePipelineConfig(graph, config, error));
    REQUIRE(error.find("nowhere") != std::string::npos);
}

TEST_CASE(pipeline_config_accepts_the_ensemble_image_input) {
    PipelineConfig config;
    std::string error;
    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "a", "model_name": "m",
         "input_map": {"input": "IMAGE"}}
    ]})";
    REQUIRE(parsePipelineConfig(graph, config, error));
}

TEST_CASE(pipeline_terminal_tensors_exclude_consumed_ones) {
    PipelineConfig config;
    std::string error;
    REQUIRE(parsePipelineConfig(twoStepGraph(), config, error));
    const auto terminal = pipelineTerminalTensors(config);
    REQUIRE_EQ(terminal.size(), 1u);
    REQUIRE_EQ(terminal.front(), "refined");
}

TEST_CASE(pipeline_executor_composes_metadata_and_reports_ensemble_platform) {
    auto registry = registryWithEchoModel();
    REQUIRE(registry.loadModel(pipelineConfig(twoStepGraph())));

    const auto metadata = registry.find("yolo_ensemble");
    REQUIRE(metadata.has_value());
    REQUIRE_EQ(metadata->platform, "ensemble");
    // First step's inputs, last step's outputs, in graph naming.
    REQUIRE_EQ(metadata->inputs.size(), 1u);
    REQUIRE_EQ(metadata->inputs.front().name, "input");
    REQUIRE_EQ(metadata->outputs.size(), 1u);
    REQUIRE_EQ(metadata->outputs.front().name, "refined");
}

TEST_CASE(pipeline_executor_routes_tensors_between_steps) {
    auto registry = registryWithEchoModel();
    REQUIRE(registry.loadModel(pipelineConfig(twoStepGraph())));

    const auto handle = registry.findHandle("yolo_ensemble");
    REQUIRE(handle != nullptr);
    REQUIRE(handle->isReady());

    auto scheduled = handle->scheduler->submit(requestWith("input", {1.0, 2.0, 3.0}));
    REQUIRE(scheduled.ok);
    REQUIRE(scheduled.response.ok);
    REQUIRE_EQ(scheduled.response.outputs.size(), 1u);

    const auto &output = scheduled.response.outputs.front();
    REQUIRE_EQ(output.name, "refined");
    // Both steps echo, so the value survives the whole graph: the second step
    // really received the first step's output.
    const auto values = tensorValuesAsDoubles(output.datatype, output.bytes);
    REQUIRE_EQ(values.size(), 3u);
    REQUIRE_EQ(values[1], 2.0);
}

TEST_CASE(pipeline_executor_reports_not_ready_for_a_missing_model) {
    auto registry = registryWithEchoModel();
    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "detect", "model_name": "absent"}
    ]})";
    // A pipeline over a model that is not loaded must fail to load, rather than
    // loading and failing on every request.
    REQUIRE(registry.loadModel(pipelineConfig(graph)));
    REQUIRE(!registry.ready("yolo_ensemble"));
}

TEST_CASE(pipeline_executor_refuses_dynamic_batching) {
    auto registry = registryWithEchoModel();
    auto config = pipelineConfig(twoStepGraph());
    config.dynamic_batching_enabled = true;
    config.max_batch_size = 4;

    REQUIRE(registry.loadModel(config));
    // max_batch_size 1 is contractual for ensembles; the load must fail loudly.
    REQUIRE(!registry.ready("yolo_ensemble"));
}

TEST_CASE(pipeline_envelope_shapes_match_the_ensemble_contract) {
    const auto detection = pipelineEnvelopeOutputs(PipelineEnvelope::Detection);
    REQUIRE_EQ(detection.size(), 4u);
    REQUIRE_EQ(detection[0].name, "NUM_DETECTIONS");
    REQUIRE_EQ(detection[0].datatype, "INT32");
    REQUIRE_EQ(detection[0].shape, (std::vector<int64_t>{1}));
    REQUIRE_EQ(detection[1].name, "BOXES");
    REQUIRE_EQ(detection[1].shape, (std::vector<int64_t>{100, 4}));
    REQUIRE_EQ(detection[2].datatype, "FP32");
    REQUIRE_EQ(detection[3].datatype, "INT32");

    const auto mask = pipelineEnvelopeOutputs(PipelineEnvelope::Mask);
    REQUIRE_EQ(mask.size(), 6u);
    REQUIRE_EQ(mask[4].name, "MASK_OFFSETS");
    REQUIRE_EQ(mask[4].datatype, "INT64");
    // 101 entries, one more than the detection cap, always.
    REQUIRE_EQ(mask[4].shape, (std::vector<int64_t>{101}));
    REQUIRE_EQ(mask[5].name, "MASK_DATA");
    REQUIRE_EQ(mask[5].datatype, "UINT8");

    const auto polygon = pipelineEnvelopeOutputs(PipelineEnvelope::Polygon);
    REQUIRE_EQ(polygon.size(), 7u);
    REQUIRE_EQ(polygon[4].name, "INSTANCE_RING_OFFSETS");
    REQUIRE_EQ(polygon[4].shape, (std::vector<int64_t>{101}));
    REQUIRE_EQ(polygon[5].name, "RING_POINT_OFFSETS");
    REQUIRE_EQ(polygon[6].name, "POLYGON_POINTS");
    REQUIRE_EQ(polygon[6].datatype, "INT32");
    REQUIRE_EQ(polygon[6].shape, (std::vector<int64_t>{-1, 2}));
}

#ifdef NEURIPLO_RUNTIME_WITH_TASKS

#include "neuriplo/tasks/core/vision/stb_io.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

// A YOLO-shaped model that returns all zeros, so every candidate falls below
// the confidence threshold and postprocessing yields no detections.
class SilentYoloExecutor final : public Executor {
  public:
    SilentYoloExecutor() {
        metadata_.name = "yolo";
        metadata_.versions.push_back("1");
        metadata_.platform = "neuriplo_stub";
        metadata_.inputs.push_back({"images", "FP32", {1, 3, 640, 640}});
        metadata_.outputs.push_back({"output0", "FP32", {1, 84, 8400}});
    }

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &) override {
        ExecutionResponse response;
        OutputTensor output;
        output.name = "output0";
        output.datatype = "FP32";
        output.shape = {1, 84, 8400};
        output.bytes.assign(static_cast<size_t>(84 * 8400) * sizeof(float), std::byte{0});
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

std::vector<std::byte> encodedTestImage() {
    neuriplo_tasks::vision::Image image(8, 6, 3, neuriplo_tasks::vision::PixelType::UInt8);
    auto *pixels = image.data<uint8_t>();
    for (size_t i = 0; i < image.sizeBytes(); ++i) {
        pixels[i] = static_cast<uint8_t>(i % 255);
    }
    const std::string path = "pipeline_test_input.png";
    neuriplo_tasks::vision::saveImage(path, image);

    std::ifstream stream(path, std::ios::binary);
    const std::vector<char> raw(std::istreambuf_iterator<char>(stream), {});
    stream.close();
    std::remove(path.c_str());

    std::vector<std::byte> bytes(raw.size());
    std::memcpy(bytes.data(), raw.data(), raw.size());
    return bytes;
}

const OutputTensor *findOutput(const std::vector<OutputTensor> &outputs, const std::string &name) {
    for (const auto &output : outputs) {
        if (output.name == name) {
            return &output;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE(pipeline_runs_preprocess_model_postprocess_end_to_end) {
    RuntimeConfig model_config;
    model_config.model_name = "yolo";
    model_config.backend = "stub";
    ModelRegistry registry(model_config, [](const RuntimeConfig &, std::string &) {
        return std::make_unique<SilentYoloExecutor>();
    });

    const std::string graph = R"({
        "steps": [
            {"kind": "preprocess", "name": "pre", "task_type": "yolo26"},
            {"kind": "model", "name": "detect", "model_name": "yolo"},
            {"kind": "postprocess", "name": "post", "task_type": "yolo26", "envelope": "mask"}
        ]
    })";
    REQUIRE(registry.loadModel(pipelineConfig(graph)));
    REQUIRE(registry.ready("yolo_ensemble"));

    const auto metadata = registry.find("yolo_ensemble");
    REQUIRE(metadata.has_value());
    REQUIRE_EQ(metadata->platform, "ensemble");
    REQUIRE_EQ(metadata->inputs.size(), 1u);
    REQUIRE_EQ(metadata->inputs.front().name, "IMAGE");
    REQUIRE_EQ(metadata->inputs.front().datatype, "UINT8");
    REQUIRE_EQ(metadata->outputs.size(), 6u);

    ExecutionRequest request;
    InputTensor image;
    image.name = "IMAGE";
    image.datatype = "UINT8";
    image.bytes = encodedTestImage();
    image.shape = {1, static_cast<int64_t>(image.bytes.size())};
    request.inputs.push_back(std::move(image));

    const auto handle = registry.findHandle("yolo_ensemble");
    REQUIRE(handle != nullptr);
    auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(scheduled.ok);
    REQUIRE(scheduled.response.ok);

    const auto *count = findOutput(scheduled.response.outputs, "NUM_DETECTIONS");
    REQUIRE(count != nullptr);
    REQUIRE_EQ(tensorScalarAt<int32_t>(count->bytes, 0), 0);

    // The empty-frame contract: a detection-free response still carries full
    // length arrays. A truncated MASK_OFFSETS here aborts any video whose first
    // frame is empty.
    const auto *offsets = findOutput(scheduled.response.outputs, "MASK_OFFSETS");
    REQUIRE(offsets != nullptr);
    REQUIRE_EQ(offsets->bytes.size(), 101u * sizeof(int64_t));
    REQUIRE_EQ(tensorScalarAt<int64_t>(offsets->bytes, 0), 0);
    REQUIRE_EQ(tensorScalarAt<int64_t>(offsets->bytes, 100), 0);

    const auto *boxes = findOutput(scheduled.response.outputs, "BOXES");
    REQUIRE(boxes != nullptr);
    REQUIRE_EQ(boxes->bytes.size(), 100u * 4u * sizeof(int32_t));

    const auto *scores = findOutput(scheduled.response.outputs, "SCORES");
    REQUIRE(scores != nullptr);
    REQUIRE_EQ(scores->bytes.size(), 100u * sizeof(float));
}

#endif

#ifdef NEURIPLO_RUNTIME_WITH_TASKS

// The end-to-end pipeline test above returns no detections, so it never
// exercises mask encoding. This drives the encoder directly with a detection
// that carries a mask, pinning the regression where the encoder read
// InstanceSegmentation::mask_data -- which the YOLO postprocessor leaves empty,
// populating the mask image instead -- and shipped an empty MASK_DATA.
// The end-to-end pipeline test above returns no detections, so it never reaches
// mask encoding. This drives the normalization directly, pinning the regression
// where the encoder read InstanceSegmentation::mask_data -- which the YOLO
// postprocessor leaves empty, populating the mask image instead -- and so
// shipped an envelope with no mask bytes at all.
TEST_CASE(pipeline_box_sized_mask_reads_a_frame_sized_mask_image) {
    neuriplo_tasks::InstanceSegmentation seg;
    seg.bbox = neuriplo_tasks::vision::Rect(4, 2, 2, 2);
    seg.mask_height = 6;
    seg.mask_width = 8;

    // Frame-sized mask with only the detection's box set; this is the layout one
    // of the two YOLO segmentation code paths produces.
    auto mask =
        neuriplo_tasks::vision::Image::zeros(8, 6, 1, neuriplo_tasks::vision::PixelType::UInt8);
    auto *pixels = mask.data<uint8_t>();
    for (int row = 2; row < 4; ++row) {
        for (int col = 4; col < 6; ++col) {
            pixels[row * 8 + col] = 255;
        }
    }
    seg.mask = neuriplo_tasks::fromImage(std::move(mask));
    REQUIRE(seg.mask_data.empty());

    // Cropped to the box: 2x2 bytes, every one set.
    const auto bytes = pipelineBoxSizedMask(seg);
    REQUIRE_EQ(bytes.size(), 4u);
    for (const auto value : bytes) {
        REQUIRE(value != 0);
    }
}

TEST_CASE(pipeline_box_sized_mask_prefers_existing_mask_data) {
    neuriplo_tasks::InstanceSegmentation seg;
    seg.bbox = neuriplo_tasks::vision::Rect(0, 0, 2, 2);
    seg.mask_data = {1, 2, 3, 4};

    const auto bytes = pipelineBoxSizedMask(seg);
    REQUIRE_EQ(bytes.size(), 4u);
    REQUIRE_EQ(bytes[3], 4);
}

TEST_CASE(pipeline_box_sized_mask_is_empty_without_any_mask) {
    neuriplo_tasks::InstanceSegmentation seg;
    seg.bbox = neuriplo_tasks::vision::Rect(0, 0, 2, 2);
    REQUIRE(pipelineBoxSizedMask(seg).empty());
}

#endif

#ifndef NEURIPLO_RUNTIME_WITH_TASKS
// Without the task layer the runtime must say so at load time instead of
// degrading silently.
TEST_CASE(pipeline_task_steps_fail_clearly_without_task_support) {
    ModelMetadata neighbour;
    neighbour.inputs.push_back({"input", "FP32", {1, 3, 640, 640}});
    neighbour.outputs.push_back({"output", "FP32", {1, 84, 8400}});

    PipelineStepConfig step;
    step.kind = PipelineStepKind::Preprocess;
    step.name = "pre";
    step.task_type = "yolo26";

    std::string error;
    REQUIRE(makeBuiltinPipelineStep(step, neighbour, error) == nullptr);
    REQUIRE(error.find("NEURIPLO_RUNTIME_ENABLE_TASKS") != std::string::npos);
}
#endif

namespace {

// Rejects any input whose declared shape differs from its metadata, standing in
// for backends (TensorRT via the neuriplo executor) that validate strictly.
class ShapeCheckingExecutor final : public Executor {
  public:
    ShapeCheckingExecutor() {
        metadata_.name = "strict";
        metadata_.versions.push_back("1");
        metadata_.platform = "neuriplo_stub";
        // Per-sample convention: no batch dimension, the way TensorRT metadata
        // reports it.
        metadata_.inputs.push_back({"input", "FP32", {3}});
        metadata_.outputs.push_back({"output", "FP32", {3}});
    }

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        ExecutionResponse response;
        if (request.inputs.size() != 1 || request.inputs[0].shape != metadata_.inputs[0].shape) {
            response.ok = false;
            response.error_code = "INVALID_ARGUMENT";
            response.error_message = "strict executor rejected input shape";
            return response;
        }
        OutputTensor output;
        output.name = "output";
        output.datatype = "FP32";
        output.shape = {3};
        output.bytes = request.inputs[0].bytes;
        response.outputs.push_back(std::move(output));
        return response;
    }

  private:
    ModelMetadata metadata_;
};

} // namespace

// Backends disagree about whether metadata shapes carry the batch dimension.
// A producer emitting [1,3] feeding a model declaring [3] must succeed: the
// pipeline seam restates the tensor in the target model's convention. Without
// that, DALI ([1,3,640,640]) feeding TensorRT ([3,640,640]) failed validation.
TEST_CASE(pipeline_reconciles_batch_dimension_conventions_between_steps) {
    RuntimeConfig config;
    config.model_name = "strict";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &, std::string &) {
        return std::make_unique<ShapeCheckingExecutor>();
    });

    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "only", "model_name": "strict"}
    ]})";
    auto pipeline_config = pipelineConfig(graph);
    REQUIRE(registry.loadModel(pipeline_config));
    REQUIRE(registry.ready("yolo_ensemble"));

    // Batch-inclusive shape [1,3]; same 3 elements.
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {1, 3};
    input.bytes = tensorBytesFromDoubles("FP32", {7.0, 8.0, 9.0});
    request.inputs.push_back(std::move(input));

    const auto handle = registry.findHandle("yolo_ensemble");
    REQUIRE(handle != nullptr);
    auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(scheduled.ok);
    REQUIRE(scheduled.response.ok);
}

// When an executor fails, the scheduler must carry the executor's error code
// and message on the SchedulerResult itself: transports read those fields on
// failure, and leaving them empty redacted every real error into a generic
// "internal error", which is what made the first ensemble failure on this
// stack undiagnosable from the outside.
TEST_CASE(scheduler_result_carries_executor_error_details) {
    RuntimeConfig config;
    config.model_name = "strict";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &, std::string &) {
        return std::make_unique<ShapeCheckingExecutor>();
    });

    // Wrong element count entirely: the strict executor rejects it.
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {2};
    input.bytes = tensorBytesFromDoubles("FP32", {1.0, 2.0});
    request.inputs.push_back(std::move(input));

    const auto handle = registry.findHandle("strict");
    REQUIRE(handle != nullptr);
    auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(!scheduled.ok);
    REQUIRE_EQ(scheduled.error_code, "INVALID_ARGUMENT");
    REQUIRE(scheduled.error_message.find("strict executor") != std::string::npos);
}

// Equal element counts are not enough to justify a reshape: NHWC and NCHW share
// a count and differ entirely in layout, so relabelling one as the other would
// feed the model transposed data. Only leading unit dimensions may be adjusted.
TEST_CASE(pipeline_refuses_to_reshape_across_incompatible_layouts) {
    RuntimeConfig config;
    config.model_name = "strict";
    config.backend = "stub";
    ModelRegistry registry(config, [](const RuntimeConfig &, std::string &) {
        return std::make_unique<ShapeCheckingExecutor>();
    });

    const std::string graph = R"({"steps": [
        {"kind": "model", "name": "only", "model_name": "strict"}
    ]})";
    REQUIRE(registry.loadModel(pipelineConfig(graph)));

    // ShapeCheckingExecutor declares [3]. A [3,1] input has the same element
    // count but is not a leading-unit-dimension difference, so it must NOT be
    // silently restated as [3]; the executor rejects it.
    ExecutionRequest request;
    InputTensor input;
    input.name = "input";
    input.datatype = "FP32";
    input.shape = {3, 1};
    input.bytes = tensorBytesFromDoubles("FP32", {1.0, 2.0, 3.0});
    request.inputs.push_back(std::move(input));

    const auto handle = registry.findHandle("yolo_ensemble");
    REQUIRE(handle != nullptr);
    auto scheduled = handle->scheduler->submit(std::move(request));
    REQUIRE(!scheduled.ok);
    REQUIRE_EQ(scheduled.error_code, "INVALID_ARGUMENT");
}

// The dynamic-axis fix in NeuriploExecutor: a model declaring [1,-1] must
// accept any concrete extent, and still reject a rank mismatch.
TEST_CASE(pipeline_dynamic_axis_accepts_any_extent_and_rejects_rank_mismatch) {
    ModelMetadata metadata;
    metadata.inputs.push_back({"IMAGE", "UINT8", {1, -1}});

    // Concrete extents of any size satisfy the dynamic axis.
    for (const int64_t extent : {1, 37, 4096}) {
        ExecutionRequest request;
        InputTensor input;
        input.name = "IMAGE";
        input.datatype = "UINT8";
        input.shape = {1, extent};
        input.bytes.assign(static_cast<size_t>(extent), std::byte{0});
        request.inputs.push_back(std::move(input));

        ExecutionResponse error;
        REQUIRE(neuriploOrderedInputs(metadata, request, error).has_value());
    }

    // A different rank is still a rejection, dynamic axis or not.
    ExecutionRequest wrong_rank;
    InputTensor input;
    input.name = "IMAGE";
    input.datatype = "UINT8";
    input.shape = {64};
    input.bytes.assign(64, std::byte{0});
    wrong_rank.inputs.push_back(std::move(input));

    ExecutionResponse error;
    REQUIRE(!neuriploOrderedInputs(metadata, wrong_rank, error).has_value());
    REQUIRE_EQ(error.error_code, "INVALID_ARGUMENT");
}
