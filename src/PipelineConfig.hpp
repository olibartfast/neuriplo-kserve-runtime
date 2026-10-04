#pragma once

// Graph description for pipeline (ensemble) models: an ordered list of steps
// that runs as one model. See the platform ensemble contract for the wire
// surface and ADR 0011 for why this lives in the runtime rather than in Triton.
//
// Tensor naming follows Triton's ensemble convention so operators can carry
// mental models across: map keys are the step's own tensor names, map values
// are graph tensor names.
//
//   input_map:  {step input name  -> graph tensor name}
//   output_map: {step output name -> graph tensor name}

#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum class PipelineStepKind {
    Model,       // execute another model in this runtime's registry
    Preprocess,  // decode + preprocess through the task layer
    Postprocess, // decode results through the task layer into an envelope
};

// Which decoded-result envelope a postprocess step emits. Shapes are fixed by
// the ensemble contract.
enum class PipelineEnvelope {
    Detection, // NUM_DETECTIONS, BOXES, SCORES, CLASSES
    Mask,      // Detection plus MASK_OFFSETS, MASK_DATA
    Polygon,   // Detection plus INSTANCE_RING_OFFSETS, RING_POINT_OFFSETS, POLYGON_POINTS
};

struct PipelineStepConfig {
    PipelineStepKind kind = PipelineStepKind::Model;
    std::string name;

    // kind == Model
    std::string model_name;
    std::string model_version; // empty = the model's default version

    std::map<std::string, std::string> input_map;
    std::map<std::string, std::string> output_map;

    // kind == Preprocess / Postprocess
    std::string task_type; // neuriplo-tasks model-type string, e.g. "yolo26"
    PipelineEnvelope envelope = PipelineEnvelope::Detection;
    float confidence_threshold = 0.25F;
    float nms_threshold = 0.45F;
    float mask_threshold = 0.50F;
};

struct PipelineConfig {
    std::vector<PipelineStepConfig> steps;
};

// The single encoded-image input every image ensemble exposes, fixed by the
// ensemble contract.
inline const char *pipelineImageInputName() {
    return "IMAGE";
}
inline const char *pipelineImageInputDatatype() {
    return "UINT8";
}

// Maximum detections carried by a decoded envelope. Baked into the [100, ...]
// and [101] envelope shapes, so it is contract, not tuning.
inline constexpr int64_t kPipelineMaxDetections = 100;

// Parses a graph from JSON text. Returns false and sets `error` on malformed
// JSON, an unknown step kind, a duplicate step name, a model step without a
// model name, or a step consuming a tensor no earlier step produces.
bool parsePipelineConfig(const std::string &json_text, PipelineConfig &config, std::string &error);

// True when the graph starts by decoding an encoded image, which is what makes
// the composed model expose the contract's IMAGE input.
bool pipelineConsumesEncodedImage(const PipelineConfig &config);

// Tensor names the graph produces that no later step consumes; these are the
// composed model's outputs.
std::vector<std::string> pipelineTerminalTensors(const PipelineConfig &config);
