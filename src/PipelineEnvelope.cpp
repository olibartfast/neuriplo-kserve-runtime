#include "PipelineSteps.hpp"

// Envelope shapes are transcribed from the platform ensemble contract, which in
// turn was transcribed from tritonic v0.4.0. The offset arrays are 101 entries
// (max detections + 1) and are always emitted at full length, including on a
// frame with no detections -- a truncated offset array there aborts any video
// whose first frame is empty.

std::vector<TensorMetadata> pipelineEnvelopeOutputs(PipelineEnvelope envelope) {
    std::vector<TensorMetadata> outputs = {
        {"NUM_DETECTIONS", "INT32", {1}},
        {"BOXES", "INT32", {kPipelineMaxDetections, 4}},
        {"SCORES", "FP32", {kPipelineMaxDetections}},
        {"CLASSES", "INT32", {kPipelineMaxDetections}},
    };

    switch (envelope) {
    case PipelineEnvelope::Detection:
        break;
    case PipelineEnvelope::Mask:
        outputs.push_back({"MASK_OFFSETS", "INT64", {kPipelineMaxDetections + 1}});
        outputs.push_back({"MASK_DATA", "UINT8", {-1}});
        break;
    case PipelineEnvelope::Polygon:
        outputs.push_back({"INSTANCE_RING_OFFSETS", "INT64", {kPipelineMaxDetections + 1}});
        outputs.push_back({"RING_POINT_OFFSETS", "INT64", {-1}});
        outputs.push_back({"POLYGON_POINTS", "INT32", {-1, 2}});
        break;
    }

    return outputs;
}
