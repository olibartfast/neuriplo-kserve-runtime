#include "PipelineSteps.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

std::vector<size_t> pipelineTopScoreIndices(const std::vector<float> &scores, size_t cap) {
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), size_t{0});
    // `left > right` alone is not a strict weak ordering once a NaN score can
    // appear (every comparison involving NaN is false), which is undefined
    // behaviour for std::stable_sort's comparator. NaN is ranked after every
    // real score, and treated as equivalent to any other NaN, so the
    // ordering stays total and NaN simply loses any tiebreak for the cap.
    std::stable_sort(indices.begin(), indices.end(), [&scores](size_t left, size_t right) {
        const bool left_nan = std::isnan(scores[left]);
        const bool right_nan = std::isnan(scores[right]);
        if (left_nan || right_nan) {
            return !left_nan && right_nan;
        }
        return scores[left] > scores[right];
    });
    if (indices.size() > cap) {
        indices.resize(cap);
    }
    return indices;
}

// Envelope shapes are transcribed from the platform ensemble contract. The
// offset arrays are 101 entries
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
