#pragma once

// Built-in pipeline steps: the preprocess and postprocess work that needs task
// knowledge (image decode, letterbox, NMS, mask/polygon decode).
//
// These are the one place the serving runtime depends on the task layer, which
// inverts the usual layering. The dependency is optional: built without
// NEURIPLO_RUNTIME_WITH_TASKS, makeBuiltinPipelineStep fails with an explicit
// message and the runtime is exactly what it is without ensembles.

#include "Executor.hpp"
#include "ModelMetadata.hpp"
#include "PipelineConfig.hpp"

#ifdef NEURIPLO_RUNTIME_WITH_TASKS
#include "neuriplo/tasks/core/result_types.hpp"
#endif

#include <memory>
#include <string>
#include <vector>

class PipelineStep {
  public:
    virtual ~PipelineStep() = default;

    // Tensors this step consumes and produces, in its own naming.
    virtual const std::vector<TensorMetadata> &inputs() const = 0;
    virtual const std::vector<TensorMetadata> &outputs() const = 0;

    // Runs the step. `inputs` arrives in the order of inputs(); `outputs` is
    // filled in the order of outputs(). Returns false and sets `error` on
    // malformed input or a task-layer failure.
    virtual bool run(const std::vector<OutputTensor> &step_inputs,
                     std::vector<OutputTensor> &step_outputs, std::string &error) = 0;
};

// Builds a preprocess or postprocess step.
//
// `neighbour` is the metadata of the model step this one is attached to: the
// following model for a preprocess step (whose input tensor it must produce),
// the preceding model for a postprocess step (whose outputs it decodes).
//
// Returns nullptr with `error` set when the step cannot be built, including
// when the runtime was built without task support.
std::unique_ptr<PipelineStep> makeBuiltinPipelineStep(const PipelineStepConfig &step,
                                                      const ModelMetadata &neighbour,
                                                      std::string &error);

// Envelope tensor metadata for a postprocess step, fixed by the ensemble
// contract. Available regardless of task support so metadata composition and
// its tests do not need the task layer.
std::vector<TensorMetadata> pipelineEnvelopeOutputs(PipelineEnvelope envelope);

#ifdef NEURIPLO_RUNTIME_WITH_TASKS
// Extracts one detection's mask as box-sized UINT8 bytes, normalizing the three
// layouts task postprocessors produce (mask_data, a box-sized mask image, or a
// frame-sized one). Exposed so the normalization can be tested directly: the
// end-to-end pipeline test returns no detections and never reaches it.
std::vector<uint8_t> pipelineBoxSizedMask(const neuriplo_tasks::InstanceSegmentation &segmentation);
#endif
