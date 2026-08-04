#include "PipelineSteps.hpp"

// Built without NEURIPLO_RUNTIME_WITH_TASKS. Model-step chaining still works;
// only the task-backed steps are unavailable, and they say so at load time
// rather than degrading silently at inference time.

std::unique_ptr<PipelineStep> makeBuiltinPipelineStep(const PipelineStepConfig &step,
                                                      const ModelMetadata & /*neighbour*/,
                                                      std::string &error) {
    error = "pipeline step '" + step.name +
            "' needs task-layer preprocessing/postprocessing, which this binary was built "
            "without; rebuild with -DNEURIPLO_RUNTIME_ENABLE_TASKS=ON";
    return nullptr;
}
