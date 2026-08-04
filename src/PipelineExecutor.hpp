#pragma once

// Executor for pipeline (ensemble) models: runs an ordered graph of steps and
// presents the whole thing to KServe as one model.

#include "Executor.hpp"
#include "InferSnapshot.hpp"
#include "PipelineConfig.hpp"
#include "RuntimeConfig.hpp"

#include <functional>
#include <memory>
#include <string>

// Resolves a `model` step to a registry entry at inference time. Resolving late
// rather than holding a reference means a referenced model can be reloaded
// underneath a loaded pipeline; a missing or not-ready model surfaces as
// MODEL_NOT_READY instead of a dangling handle.
using PipelineStepResolver = std::function<std::shared_ptr<const InferSnapshot>(
    const std::string &model_name, const std::string &model_version)>;

// Builds a pipeline executor. The graph comes from config.model_path (a file)
// or config.pipeline_graph (inline JSON). Returns nullptr with `error` set when
// the graph is malformed, references a model that is not loaded, or needs
// task-layer steps in a binary built without them.
std::unique_ptr<Executor> makePipelineExecutor(const RuntimeConfig &config,
                                               PipelineStepResolver resolver, std::string &error);

// Backend id a pipeline model is loaded under.
inline const char *pipelineBackendId() {
    return "ensemble";
}
