#pragma once

#include "RuntimeConfig.hpp"

#include <string>
#include <vector>

// Discovery for a Triton-style model repository tree:
//
//   <root>/<model-name>/[config.pbtxt]
//   <root>/<model-name>/<version>/<model file>
//
// Versions are numeric directories; the highest one is served. The backend is
// inferred from the model file's extension, so an engine produced by a TensorRT
// conversion step (.plan) is served by the tensorrt backend without the
// deployment restating it.
//
// This is the layout the deployment procedure builds: an init container stages
// the .onnx, a conversion step writes the engine into <model>/<version>/, and
// the runtime is pointed at the root rather than at one file.

struct DiscoveredModel {
    std::string name;
    std::string version;
    std::string path;
    std::string backend;
};

// Maps a model file extension to a backend id. Empty when unrecognized.
std::string backendForModelFile(const std::string &path);

// Scans `root` and returns one config per servable model, each inheriting
// `defaults` for the settings a repository tree does not express (use_gpu,
// instances, batching, timeouts). Directories that contain nothing servable are
// reported through `warnings` rather than failing the scan, so one malformed
// model cannot stop a server from bringing up the rest.
std::vector<RuntimeConfig> scanModelRepository(const std::string &root,
                                               const RuntimeConfig &defaults,
                                               std::vector<std::string> &warnings);
