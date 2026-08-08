#include "ModelRepository.hpp"

#include "PipelineExecutor.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <system_error>

namespace fs = std::filesystem;

namespace {

bool isNumericVersion(const std::string &name) {
    return !name.empty() && std::all_of(name.begin(), name.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Model files are recognized by extension. Anything else in a version directory
// (weight blobs like OpenVINO's .bin, label files, notes) is ignored.
const std::map<std::string, std::string> &extensionBackends() {
    static const std::map<std::string, std::string> map{
        {".plan", "tensorrt"},
        {".engine", "tensorrt"},
        {".onnx", "onnx_runtime"},
        {".torchscript", "libtorch"},
        {".pt", "libtorch"},
        {".tflite", "litert"},
        {".pb", "libtensorflow"},
        {".xml", "openvino"},
        {".pte", "executorch"},
        {".dali", "dali"},
        {".json", pipelineBackendId()},
    };
    return map;
}

// Order used when one version directory holds artifacts for several backends,
// which is normal for a repository that keeps every export of a model side by
// side. Earliest wins. A config.pbtxt `backend:` line overrides this, and so
// does NEURIPLO_REPOSITORY_BACKEND_PRIORITY, because no fixed order can be
// right for every deployment -- an OpenVINO host and a CUDA host want opposite
// answers from the same directory.
const char *kDefaultBackendPriority = "tensorrt,onnx_runtime,openvino,executorch,litert,"
                                      "libtorch,libtensorflow,dali,ensemble";

std::vector<std::string> backendPriority() {
    const char *override_value = std::getenv("NEURIPLO_REPOSITORY_BACKEND_PRIORITY");
    const std::string source = override_value != nullptr && *override_value != '\0'
                                   ? override_value
                                   : kDefaultBackendPriority;
    std::vector<std::string> order;
    std::string current;
    for (const char c : source) {
        if (c == ',' || c == ';' || c == ' ') {
            if (!current.empty()) {
                order.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        order.push_back(current);
    }
    return order;
}

// Lower is better; unlisted backends sort after every listed one.
size_t backendRank(const std::string &backend) {
    static const std::vector<std::string> order = backendPriority();
    const auto found = std::find(order.begin(), order.end(), backend);
    return found == order.end() ? order.size() : static_cast<size_t>(found - order.begin());
}

std::vector<std::string> sortedEntries(const fs::path &dir, bool directories) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        std::error_code kind_ec;
        const bool is_dir = entry.is_directory(kind_ec);
        if (is_dir == directories) {
            names.push_back(entry.path().filename().string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

// Highest numeric version directory, which is the one served.
std::string latestVersion(const fs::path &model_dir) {
    std::string best;
    for (const auto &name : sortedEntries(model_dir, true)) {
        if (!isNumericVersion(name)) {
            continue;
        }
        if (best.empty() || std::stoull(name) > std::stoull(best)) {
            best = name;
        }
    }
    return best;
}

} // namespace

std::string backendForModelFile(const std::string &path) {
    const auto extension = fs::path(path).extension().string();
    std::string lowered;
    lowered.reserve(extension.size());
    for (const unsigned char c : extension) {
        lowered.push_back(static_cast<char>(std::tolower(c)));
    }
    const auto &map = extensionBackends();
    const auto found = map.find(lowered);
    return found == map.end() ? std::string() : found->second;
}

std::vector<RuntimeConfig> scanModelRepository(const std::string &root,
                                               const RuntimeConfig &defaults,
                                               std::vector<std::string> &warnings) {
    std::vector<RuntimeConfig> configs;

    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        warnings.push_back("model repository is not a directory: " + root);
        return configs;
    }

    for (const auto &model_name : sortedEntries(root, true)) {
        const auto model_dir = fs::path(root) / model_name;

        const auto version = latestVersion(model_dir);
        if (version.empty()) {
            warnings.push_back("no numeric version directory under " + model_dir.string() +
                               "; skipping");
            continue;
        }

        const auto version_dir = model_dir / version;
        std::string model_file;
        std::string backend;
        for (const auto &file_name : sortedEntries(version_dir, false)) {
            const auto candidate = backendForModelFile(file_name);
            if (candidate.empty()) {
                continue;
            }
            // A directory may hold one export per backend. Pick by explicit
            // priority rather than directory order, so the choice is the same
            // on every host and can be steered without renaming files.
            if (model_file.empty() || backendRank(candidate) < backendRank(backend)) {
                model_file = (version_dir / file_name).string();
                backend = candidate;
            }
        }

        if (model_file.empty()) {
            warnings.push_back("no recognized model file under " + version_dir.string() +
                               "; skipping");
            continue;
        }

        RuntimeConfig config = defaults;
        config.model_name = model_name;
        // The version directory is the authoritative version for a tree.
        config.model_version = version;
        config.model_version_explicit = true;
        config.model_path = model_file;
        config.backend = backend;
        // An ensemble's graph lives in the JSON file itself; leaving
        // pipeline_graph empty tells the executor to read it from model_path.
        config.pipeline_graph.clear();
        configs.push_back(std::move(config));
    }

    if (configs.empty()) {
        warnings.push_back("model repository contains no servable models: " + root);
    }
    return configs;
}
