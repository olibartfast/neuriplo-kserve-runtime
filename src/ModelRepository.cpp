#include "ModelRepository.hpp"

#include "PipelineExecutor.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace {

bool isNumericVersion(const std::string &name) {
    return !name.empty() && std::all_of(name.begin(), name.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; });
}

// Strips leading zeros so "01" and "1" report the same, routable version
// label ("1"), never "01". All-zero input ("0", "000") canonicalizes to "0".
std::string canonicalVersion(const std::string &raw) {
    const auto first = raw.find_first_not_of('0');
    return first == std::string::npos ? std::string("0") : raw.substr(first);
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
// side. Earliest wins, and NEURIPLO_REPOSITORY_BACKEND_PRIORITY overrides it,
// because no fixed order can be right for every deployment -- an OpenVINO host
// and a CUDA host want opposite answers from the same directory. A config.pbtxt
// does not participate: it is an I/O name overlay only (see applyConfigPbtxt in
// RealNeuriploAdapter.cpp), not a Triton model config.
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

// Non-throwing directory listing: a race that removes `dir` mid-scan (or a
// `dir` that never existed) reports through `ec`/an empty result rather than
// escaping as a filesystem_error and aborting the whole scan over one
// directory. Both directory_iterator's construction *and* its increment can
// throw when used the ordinary way (range-for uses the throwing ++), so this
// drives the iterator manually with the error_code-returning increment.
std::vector<std::string> sortedEntries(const fs::path &dir, bool directories) {
    std::vector<std::string> names;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    const fs::directory_iterator end;
    while (!ec && it != end) {
        std::error_code kind_ec;
        const bool is_dir = it->is_directory(kind_ec);
        if (!kind_ec && is_dir == directories) {
            names.push_back(it->path().filename().string());
        }
        it.increment(ec);
    }
    std::sort(names.begin(), names.end());
    return names;
}

// Compares two all-digit version names without converting them to an integer.
// A directory name can hold more digits than any integer type, and std::stoull
// would throw std::out_of_range on it -- escaping the scan and aborting startup,
// which is precisely the "warn and skip" contract this file promises. Comparing
// the digits directly also means an oversized version still sorts correctly
// instead of being discarded.
bool isHigherVersion(const std::string &candidate, const std::string &current) {
    const auto lhs = canonicalVersion(candidate);
    const auto rhs = canonicalVersion(current);
    if (lhs.size() != rhs.size()) {
        return lhs.size() > rhs.size();
    }
    // Equal digit counts, so lexicographic order is numeric order.
    return lhs > rhs;
}

// Picks the recognized model file in one version directory, by backend
// priority when more than one is present. nullopt when the directory holds
// nothing servable.
std::optional<std::pair<std::string, std::string>> pickModelFile(const fs::path &version_dir) {
    std::string model_file;
    std::string backend;
    for (const auto &file_name : sortedEntries(version_dir, false)) {
        const auto candidate = backendForModelFile(file_name);
        if (candidate.empty()) {
            continue;
        }
        if (model_file.empty() || backendRank(candidate) < backendRank(backend)) {
            model_file = (version_dir / file_name).string();
            backend = candidate;
        }
    }
    if (model_file.empty()) {
        return std::nullopt;
    }
    return std::make_pair(std::move(model_file), std::move(backend));
}

// Groups a model directory's numeric version subdirectories by their
// canonical (leading-zero-stripped) version, highest canonical version
// first. A canonical group with more than one raw directory name (e.g. "01"
// and "1" together) is a duplicate; its raw names are sorted so the pick is
// deterministic across runs rather than depending on filesystem order.
std::vector<std::pair<std::string, std::vector<std::string>>>
groupVersionsDescending(const fs::path &model_dir, const std::string &model_name,
                        std::vector<std::string> &warnings) {
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto &name : sortedEntries(model_dir, true)) {
        if (!isNumericVersion(name)) {
            warnings.push_back("model " + model_name + ": skipping non-numeric version directory " +
                               (model_dir / name).string());
            continue;
        }
        groups[canonicalVersion(name)].push_back(name);
    }

    std::vector<std::pair<std::string, std::vector<std::string>>> ordered;
    ordered.reserve(groups.size());
    for (auto &entry : groups) {
        std::sort(entry.second.begin(), entry.second.end());
        if (entry.second.size() > 1) {
            std::string raw_list;
            for (size_t i = 0; i < entry.second.size(); ++i) {
                if (i > 0) {
                    raw_list += ", ";
                }
                raw_list += entry.second[i];
            }
            warnings.push_back("model " + model_name +
                               ": duplicate version directories for version " + entry.first + " (" +
                               raw_list + "); using " + entry.second.front());
        }
        ordered.emplace_back(entry.first, std::move(entry.second));
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &a, const auto &b) { return isHigherVersion(a.first, b.first); });
    return ordered;
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

        // Highest version *that holds a recognized model file* is served: an
        // empty highest-numbered directory (e.g. staged but not yet
        // populated) must not drop the whole model when a lower version is
        // perfectly servable. Every version skipped along the way -- for
        // having no file, or for being a duplicate of a higher pick -- is
        // warned about individually.
        const auto groups = groupVersionsDescending(model_dir, model_name, warnings);

        std::string model_file;
        std::string backend;
        std::string chosen_version;
        for (const auto &[canonical, raw_names] : groups) {
            const auto version_dir = model_dir / raw_names.front();
            auto found = pickModelFile(version_dir);
            if (!found) {
                warnings.push_back("no recognized model file under " + version_dir.string() +
                                   "; skipping version " + canonical);
                continue;
            }
            model_file = std::move(found->first);
            backend = std::move(found->second);
            chosen_version = canonical;
            break;
        }

        if (model_file.empty()) {
            // Every version group already warned individually above (one
            // version directory, one warning); only add a summary warning
            // when there was no version directory at all to warn about in
            // the first place (e.g. only non-numeric ones, each already
            // warned about by groupVersionsDescending).
            if (groups.empty()) {
                warnings.push_back("no recognized model file under any version of " +
                                   model_dir.string() + "; skipping");
            }
            continue;
        }

        RuntimeConfig config = defaults;
        config.model_name = model_name;
        // The version directory is the authoritative version for a tree,
        // canonicalized so a client is never handed a "01" to route by.
        config.model_version = chosen_version;
        config.model_version_explicit = true;
        config.model_path = model_file;
        config.backend = backend;
        // An ensemble's graph lives in the JSON file itself; leaving
        // pipeline_graph empty tells the executor to read it from model_path.
        config.pipeline_graph.clear();
        if (config.backend == pipelineBackendId()) {
            // max_batch_size 1 / batching off is contractual for ensembles
            // (PipelineExecutor rejects anything else), not a tuning
            // default. A repository-wide --dynamic-batching-enabled/
            // --max-batch-size default is meant for the tensor models it was
            // set for, not for an ensemble discovered alongside them; an
            // ensemble never inherits it, since discovery never carries an
            // operator's per-model intent the way an explicit admin load
            // body does.
            config.dynamic_batching_enabled = false;
            config.max_batch_size = 1;
        }
        configs.push_back(std::move(config));
    }

    if (configs.empty()) {
        warnings.push_back("model repository contains no servable models: " + root);
    }
    return configs;
}

std::optional<ResolvedRepositoryVersion>
resolveRepositoryModelVersion(const std::string &root, const std::string &model_name,
                              const std::string &version) {
    // `version` is untrusted request input. isNumericVersion rejects
    // anything containing '.', '/', or any other non-digit character -- "..",
    // "../x/1", "1/../../y" included -- before it is ever used to build a
    // path. It is only ever compared (after canonicalizing) against
    // directory names this function discovers itself below; it is never
    // joined onto a filesystem path.
    if (!isNumericVersion(version)) {
        return std::nullopt;
    }
    // Defends the model_name segment the same way: "." or ".." here would
    // resolve to the repository root or its parent, and a path separator
    // would let it name a nested path component instead of one directory
    // under root -- routing already splits on '/' before model_name ever
    // reaches here, but this must not depend on that.
    if (model_name.empty() || model_name == "." || model_name == ".." ||
        model_name.find('/') != std::string::npos || model_name.find('\\') != std::string::npos) {
        return std::nullopt;
    }

    const auto model_dir = fs::path(root) / model_name;
    std::error_code ec;
    if (!fs::is_directory(model_dir, ec)) {
        return std::nullopt;
    }

    // Reuses the exact same grouping/pick that scanModelRepository uses, so
    // a duplicate such as "01"/"1" resolves to the same directory here that
    // the initial scan served it from.
    std::vector<std::string> discarded_warnings;
    const auto groups = groupVersionsDescending(model_dir, model_name, discarded_warnings);
    const auto target = canonicalVersion(version);
    for (const auto &[canonical, raw_names] : groups) {
        if (canonical != target) {
            continue;
        }
        auto found = pickModelFile(model_dir / raw_names.front());
        if (!found) {
            return std::nullopt;
        }
        return ResolvedRepositoryVersion{std::move(found->first), std::move(found->second),
                                         canonical};
    }
    return std::nullopt;
}
