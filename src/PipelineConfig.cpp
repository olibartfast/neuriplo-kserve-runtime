#include "PipelineConfig.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>
#include <unordered_set>

namespace {

using nlohmann::json;

bool parseKind(const std::string &text, PipelineStepKind &kind, std::string &error) {
    if (text == "model") {
        kind = PipelineStepKind::Model;
        return true;
    }
    if (text == "preprocess") {
        kind = PipelineStepKind::Preprocess;
        return true;
    }
    if (text == "postprocess") {
        kind = PipelineStepKind::Postprocess;
        return true;
    }
    error = "unknown pipeline step kind: " + text;
    return false;
}

bool parseEnvelope(const std::string &text, PipelineEnvelope &envelope, std::string &error) {
    if (text == "detection") {
        envelope = PipelineEnvelope::Detection;
        return true;
    }
    if (text == "mask") {
        envelope = PipelineEnvelope::Mask;
        return true;
    }
    if (text == "polygon") {
        envelope = PipelineEnvelope::Polygon;
        return true;
    }
    error = "unknown pipeline envelope: " + text;
    return false;
}

bool readTensorMap(const json &node, const char *field, std::map<std::string, std::string> &target,
                   const std::string &step_name, std::string &error) {
    if (!node.contains(field)) {
        return true;
    }
    if (!node[field].is_object()) {
        error = "pipeline step '" + step_name + "': " + field + " must be an object";
        return false;
    }
    for (const auto &entry : node[field].items()) {
        if (!entry.value().is_string()) {
            error = "pipeline step '" + step_name + "': " + field + " values must be strings";
            return false;
        }
        target.emplace(entry.key(), entry.value().get<std::string>());
    }
    return true;
}

void readFloat(const json &node, const char *field, float &target) {
    if (node.contains(field) && node[field].is_number()) {
        target = node[field].get<float>();
    }
}

} // namespace

bool parsePipelineConfig(const std::string &json_text, PipelineConfig &config, std::string &error) {
    json root = json::parse(json_text, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        error = "pipeline graph must be a JSON object";
        return false;
    }
    if (!root.contains("steps") || !root["steps"].is_array() || root["steps"].empty()) {
        error = "pipeline graph must declare a non-empty 'steps' array";
        return false;
    }

    PipelineConfig parsed;
    std::unordered_set<std::string> step_names;
    // Tensors available to a step: the ensemble input, plus everything earlier
    // steps mapped into the graph. Checking this at parse time turns a mis-wired
    // graph into a load failure instead of a first-request failure.
    //
    // A model step with no output_map publishes its outputs under whatever names
    // its metadata declares, which is not knowable from the graph alone. Once
    // one appears, the name check goes quiet here; PipelineExecutor repeats it
    // exactly at load time, where model metadata is resolved.
    std::unordered_set<std::string> available{pipelineImageInputName()};
    bool names_knowable = true;

    for (const auto &node : root["steps"]) {
        if (!node.is_object()) {
            error = "pipeline steps must be objects";
            return false;
        }

        PipelineStepConfig step;
        step.name = node.value("name", std::string{});
        if (step.name.empty()) {
            error = "every pipeline step needs a name";
            return false;
        }
        if (!step_names.insert(step.name).second) {
            error = "duplicate pipeline step name: " + step.name;
            return false;
        }

        const std::string kind_text = node.value("kind", std::string{});
        if (kind_text.empty()) {
            error = "pipeline step '" + step.name + "' has no kind";
            return false;
        }
        if (!parseKind(kind_text, step.kind, error)) {
            return false;
        }

        if (step.kind == PipelineStepKind::Model) {
            step.model_name = node.value("model_name", std::string{});
            if (step.model_name.empty()) {
                error = "pipeline step '" + step.name + "' is a model step without a model_name";
                return false;
            }
            step.model_version = node.value("model_version", std::string{});
        } else {
            step.task_type = node.value("task_type", std::string{});
            if (step.task_type.empty()) {
                error = "pipeline step '" + step.name + "' needs a task_type";
                return false;
            }
        }

        if (step.kind == PipelineStepKind::Postprocess && node.contains("envelope")) {
            if (!node["envelope"].is_string()) {
                error = "pipeline step '" + step.name + "': envelope must be a string";
                return false;
            }
            if (!parseEnvelope(node["envelope"].get<std::string>(), step.envelope, error)) {
                return false;
            }
        }

        readFloat(node, "confidence_threshold", step.confidence_threshold);
        readFloat(node, "nms_threshold", step.nms_threshold);
        readFloat(node, "mask_threshold", step.mask_threshold);

        if (!readTensorMap(node, "input_map", step.input_map, step.name, error) ||
            !readTensorMap(node, "output_map", step.output_map, step.name, error)) {
            return false;
        }

        if (names_knowable) {
            for (const auto &entry : step.input_map) {
                if (available.find(entry.second) == available.end()) {
                    error = "pipeline step '" + step.name + "' consumes tensor '" + entry.second +
                            "' that no earlier step produces";
                    return false;
                }
            }
        }
        for (const auto &entry : step.output_map) {
            available.insert(entry.second);
        }
        if (step.output_map.empty()) {
            names_knowable = false;
        }

        parsed.steps.push_back(std::move(step));
    }

    config = std::move(parsed);
    return true;
}

bool pipelineConsumesEncodedImage(const PipelineConfig &config) {
    return !config.steps.empty() && config.steps.front().kind == PipelineStepKind::Preprocess;
}

std::vector<std::string> pipelineTerminalTensors(const PipelineConfig &config) {
    std::set<std::string> consumed;
    for (const auto &step : config.steps) {
        for (const auto &entry : step.input_map) {
            consumed.insert(entry.second);
        }
    }

    std::vector<std::string> terminal;
    for (const auto &step : config.steps) {
        for (const auto &entry : step.output_map) {
            if (consumed.find(entry.second) == consumed.end() &&
                std::find(terminal.begin(), terminal.end(), entry.second) == terminal.end()) {
                terminal.push_back(entry.second);
            }
        }
    }
    return terminal;
}
