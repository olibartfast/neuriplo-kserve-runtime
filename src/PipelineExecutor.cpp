#include "PipelineExecutor.hpp"

#include "KServeErrors.hpp"
#include "PipelineSteps.hpp"
#include "Scheduler.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace {

// Absent maps mean identity, so a graph whose tensor names already line up
// needs no mapping at all.
std::string mapped(const std::map<std::string, std::string> &map, const std::string &name) {
    const auto found = map.find(name);
    return found == map.end() ? name : found->second;
}

bool readGraphText(const RuntimeConfig &config, std::string &text, std::string &error) {
    if (!config.pipeline_graph.empty()) {
        text = config.pipeline_graph;
        return true;
    }
    if (config.model_path.empty()) {
        error = "pipeline model needs a graph: set model_path to a graph file or supply "
                "pipeline_graph inline";
        return false;
    }
    std::ifstream stream(config.model_path);
    if (!stream.is_open()) {
        error = "could not open pipeline graph: " + config.model_path;
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    text = buffer.str();
    return true;
}

class PipelineExecutor : public Executor {
  public:
    struct Step {
        PipelineStepConfig config;
        std::unique_ptr<PipelineStep> builtin; // null for model steps
        std::vector<TensorMetadata> inputs;
        std::vector<TensorMetadata> outputs;
    };

    PipelineExecutor(std::string model_name, std::vector<Step> steps, PipelineStepResolver resolver)
        : steps_(std::move(steps)), resolver_(std::move(resolver)) {
        metadata_.name = std::move(model_name);
        metadata_.versions.push_back("1");
        metadata_.platform = "ensemble";

        // The composed model exposes the first step's inputs and the last
        // step's outputs, both in graph naming.
        for (const auto &input : steps_.front().inputs) {
            auto entry = input;
            entry.name = mapped(steps_.front().config.input_map, input.name);
            metadata_.inputs.push_back(std::move(entry));
        }
        for (const auto &output : steps_.back().outputs) {
            auto entry = output;
            entry.name = mapped(steps_.back().config.output_map, output.name);
            metadata_.outputs.push_back(std::move(entry));
        }

        indexLastUse();
    }

    void indexLastUse() {
        for (size_t index = 0; index < steps_.size(); ++index) {
            for (const auto &declared : steps_[index].inputs) {
                last_use_[mapped(steps_[index].config.input_map, declared.name)] = index;
            }
        }
        for (const auto &declared : metadata_.outputs) {
            last_use_.erase(declared.name);
        }
    }

    const ModelMetadata &metadata() const override {
        return metadata_;
    }

    ExecutionResponse infer(const ExecutionRequest &request) override {
        std::unordered_map<std::string, OutputTensor> values;
        for (const auto &input : request.inputs) {
            OutputTensor tensor;
            tensor.name = input.name;
            tensor.datatype = input.datatype;
            tensor.shape = input.shape;
            tensor.bytes = input.bytes;
            tensor.string_data = input.string_data;
            values.emplace(input.name, std::move(tensor));
        }

        size_t step_index = 0;
        for (const auto &step : steps_) {
            std::vector<OutputTensor> step_inputs;
            step_inputs.reserve(step.inputs.size());
            for (const auto &declared : step.inputs) {
                const auto graph_name = mapped(step.config.input_map, declared.name);
                const auto found = values.find(graph_name);
                if (found == values.end()) {
                    return failure(KServeErrors::InvalidArgument,
                                   "pipeline step '" + step.config.name +
                                       "' is missing input tensor '" + graph_name + "'");
                }
                // Move rather than copy when no later step and no declared
                // output needs this tensor again. Model tensors are megabytes
                // (a 640x640 FP32 image is 4.9 MB), so a deep copy per step is
                // a measurable share of pipeline latency.
                const auto last = last_use_.find(graph_name);
                if (last != last_use_.end() && last->second == step_index) {
                    auto tensor = std::move(found->second);
                    tensor.name = declared.name;
                    values.erase(found);
                    step_inputs.push_back(std::move(tensor));
                } else {
                    auto tensor = found->second;
                    tensor.name = declared.name;
                    step_inputs.push_back(std::move(tensor));
                }
            }

            std::vector<OutputTensor> step_outputs;
            if (step.builtin) {
                std::string error;
                if (!step.builtin->run(step_inputs, step_outputs, error)) {
                    return failure(KServeErrors::InvalidArgument, error);
                }
            } else {
                auto response = runModelStep(step, std::move(step_inputs));
                if (!response.ok) {
                    return response;
                }
                step_outputs = std::move(response.outputs);
            }

            for (auto &produced : step_outputs) {
                const auto graph_name = mapped(step.config.output_map, produced.name);
                produced.name = graph_name;
                values[graph_name] = std::move(produced);
            }
            ++step_index;
        }

        ExecutionResponse response;
        for (const auto &declared : metadata_.outputs) {
            if (!request.requested_outputs.empty() &&
                std::find(request.requested_outputs.begin(), request.requested_outputs.end(),
                          declared.name) == request.requested_outputs.end()) {
                continue;
            }
            const auto found = values.find(declared.name);
            if (found == values.end()) {
                return failure(KServeErrors::Internal,
                               "pipeline produced no tensor named '" + declared.name + "'");
            }
            response.outputs.push_back(found->second);
        }
        return response;
    }

  private:
    static ExecutionResponse failure(const std::string &code, const std::string &message) {
        ExecutionResponse response;
        response.ok = false;
        response.error_code = code;
        response.error_message = message;
        return response;
    }

    // Backends disagree about whether metadata shapes include the batch
    // dimension (ONNX Runtime reports [1,3,640,640], TensorRT [3,640,640]), so
    // a tensor produced by one step can carry a shape the next model's
    // validation rejects even though the bytes are exactly right.
    //
    // Only leading dimensions of extent 1 may be added or removed, and every
    // remaining dimension must match exactly. Equal element counts are NOT
    // sufficient: NHWC [640,640,3] and NCHW [3,640,640] have the same count and
    // completely different memory layouts, and silently relabelling one as the
    // other would feed the model transposed data.
    static bool sameAfterDroppingLeadingOnes(const std::vector<int64_t> &left,
                                             const std::vector<int64_t> &right) {
        auto trim = [](const std::vector<int64_t> &shape) {
            size_t begin = 0;
            while (begin + 1 < shape.size() && shape[begin] == 1) {
                ++begin;
            }
            return std::vector<int64_t>(shape.begin() + static_cast<std::ptrdiff_t>(begin),
                                        shape.end());
        };
        return trim(left) == trim(right);
    }

    static void adoptDeclaredShape(OutputTensor &tensor,
                                   const std::vector<TensorMetadata> &declared_inputs) {
        for (const auto &declared : declared_inputs) {
            if (declared.name != tensor.name) {
                continue;
            }
            if (declared.shape == tensor.shape) {
                return;
            }
            for (const auto dim : declared.shape) {
                if (dim < 0) {
                    return; // dynamic axis: validation already accepts any extent
                }
            }
            if (sameAfterDroppingLeadingOnes(declared.shape, tensor.shape)) {
                tensor.shape = declared.shape;
            }
            return;
        }
    }

    ExecutionResponse runModelStep(const Step &step, std::vector<OutputTensor> step_inputs) const {
        for (auto &tensor : step_inputs) {
            adoptDeclaredShape(tensor, step.inputs);
        }
        // Resolved per request: the referenced model may have been reloaded or
        // unloaded since this pipeline was loaded.
        auto snapshot = resolver_(step.config.model_name, step.config.model_version);
        if (!snapshot || !snapshot->isReady()) {
            return failure(KServeErrors::ModelNotReady,
                           "pipeline step '" + step.config.name + "' references model '" +
                               step.config.model_name + "' which is not ready");
        }

        ExecutionRequest inner;
        inner.inputs.reserve(step_inputs.size());
        for (auto &tensor : step_inputs) {
            InputTensor input;
            input.name = std::move(tensor.name);
            input.datatype = std::move(tensor.datatype);
            input.shape = std::move(tensor.shape);
            input.bytes = std::move(tensor.bytes);
            input.string_data = std::move(tensor.string_data);
            inner.inputs.push_back(std::move(input));
        }

        auto scheduled = snapshot->scheduler->submit(std::move(inner));
        if (!scheduled.ok) {
            return failure(scheduled.error_code.empty() ? KServeErrors::Internal
                                                        : scheduled.error_code,
                           "pipeline step '" + step.config.name + "': " + scheduled.error_message);
        }
        if (!scheduled.response.ok) {
            return scheduled.response;
        }
        return scheduled.response;
    }

    ModelMetadata metadata_;
    std::vector<Step> steps_;
    PipelineStepResolver resolver_;
    // Index of the last step consuming each graph tensor; tensors leaving as
    // model outputs are absent, since they must survive to the response.
    std::unordered_map<std::string, size_t> last_use_;
};

// Metadata of the model step nearest to `index` in `direction`, which is the
// step a built-in preprocess/postprocess step is attached to.
bool neighbourModelMetadata(const PipelineConfig &config, const PipelineStepResolver &resolver,
                            size_t index, int direction, ModelMetadata &metadata,
                            std::string &error) {
    for (int64_t cursor = static_cast<int64_t>(index) + direction;
         cursor >= 0 && cursor < static_cast<int64_t>(config.steps.size()); cursor += direction) {
        const auto &candidate = config.steps[static_cast<size_t>(cursor)];
        if (candidate.kind != PipelineStepKind::Model) {
            continue;
        }
        auto snapshot = resolver(candidate.model_name, candidate.model_version);
        if (!snapshot) {
            error = "pipeline references model '" + candidate.model_name +
                    "' which is not loaded; load it before the pipeline";
            return false;
        }
        metadata = snapshot->metadata;
        return true;
    }
    error = "pipeline step at position " + std::to_string(index) + " has no adjacent model step";
    return false;
}

} // namespace

std::unique_ptr<Executor> makePipelineExecutor(const RuntimeConfig &config,
                                               PipelineStepResolver resolver, std::string &error) {
    if (!resolver) {
        error = "pipeline models need a model resolver";
        return nullptr;
    }

    // max_batch_size 1 is contractual for ensembles, not a tuning default:
    // encoded images have no common shape, so batching them is meaningless.
    // Refusing the load is the only way an operator finds out.
    if (config.dynamic_batching_enabled || config.max_batch_size > 1) {
        error = "pipeline models cannot batch: max_batch_size is 1 by contract, but this model "
                "was configured with dynamic batching";
        return nullptr;
    }

    std::string graph_text;
    if (!readGraphText(config, graph_text, error)) {
        return nullptr;
    }

    PipelineConfig graph;
    if (!parsePipelineConfig(graph_text, graph, error)) {
        return nullptr;
    }

    std::vector<PipelineExecutor::Step> steps;
    steps.reserve(graph.steps.size());
    for (size_t index = 0; index < graph.steps.size(); ++index) {
        const auto &step_config = graph.steps[index];
        PipelineExecutor::Step step;
        step.config = step_config;

        if (step_config.kind == PipelineStepKind::Model) {
            // Resolved once here so the composed metadata is known at load
            // time and a pipeline over a missing model fails to load rather
            // than failing on first request.
            auto snapshot = resolver(step_config.model_name, step_config.model_version);
            if (!snapshot) {
                error = "pipeline step '" + step_config.name + "' references model '" +
                        step_config.model_name + "' which is not loaded";
                return nullptr;
            }
            step.inputs = snapshot->metadata.inputs;
            step.outputs = snapshot->metadata.outputs;
        } else {
            const int direction = step_config.kind == PipelineStepKind::Preprocess ? 1 : -1;
            ModelMetadata neighbour;
            if (!neighbourModelMetadata(graph, resolver, index, direction, neighbour, error)) {
                return nullptr;
            }
            step.builtin = makeBuiltinPipelineStep(step_config, neighbour, error);
            if (!step.builtin) {
                return nullptr;
            }
            step.inputs = step.builtin->inputs();
            step.outputs = step.builtin->outputs();
        }

        steps.push_back(std::move(step));
    }

    // Now that every step's real tensor names are known, repeat the
    // availability check exactly. Parse-time validation goes quiet as soon as a
    // model step relies on identity naming; this is where a graph wired to a
    // tensor nothing produces is actually caught.
    std::set<std::string> available;
    for (const auto &declared : steps.front().inputs) {
        available.insert(mapped(steps.front().config.input_map, declared.name));
    }
    for (const auto &step : steps) {
        for (const auto &declared : step.inputs) {
            const auto graph_name = mapped(step.config.input_map, declared.name);
            if (available.find(graph_name) == available.end()) {
                error = "pipeline step '" + step.config.name + "' consumes tensor '" + graph_name +
                        "' that no earlier step produces";
                return nullptr;
            }
        }
        for (const auto &produced : step.outputs) {
            available.insert(mapped(step.config.output_map, produced.name));
        }
    }

    return std::make_unique<PipelineExecutor>(config.model_name, std::move(steps),
                                              std::move(resolver));
}
