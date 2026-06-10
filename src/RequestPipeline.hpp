#pragma once

#include "HttpTypes.hpp"
#include "InferSnapshot.hpp"
#include "KServeV2Codec.hpp"
#include "Scheduler.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

struct InferContext {
    const HttpRequest *request = nullptr;
    std::string model_name;
    std::string model_version;
    std::shared_ptr<const InferSnapshot> handle;
    InferenceParseResult parsed;
    SchedulerResult scheduled;
};

using PipelineStage = std::function<std::optional<HttpResponse>(InferContext &ctx)>;

class RequestPipeline {
  public:
    RequestPipeline() = default;

    RequestPipeline &addStage(PipelineStage stage) {
        stages_.push_back(std::move(stage));
        return *this;
    }

    HttpResponse run(InferContext &ctx) const {
        for (const auto &stage : stages_) {
            if (!stage) {
                continue;
            }
            if (auto error = stage(ctx)) {
                return *error;
            }
        }
        return {};
    }

  private:
    std::vector<PipelineStage> stages_;
};
