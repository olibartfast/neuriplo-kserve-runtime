#pragma once

#include "HttpTypes.hpp"
#include "KServeV2Codec.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "Scheduler.hpp"

#include <optional>
#include <string>

class KServeRuntime {
  public:
    explicit KServeRuntime(ModelRegistry registry, MetricsRegistry &metrics);

    HttpResponse handle(const HttpRequest &request) const;

  private:
    HttpResponse serverMetadata() const;
    HttpResponse live() const;
    HttpResponse ready() const;
    HttpResponse metricsPage() const;
    HttpResponse modelMetadata(const std::string &model_name,
                               const std::string &model_version = "") const;
    HttpResponse modelReady(const std::string &model_name,
                            const std::string &model_version = "") const;
    HttpResponse infer(const std::string &model_name, const std::string &model_version,
                       const HttpRequest &request) const;
    HttpResponse completions(const HttpRequest &request) const;

    std::optional<HttpResponse> validateInferModel(const std::string &model_name,
                                                   const std::string &model_version,
                                                   const ModelHandle *&out_handle) const;
    std::optional<HttpResponse> decodeAndAdmit(const HttpRequest &request,
                                               const std::string &model_name,
                                               const std::string &model_version,
                                               const ModelHandle &handle,
                                               InferenceParseResult &out_parsed) const;
    SchedulerResult scheduleInfer(const std::string &model_name, const std::string &model_version,
                                  const InferenceParseResult &parsed,
                                  const ModelHandle &handle) const;
    HttpResponse encodeInferResponse(const std::string &model_name,
                                     const std::string &model_version, const HttpRequest &request,
                                     const InferenceParseResult &parsed,
                                     const SchedulerResult &scheduled) const;

    ModelRegistry registry_;
    MetricsRegistry &metrics_;
};
