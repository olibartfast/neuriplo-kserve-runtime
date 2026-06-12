#pragma once

#include "HttpTypes.hpp"
#include "InferSnapshot.hpp"
#include "KServeV2Codec.hpp"
#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"
#include "OpenAiCodec.hpp"
#include "Scheduler.hpp"

#include <optional>
#include <string>

class KServeRuntime {
  public:
    explicit KServeRuntime(ModelRegistry &registry, MetricsRegistry &metrics);

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
    HttpResponse completionsStreaming(const HttpRequest &request,
                                      const OpenAiCompletionRequest &comp_req,
                                      const InferSnapshot &handle) const;
    HttpResponse chatCompletions(const HttpRequest &request) const;
    HttpResponse chatCompletionsStreaming(const HttpRequest &request,
                                          const OpenAiChatRequest &chat_req,
                                          const InferSnapshot &handle) const;
    HttpResponse embeddings(const HttpRequest &request) const;
    HttpResponse handleAdmin(const HttpRequest &request) const;

    std::optional<HttpResponse>
    validateInferModel(const std::string &model_name, const std::string &model_version,
                       std::shared_ptr<const InferSnapshot> &out_handle) const;
    std::optional<HttpResponse> decodeAndAdmit(const HttpRequest &request,
                                               const std::string &model_name,
                                               const std::string &model_version,
                                               const InferSnapshot &handle,
                                               InferenceParseResult &out_parsed) const;
    SchedulerResult scheduleInfer(const std::string &model_name, const std::string &model_version,
                                  const InferenceParseResult &parsed,
                                  const InferSnapshot &handle) const;
    HttpResponse encodeInferResponse(const std::string &model_name,
                                     const std::string &model_version, const HttpRequest &request,
                                     const InferenceParseResult &parsed,
                                     const SchedulerResult &scheduled) const;

    ModelRegistry &registry_;
    MetricsRegistry &metrics_;
};