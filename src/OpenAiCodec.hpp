#pragma once

#include "Executor.hpp"

#include <optional>
#include <string>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string content;
};

struct OpenAiCompletionRequest {
    std::string model;
    std::string prompt;
    std::optional<size_t> max_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<size_t> top_k;
    bool stream = false;
};

struct OpenAiChatRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    std::optional<size_t> max_tokens;
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<size_t> top_k;
    bool stream = false;
};

struct OpenAiEmbeddingRequest {
    std::string model;
    std::string input;
};

struct OpenAiCompletionResponse {
    std::string id;
    std::string object = "text_completion";
    int64_t created = 0;
    std::string model;
    std::string text;
    std::string finish_reason = "stop";
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
};

struct OpenAiChatResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    ChatMessage message;
    std::string finish_reason = "stop";
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
};

struct OpenAiEmbeddingResponse {
    std::string object = "list";
    std::string model;
    std::vector<double> embedding;
    size_t prompt_tokens = 0;
};

struct OpenAiParseResult {
    bool ok = false;
    std::string error_message;
    std::string error_code;
};

OpenAiParseResult parseCompletionRequest(const std::string &body, OpenAiCompletionRequest &request);

OpenAiParseResult parseChatRequest(const std::string &body, OpenAiChatRequest &request);

OpenAiParseResult parseEmbeddingRequest(const std::string &body, OpenAiEmbeddingRequest &request);

std::string completionResponseJson(const OpenAiCompletionResponse &response);

std::string chatResponseJson(const OpenAiChatResponse &response);

std::string embeddingResponseJson(const OpenAiEmbeddingResponse &response);

std::string streamingChunkJson(const std::string &id, const std::string &object,
                               const std::string &model, int64_t created,
                               const std::string &delta_content, const std::string &finish_reason);

std::string streamingDoneJson();