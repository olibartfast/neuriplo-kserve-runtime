#include "OpenAiCodec.hpp"

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

} // namespace

OpenAiParseResult parseCompletionRequest(const std::string &body,
                                         OpenAiCompletionRequest &request) {
    Json root;
    try {
        root = Json::parse(body);
    } catch (const Json::parse_error &) {
        return {false, "invalid JSON request body", "INVALID_ARGUMENT"};
    }

    if (!root.is_object()) {
        return {false, "request body must be a JSON object", "INVALID_ARGUMENT"};
    }

    if (!root.contains("model") || !root["model"].is_string()) {
        return {false, "model name must be a string", "INVALID_ARGUMENT"};
    }
    request.model = root["model"].get<std::string>();

    if (!root.contains("prompt")) {
        return {false, "prompt is required", "INVALID_ARGUMENT"};
    }
    if (root["prompt"].is_string()) {
        request.prompt = root["prompt"].get<std::string>();
    } else if (root["prompt"].is_array()) {
        for (const auto &elem : root["prompt"]) {
            if (!elem.is_string()) {
                return {false, "prompt array elements must be strings", "INVALID_ARGUMENT"};
            }
            request.prompt += elem.get<std::string>();
        }
    } else {
        return {false, "prompt must be a string or array", "INVALID_ARGUMENT"};
    }

    if (root.contains("max_tokens") && root["max_tokens"].is_number_unsigned()) {
        request.max_tokens = root["max_tokens"].get<size_t>();
    }
    if (root.contains("temperature") && root["temperature"].is_number()) {
        request.temperature = root["temperature"].get<double>();
    }
    if (root.contains("top_p") && root["top_p"].is_number()) {
        request.top_p = root["top_p"].get<double>();
    }
    if (root.contains("stream") && root["stream"].is_boolean()) {
        request.stream = root["stream"].get<bool>();
    }

    return {true, {}, {}};
}

OpenAiParseResult parseChatRequest(const std::string &body, OpenAiChatRequest &request) {
    Json root;
    try {
        root = Json::parse(body);
    } catch (const Json::parse_error &) {
        return {false, "invalid JSON request body", "INVALID_ARGUMENT"};
    }

    if (!root.is_object()) {
        return {false, "request body must be a JSON object", "INVALID_ARGUMENT"};
    }

    if (!root.contains("model") || !root["model"].is_string()) {
        return {false, "model name must be a string", "INVALID_ARGUMENT"};
    }
    request.model = root["model"].get<std::string>();

    if (!root.contains("messages") || !root["messages"].is_array()) {
        return {false, "messages must be an array", "INVALID_ARGUMENT"};
    }
    if (root["messages"].empty()) {
        return {false, "messages must not be empty", "INVALID_ARGUMENT"};
    }
    for (const auto &msg : root["messages"]) {
        if (!msg.is_object() || !msg.contains("role") || !msg.contains("content")) {
            return {false, "each message must have role and content", "INVALID_ARGUMENT"};
        }
        if (!msg["role"].is_string() || !msg["content"].is_string()) {
            return {false, "role and content must be strings", "INVALID_ARGUMENT"};
        }
        ChatMessage message;
        message.role = msg["role"].get<std::string>();
        message.content = msg["content"].get<std::string>();
        request.messages.push_back(std::move(message));
    }

    if (root.contains("max_tokens") && root["max_tokens"].is_number_unsigned()) {
        request.max_tokens = root["max_tokens"].get<size_t>();
    }
    if (root.contains("temperature") && root["temperature"].is_number()) {
        request.temperature = root["temperature"].get<double>();
    }
    if (root.contains("top_p") && root["top_p"].is_number()) {
        request.top_p = root["top_p"].get<double>();
    }
    if (root.contains("stream") && root["stream"].is_boolean()) {
        request.stream = root["stream"].get<bool>();
    }

    return {true, {}, {}};
}

OpenAiParseResult parseEmbeddingRequest(const std::string &body, OpenAiEmbeddingRequest &request) {
    Json root;
    try {
        root = Json::parse(body);
    } catch (const Json::parse_error &) {
        return {false, "invalid JSON request body", "INVALID_ARGUMENT"};
    }

    if (!root.is_object()) {
        return {false, "request body must be a JSON object", "INVALID_ARGUMENT"};
    }

    if (!root.contains("model") || !root["model"].is_string()) {
        return {false, "model name must be a string", "INVALID_ARGUMENT"};
    }
    request.model = root["model"].get<std::string>();

    if (!root.contains("input")) {
        return {false, "input is required", "INVALID_ARGUMENT"};
    }
    if (root["input"].is_string()) {
        request.input = root["input"].get<std::string>();
    } else {
        return {false, "input must be a string", "INVALID_ARGUMENT"};
    }

    return {true, {}, {}};
}

std::string completionResponseJson(const OpenAiCompletionResponse &response) {
    Json body;
    body["id"] = response.id;
    body["object"] = response.object;
    body["created"] = response.created;
    body["model"] = response.model;

    Json choices = Json::array();
    Json choice;
    choice["index"] = 0;
    choice["text"] = response.text;
    choice["finish_reason"] = response.finish_reason;
    choices.push_back(choice);
    body["choices"] = choices;

    body["usage"] = Json{{"prompt_tokens", response.prompt_tokens},
                         {"completion_tokens", response.completion_tokens},
                         {"total_tokens", response.prompt_tokens + response.completion_tokens}};

    return body.dump();
}

std::string chatResponseJson(const OpenAiChatResponse &response) {
    Json body;
    body["id"] = response.id;
    body["object"] = response.object;
    body["created"] = response.created;
    body["model"] = response.model;

    Json choices = Json::array();
    Json choice;
    choice["index"] = 0;
    Json message;
    message["role"] = response.message.role;
    message["content"] = response.message.content;
    choice["message"] = message;
    choice["finish_reason"] = response.finish_reason;
    choices.push_back(choice);
    body["choices"] = choices;

    body["usage"] = Json{{"prompt_tokens", response.prompt_tokens},
                         {"completion_tokens", response.completion_tokens},
                         {"total_tokens", response.prompt_tokens + response.completion_tokens}};

    return body.dump();
}

std::string embeddingResponseJson(const OpenAiEmbeddingResponse &response) {
    Json body;
    body["object"] = "list";
    body["model"] = response.model;

    Json data = Json::array();
    Json embedding_obj;
    embedding_obj["object"] = "embedding";
    embedding_obj["index"] = 0;
    embedding_obj["embedding"] = response.embedding;
    data.push_back(embedding_obj);
    body["data"] = data;

    body["usage"] =
        Json{{"prompt_tokens", response.prompt_tokens}, {"total_tokens", response.prompt_tokens}};

    return body.dump();
}

std::string streamingChunkJson(const std::string &id, const std::string &object,
                               const std::string &model, int64_t created,
                               const std::string &delta_content, const std::string &finish_reason) {
    Json body;
    body["id"] = id;
    body["object"] = object;
    body["created"] = created;
    body["model"] = model;

    Json choices = Json::array();
    Json choice;
    choice["index"] = 0;
    Json delta;
    delta["content"] = delta_content;
    choice["delta"] = delta;
    if (!finish_reason.empty()) {
        choice["finish_reason"] = finish_reason;
    } else {
        choice["finish_reason"] = nullptr;
    }
    choices.push_back(choice);
    body["choices"] = choices;

    return "data: " + body.dump() + "\n\n";
}

std::string streamingDoneJson() {
    return "data: [DONE]\n\n";
}