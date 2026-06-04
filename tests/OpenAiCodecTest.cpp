#include "OpenAiCodec.hpp"
#include "Test.hpp"

TEST_CASE(parse_completion_request_valid) {
    OpenAiCompletionRequest request;
    const auto result = parseCompletionRequest(
        R"({"model":"llm-model","prompt":"Hello world","max_tokens":64,"temperature":0.8})",
        request);
    REQUIRE(result.ok);
    REQUIRE_EQ(request.model, "llm-model");
    REQUIRE_EQ(request.prompt, "Hello world");
    REQUIRE(request.max_tokens.has_value());
    REQUIRE_EQ(*request.max_tokens, static_cast<size_t>(64));
    REQUIRE(request.temperature.has_value());
}

TEST_CASE(parse_completion_request_missing_model) {
    OpenAiCompletionRequest request;
    const auto result = parseCompletionRequest(R"({"prompt":"Hello"})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_completion_request_missing_prompt) {
    OpenAiCompletionRequest request;
    const auto result = parseCompletionRequest(R"({"model":"test"})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_completion_request_invalid_json) {
    OpenAiCompletionRequest request;
    const auto result = parseCompletionRequest("{invalid", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_completion_request_prompt_array) {
    OpenAiCompletionRequest request;
    const auto result =
        parseCompletionRequest(R"({"model":"test","prompt":["Hello ","world"]})", request);
    REQUIRE(result.ok);
    REQUIRE_EQ(request.prompt, "Hello world");
}

TEST_CASE(parse_chat_request_valid) {
    OpenAiChatRequest request;
    const auto result = parseChatRequest(
        R"({"model":"llm-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":128})",
        request);
    REQUIRE(result.ok);
    REQUIRE_EQ(request.model, "llm-model");
    REQUIRE_EQ(request.messages.size(), static_cast<size_t>(1));
    REQUIRE_EQ(request.messages[0].role, "user");
    REQUIRE_EQ(request.messages[0].content, "Hello");
}

TEST_CASE(parse_chat_request_missing_messages) {
    OpenAiChatRequest request;
    const auto result = parseChatRequest(R"({"model":"test"})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_chat_request_empty_messages) {
    OpenAiChatRequest request;
    const auto result = parseChatRequest(R"({"model":"test","messages":[]})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_chat_request_invalid_message) {
    OpenAiChatRequest request;
    const auto result =
        parseChatRequest(R"({"model":"test","messages":[{"role":"user"}]})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(parse_embedding_request_valid) {
    OpenAiEmbeddingRequest request;
    const auto result = parseEmbeddingRequest(R"({"model":"test","input":"Hello world"})", request);
    REQUIRE(result.ok);
    REQUIRE_EQ(request.model, "test");
    REQUIRE_EQ(request.input, "Hello world");
}

TEST_CASE(parse_embedding_request_missing_input) {
    OpenAiEmbeddingRequest request;
    const auto result = parseEmbeddingRequest(R"({"model":"test"})", request);
    REQUIRE(!result.ok);
}

TEST_CASE(completion_response_json_format) {
    OpenAiCompletionResponse response;
    response.id = "cmpl-123";
    response.object = "text_completion";
    response.created = 1700000000;
    response.model = "test";
    response.text = "Hello response";
    response.finish_reason = "stop";
    response.prompt_tokens = 3;
    response.completion_tokens = 10;

    const auto json_str = completionResponseJson(response);
    REQUIRE(json_str.find("\"id\":\"cmpl-123\"") != std::string::npos);
    REQUIRE(json_str.find("\"object\":\"text_completion\"") != std::string::npos);
    REQUIRE(json_str.find("\"model\":\"test\"") != std::string::npos);
    REQUIRE(json_str.find("\"prompt_tokens\":3") != std::string::npos);
    REQUIRE(json_str.find("\"completion_tokens\":10") != std::string::npos);
}

TEST_CASE(chat_response_json_format) {
    OpenAiChatResponse response;
    response.id = "chatcmpl-123";
    response.object = "chat.completion";
    response.created = 1700000000;
    response.model = "test";
    response.message.role = "assistant";
    response.message.content = "Hello";
    response.finish_reason = "stop";
    response.prompt_tokens = 5;
    response.completion_tokens = 3;

    const auto json_str = chatResponseJson(response);
    REQUIRE(json_str.find("\"id\":\"chatcmpl-123\"") != std::string::npos);
    REQUIRE(json_str.find("\"object\":\"chat.completion\"") != std::string::npos);
    REQUIRE(json_str.find("\"role\":\"assistant\"") != std::string::npos);
    REQUIRE(json_str.find("\"content\":\"Hello\"") != std::string::npos);
}

TEST_CASE(streaming_chunk_json_format) {
    const auto chunk = streamingChunkJson("chatcmpl-123", "chat.completion.chunk", "test",
                                          1700000000, "Hello", "");
    REQUIRE(chunk.find("data: ") != std::string::npos);
    REQUIRE(chunk.find("\"id\":\"chatcmpl-123\"") != std::string::npos);
    REQUIRE(chunk.find("\"delta\"") != std::string::npos);
    REQUIRE(chunk.find("\"content\":\"Hello\"") != std::string::npos);
}

TEST_CASE(streaming_done_json_format) {
    const auto done = streamingDoneJson();
    REQUIRE_EQ(done, "data: [DONE]\n\n");
}