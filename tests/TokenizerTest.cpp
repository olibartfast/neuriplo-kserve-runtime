#include "Tokenizer.hpp"
#include "Test.hpp"

TEST_CASE(char_ratio_tokenizer_counts_basic) {
    CharRatioTokenizer tokenizer(0.25);
    REQUIRE_EQ(tokenizer.countTokens("hello"), static_cast<size_t>(1));
    REQUIRE_EQ(tokenizer.countTokens("hello world!"), static_cast<size_t>(3));
}

TEST_CASE(char_ratio_tokenizer_empty_string) {
    CharRatioTokenizer tokenizer(0.25);
    REQUIRE_EQ(tokenizer.countTokens(""), static_cast<size_t>(0));
}

TEST_CASE(char_ratio_tokenizer_rounds) {
    CharRatioTokenizer tokenizer(0.3);
    const auto count = tokenizer.countTokens("abcdefghij");
    REQUIRE(count >= static_cast<size_t>(1));
}

TEST_CASE(whitespace_tokenizer_counts_words) {
    WhitespaceTokenizer tokenizer;
    REQUIRE_EQ(tokenizer.countTokens("hello world"), static_cast<size_t>(2));
    REQUIRE_EQ(tokenizer.countTokens("one two three four"), static_cast<size_t>(4));
}

TEST_CASE(whitespace_tokenizer_empty_string) {
    WhitespaceTokenizer tokenizer;
    REQUIRE_EQ(tokenizer.countTokens(""), static_cast<size_t>(0));
}

TEST_CASE(whitespace_tokenizer_whitespace_only) {
    WhitespaceTokenizer tokenizer;
    REQUIRE_EQ(tokenizer.countTokens("   "), static_cast<size_t>(0));
}

TEST_CASE(whitespace_tokenizer_single_word) {
    WhitespaceTokenizer tokenizer;
    REQUIRE_EQ(tokenizer.countTokens("hello"), static_cast<size_t>(1));
}

TEST_CASE(whitespace_tokenizer_newlines_and_tabs) {
    WhitespaceTokenizer tokenizer;
    REQUIRE_EQ(tokenizer.countTokens("hello\tworld\nfoo"), static_cast<size_t>(3));
}