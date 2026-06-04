#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Tokenizer {
  public:
    virtual ~Tokenizer() = default;
    virtual size_t countTokens(const std::string &text) const = 0;
    virtual std::vector<size_t> encode(const std::string &text) const = 0;
};

class CharRatioTokenizer final : public Tokenizer {
  public:
    explicit CharRatioTokenizer(double tokens_per_char = 0.25)
        : tokens_per_char_(tokens_per_char) {}

    size_t countTokens(const std::string &text) const override {
        return static_cast<size_t>(static_cast<double>(text.size()) * tokens_per_char_ + 0.5);
    }

    std::vector<size_t> encode(const std::string &text) const override {
        const auto token_count = countTokens(text);
        std::vector<size_t> ids(token_count);
        for (size_t i = 0; i < token_count; ++i) {
            ids[i] = i;
        }
        return ids;
    }

  private:
    double tokens_per_char_;
};

class WhitespaceTokenizer final : public Tokenizer {
  public:
    size_t countTokens(const std::string &text) const override {
        if (text.empty()) {
            return 0;
        }
        size_t count = 0;
        bool in_word = false;
        for (const char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                in_word = false;
            } else if (!in_word) {
                in_word = true;
                ++count;
            }
        }
        return count;
    }

    std::vector<size_t> encode(const std::string &text) const override {
        const auto count = countTokens(text);
        std::vector<size_t> ids(count);
        for (size_t i = 0; i < count; ++i) {
            ids[i] = i;
        }
        return ids;
    }
};