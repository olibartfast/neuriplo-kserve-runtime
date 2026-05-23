#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct TestCase {
    std::string name;
    std::function<void()> run;
};

std::vector<TestCase> &testRegistry();
void registerTest(std::string name, std::function<void()> run);

#define TEST_CASE(name)                                                                            \
    void name();                                                                                   \
    namespace {                                                                                    \
    const bool name##_registered = [] {                                                            \
        registerTest(#name, name);                                                                 \
        return true;                                                                               \
    }();                                                                                           \
    }                                                                                              \
    void name()

#define REQUIRE(condition)                                                                         \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +      \
                                     ": requirement failed: " #condition);                         \
        }                                                                                          \
    } while (false)

#define REQUIRE_EQ(actual, expected) REQUIRE((actual) == (expected))
