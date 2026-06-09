#include "Test.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

std::vector<TestCase> &testRegistry() {
    static std::vector<TestCase> tests;
    return tests;
}

void registerTest(std::string name, std::function<void()> run) {
    testRegistry().push_back({std::move(name), std::move(run)});
}

int main() {
    const char *filter = std::getenv("NEURIPLO_TEST_FILTER");
    int failures = 0;
    int ran = 0;
    for (const auto &test : testRegistry()) {
        if (filter != nullptr && test.name != filter) {
            continue;
        }
        ++ran;
        try {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
        }
    }

    if (filter != nullptr && ran == 0) {
        std::cerr << "No test matched NEURIPLO_TEST_FILTER=" << filter << "\n";
        return 1;
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed" << '\n';
        return 1;
    }

    std::cout << ran << " test(s) passed" << '\n';
    return 0;
}
