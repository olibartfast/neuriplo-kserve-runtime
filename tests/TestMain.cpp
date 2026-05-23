#include "Test.hpp"

#include <exception>
#include <iostream>
#include <utility>

std::vector<TestCase> &testRegistry() {
    static std::vector<TestCase> tests;
    return tests;
}

void registerTest(std::string name, std::function<void()> run) {
    testRegistry().push_back({std::move(name), std::move(run)});
}

int main() {
    int failures = 0;
    for (const auto &test : testRegistry()) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << "\n";
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed" << '\n';
        return 1;
    }

    std::cout << testRegistry().size() << " test(s) passed" << '\n';
    return 0;
}
