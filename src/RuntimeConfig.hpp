#pragma once

#include <string>

struct RuntimeConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string model_name = "demo";
    std::string model_path;
    std::string backend = "stub";
};

RuntimeConfig parseRuntimeConfig(int argc, char **argv);
