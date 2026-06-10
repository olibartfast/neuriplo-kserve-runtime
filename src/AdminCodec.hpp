#pragma once

#include "RuntimeConfig.hpp"

#include <optional>
#include <string>

struct AdminParseResult {
    bool ok = false;
    std::string error_message;
    RuntimeConfig config;
    std::string model_name;
    std::string version;
};

AdminParseResult parseLoadModelRequest(const std::string &body, const RuntimeConfig &defaults);
AdminParseResult parseReloadModelRequest(const std::string &body, const RuntimeConfig &defaults);
AdminParseResult parseSwitchVersionRequest(const std::string &body, const RuntimeConfig &defaults);
