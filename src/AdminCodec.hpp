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
    // Whether the request body named "backend" explicitly, whatever its
    // type (a wrong-typed value still fails parsing with ok == false; this
    // only distinguishes "present" from "absent"). A caller resolving a bare
    // model name against a repository catalog needs this: an empty/"{}" body
    // must not be mistaken for an explicit backend choice.
    bool backend_provided = false;
};

AdminParseResult parseLoadModelRequest(const std::string &body, const RuntimeConfig &defaults);
AdminParseResult parseReloadModelRequest(const std::string &body, const RuntimeConfig &defaults);
AdminParseResult parseSwitchVersionRequest(const std::string &body, const RuntimeConfig &defaults);
