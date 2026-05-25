#pragma once

#include "Executor.hpp"
#include "RuntimeConfig.hpp"

#include <memory>
#include <string>

std::unique_ptr<Executor> makeStubExecutor(const RuntimeConfig &config, std::string &error);
