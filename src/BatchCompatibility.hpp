#pragma once

#include "Executor.hpp"

#include <optional>
#include <string>

std::optional<std::string> batchCompatibilityError(const ExecutionRequest &left,
                                                   const ExecutionRequest &right);

bool areBatchCompatible(const ExecutionRequest &left, const ExecutionRequest &right);

int64_t requestBatchSize(const ExecutionRequest &request);
