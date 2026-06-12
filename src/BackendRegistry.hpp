#pragma once

#include "Executor.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class BackendKind { Tensor, Llm };

using ExecutorFactoryFn =
    std::function<std::unique_ptr<Executor>(const class RuntimeConfig &, std::string &error)>;

struct BackendCapability {
    std::string id;
    BackendKind kind = BackendKind::Tensor;
    bool uses_neuriplo = false;
    ExecutorFactoryFn factory;
};

bool registerBackend(BackendCapability capability);

std::optional<BackendCapability> findBackendCapability(const std::string &backend_id);

BackendKind backendKind(const std::string &backend_id);

bool isKnownBackend(const std::string &backend_id);

bool usesLlmScheduler(const std::string &scheduler_strategy, const std::string &backend_id);

// Backend ids this binary can actually serve: non-neuriplo backends (stub)
// plus the neuriplo backends compiled in or loaded as plugins.
std::vector<std::string> availableBackendIds(const std::string &plugin_dir = "");

std::unique_ptr<Executor> createExecutorFor(const std::string &backend_id,
                                            const class RuntimeConfig &config, std::string &error);
