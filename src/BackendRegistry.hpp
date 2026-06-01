#pragma once

#include <optional>
#include <string>

enum class BackendKind { Tensor, Llm };

struct BackendCapability {
    std::string id;
    BackendKind kind = BackendKind::Tensor;
    bool uses_neuriplo = false;
};

bool registerBackend(BackendCapability capability);

std::optional<BackendCapability> findBackendCapability(const std::string &backend_id);

BackendKind backendKind(const std::string &backend_id);

bool isKnownBackend(const std::string &backend_id);

bool usesLlmScheduler(const std::string &scheduler_strategy, const std::string &backend_id);
