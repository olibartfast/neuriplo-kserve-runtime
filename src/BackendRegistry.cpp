#include "BackendRegistry.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

std::mutex registry_mutex;
std::unordered_map<std::string, BackendCapability> capabilities;
std::once_flag default_init_flag;

void registerDefaultBackends() {
    std::call_once(default_init_flag, []() {
        const std::vector<BackendCapability> defaults = {
            {"stub", BackendKind::Tensor, false},
            {"onnx_runtime", BackendKind::Tensor, true},
            {"opencv_dnn", BackendKind::Tensor, true},
            {"openvino", BackendKind::Tensor, true},
            {"tensorrt", BackendKind::Tensor, true},
            {"libtorch", BackendKind::Tensor, true},
            {"libtensorflow", BackendKind::Tensor, true},
            {"migraphx", BackendKind::Tensor, true},
            {"executorch", BackendKind::Tensor, true},
            {"ggml", BackendKind::Llm, true},
            {"llamacpp", BackendKind::Llm, true},
            {"cactus", BackendKind::Llm, true},
        };

        std::lock_guard<std::mutex> lock(registry_mutex);
        for (const auto &capability : defaults) {
            capabilities.emplace(capability.id, capability);
        }
    });
}

} // namespace

bool registerBackend(BackendCapability capability) {
    registerDefaultBackends();
    if (capability.id.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto inserted = capabilities.emplace(capability.id, std::move(capability));
    return inserted.second;
}

std::optional<BackendCapability> findBackendCapability(const std::string &backend_id) {
    registerDefaultBackends();
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = capabilities.find(backend_id);
    if (found == capabilities.end()) {
        return std::nullopt;
    }
    return found->second;
}

BackendKind backendKind(const std::string &backend_id) {
    const auto capability = findBackendCapability(backend_id);
    if (!capability) {
        return BackendKind::Tensor;
    }
    return capability->kind;
}

bool isKnownBackend(const std::string &backend_id) {
    return findBackendCapability(backend_id).has_value();
}

bool usesLlmScheduler(const std::string &scheduler_strategy, const std::string &backend_id) {
    if (scheduler_strategy == "llm") {
        return true;
    }
    if (scheduler_strategy != "tensor") {
        return false;
    }
    return backendKind(backend_id) == BackendKind::Llm;
}
