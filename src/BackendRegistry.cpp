#include "BackendRegistry.hpp"

#include "NeuriploAdapter.hpp"
#include "NeuriploExecutor.hpp"
#include "StubExecutor.hpp"

#include <algorithm>
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
            {"stub", BackendKind::Tensor, false,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeStubExecutor(config, error);
             }},
            {"onnx_runtime", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"opencv_dnn", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"openvino", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"tensorrt", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"libtorch", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"libtensorflow", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"migraphx", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"executorch", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"litert", BackendKind::Tensor, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"ggml", BackendKind::Llm, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"llamacpp", BackendKind::Llm, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
            {"cactus", BackendKind::Llm, true,
             [](const RuntimeConfig &config, std::string &error) {
                 return makeNeuriploExecutor(config, error);
             }},
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

std::vector<std::string> availableBackendIds(const std::string &plugin_dir) {
    registerDefaultBackends();

    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        for (const auto &entry : capabilities) {
            if (!entry.second.uses_neuriplo) {
                ids.push_back(entry.first);
            }
        }
    }
    for (const auto &id : realNeuriploAvailableBackends(plugin_dir)) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::unique_ptr<Executor> createExecutorFor(const std::string &backend_id,
                                            const RuntimeConfig &config, std::string &error) {
    const auto capability = findBackendCapability(backend_id);
    if (!capability) {
        error = "unsupported backend: " + backend_id;
        return nullptr;
    }
    if (!capability->factory) {
        error = "no executor factory registered for backend: " + backend_id;
        return nullptr;
    }

    // Fail fast for known backends that are not compiled into (or loaded as a
    // plugin by) this binary, so admin loads report a clear 4xx instead of a
    // deep adapter failure. Stub builds keep their existing error contract.
    if (capability->uses_neuriplo && realNeuriploSupportEnabled()) {
        const auto available = realNeuriploAvailableBackends(config.plugin_dir);
        if (std::find(available.begin(), available.end(), backend_id) == available.end()) {
            std::string joined;
            for (const auto &id : availableBackendIds(config.plugin_dir)) {
                if (!joined.empty()) {
                    joined += ", ";
                }
                joined += id;
            }
            error = "backend '" + backend_id +
                    "' is not available in this binary; available backends: " + joined;
            return nullptr;
        }
    }

    return capability->factory(config, error);
}
