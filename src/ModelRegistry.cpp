#include "ModelRegistry.hpp"

#include "BackendRegistry.hpp"
#include "NeuriploExecutor.hpp"
#include "Scheduler.hpp"
#include "StubExecutor.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::unique_ptr<Executor> defaultExecutorFactory(const RuntimeConfig &config, std::string &error) {
    if (config.backend == "stub") {
        return makeStubExecutor(config, error);
    }

    const auto capability = findBackendCapability(config.backend);
    if (!capability) {
        error = "unsupported backend: " + config.backend;
        return nullptr;
    }
    if (!capability->uses_neuriplo) {
        error = "unsupported backend: " + config.backend;
        return nullptr;
    }
    return makeNeuriploExecutor(config, error);
}

} // namespace

ModelRegistry::ModelRegistry(const RuntimeConfig &config)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config, defaultExecutorFactory);
}

ModelRegistry::ModelRegistry(const RuntimeConfig &config, ExecutorFactory factory)
    : log_payloads_(config.log_payloads), tokens_per_char_(config.tokens_per_char) {
    loadModel(config, std::move(factory));
}

void ModelRegistry::loadModel(const RuntimeConfig &config, ExecutorFactory factory) {
    handle_.name = config.model_name;
    handle_.versions = {"1"};
    handle_.state.startLoad();

    std::vector<std::unique_ptr<Executor>> executors;
    executors.reserve(config.instances);
    std::string error;
    for (size_t instance_index = 0; instance_index < config.instances; ++instance_index) {
        (void)instance_index;
        auto executor = factory(config, error);
        if (!executor) {
            handle_.state.markFailed();
            handle_.load_error = error.empty() ? "failed to create executor" : error;
            handle_.metadata.name = config.model_name;
            handle_.metadata.versions = handle_.versions;
            handle_.metadata.platform = "neuriplo_" + config.backend;
            return;
        }
        executors.push_back(std::move(executor));
    }

    handle_.metadata = executors.front()->metadata();
    handle_.name = handle_.metadata.name;
    handle_.versions = handle_.metadata.versions;

    if (usesLlmScheduler(config.scheduler_strategy, config.backend)) {
        LlmSchedulerConfig scheduler_config;
        scheduler_config.max_queue_size = config.max_queue_size;
        scheduler_config.request_timeout_ms = config.request_timeout_ms;
        scheduler_config.instances = config.instances;
        scheduler_config.context_length = config.context_length;
        scheduler_config.kv_cache_slots = config.kv_cache_slots;
        scheduler_config.max_tokens = config.max_tokens;
        scheduler_config.tokens_per_char = config.tokens_per_char;
        scheduler_config.memory_budget_bytes = config.memory_budget_bytes;
        handle_.scheduler =
            makeLlmScheduler(std::move(executors), scheduler_config, config.model_name);
    } else {
        SchedulerConfig scheduler_config;
        scheduler_config.max_queue_size = config.max_queue_size;
        scheduler_config.request_timeout_ms = config.request_timeout_ms;
        scheduler_config.instances = config.instances;
        scheduler_config.dynamic_batching.enabled = config.dynamic_batching_enabled;
        scheduler_config.dynamic_batching.max_batch_size = config.max_batch_size;
        scheduler_config.dynamic_batching.max_queue_delay_us = config.max_queue_delay_us;
        scheduler_config.dynamic_batching.preferred_batch_sizes = config.preferred_batch_sizes;
        handle_.scheduler =
            makeModelScheduler(std::move(executors), scheduler_config, config.model_name);
    }
    handle_.state.markReady();
}

std::optional<ModelMetadata> ModelRegistry::find(const std::string &model_name) const {
    if (model_name != handle_.name) {
        return std::nullopt;
    }
    return handle_.metadata;
}

std::optional<ModelMetadata> ModelRegistry::findVersion(const std::string &model_name,
                                                        const std::string &version) const {
    const auto model = find(model_name);
    if (!model) {
        return std::nullopt;
    }
    for (const auto &known_version : model->versions) {
        if (known_version == version) {
            return model;
        }
    }
    return std::nullopt;
}

const ModelHandle *ModelRegistry::findHandle(const std::string &model_name) const {
    if (model_name != handle_.name) {
        return nullptr;
    }
    return &handle_;
}

const ModelHandle *ModelRegistry::findHandleVersion(const std::string &model_name,
                                                    const std::string &version) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr) {
        return nullptr;
    }
    const auto &known_versions =
        handle->metadata.versions.empty() ? handle->versions : handle->metadata.versions;
    for (const auto &known_version : known_versions) {
        if (known_version == version) {
            return handle;
        }
    }
    return nullptr;
}

bool ModelRegistry::ready(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    return handle != nullptr && handle->isReady();
}

bool ModelRegistry::readyVersion(const std::string &model_name, const std::string &version) const {
    const auto *handle = findHandleVersion(model_name, version);
    return handle != nullptr && handle->isReady();
}

bool ModelRegistry::allReady() const {
    return handle_.isReady();
}

std::optional<std::string> ModelRegistry::defaultVersion(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr) {
        return std::nullopt;
    }
    if (!handle->metadata.versions.empty()) {
        return handle->metadata.versions.front();
    }
    if (!handle->versions.empty()) {
        return handle->versions.front();
    }
    return std::nullopt;
}

bool ModelRegistry::beginDrain(const std::string &model_name) {
    if (model_name != handle_.name || handle_.scheduler == nullptr) {
        return false;
    }
    if (handle_.state.current() == ModelState::Ready) {
        handle_.state.beginUnload();
    }
    handle_.scheduler->beginDrain();
    return true;
}

SchedulerMetricsSnapshot ModelRegistry::schedulerMetrics(const std::string &model_name) const {
    const auto *handle = findHandle(model_name);
    if (handle == nullptr || handle->scheduler == nullptr) {
        return {};
    }
    return handle->scheduler->metrics();
}

std::string ModelRegistry::modelName() const {
    return handle_.name;
}

bool ModelRegistry::logPayloads() const {
    return log_payloads_;
}

double ModelRegistry::tokensPerChar() const {
    return tokens_per_char_;
}
