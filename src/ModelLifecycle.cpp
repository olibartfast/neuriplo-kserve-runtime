#include "ModelLifecycle.hpp"

#include "BackendRegistry.hpp"
#include "Scheduler.hpp"

#include <utility>
#include <vector>

void ModelLifecycle::load(ModelHandle &handle, const RuntimeConfig &config,
                          ExecutorFactory factory) {
    if (handle.state.current() != ModelState::Unloaded) {
        return;
    }

    handle.name = config.model_name;
    handle.versions = {"1"};
    handle.load_error.reset();
    handle.state.startLoad();

    std::vector<std::unique_ptr<Executor>> executors;
    executors.reserve(config.instances);
    std::string error;
    for (size_t instance_index = 0; instance_index < config.instances; ++instance_index) {
        (void)instance_index;
        auto executor = factory(config, error);
        if (!executor) {
            handle.state.markFailed();
            handle.load_error = error.empty() ? "failed to create executor" : error;
            handle.metadata.name = config.model_name;
            handle.metadata.versions = handle.versions;
            handle.metadata.platform = "neuriplo_" + config.backend;
            handle.scheduler.reset();
            return;
        }
        executors.push_back(std::move(executor));
    }

    handle.metadata = executors.front()->metadata();
    handle.name = handle.metadata.name;
    handle.versions = handle.metadata.versions;

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
        handle.scheduler =
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
        handle.scheduler =
            makeModelScheduler(std::move(executors), scheduler_config, config.model_name);
    }
    handle.state.markReady();
}

bool ModelLifecycle::beginDrain(ModelHandle &handle) {
    if (handle.scheduler == nullptr) {
        return false;
    }
    if (handle.state.current() == ModelState::Ready) {
        handle.state.beginUnload();
    }
    handle.scheduler->beginDrain();
    return true;
}

bool ModelLifecycle::completeUnload(ModelHandle &handle) {
    if (handle.state.current() != ModelState::Unloading) {
        return false;
    }
    if (handle.scheduler != nullptr) {
        handle.scheduler->beginDrain();
        handle.scheduler.reset();
    }
    handle.metadata = {};
    handle.versions.clear();
    handle.load_error.reset();
    return handle.state.completeUnload();
}

bool ModelLifecycle::reload(ModelHandle &handle, const RuntimeConfig &config,
                            ExecutorFactory factory) {
    switch (handle.state.current()) {
    case ModelState::Loading:
        return false;
    case ModelState::Ready:
        if (!beginDrain(handle)) {
            return false;
        }
        if (!completeUnload(handle)) {
            return false;
        }
        break;
    case ModelState::Unloading:
        if (!completeUnload(handle)) {
            return false;
        }
        break;
    case ModelState::Failed:
    case ModelState::Unavailable:
        handle.state.reset();
        break;
    case ModelState::Unloaded:
        break;
    }
    load(handle, config, std::move(factory));
    return handle.state.current() == ModelState::Ready;
}
