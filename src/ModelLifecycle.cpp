#include "ModelLifecycle.hpp"

#include "BackendRegistry.hpp"
#include "Scheduler.hpp"

#include <utility>
#include <vector>

namespace {

// Leaves `handle` in a reportable Failed state with `message` recorded,
// instead of whatever half-built state an exception interrupted it in.
// Shared by every early-return/failure path in load() below, including the
// catch blocks, so a caller (ModelRegistry) never has to special-case
// "the build threw" versus "the build returned nullptr".
void markLoadFailed(ModelHandle &handle, const RuntimeConfig &config, std::string message) {
    handle.state.markFailed();
    handle.load_error = std::move(message);
    handle.metadata.name = config.model_name;
    handle.metadata.versions = handle.versions;
    handle.metadata.platform = "neuriplo_" + config.backend;
    handle.scheduler.reset();
}

} // namespace

void ModelLifecycle::load(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory,
                          const std::optional<std::string> &version_override) {
    if (handle.state.current() != ModelState::Unloaded) {
        return;
    }

    handle.name = config.model_name;
    handle.versions = {version_override.value_or("1")};
    handle.load_error.reset();
    handle.state.startLoad();

    // The factory and scheduler/backend construction below run third-party
    // and plugin code (real neuriplo backends, pipeline step resolution) that
    // can throw -- bad_alloc, invalid_argument from a malformed model, a
    // plugin's own exception types. ModelRegistry builds this with no lock
    // held and a waiting load_cv_ on the other side, so this function must
    // never propagate an exception: it has to leave `handle` Failed instead,
    // the same as a factory returning nullptr.
    try {
        std::vector<std::unique_ptr<Executor>> executors;
        executors.reserve(config.instances);
        std::string error;
        for (size_t instance_index = 0; instance_index < config.instances; ++instance_index) {
            (void)instance_index;
            auto executor = factory(config, error);
            if (!executor) {
                markLoadFailed(handle, config, error.empty() ? "failed to create executor" : error);
                return;
            }
            executors.push_back(std::move(executor));
        }

        // Defensive: config.instances == 0 (or any path that reaches here
        // with no built executors) must fail cleanly rather than dereference
        // an empty vector. AdminCodec rejects instances < 1 before a load
        // ever reaches this class, but ModelLifecycle does not get to assume
        // that -- it is called directly from tests and from any other config
        // source.
        if (executors.empty()) {
            markLoadFailed(handle, config, "no executor instances configured");
            return;
        }

        handle.metadata = executors.front()->metadata();
        if (version_override.has_value()) {
            handle.metadata.versions = {*version_override};
        }
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
            handle.scheduler = std::shared_ptr<Scheduler>(
                makeLlmScheduler(std::move(executors), scheduler_config, config.model_name)
                    .release());
        } else {
            SchedulerConfig scheduler_config;
            scheduler_config.max_queue_size = config.max_queue_size;
            scheduler_config.request_timeout_ms = config.request_timeout_ms;
            scheduler_config.instances = config.instances;
            scheduler_config.dynamic_batching.enabled = config.dynamic_batching_enabled;
            scheduler_config.dynamic_batching.max_batch_size = config.max_batch_size;
            scheduler_config.dynamic_batching.max_queue_delay_us = config.max_queue_delay_us;
            scheduler_config.dynamic_batching.preferred_batch_sizes = config.preferred_batch_sizes;
            handle.scheduler = std::shared_ptr<Scheduler>(
                makeModelScheduler(std::move(executors), scheduler_config, config.model_name)
                    .release());
        }
        handle.state.markReady();
    } catch (const std::exception &caught) {
        markLoadFailed(handle, config, caught.what());
    } catch (...) {
        markLoadFailed(handle, config, "executor build threw");
    }
}

bool ModelLifecycle::beginDrain(ModelHandle &handle) {
    if (handle.scheduler == nullptr) {
        return false;
    }
    if (handle.state.current() == ModelState::Ready) {
        handle.state.beginUnload();
    }
    handle.scheduler->stopAccepting();
    return true;
}

bool ModelLifecycle::completeUnload(ModelHandle &handle,
                                    std::shared_ptr<Scheduler> *retired_scheduler) {
    if (handle.state.current() != ModelState::Unloading) {
        return false;
    }
    if (handle.scheduler != nullptr) {
        handle.scheduler->stopAccepting();
        if (retired_scheduler != nullptr) {
            *retired_scheduler = handle.scheduler;
        }
        handle.scheduler.reset();
    }
    handle.metadata = {};
    handle.versions.clear();
    handle.load_error.reset();
    return handle.state.completeUnload();
}

bool ModelLifecycle::beginReload(ModelHandle &handle) {
    switch (handle.state.current()) {
    case ModelState::Loading:
        // A load/reload is already building a result for this handle; a
        // second one cannot safely build another behind its back.
        return false;
    case ModelState::Ready:
        // Left exactly as it is -- still Ready, still serving on its current
        // scheduler -- until finishReload() has a replacement to swap in.
        return true;
    case ModelState::Unloading:
        return completeUnload(handle);
    case ModelState::Failed:
    case ModelState::Unavailable:
        handle.state.reset();
        return true;
    case ModelState::Unloaded:
        return true;
    }
    return false;
}

bool ModelLifecycle::finishReload(ModelHandle &handle, ModelHandle &&next) {
    if (next.state.current() != ModelState::Ready) {
        // The replacement never came up: the Triton-compatible contract is
        // that a failed reload of an already-loaded model leaves the serving
        // model untouched and Ready, so nothing about `handle` changes here
        // beyond recording why the attempt failed -- *except* when
        // beginReload() had to reset a Failed/Unavailable/Unloading handle
        // to Unloaded to attempt this rebuild at all: landing back on
        // Unloaded would make a model that failed twice look like one that
        // was simply never loaded. Put it back on Failed (via Loading, the
        // only transition that reaches it) instead.
        handle.load_error = next.load_error;
        if (handle.state.current() == ModelState::Unloaded) {
            handle.state.startLoad();
            handle.state.markFailed();
        }
        return false;
    }

    // The replacement is Ready: only now is it safe to retire whatever
    // `handle` was serving before. The caller (ModelRegistry) owns the old
    // scheduler from this point and is responsible for letting in-flight
    // requests that already captured it finish before it is destroyed.
    handle.name = next.name;
    handle.versions = next.versions;
    handle.metadata = next.metadata;
    handle.scheduler = next.scheduler;
    handle.load_error.reset();
    if (handle.state.current() != ModelState::Ready) {
        if (handle.state.current() != ModelState::Unloaded) {
            handle.state.reset();
        }
        handle.state.startLoad();
        handle.state.markReady();
    }
    return true;
}

bool ModelLifecycle::reload(ModelHandle &handle, const RuntimeConfig &config,
                            ExecutorFactory factory,
                            const std::optional<std::string> &version_override) {
    if (!beginReload(handle)) {
        return false;
    }

    // Built into a fresh handle, never touching `handle` itself, so a caller
    // with a lock to release (ModelRegistry) can do exactly that around this
    // call without any risk of exposing a half-updated `handle` to a reader.
    // load() above never throws (it catches everything and reports Failed
    // instead), so there is no exception-safety gap between this call and
    // finishReload() below.
    ModelHandle next;
    load(next, config, std::move(factory), version_override);
    return finishReload(handle, std::move(next));
}
