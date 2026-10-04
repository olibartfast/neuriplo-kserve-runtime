#pragma once

#include "ModelHandle.hpp"
#include "RuntimeConfig.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>

class ModelLifecycle {
  public:
    using ExecutorFactory =
        std::function<std::unique_ptr<Executor>(const RuntimeConfig &, std::string &error)>;

    // version_override pins the published version (e.g. admin switch-version);
    // without it the executor-reported metadata versions win.
    void load(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory,
              const std::optional<std::string> &version_override = std::nullopt);
    bool beginDrain(ModelHandle &handle);
    bool completeUnload(ModelHandle &handle,
                        std::shared_ptr<Scheduler> *retired_scheduler = nullptr);

    // reload() is beginReload() + building the replacement + finishReload(),
    // all in one call. Callers that need to build the replacement without
    // holding a lock (ModelRegistry does, so a slow build never blocks
    // inference or readiness on other models) call the two halves directly
    // with the build in between; reload() itself is kept for callers, and
    // tests, that have no lock to release.
    bool reload(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory,
                const std::optional<std::string> &version_override = std::nullopt);

    // First half of a reload: resolves the handle's current state (draining
    // an Unloading handle, resetting a Failed/Unavailable one) so it is safe
    // to attempt a reload. Returns false if a reload cannot proceed right now
    // (already Loading). Never touches the handle's scheduler -- a Ready
    // handle is left exactly as it is, still serving, until finishReload()
    // confirms a replacement actually came up.
    bool beginReload(ModelHandle &handle);

    // Second half: given a handle already built by load() into `next`
    // (built with no lock held), swaps it into `handle` only if it is Ready.
    // On failure, `handle` is untouched except for load_error -- whatever it
    // was serving (or not) before the attempt, it still is.
    bool finishReload(ModelHandle &handle, ModelHandle &&next);
};
