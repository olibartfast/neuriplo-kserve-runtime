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
    bool reload(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory,
                const std::optional<std::string> &version_override = std::nullopt);
};
