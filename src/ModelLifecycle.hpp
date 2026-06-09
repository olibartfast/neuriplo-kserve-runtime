#pragma once

#include "ModelHandle.hpp"
#include "RuntimeConfig.hpp"

#include <functional>
#include <memory>
#include <string>

class ModelLifecycle {
  public:
    using ExecutorFactory =
        std::function<std::unique_ptr<Executor>(const RuntimeConfig &, std::string &error)>;

    void load(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory);
    bool beginDrain(ModelHandle &handle);
    bool completeUnload(ModelHandle &handle);
    bool reload(ModelHandle &handle, const RuntimeConfig &config, ExecutorFactory factory);
};
