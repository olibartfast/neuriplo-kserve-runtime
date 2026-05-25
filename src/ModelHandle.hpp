#pragma once

#include "Executor.hpp"
#include "ModelMetadata.hpp"
#include "ModelState.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ModelHandle {
    std::string name;
    std::vector<std::string> versions;
    ModelState state = ModelState::Loading;
    ModelMetadata metadata;
    std::unique_ptr<Executor> executor;
    std::optional<std::string> load_error;

    bool isReady() const {
        return state == ModelState::Ready;
    }
};
