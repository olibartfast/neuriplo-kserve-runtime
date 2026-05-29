#pragma once

#include "ModelMetadata.hpp"
#include "ModelState.hpp"
#include "Scheduler.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ModelHandle {
    std::string name;
    std::vector<std::string> versions;
    ModelState state = ModelState::Loading;
    ModelMetadata metadata;
    std::unique_ptr<Scheduler> scheduler;
    std::optional<std::string> load_error;

    bool isReady() const {
        return state == ModelState::Ready && scheduler != nullptr && scheduler->isReady();
    }
};
