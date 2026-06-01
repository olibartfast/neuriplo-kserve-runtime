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
    ModelState state = ModelState::Unloaded;
    ModelMetadata metadata;
    std::unique_ptr<Scheduler> scheduler;
    std::optional<std::string> load_error;

    bool isReady() const {
        return state == ModelState::Ready && scheduler != nullptr && scheduler->isReady();
    }

    bool canTransitionTo(ModelState next) const {
        return isValidTransition(state, next);
    }

    bool transitionTo(ModelState next) {
        if (!canTransitionTo(next)) {
            return false;
        }
        state = next;
        return true;
    }
};
