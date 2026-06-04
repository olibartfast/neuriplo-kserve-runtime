#pragma once

#include "ModelMetadata.hpp"
#include "ModelState.hpp"
#include "ModelStateMachine.hpp"
#include "Scheduler.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct ModelHandle {
    std::string name;
    std::vector<std::string> versions;
    ModelStateMachine state;
    ModelMetadata metadata;
    std::unique_ptr<Scheduler> scheduler;
    std::optional<std::string> load_error;

    bool isReady() const {
        return state.current() == ModelState::Ready && scheduler != nullptr && scheduler->isReady();
    }
};
