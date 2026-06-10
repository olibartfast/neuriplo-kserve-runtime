#pragma once

#include "ModelHandle.hpp"
#include "ModelMetadata.hpp"
#include "ModelState.hpp"
#include "Scheduler.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct InferSnapshot {
    std::string name;
    std::vector<std::string> versions;
    ModelState state = ModelState::Unloaded;
    ModelMetadata metadata;
    std::shared_ptr<Scheduler> scheduler;
    std::optional<std::string> load_error;

    bool isReady() const {
        return state == ModelState::Ready && scheduler != nullptr && scheduler->isReady();
    }

    static std::shared_ptr<const InferSnapshot> fromHandle(const ModelHandle &handle) {
        auto snapshot = std::make_shared<InferSnapshot>();
        snapshot->name = handle.name;
        snapshot->versions = handle.versions;
        snapshot->state = handle.state.current();
        snapshot->metadata = handle.metadata;
        snapshot->scheduler = handle.scheduler;
        snapshot->load_error = handle.load_error;
        return snapshot;
    }
};
