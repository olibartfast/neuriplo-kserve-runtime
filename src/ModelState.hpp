#pragma once

#include <string>

enum class ModelState {
    Unloaded,
    Loading,
    Ready,
    Failed,
    Unavailable,
    Unloading,
};

inline const char *modelStateName(ModelState state) {
    switch (state) {
    case ModelState::Unloaded:
        return "UNLOADED";
    case ModelState::Loading:
        return "LOADING";
    case ModelState::Ready:
        return "READY";
    case ModelState::Failed:
        return "FAILED";
    case ModelState::Unavailable:
        return "UNAVAILABLE";
    case ModelState::Unloading:
        return "UNLOADING";
    }
    return "UNKNOWN";
}

inline bool isValidTransition(ModelState current, ModelState next) {
    switch (current) {
    case ModelState::Unloaded:
        return next == ModelState::Loading;
    case ModelState::Loading:
        return next == ModelState::Ready || next == ModelState::Failed ||
               next == ModelState::Unavailable;
    case ModelState::Ready:
        return next == ModelState::Unloading || next == ModelState::Unavailable;
    case ModelState::Unloading:
        return next == ModelState::Unloaded;
    case ModelState::Failed:
        return next == ModelState::Unloaded;
    case ModelState::Unavailable:
        return next == ModelState::Ready || next == ModelState::Unloaded;
    }
    return false;
}
