#pragma once

#include "ModelState.hpp"

#include <functional>
#include <string>
#include <vector>

class ModelStateMachine {
  public:
    using Observer = std::function<void(ModelState from, ModelState to)>;

    ModelStateMachine() : state_(ModelState::Unloaded) {}

    ModelState current() const {
        return state_;
    }

    bool startLoad();
    bool markReady();
    bool markFailed();
    bool markUnavailable();
    bool beginUnload();
    bool completeUnload();
    bool reset();

    bool isLoaded() const {
        return state_ == ModelState::Ready;
    }

    bool isTerminal() const {
        return state_ == ModelState::Failed || state_ == ModelState::Unavailable;
    }

    const char *stateName() const {
        return modelStateName(state_);
    }

    void onTransition(Observer observer) {
        observers_.push_back(std::move(observer));
    }

  private:
    bool transitionTo(ModelState next);
    void notify(ModelState from, ModelState to);

    ModelState state_;
    std::vector<Observer> observers_;
};

inline bool ModelStateMachine::transitionTo(ModelState next) {
    if (!isValidTransition(state_, next)) {
        return false;
    }
    const auto prev = state_;
    state_ = next;
    notify(prev, next);
    return true;
}

inline void ModelStateMachine::notify(ModelState from, ModelState to) {
    for (const auto &obs : observers_) {
        if (obs) {
            obs(from, to);
        }
    }
}

inline bool ModelStateMachine::startLoad() {
    return transitionTo(ModelState::Loading);
}

inline bool ModelStateMachine::markReady() {
    return transitionTo(ModelState::Ready);
}

inline bool ModelStateMachine::markFailed() {
    return transitionTo(ModelState::Failed);
}

inline bool ModelStateMachine::markUnavailable() {
    return transitionTo(ModelState::Unavailable);
}

inline bool ModelStateMachine::beginUnload() {
    return transitionTo(ModelState::Unloading);
}

inline bool ModelStateMachine::completeUnload() {
    return transitionTo(ModelState::Unloaded);
}

inline bool ModelStateMachine::reset() {
    if (state_ == ModelState::Unloaded) {
        return false;
    }
    return transitionTo(ModelState::Unloaded);
}
