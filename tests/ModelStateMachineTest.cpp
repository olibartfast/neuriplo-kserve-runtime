#include "ModelStateMachine.hpp"
#include "Test.hpp"

#include <string>
#include <vector>

TEST_CASE(state_machine_starts_unloaded) {
    ModelStateMachine machine;
    REQUIRE(machine.current() == ModelState::Unloaded);
}

TEST_CASE(state_machine_normal_lifecycle) {
    ModelStateMachine machine;

    REQUIRE(machine.startLoad());
    REQUIRE(machine.current() == ModelState::Loading);

    REQUIRE(machine.markReady());
    REQUIRE(machine.current() == ModelState::Ready);
    REQUIRE(machine.isLoaded());

    REQUIRE(machine.beginUnload());
    REQUIRE(machine.current() == ModelState::Unloading);

    REQUIRE(machine.completeUnload());
    REQUIRE(machine.current() == ModelState::Unloaded);
    REQUIRE(!machine.isLoaded());
}

TEST_CASE(state_machine_load_failure) {
    ModelStateMachine machine;

    REQUIRE(machine.startLoad());
    REQUIRE(machine.markFailed());
    REQUIRE(machine.current() == ModelState::Failed);
    REQUIRE(machine.isTerminal());

    REQUIRE(machine.reset());
    REQUIRE(machine.current() == ModelState::Unloaded);
}

TEST_CASE(state_machine_unavailable_from_loading) {
    ModelStateMachine machine;

    REQUIRE(machine.startLoad());
    REQUIRE(machine.markUnavailable());
    REQUIRE(machine.current() == ModelState::Unavailable);
    REQUIRE(machine.isTerminal());
}

TEST_CASE(state_machine_unavailable_from_ready) {
    ModelStateMachine machine;

    REQUIRE(machine.startLoad());
    REQUIRE(machine.markReady());
    REQUIRE(machine.markUnavailable());
    REQUIRE(machine.current() == ModelState::Unavailable);
    REQUIRE(machine.isTerminal());

    REQUIRE(machine.reset());
    REQUIRE(machine.current() == ModelState::Unloaded);
}

TEST_CASE(state_machine_rejects_invalid_transitions) {
    ModelStateMachine machine;

    REQUIRE(!machine.markReady());
    REQUIRE(!machine.markFailed());
    REQUIRE(!machine.beginUnload());
    REQUIRE(!machine.completeUnload());

    REQUIRE(machine.current() == ModelState::Unloaded);
}

TEST_CASE(state_machine_rejects_double_load) {
    ModelStateMachine machine;

    REQUIRE(machine.startLoad());
    REQUIRE(machine.current() == ModelState::Loading);
    REQUIRE(!machine.startLoad());
    REQUIRE(machine.current() == ModelState::Loading);
}

TEST_CASE(state_machine_reset_noop_when_unloaded) {
    ModelStateMachine machine;
    REQUIRE(!machine.reset());
    REQUIRE(machine.current() == ModelState::Unloaded);
}

TEST_CASE(state_machine_observes_transitions) {
    ModelStateMachine machine;
    std::vector<std::string> events;

    machine.onTransition([&events](ModelState from, ModelState to) {
        events.push_back(std::string(modelStateName(from)) + " -> " + modelStateName(to));
    });

    machine.startLoad();
    machine.markReady();
    machine.beginUnload();
    machine.completeUnload();

    REQUIRE_EQ(events.size(), static_cast<size_t>(4));
    REQUIRE_EQ(events[0], "UNLOADED -> LOADING");
    REQUIRE_EQ(events[1], "LOADING -> READY");
    REQUIRE_EQ(events[2], "READY -> UNLOADING");
    REQUIRE_EQ(events[3], "UNLOADING -> UNLOADED");
}

TEST_CASE(state_machine_observer_sees_failure) {
    ModelStateMachine machine;
    std::string last_to;

    machine.onTransition([&last_to](ModelState, ModelState to) { last_to = modelStateName(to); });

    machine.startLoad();
    machine.markFailed();

    REQUIRE_EQ(last_to, "FAILED");
}

TEST_CASE(state_machine_state_name) {
    ModelStateMachine machine;
    REQUIRE_EQ(std::string(machine.stateName()), "UNLOADED");

    machine.startLoad();
    REQUIRE_EQ(std::string(machine.stateName()), "LOADING");

    machine.markReady();
    REQUIRE_EQ(std::string(machine.stateName()), "READY");
}

TEST_CASE(state_machine_multiple_observers) {
    ModelStateMachine machine;
    int count_a = 0;
    int count_b = 0;

    machine.onTransition([&count_a](ModelState, ModelState) { ++count_a; });
    machine.onTransition([&count_b](ModelState, ModelState) { ++count_b; });

    machine.startLoad();
    machine.markReady();

    REQUIRE(count_a == 2);
    REQUIRE(count_b == 2);
}
