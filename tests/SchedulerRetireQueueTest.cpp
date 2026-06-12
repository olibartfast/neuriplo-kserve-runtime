#include "SchedulerRetireQueue.hpp"
#include "RuntimeConfig.hpp"
#include "Scheduler.hpp"
#include "StubExecutor.hpp"
#include "Test.hpp"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

TEST_CASE(scheduler_retire_queue_completes_background_drain) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";

    std::string error;
    std::vector<std::unique_ptr<Executor>> executors;
    executors.push_back(makeStubExecutor(config, error));

    SchedulerConfig scheduler_config;
    auto scheduler = std::shared_ptr<Scheduler>(
        makeModelScheduler(std::move(executors), scheduler_config, "demo").release());

    SchedulerRetireQueue queue;
    queue.retire(scheduler);

    for (int attempt = 0; attempt < 100; ++attempt) {
        if (queue.pendingCount() == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    REQUIRE_EQ(queue.pendingCount(), 0);
}
