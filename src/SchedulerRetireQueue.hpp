#pragma once

#include "Scheduler.hpp"

#include <future>
#include <memory>
#include <mutex>
#include <vector>

class SchedulerRetireQueue {
  public:
    SchedulerRetireQueue() = default;
    ~SchedulerRetireQueue();

    SchedulerRetireQueue(const SchedulerRetireQueue &) = delete;
    SchedulerRetireQueue &operator=(const SchedulerRetireQueue &) = delete;

    void retire(std::shared_ptr<Scheduler> scheduler);
    size_t pendingCount() const;

  private:
    void pruneCompleted();

    mutable std::mutex mutex_;
    std::vector<std::future<void>> pending_;
};
