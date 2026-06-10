#include "SchedulerRetireQueue.hpp"

#include <algorithm>

SchedulerRetireQueue::~SchedulerRetireQueue() {
    std::vector<std::future<void>> pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.swap(pending_);
    }
    for (auto &task : pending) {
        if (task.valid()) {
            task.wait();
        }
    }
}

void SchedulerRetireQueue::retire(std::shared_ptr<Scheduler> scheduler) {
    if (!scheduler) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    pending_.emplace_back(std::async(
        std::launch::async, [scheduler = std::move(scheduler)]() { scheduler->beginDrain(); }));
    pruneCompleted();
}

size_t SchedulerRetireQueue::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *self = const_cast<SchedulerRetireQueue *>(this);
    self->pruneCompleted();
    return pending_.size();
}

void SchedulerRetireQueue::pruneCompleted() {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [](const std::future<void> &task) {
                                      return !task.valid() ||
                                             task.wait_for(std::chrono::seconds(0)) ==
                                                 std::future_status::ready;
                                  }),
                   pending_.end());
}
