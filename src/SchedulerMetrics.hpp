#pragma once

#include <cstdint>
#include <string>

struct SchedulerMetricsSnapshot {
    size_t queue_depth = 0;
    size_t in_flight = 0;
    uint64_t requests_accepted = 0;
    uint64_t requests_rejected = 0;
    uint64_t requests_timed_out = 0;
    uint64_t requests_memory_pressure_rejected = 0;
    // Latency totals are stored in nanoseconds. Step 7 Prometheus export must
    // divide by 1e9 before exposing the *_seconds metric names below.
    uint64_t queue_wait_ns_total = 0;
    uint64_t execution_ns_total = 0;
    uint64_t total_ns_total = 0;
    uint64_t completed_requests = 0;
    uint64_t batches_formed = 0;
    uint64_t batched_requests_total = 0;
    // Nanoseconds; divide by 1e9 at Step 7 Prometheus export.
    uint64_t batch_formation_ns_total = 0;
    uint64_t batch_execution_ns_total = 0;
    // KV cache metrics for LLM scheduler path
    size_t kv_cache_slots_total = 0;
    size_t kv_cache_slots_active = 0;
};

namespace SchedulerMetricNames {

inline constexpr const char *queueDepth = "neuriplo_scheduler_queue_depth";
inline constexpr const char *inFlight = "neuriplo_scheduler_in_flight_requests";
inline constexpr const char *requestsAccepted = "neuriplo_scheduler_requests_accepted_total";
inline constexpr const char *requestsRejected = "neuriplo_scheduler_requests_rejected_total";
inline constexpr const char *requestsTimedOut = "neuriplo_scheduler_requests_timed_out_total";
inline constexpr const char *queueWaitSeconds = "neuriplo_scheduler_queue_wait_seconds";
inline constexpr const char *executionSeconds = "neuriplo_scheduler_execution_seconds";
inline constexpr const char *totalSeconds = "neuriplo_scheduler_request_total_seconds";
inline constexpr const char *batchesFormed = "neuriplo_scheduler_batches_formed_total";
inline constexpr const char *batchedRequests = "neuriplo_scheduler_batched_requests_total";
inline constexpr const char *batchFormationSeconds = "neuriplo_scheduler_batch_formation_seconds";
inline constexpr const char *batchExecutionSeconds = "neuriplo_scheduler_batch_execution_seconds";
inline constexpr const char *batchSize = "neuriplo_scheduler_batch_size";
inline constexpr const char *kvCacheSlotsTotal = "neuriplo_kv_cache_slots_total";
inline constexpr const char *kvCacheSlotsActive = "neuriplo_kv_cache_slots_active";
inline constexpr const char *requestsMemoryPressureRejected =
    "neuriplo_scheduler_requests_memory_pressure_rejected_total";

inline constexpr const char *modelLabel = "model";

} // namespace SchedulerMetricNames
