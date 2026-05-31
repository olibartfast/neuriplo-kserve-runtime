#pragma once

#include "SchedulerMetrics.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct Histogram {
    std::vector<double> buckets;
    std::vector<uint64_t> counts;
    double sum = 0.0;
    uint64_t count = 0;

    Histogram() = default;
    Histogram(std::vector<double> b) : buckets(std::move(b)), counts(buckets.size() + 1, 0) {}

    void observe(double value) {
        sum += value;
        count++;
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (value <= buckets[i]) {
                counts[i]++;
            }
        }
        counts.back()++; // +Inf
    }
};

struct HttpMetricsSnapshot {
    uint64_t requests_total = 0;
    uint64_t requests_by_path_total = 0;
    uint64_t requests_by_status_total = 0;
};

class MetricsRegistry {
  public:
    MetricsRegistry();

    void recordInferRequest(const std::string &model, const std::string &method, int status_code,
                            uint64_t queue_latency_ns, uint64_t infer_latency_ns,
                            uint64_t total_latency_ns, uint64_t batch_size = 0);

    void recordModelLoadSuccess(const std::string &model, const std::string &backend);
    void recordModelLoadFailure(const std::string &model, const std::string &backend);

    std::string renderMetrics() const;

    void setSchedulerMetrics(const SchedulerMetricsSnapshot &snapshot);
    SchedulerMetricsSnapshot schedulerMetrics() const;

    void recordProcessMemory();

  private:
    mutable std::mutex mutex_;

    SchedulerMetricsSnapshot scheduler_metrics_;
    std::string active_model_name_ = "demo";

    // Request metrics maps
    std::map<std::string, std::map<std::string, uint64_t>>
        requests_by_status_; // model -> status -> count
    std::map<std::string, std::map<std::string, uint64_t>>
        requests_by_method_;                           // model -> method -> count
    std::map<std::string, uint64_t> requests_total_;   // model -> count
    std::map<std::string, uint64_t> requests_success_; // model -> count
    std::map<std::string, uint64_t> requests_failure_; // model -> count
    std::map<std::string, uint64_t> backend_errors_;   // model -> count

    std::map<std::string, uint64_t> model_load_success_; // model -> count
    std::map<std::string, uint64_t> model_load_failure_; // model -> count

    // Histograms
    std::map<std::string, Histogram> queue_latency_histograms_;
    std::map<std::string, Histogram> infer_latency_histograms_;
    std::map<std::string, Histogram> total_latency_histograms_;
    std::map<std::string, Histogram> batch_size_histograms_;

    double nsToSeconds(uint64_t ns) const;
    void appendLine(std::string &out, const std::string &name,
                    const std::map<std::string, std::string> &labels, double value) const;
    void appendCounterLine(std::string &out, const std::string &name,
                           const std::map<std::string, std::string> &labels, uint64_t value) const;
    void appendHistogram(std::string &out, const std::string &name,
                         const std::map<std::string, std::string> &labels,
                         const Histogram &hist) const;
};
