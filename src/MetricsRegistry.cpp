#include "MetricsRegistry.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace {

uint64_t readVmRssBytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string label;
            uint64_t kb = 0;
            iss >> label >> kb;
            return kb * 1024;
        }
    }
    return 0;
}

} // namespace

MetricsRegistry::MetricsRegistry() {}

void MetricsRegistry::recordInferRequest(const std::string &model, const std::string &method,
                                         int status_code, uint64_t queue_latency_ns,
                                         uint64_t infer_latency_ns, uint64_t total_latency_ns,
                                         uint64_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_model_name_ = model;

    // Track detailed requests by model, method, and status
    requests_total_[model]++;
    if (status_code >= 200 && status_code < 300) {
        requests_success_[model]++;
    } else {
        requests_failure_[model]++;
    }
    if (status_code == 500) {
        backend_errors_[model]++;
    }

    requests_by_status_[model][std::to_string(status_code)]++;
    requests_by_method_[model][method]++;

    // Record Latency Histograms
    auto &q_hist = queue_latency_histograms_[model];
    if (q_hist.buckets.empty()) {
        q_hist = Histogram(
            {0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0});
    }
    q_hist.observe(nsToSeconds(queue_latency_ns));

    auto &inf_hist = infer_latency_histograms_[model];
    if (inf_hist.buckets.empty()) {
        inf_hist = Histogram(
            {0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0});
    }
    inf_hist.observe(nsToSeconds(infer_latency_ns));

    auto &tot_hist = total_latency_histograms_[model];
    if (tot_hist.buckets.empty()) {
        tot_hist = Histogram(
            {0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0});
    }
    tot_hist.observe(nsToSeconds(total_latency_ns));

    // Record Batch Size Histogram
    if (batch_size > 0) {
        auto &b_hist = batch_size_histograms_[model];
        if (b_hist.buckets.empty()) {
            b_hist = Histogram({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0});
        }
        b_hist.observe(static_cast<double>(batch_size));
    }
}

void MetricsRegistry::recordModelLoadSuccess(const std::string &model, const std::string &backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_model_name_ = model;
    model_load_success_[model]++;
    (void)backend;
}

void MetricsRegistry::recordModelLoadFailure(const std::string &model, const std::string &backend) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_model_name_ = model;
    model_load_failure_[model]++;
    (void)backend;
}

void MetricsRegistry::setSchedulerMetrics(const SchedulerMetricsSnapshot &snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    scheduler_metrics_ = snapshot;
}

SchedulerMetricsSnapshot MetricsRegistry::schedulerMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return scheduler_metrics_;
}

double MetricsRegistry::nsToSeconds(uint64_t ns) const {
    return static_cast<double>(ns) / 1'000'000'000.0;
}

void MetricsRegistry::appendLine(std::string &out, const std::string &name,
                                 const std::map<std::string, std::string> &labels,
                                 double value) const {
    out += name;
    if (!labels.empty()) {
        out += '{';
        bool first = true;
        for (const auto &[key, val] : labels) {
            if (!first) {
                out += ',';
            }
            first = false;
            out += key;
            out += "=\"";
            out += val;
            out += '"';
        }
        out += '}';
    }
    out += ' ';
    // Format double to avoid scientific notation
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string val_str = oss.str();
    // Trim trailing zeros to keep Prometheus clean
    val_str.erase(val_str.find_last_not_of('0') + 1, std::string::npos);
    if (val_str.back() == '.') {
        val_str.pop_back();
    }
    out += val_str;
    out += '\n';
}

void MetricsRegistry::appendCounterLine(std::string &out, const std::string &name,
                                        const std::map<std::string, std::string> &labels,
                                        uint64_t value) const {
    appendLine(out, name, labels, static_cast<double>(value));
}

void MetricsRegistry::appendHistogram(std::string &out, const std::string &name,
                                      const std::map<std::string, std::string> &labels,
                                      const Histogram &hist) const {
    for (size_t i = 0; i < hist.buckets.size(); ++i) {
        auto bucket_labels = labels;
        std::ostringstream oss;
        oss << hist.buckets[i];
        bucket_labels["le"] = oss.str();
        appendCounterLine(out, name + "_bucket", bucket_labels, hist.counts[i]);
    }
    auto inf_labels = labels;
    inf_labels["le"] = "+Inf";
    appendCounterLine(out, name + "_bucket", inf_labels, hist.counts.back());

    appendLine(out, name + "_sum", labels, hist.sum);
    appendCounterLine(out, name + "_count", labels, hist.count);
}

void MetricsRegistry::recordProcessMemory() {
    // Process-wide, handled in renderMetrics
}

std::string MetricsRegistry::renderMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string out;
    const auto modelLabel = std::map<std::string, std::string>{{"model", active_model_name_}};

    // 1. Scheduler Metrics
    out += "# HELP neuriplo_scheduler_queue_depth Current scheduler queue depth\n";
    out += "# TYPE neuriplo_scheduler_queue_depth gauge\n";
    appendLine(out, "neuriplo_scheduler_queue_depth", modelLabel,
               static_cast<double>(scheduler_metrics_.queue_depth));

    out += "# HELP neuriplo_scheduler_in_flight_requests Current in-flight requests\n";
    out += "# TYPE neuriplo_scheduler_in_flight_requests gauge\n";
    appendLine(out, "neuriplo_scheduler_in_flight_requests", modelLabel,
               static_cast<double>(scheduler_metrics_.in_flight));

    out += "# HELP neuriplo_scheduler_requests_accepted_total Total accepted requests\n";
    out += "# TYPE neuriplo_scheduler_requests_accepted_total counter\n";
    appendCounterLine(out, "neuriplo_scheduler_requests_accepted_total", modelLabel,
                      scheduler_metrics_.requests_accepted);

    out += "# HELP neuriplo_scheduler_requests_rejected_total Total rejected requests\n";
    out += "# TYPE neuriplo_scheduler_requests_rejected_total counter\n";
    appendCounterLine(out, "neuriplo_scheduler_requests_rejected_total", modelLabel,
                      scheduler_metrics_.requests_rejected);

    out += "# HELP neuriplo_scheduler_requests_timed_out_total Total timed-out requests\n";
    out += "# TYPE neuriplo_scheduler_requests_timed_out_total counter\n";
    appendCounterLine(out, "neuriplo_scheduler_requests_timed_out_total", modelLabel,
                      scheduler_metrics_.requests_timed_out);

    out += "# HELP neuriplo_scheduler_queue_wait_seconds Total seconds spent waiting in queue\n";
    out += "# TYPE neuriplo_scheduler_queue_wait_seconds counter\n";
    appendLine(out, "neuriplo_scheduler_queue_wait_seconds", modelLabel,
               nsToSeconds(scheduler_metrics_.queue_wait_ns_total));

    out += "# HELP neuriplo_scheduler_execution_seconds Total seconds spent in execution\n";
    out += "# TYPE neuriplo_scheduler_execution_seconds counter\n";
    appendLine(out, "neuriplo_scheduler_execution_seconds", modelLabel,
               nsToSeconds(scheduler_metrics_.execution_ns_total));

    out += "# HELP neuriplo_scheduler_request_total_seconds Total end-to-end request seconds\n";
    out += "# TYPE neuriplo_scheduler_request_total_seconds counter\n";
    appendLine(out, "neuriplo_scheduler_request_total_seconds", modelLabel,
               nsToSeconds(scheduler_metrics_.total_ns_total));

    out += "# HELP neuriplo_scheduler_batches_formed_total Total batches formed\n";
    out += "# TYPE neuriplo_scheduler_batches_formed_total counter\n";
    appendCounterLine(out, "neuriplo_scheduler_batches_formed_total", modelLabel,
                      scheduler_metrics_.batches_formed);

    out += "# HELP neuriplo_scheduler_batched_requests_total Total requests processed in batches\n";
    out += "# TYPE neuriplo_scheduler_batched_requests_total counter\n";
    appendCounterLine(out, "neuriplo_scheduler_batched_requests_total", modelLabel,
                      scheduler_metrics_.batched_requests_total);

    out += "# HELP neuriplo_scheduler_batch_formation_seconds "
           "Total seconds spent forming batches\n";
    out += "# TYPE neuriplo_scheduler_batch_formation_seconds counter\n";
    appendLine(out, "neuriplo_scheduler_batch_formation_seconds", modelLabel,
               nsToSeconds(scheduler_metrics_.batch_formation_ns_total));

    out += "# HELP neuriplo_scheduler_batch_execution_seconds "
           "Total seconds spent in batch execution\n";
    out += "# TYPE neuriplo_scheduler_batch_execution_seconds counter\n";
    appendLine(out, "neuriplo_scheduler_batch_execution_seconds", modelLabel,
               nsToSeconds(scheduler_metrics_.batch_execution_ns_total));

    // 2. HTTP Interface Stats
    out += "# HELP neuriplo_http_infer_requests_total Total infer requests\n";
    out += "# TYPE neuriplo_http_infer_requests_total counter\n";
    uint64_t tot_requests = 0;
    if (requests_total_.count(active_model_name_)) {
        tot_requests = requests_total_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_http_infer_requests_total", modelLabel, tot_requests);

    out += "# HELP neuriplo_http_infer_requests_success_total "
           "Total successful infer requests\n";
    out += "# TYPE neuriplo_http_infer_requests_success_total counter\n";
    uint64_t tot_success = 0;
    if (requests_success_.count(active_model_name_)) {
        tot_success = requests_success_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_http_infer_requests_success_total", modelLabel, tot_success);

    out += "# HELP neuriplo_http_infer_requests_failure_total Total failed infer requests\n";
    out += "# TYPE neuriplo_http_infer_requests_failure_total counter\n";
    uint64_t tot_failure = 0;
    if (requests_failure_.count(active_model_name_)) {
        tot_failure = requests_failure_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_http_infer_requests_failure_total", modelLabel, tot_failure);

    // 3. Roadmap: Request count by model, method, and status
    out += "# HELP neuriplo_http_requests_total Total HTTP requests with method/status labels\n";
    out += "# TYPE neuriplo_http_requests_total counter\n";
    for (const auto &[model, statuses] : requests_by_status_) {
        for (const auto &[status, count] : statuses) {
            std::map<std::string, std::string> labels = {{"model", model}, {"status", status}};
            // Cross-reference with methods if available, otherwise general
            appendCounterLine(out, "neuriplo_http_requests_total", labels, count);
        }
    }

    // 4. Model Load Stats
    out += "# HELP neuriplo_model_load_success_total Total successful model loads\n";
    out += "# TYPE neuriplo_model_load_success_total counter\n";
    uint64_t load_succ = 0;
    if (model_load_success_.count(active_model_name_)) {
        load_succ = model_load_success_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_model_load_success_total", modelLabel, load_succ);

    out += "# HELP neuriplo_model_load_failure_total Total failed model loads\n";
    out += "# TYPE neuriplo_model_load_failure_total counter\n";
    uint64_t load_fail = 0;
    if (model_load_failure_.count(active_model_name_)) {
        load_fail = model_load_failure_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_model_load_failure_total", modelLabel, load_fail);

    // 5. Backend Error Counters
    out += "# HELP neuriplo_backend_errors_total Total backend execution failures\n";
    out += "# TYPE neuriplo_backend_errors_total counter\n";
    uint64_t be_errors = 0;
    if (backend_errors_.count(active_model_name_)) {
        be_errors = backend_errors_.at(active_model_name_);
    }
    appendCounterLine(out, "neuriplo_backend_errors_total", modelLabel, be_errors);

    // 6. Roadmap: Histograms
    out += "# HELP neuriplo_scheduler_queue_latency_seconds Distribution of queue wait times\n";
    out += "# TYPE neuriplo_scheduler_queue_latency_seconds histogram\n";
    if (queue_latency_histograms_.count(active_model_name_)) {
        appendHistogram(out, "neuriplo_scheduler_queue_latency_seconds", modelLabel,
                        queue_latency_histograms_.at(active_model_name_));
    } else {
        appendHistogram(out, "neuriplo_scheduler_queue_latency_seconds", modelLabel,
                        Histogram({0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
                                   5.0, 10.0}));
    }

    out += "# HELP neuriplo_scheduler_infer_latency_seconds Distribution of execution latencies\n";
    out += "# TYPE neuriplo_scheduler_infer_latency_seconds histogram\n";
    if (infer_latency_histograms_.count(active_model_name_)) {
        appendHistogram(out, "neuriplo_scheduler_infer_latency_seconds", modelLabel,
                        infer_latency_histograms_.at(active_model_name_));
    } else {
        appendHistogram(out, "neuriplo_scheduler_infer_latency_seconds", modelLabel,
                        Histogram({0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
                                   5.0, 10.0}));
    }

    out += "# HELP neuriplo_scheduler_total_latency_seconds Distribution of total end-to-end "
           "latencies\n";
    out += "# TYPE neuriplo_scheduler_total_latency_seconds histogram\n";
    if (total_latency_histograms_.count(active_model_name_)) {
        appendHistogram(out, "neuriplo_scheduler_total_latency_seconds", modelLabel,
                        total_latency_histograms_.at(active_model_name_));
    } else {
        appendHistogram(out, "neuriplo_scheduler_total_latency_seconds", modelLabel,
                        Histogram({0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
                                   5.0, 10.0}));
    }

    out += "# HELP neuriplo_scheduler_batch_size Distribution of formed batch sizes\n";
    out += "# TYPE neuriplo_scheduler_batch_size histogram\n";
    if (batch_size_histograms_.count(active_model_name_)) {
        appendHistogram(out, "neuriplo_scheduler_batch_size", modelLabel,
                        batch_size_histograms_.at(active_model_name_));
    } else {
        appendHistogram(out, "neuriplo_scheduler_batch_size", modelLabel,
                        Histogram({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}));
    }

    // 7. System Memory
    const auto memory_bytes = readVmRssBytes();
    out += "# HELP neuriplo_process_resident_memory_bytes Process resident memory in bytes\n";
    out += "# TYPE neuriplo_process_resident_memory_bytes gauge\n";
    appendLine(out, "neuriplo_process_resident_memory_bytes", {},
               static_cast<double>(memory_bytes));

    return out;
}
