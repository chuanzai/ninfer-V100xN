#pragma once

// Read-only data aggregator for the embedded management console. Kept free of
// CUDA / Engine includes on purpose: it can be syntax-checked and unit-tested
// on a non-GPU host, and the HTTP layer feeds it a plain snapshot of engine
// counters gathered from GenerationService's existing getters.

#include "ninfer/types.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

// Engine-side values the console needs. HttpServer fills this from
// GenerationService::runtime_stats()/model_residency() on every poll; no CUDA
// type appears in this structure so this file stays host-compilable.
struct ConsoleEngineStats {
    bool attached                    = false;
    std::string model_id;
    std::string residency_state      = "unknown";
    std::uint64_t retained_host_bytes = 0;
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t committed_decode_tokens = 0;
    std::uint32_t running_requests   = 0;
    std::uint32_t prefilling_requests = 0;
    std::uint32_t decode_ready_requests = 0;
    std::uint32_t waiting_requests   = 0;
};

class ConsoleApi {
public:
    explicit ConsoleApi(std::string log_path);

    // Full view model consumed by app.js renderMonitor(). Never throws: any
    // failing collector degrades to null/empty fields so the console stays up.
    [[nodiscard]] nlohmann::json overview(const ServeOptions& options,
                                          const ConsoleEngineStats& engine);

    // Last JSONL lines for the raw-log panel.
    [[nodiscard]] nlohmann::json log_tail(int lines) const;

private:
    struct HostSample {
        double at                = 0;
        double gpu_util          = 0;
        double gpu_mem_used      = 0;
        double gpu_mem_total     = 0;
        double gpu_temp          = 0;
        double gpu_power         = 0;
        double cpu_pct           = 0;
        double ram_pct           = 0;
        double disk_read_mb      = 0;
        double disk_write_mb     = 0;
        double pcie_rx_mb        = 0; // summed GPU PCIe receive, MB/s
        double pcie_tx_mb        = 0; // summed GPU PCIe transmit, MB/s
    };

    struct RequestRow {
        std::string id;
        double at                = 0;
        std::string protocol;
        int prompt_tokens        = 0;
        int reused_tokens        = 0;
        int output_tokens        = 0;
        double ttft_ms           = 0;
        double prefill_ms        = 0;
        double decode_ms         = 0;
        double total_s           = 0;
        double decode_tps        = 0;
        double prefill_tps       = 0;
        double accept_rate       = 0;
        std::string finish;
    };

    struct LogTotals {
        double requests      = 0;
        double errors        = 0;
        double prompt_tokens = 0;
        double reused        = 0;
        double output_tokens = 0;
        double prefill_ms    = 0;
        double decode_ms     = 0;
        double since         = 0;
    };

    void refresh_log();
    void sample_host();
    [[nodiscard]] nlohmann::json machine_json() const;
    [[nodiscard]] nlohmann::json history_json() const;

    std::string log_path_;
    mutable std::mutex mutex_;
    std::size_t log_size_       = 0;
    bool log_loaded_            = false;
    std::deque<RequestRow> requests_;
    LogTotals totals_;
    std::deque<HostSample> history_;
    std::vector<double> throughput_decode_;
    std::vector<double> throughput_prefill_;
};

} // namespace ninfer::serve
