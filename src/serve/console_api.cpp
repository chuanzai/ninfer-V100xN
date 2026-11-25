#include "serve/console_api.h"

#include <httplib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <limits>
#include <sstream>

namespace ninfer::serve {
namespace {

// Run a shell command and capture stdout. Console telemetry is deliberately
// shell-based: nvidia-smi/free/awk are already on the GPU host and keep this
// module independent of NVML and CUDA. Never call with user-supplied input.
std::string capture(const std::string& command) {
    std::array<char, 256> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return output;
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);
    return output;
}

double to_double(const std::string& text) {
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (used == 0) {
            return 0.0;
        }
        return std::isfinite(value) ? value : 0.0;
    } catch (...) {
        return 0.0;
    }
}

double json_number(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_number()) {
        return 0.0;
    }
    return object.at(key).get<double>();
}

int json_int(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        return 0;
    }
    return object.at(key).get<int>();
}

std::string json_string(const nlohmann::json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        return {};
    }
    return object.at(key).get<std::string>();
}

// PCIe throughput (KB/s per GPU) is already sampled by gpu-monitor.service on
// 127.0.0.1:80 via NVML. Read it rather than duplicate NVML here; a failure just
// leaves the two rates at zero and the console stays usable.
void read_pcie_throughput(double& rx_mb, double& tx_mb) {
    rx_mb = tx_mb = 0;
    try {
        httplib::Client client("127.0.0.1", 80);
        client.set_connection_timeout(0, 300000); // 300 ms
        client.set_read_timeout(0, 500000);        // 500 ms
        const auto response = client.Get("/api/data");
        if (!response || response->status != 200) {
            return;
        }
        const auto data = nlohmann::json::parse(response->body);
        if (!data.contains("gpus") || !data.at("gpus").is_array()) {
            return;
        }
        for (const auto& gpu : data.at("gpus")) {
            if (!gpu.is_object()) {
                continue;
            }
            const auto rx = gpu.value("pcie_rx", nlohmann::json());
            const auto tx = gpu.value("pcie_tx", nlohmann::json());
            if (rx.is_number()) {
                rx_mb += rx.get<double>() / 1024.0; // KB/s -> MB/s
            }
            if (tx.is_number()) {
                tx_mb += tx.get<double>() / 1024.0;
            }
        }
    } catch (...) {
        // monitor service down or payload changed: keep zero, do not fail overview.
    }
}

double now_seconds() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

} // namespace

ConsoleApi::ConsoleApi(std::string log_path) : log_path_(std::move(log_path)) {}

void ConsoleApi::refresh_log() {
    if (log_path_.empty()) {
        return;
    }
    std::ifstream probe(log_path_, std::ios::binary | std::ios::ate);
    if (!probe.good()) {
        return;
    }
    const std::size_t size = static_cast<std::size_t>(probe.tellg());
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_loaded_ && size == log_size_) {
        return;
    }
    const bool full_scan = !log_loaded_ || size < log_size_;
    const std::size_t offset = full_scan ? 0 : log_size_;
    if (full_scan) {
        requests_.clear();
        totals_ = LogTotals{};
        throughput_decode_.clear();
        throughput_prefill_.clear();
    }
    probe.close();

    std::ifstream input(log_path_, std::ios::binary);
    if (!input.good()) {
        return;
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::string line;
    bool first_line = true;
    while (std::getline(input, line)) {
        if (offset != 0 && first_line) {
            first_line = false; // partial line left from the previous read
            continue;
        }
        first_line = false;
        if (line.empty()) {
            continue;
        }
        nlohmann::json event;
        try {
            event = nlohmann::json::parse(line);
        } catch (...) {
            continue;
        }
        const std::string kind = json_string(event, "event");
        if (kind == "throughput") {
            const auto rates = event.value("throughput_tokens_per_second", nlohmann::json::object());
            throughput_decode_.push_back(json_number(rates, "decode"));
            throughput_prefill_.push_back(json_number(rates, "prefill"));
            if (throughput_decode_.size() > 120) {
                throughput_decode_.erase(throughput_decode_.begin());
                throughput_prefill_.erase(throughput_prefill_.begin());
            }
            continue;
        }
        if (kind == "request_error" || kind == "request_rejected") {
            totals_.errors += 1;
            continue;
        }
        if (kind != "request_done") {
            continue;
        }

        const auto req = event.value("request", nlohmann::json::object());
        const auto res = event.value("result", nlohmann::json::object());
        const auto timings = event.value("timings_seconds", nlohmann::json::object());
        const auto spec = event.value("speculative", nlohmann::json::object());

        RequestRow row;
        row.id       = std::to_string(json_int(event, "request_id"));
        row.at       = json_number(event, "timestamp_unix_ms") / 1000.0;
        row.protocol = json_string(req, "protocol");
        row.prompt_tokens  = json_int(res, "prompt_tokens");
        row.reused_tokens  = json_int(res, "prefix_cache_hit_tokens");
        row.output_tokens  = json_int(res, "completion_tokens");
        row.ttft_ms        = json_number(timings, "ttft") * 1000.0;
        row.prefill_ms     = json_number(timings, "prefill") * 1000.0;
        row.decode_ms      = json_number(timings, "decode") * 1000.0;
        row.total_s        = json_number(timings, "total");
        row.decode_tps     = row.decode_ms > 0 ? row.output_tokens * 1000.0 / row.decode_ms : 0.0;
        const int fresh    = std::max(0, row.prompt_tokens - row.reused_tokens);
        row.prefill_tps    = row.prefill_ms > 0 ? fresh * 1000.0 / row.prefill_ms : 0.0;
        row.accept_rate    = json_int(spec, "drafted_tokens") > 0
                                 ? static_cast<double>(json_int(spec, "accepted_tokens")) /
                                       static_cast<double>(json_int(spec, "drafted_tokens"))
                                 : 0.0;
        row.finish         = json_string(res, "finish_reason");

        if (totals_.requests == 0 && totals_.since == 0) {
            totals_.since = row.at;
        }
        totals_.requests += 1;
        totals_.prompt_tokens += row.prompt_tokens;
        totals_.reused += row.reused_tokens;
        totals_.output_tokens += row.output_tokens;
        totals_.prefill_ms += row.prefill_ms;
        totals_.decode_ms += row.decode_ms;
        requests_.push_back(std::move(row));
        if (requests_.size() > 500) {
            requests_.pop_front();
        }
    }
    log_size_  = size;
    log_loaded_ = true;
}

void ConsoleApi::sample_host() {
    // Throttle: capture() spawns several shells (with a 200 ms sleep for CPU
    // deltas). Polling the console every 3 s must not fork that storm per call.
    // Called without the lock held.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!history_.empty() && now_seconds() - history_.back().at < 2.0) {
            return;
        }
    }

    HostSample sample;
    sample.at = now_seconds();

    // nvidia-smi: index,name,mem.used,mem.total,util,temp,power.draw,power.limit
    const std::string smi = capture(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,"
        "power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null");
    double used = 0, total = 0, power = 0;
    std::istringstream gpu_lines(smi);
    std::string gpu_line;
    while (std::getline(gpu_lines, gpu_line)) {
        std::vector<std::string> parts;
        std::istringstream split(gpu_line);
        std::string part;
        while (std::getline(split, part, ',')) {
            parts.push_back(part);
        }
        if (parts.size() < 6) {
            continue;
        }
        const double card_used  = to_double(parts[1]);
        const double card_total = to_double(parts[2]);
        used += card_used;
        total += card_total;
        power += to_double(parts[5]);
        sample.gpu_util = std::max(sample.gpu_util, to_double(parts[3]));
        sample.gpu_temp = std::max(sample.gpu_temp, to_double(parts[4]));
    }
    sample.gpu_mem_used  = used;
    sample.gpu_mem_total = total;
    sample.gpu_power     = power;

    // free -b: Mem: total used free ...
    const std::string memory = capture("free -b 2>/dev/null | awk '/^Mem:/{print $2, $3}'");
    std::istringstream mem_stream(memory);
    double mem_total = 0, mem_used = 0;
    mem_stream >> mem_total >> mem_used;
    sample.ram_pct = mem_total > 0 ? mem_used * 100.0 / mem_total : 0;

    // CPU and disk deltas over a short window. /proc is Linux-only; on other
    // hosts these stay at zero rather than failing the whole overview.
    const std::string cpu_before = capture("awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;print t,$5+$6}' /proc/stat 2>/dev/null");
    const std::string disk_before = capture("awk '{r+=$6;w+=$10}END{print r*512,w*512}' /proc/diskstats 2>/dev/null");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::string cpu_after = capture("awk '/^cpu /{t=0;for(i=2;i<=NF;i++)t+=$i;print t,$5+$6}' /proc/stat 2>/dev/null");
    const std::string disk_after = capture("awk '{r+=$6;w+=$10}END{print r*512,w*512}' /proc/diskstats 2>/dev/null");

    std::istringstream c0(cpu_before), c1(cpu_after);
    double t0 = 0, i0 = 0, t1 = 0, i1 = 0;
    c0 >> t0 >> i0;
    c1 >> t1 >> i1;
    if (t1 > t0) {
        sample.cpu_pct = 100.0 * (1.0 - (i1 - i0) / (t1 - t0));
    }

    std::istringstream d0(disk_before), d1(disk_after);
    double r0 = 0, w0 = 0, r1 = 0, w1 = 0;
    d0 >> r0 >> w0;
    d1 >> r1 >> w1;
    sample.disk_read_mb  = std::max(0.0, r1 - r0) / 1048576.0 / 0.2;
    sample.disk_write_mb = std::max(0.0, w1 - w0) / 1048576.0 / 0.2;
    read_pcie_throughput(sample.pcie_rx_mb, sample.pcie_tx_mb);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.push_back(sample);
        if (history_.size() > 90) {
            history_.pop_front();
        }
    }
}

// Callers hold mutex_. These helpers must not lock it again: std::mutex is
// non-recursive, and overview() already owns the lock while assembling fields.
nlohmann::json ConsoleApi::machine_json() const {
    const HostSample latest = history_.empty() ? HostSample{} : history_.back();
    const std::string host = capture("hostname 2>/dev/null");
    const std::string threads = capture("nproc 2>/dev/null");
    const std::string gpus = capture(
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
        "temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null");

    nlohmann::json gpu_array = nlohmann::json::array();
    std::istringstream lines(gpus);
    std::string line;
    while (std::getline(lines, line)) {
        std::vector<std::string> parts;
        std::istringstream split(line);
        std::string part;
        while (std::getline(split, part, ',')) {
            parts.push_back(part);
        }
        if (parts.size() < 11) {
            continue;
        }
        gpu_array.push_back({
            {"index", to_double(parts[0])},
            {"name", parts.size() > 1 ? parts[1] : ""},
            {"vram_used_mib", to_double(parts[1 + 1])},
            {"vram_total_mib", to_double(parts[2 + 1])},
            {"util_pct", to_double(parts[3 + 1])},
            {"temp_c", to_double(parts[4 + 1])},
            {"power_w", to_double(parts[5 + 1])},
            {"power_limit_w", to_double(parts[6 + 1])},
            {"pcie_gen", static_cast<int>(to_double(parts[7 + 1]))},
            {"pcie_gen_max", static_cast<int>(to_double(parts[8 + 1]))},
            {"pcie_width", static_cast<int>(to_double(parts[9 + 1]))},
            {"pcie_width_max", static_cast<int>(to_double(parts[10 + 1]))},
        });
    }

    double pcie_gen = 0, pcie_gen_max = 0, pcie_width = 0, pcie_width_max = 0;
    if (!gpu_array.empty()) {
        const auto& first = gpu_array.front();
        pcie_gen       = json_number(first, "pcie_gen");
        pcie_gen_max   = json_number(first, "pcie_gen_max");
        pcie_width     = json_number(first, "pcie_width");
        pcie_width_max = json_number(first, "pcie_width_max");
    }

    std::string mem_line = capture("free -b 2>/dev/null | awk '/^Mem:/{print $2, $3}'");
    std::istringstream mem_stream(mem_line);
    double mem_total = 0, mem_used = 0;
    mem_stream >> mem_total >> mem_used;

    return {
        {"host", host.empty() ? "local" : host},
        {"gpus", gpu_array},
        {"ram", {{"total_bytes", mem_total}, {"used_bytes", mem_used}}},
        {"gpu_util", latest.gpu_util},
        {"gpu_mem_used", latest.gpu_mem_used},
        {"gpu_mem_total", latest.gpu_mem_total},
        {"gpu_temp", latest.gpu_temp},
        {"gpu_power", latest.gpu_power},
        {"gpu_power_limit", gpu_array.empty() ? 0 : json_number(gpu_array.front(), "power_limit_w")},
        {"gpu_pcie_gen", pcie_gen},
        {"gpu_pcie_gen_max", pcie_gen_max},
        {"gpu_pcie_width", pcie_width},
        {"gpu_pcie_width_max", pcie_width_max},
        {"gpu_pcie_rx_mb", latest.pcie_rx_mb},
        {"gpu_pcie_tx_mb", latest.pcie_tx_mb},
        {"gpu_pcie_bottleneck", gpu_array.empty() ? nlohmann::json() : gpu_array.front()},
        {"pcie", gpu_array.empty() ? nlohmann::json() : gpu_array.front()},
        {"cores", nullptr},
        {"threads", threads.empty() ? nullptr : nlohmann::json(threads)},
    };
}

nlohmann::json ConsoleApi::history_json() const {
    nlohmann::json result;
    auto push = [&](const char* key, auto extractor) {
        nlohmann::json values = nlohmann::json::array();
        for (const auto& sample : history_) {
            values.push_back(extractor(sample));
        }
        result[key] = std::move(values);
    };
    push("gpu_util", [](const HostSample& s) { return s.gpu_util; });
    push("gpu_mem_used", [](const HostSample& s) { return s.gpu_mem_used; });
    push("gpu_mem_total", [](const HostSample& s) { return s.gpu_mem_total; });
    push("gpu_temp", [](const HostSample& s) { return s.gpu_temp; });
    push("gpu_power", [](const HostSample& s) { return s.gpu_power; });
    push("cpu_pct", [](const HostSample& s) { return s.cpu_pct; });
    push("ram_pct", [](const HostSample& s) { return s.ram_pct; });
    push("disk_read_mb", [](const HostSample& s) { return s.disk_read_mb; });
    push("disk_write_mb", [](const HostSample& s) { return s.disk_write_mb; });
    push("pcie_rx_mb", [](const HostSample& s) { return s.pcie_rx_mb; });
    push("pcie_tx_mb", [](const HostSample& s) { return s.pcie_tx_mb; });
    result["tok_s"] = throughput_decode_;
    result["prefill_tok_s_mean"] = throughput_prefill_;
    return result;
}

nlohmann::json ConsoleApi::overview(const ServeOptions& options,
                                    const ConsoleEngineStats& engine) {
    refresh_log();
    sample_host();

    std::lock_guard<std::mutex> lock(mutex_);

    nlohmann::json requests = nlohmann::json::array();
    for (auto it = requests_.rbegin(); it != requests_.rend(); ++it) {
        const auto& r = *it;
        requests.push_back({
            {"id", r.id}, {"at", r.at}, {"time", r.at}, {"protocol", r.protocol},
            {"prompt_tokens", r.prompt_tokens}, {"reused_tokens", r.reused_tokens},
            {"reused", r.reused_tokens}, {"output_tokens", r.output_tokens},
            {"ttft_ms", r.ttft_ms}, {"prefill_ms", r.prefill_ms}, {"prompt_ms", r.prefill_ms},
            {"decode_ms", r.decode_ms}, {"total_s", r.total_s}, {"duration_s", r.total_s},
            {"decode_tps", r.decode_tps}, {"decode_tok_s", r.decode_tps},
            {"prefill_tps", r.prefill_tps}, {"accept_rate", r.accept_rate},
            {"finish", r.finish}, {"hit_rate", nullptr}, {"pcie_share", nullptr},
        });
    }

    nlohmann::json totals = {
        {"requests", totals_.requests}, {"errors", totals_.errors},
        {"prompt_tokens", totals_.prompt_tokens}, {"reused", totals_.reused},
        {"output_tokens", totals_.output_tokens}, {"prefill_ms", totals_.prefill_ms},
        {"prompt_ms", totals_.prefill_ms}, {"decode_ms", totals_.decode_ms},
        {"since", totals_.since},
    };

    const auto last = requests_.empty() ? RequestRow{} : requests_.back();
    const bool active = engine.running_requests > 0 || engine.prefilling_requests > 0;
    const std::string state = !engine.attached ? "error"
                              : engine.residency_state != "gpu" ? "unloaded"
                              : engine.prefilling_requests > 0 && engine.running_requests == 0 ? "reading"
                              : active ? "generating" : "idle";

    nlohmann::json live = {
        {"state", state}, {"running", engine.running_requests},
        {"prefilling", engine.prefilling_requests}, {"waiting", engine.waiting_requests},
        {"queued", engine.waiting_requests},
        {"prompt_tokens", active ? last.prompt_tokens : 0},
        {"prompt_total", active ? last.prompt_tokens : 0},
        {"reused", active ? last.reused_tokens : 0},
        {"prompt_read", active ? std::max(0, last.prompt_tokens - last.reused_tokens) : 0},
        {"generated", state == "generating" ? last.output_tokens : 0},
        {"tok_s", state == "generating" ? nlohmann::json(last.decode_tps) : nlohmann::json(nullptr)},
        {"prefill_tok_s_mean", state == "reading" ? nlohmann::json(last.prefill_tps) : nlohmann::json(nullptr)},
    };

    const int max_context = static_cast<int>(options.max_context);
    nlohmann::json health = {
        {"service", "ninfer"}, {"model", engine.model_id},
        {"max_context", options.max_context}, {"images", options.enable_vision},
        {"api_key", !options.api_key.empty()}, {"loaded", engine.attached},
        {"residency", engine.residency_state}, {"max_concurrency", options.max_concurrency},
        {"tp", options.tp},
    };
    nlohmann::json residency = {
        {"state", engine.residency_state}, {"retained_host_bytes", engine.retained_host_bytes},
    };
    nlohmann::json engine_json = {
        {"model", engine.model_id}, {"max_context", options.max_context},
        {"images", options.enable_vision}, {"kv", "int8"}, {"tp", options.tp},
        {"spec", options.speculative.backend == SpeculativeBackend::Mtp ? "mtp" : nullptr},
        {"draft_tokens", options.speculative.draft_tokens},
    };

    nlohmann::json cache = {
        {"enabled", options.prefix_cache_tokens > 0},
        {"budget_tokens", options.prefix_cache_tokens},
        {"archive", options.prefix_archive_path},
        {"archive_max_mib", options.prefix_archive_max_bytes / (1024ULL * 1024ULL)},
        {"requests", totals_.requests}, {"reused_tokens", totals_.reused},
        {"prompt_tokens", totals_.prompt_tokens},
        {"note", "ninfer 使用前缀缓存与磁盘归档；这里展示提示词复用统计，没有 Strata 的会话停车槽位概念。"},
    };

    nlohmann::json context = {
        {"last_prompt_tokens", last.prompt_tokens}, {"last_reused_tokens", last.reused_tokens},
        {"last_output_tokens", last.output_tokens}, {"max_tokens", max_context},
        {"pct", max_context > 0 ? std::min(100.0, (last.prompt_tokens + last.output_tokens) * 100.0 / max_context) : 0},
        {"cached", last.prompt_tokens + last.output_tokens}, {"reading", 0}, {"answering", 0},
        {"free", max_context > 0 ? std::max(0.0, static_cast<double>(max_context) - last.prompt_tokens - last.output_tokens) : 0},
    };

    return {
        {"ts", now_seconds()}, {"upstream_ok", engine.attached}, {"base_url", ""},
        {"health", health}, {"residency", residency}, {"engine", engine_json},
        {"service", {{"flags", nlohmann::json::object()}}},
        {"machine", machine_json()}, {"history", history_json()},
        {"live", live}, {"series", {{"decode", throughput_decode_}, {"prefill", throughput_prefill_}}},
        {"requests", requests}, {"requests_kept", requests_.size()},
        {"totals", totals}, {"speed_window", nlohmann::json::object()},
        {"conversation_cache", cache}, {"context", context},
    };
}

nlohmann::json ConsoleApi::log_tail(int lines) const {
    nlohmann::json result = nlohmann::json::array();
    if (log_path_.empty()) {
        return {{"lines", result}};
    }
    std::ifstream input(log_path_);
    if (!input.good()) {
        return {{"lines", result}};
    }
    std::vector<std::string> window;
    std::string line;
    while (std::getline(input, line)) {
        window.push_back(line);
        if (static_cast<int>(window.size()) > lines) {
            window.erase(window.begin());
        }
    }
    for (const auto& entry : window) {
        result.push_back(entry);
    }
    return {{"lines", result}};
}

} // namespace ninfer::serve
