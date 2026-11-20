#include "mobilellm/engine.h"
#include "mobilellm/model_loader.h"
#include "mobilellm/scheduler/memory_manager.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <future>
#include <sstream>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__) || defined(__ANDROID__)
#include <sys/utsname.h>
#endif

namespace mobilellm::benchmark {

struct BenchmarkConfig {
    std::string model_path;
    std::string prompt{"Explain the principles of aerodynamics in clear, simple terms."};
    uint32_t prompt_tokens_target{0}; // 0 = use prompt string as-is
    uint32_t max_tokens{64};
    uint32_t threads{4};
    uint32_t context_length{2048};
    float temperature{0.0f}; // deterministic by default
    uint32_t warmup_iterations{1};
    uint32_t measurement_iterations{3};
    std::string output_json_path;
    bool verbose{false};
};

struct IterationMetrics {
    uint32_t iteration_index{0};
    bool is_warmup{false};
    uint32_t prompt_tokens{0};
    uint32_t completion_tokens{0};
    double queue_wait_time_ms{0.0};
    double prefill_time_ms{0.0};
    double prefill_tokens_per_sec{0.0};
    double first_token_sampling_time_ms{0.0};
    double decode_time_ms{0.0};
    double decode_tokens_per_sec{0.0};
    double decode_compute_time_ms{0.0};
    double pure_decode_tokens_per_sec{0.0};
    double ttft_pure_ms{0.0};
    double ttft_e2e_ms{0.0};
    double total_duration_ms{0.0};
    double peak_rss_mb{0.0};
    std::vector<double> token_latencies_ms;
    double p50_latency_ms{0.0};
    double p90_latency_ms{0.0};
    double p95_latency_ms{0.0};
    double p99_latency_ms{0.0};
    double min_latency_ms{0.0};
    double max_latency_ms{0.0};
    double jitter_stddev_ms{0.0};
};

struct AggregatedStats {
    double mean{0.0};
    double stddev{0.0};
    double min{0.0};
    double max{0.0};
    double cv_percent{0.0}; // coefficient of variation
};

AggregatedStats compute_stats(const std::vector<double>& values) {
    AggregatedStats res;
    if (values.empty()) return res;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    res.mean = sum / values.size();
    res.min = *std::min_element(values.begin(), values.end());
    res.max = *std::max_element(values.begin(), values.end());

    if (values.size() > 1) {
        double variance = 0.0;
        for (double v : values) {
            variance += (v - res.mean) * (v - res.mean);
        }
        res.stddev = std::sqrt(variance / (values.size() - 1));
        if (res.mean > 0.0) {
            res.cv_percent = (res.stddev / res.mean) * 100.0;
        }
    }
    return res;
}

double compute_percentile(std::vector<double>& sorted_data, double percentile) {
    if (sorted_data.empty()) return 0.0;
    if (sorted_data.size() == 1) return sorted_data[0];
    double rank = (percentile / 100.0) * (sorted_data.size() - 1);
    size_t low = static_cast<size_t>(std::floor(rank));
    size_t high = static_cast<size_t>(std::ceil(rank));
    double weight = rank - low;
    return sorted_data[low] * (1.0 - weight) + sorted_data[high] * weight;
}

std::string get_system_arch() {
#if defined(__APPLE__)
    char buffer[128];
    size_t size = sizeof(buffer);
    if (sysctlbyname("hw.machine", buffer, &size, nullptr, 0) == 0) {
        return std::string(buffer);
    }
    return "darwin_arm64";
#elif defined(__ANDROID__)
    return "android_arm64";
#elif defined(__linux__)
    struct utsname u;
    if (uname(&u) == 0) return std::string(u.machine);
    return "linux_arch";
#else
    return "unknown_arch";
#endif
}

std::string get_os_info() {
#if defined(__APPLE__)
    return "macOS (Apple Darwin)";
#elif defined(__ANDROID__)
    return "Android NDK (Bionic)";
#elif defined(__linux__)
    return "Linux (GNU/POSIX)";
#else
    return "Generic OS";
#endif
}

} // namespace mobilellm::benchmark

void print_help() {
    std::cout << "MobileLLM Inference Benchmark Runner\n"
              << "Usage: mobilellm_benchmark --model <path> [options]\n\n"
              << "Options:\n"
              << "  --model <path>              Path to GGUF model file (required)\n"
              << "  --prompt <text>             Input prompt text for prefill\n"
              << "  --prompt-length <n>         Synthesize prompt with approximately N tokens\n"
              << "  --max-tokens <n>            Maximum tokens to generate per iteration (default: 64)\n"
              << "  --threads <n>               Number of CPU worker threads (default: 4)\n"
              << "  --context <n>               Context window length (default: 2048)\n"
              << "  --temp <float>              Sampling temperature (default: 0.0 - greedy)\n"
              << "  --warmup <n>                Warmup iteration count (default: 1)\n"
              << "  --iterations <n>            Measured iteration count (default: 3)\n"
              << "  --output-json <path>        Export detailed benchmark results as JSON\n"
              << "  --verbose                   Print tokens and per-step progress\n"
              << "  --help                      Show this help message\n";
}

int main(int argc, char* argv[]) {
    using namespace mobilellm;
    using namespace mobilellm::benchmark;

    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if (arg == "--prompt-length" && i + 1 < argc) {
            cfg.prompt_tokens_target = std::stoul(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            cfg.max_tokens = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = std::stoul(argv[++i]);
        } else if (arg == "--context" && i + 1 < argc) {
            cfg.context_length = std::stoul(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            cfg.temperature = std::stof(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iterations = std::stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.measurement_iterations = std::stoul(argv[++i]);
        } else if (arg == "--output-json" && i + 1 < argc) {
            cfg.output_json_path = argv[++i];
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model argument is required.\n";
        print_help();
        return 1;
    }

    // Synthesize prompt of target length if requested
    if (cfg.prompt_tokens_target > 0) {
        std::string base_pattern = "The quick brown fox jumps over the lazy dog. Local LLM inference on mobile devices requires low memory bandwidth and high compute efficiency. ";
        std::string repeated;
        while (repeated.size() < cfg.prompt_tokens_target * 4) {
            repeated += base_pattern;
        }
        cfg.prompt = repeated;
    }

    std::cout << "================================================================================\n"
              << "                 MobileLLM Inference Performance Benchmark                      \n"
              << "================================================================================\n"
              << "System OS:     " << get_os_info() << "\n"
              << "Architecture:  " << get_system_arch() << "\n"
              << "Model Path:    " << cfg.model_path << "\n"
              << "Threads:       " << cfg.threads << "\n"
              << "Max Context:   " << cfg.context_length << "\n"
              << "Max Tokens:    " << cfg.max_tokens << "\n"
              << "Temperature:   " << cfg.temperature << "\n"
              << "Warmup Runs:   " << cfg.warmup_iterations << "\n"
              << "Measured Runs: " << cfg.measurement_iterations << "\n"
              << "--------------------------------------------------------------------------------\n";

    // 1. Inspect GGUF Model File
    auto inspect_res = ModelLoader::inspect_file(cfg.model_path);
    if (!inspect_res.is_ok()) {
        std::cerr << "Failed to inspect model file: " << inspect_res.status().message << "\n";
        return 1;
    }
    const auto& meta = inspect_res.value();
    std::cout << "GGUF Architecture: " << meta.architecture << "\n"
              << "GGUF Version:      " << meta.gguf_version << "\n"
              << "Model File Size:   " << std::fixed << std::setprecision(2)
              << (meta.file_size_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "Tensor Count:      " << meta.tensor_count << "\n"
              << "Native Context:    " << meta.context_length << "\n"
              << "--------------------------------------------------------------------------------\n";

    // 2. Measure Model Loading Latency & Memory Footprint
    auto mem_before_load = scheduler::MemoryBudgetManager::query_process_memory();
    auto t_load_start = std::chrono::steady_clock::now();

    auto engine = create_engine();
    EngineConfig e_cfg;
    Status init_st = engine->initialize(e_cfg);
    if (!init_st.is_ok()) {
        std::cerr << "Failed to initialize engine: " << init_st.message << "\n";
        return 1;
    }

    ModelLoadConfig m_cfg;
    m_cfg.model_path = cfg.model_path;
    m_cfg.context_length = cfg.context_length;
    m_cfg.n_threads = cfg.threads;
    auto load_res = engine->load_model(m_cfg);
    if (!load_res.is_ok()) {
        std::cerr << "Failed to load model: " << load_res.status().message << "\n";
        return 1;
    }
    auto t_load_end = std::chrono::steady_clock::now();
    double model_load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    auto mem_after_load = scheduler::MemoryBudgetManager::query_process_memory();
    double model_rss_delta_mb = static_cast<double>(mem_after_load.current_rss_bytes - mem_before_load.current_rss_bytes) / (1024.0 * 1024.0);

    std::cout << "Model Loading Latency: " << std::fixed << std::setprecision(2) << model_load_ms << " ms\n"
              << "Process RSS Post-Load: " << (mem_after_load.current_rss_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "Estimated Model RSS:   " << (model_rss_delta_mb > 0.0 ? model_rss_delta_mb : 0.0) << " MB\n"
              << "--------------------------------------------------------------------------------\n";

    // 3. Create Session
    SessionConfig s_cfg;
    s_cfg.max_context_tokens = cfg.context_length;
    auto t_sess_start = std::chrono::steady_clock::now();
    auto sess_res = engine->create_session(load_res.value(), s_cfg);
    if (!sess_res.is_ok()) {
        std::cerr << "Failed to create session: " << sess_res.status().message << "\n";
        return 1;
    }
    auto t_sess_end = std::chrono::steady_clock::now();
    double session_create_ms = std::chrono::duration<double, std::milli>(t_sess_end - t_sess_start).count();
    std::cout << "Session Init Latency:  " << session_create_ms << " ms\n"
              << "--------------------------------------------------------------------------------\n\n";

    // 4. Benchmark Loop (Warmup + Measured Iterations)
    uint32_t total_runs = cfg.warmup_iterations + cfg.measurement_iterations;
    std::vector<IterationMetrics> all_metrics;

    for (uint32_t r = 0; r < total_runs; ++r) {
        bool is_warmup = (r < cfg.warmup_iterations);
        std::cout << (is_warmup ? "[WARMUP]  " : "[MEASURE] ")
                  << "Iteration " << (is_warmup ? (r + 1) : (r - cfg.warmup_iterations + 1))
                  << "/" << (is_warmup ? cfg.warmup_iterations : cfg.measurement_iterations) << "... " << std::flush;

        IterationMetrics iter;
        iter.iteration_index = r;
        iter.is_warmup = is_warmup;

        GenerationRequest req;
        req.prompt = cfg.prompt;
        req.sampling.max_tokens = cfg.max_tokens;
        req.sampling.temperature = cfg.temperature;

        std::promise<void> comp_promise;
        auto comp_future = comp_promise.get_future();

        Status final_status = Status::ok();
        GenerationMetrics runtime_metrics;
        std::vector<std::chrono::steady_clock::time_point> token_arrival_times;
        token_arrival_times.reserve(cfg.max_tokens + 4);

        auto t_enqueue = std::chrono::steady_clock::now();

        auto token_cb = [&](const TokenChunk& chunk) -> bool {
            auto now = std::chrono::steady_clock::now();
            token_arrival_times.push_back(now);
            if (cfg.verbose) {
                std::cout << chunk.text << std::flush;
            }
            return true;
        };

        auto complete_cb = [&](const Status& st, const GenerationMetrics& m) {
            final_status = st;
            runtime_metrics = m;
            comp_promise.set_value();
        };

        auto gen_res = engine->generate_async(sess_res.value(), req, token_cb, complete_cb);
        if (!gen_res.is_ok()) {
            std::cerr << "Generation dispatch failed: " << gen_res.status().message << "\n";
            return 1;
        }

        comp_future.wait();
        auto t_complete = std::chrono::steady_clock::now();

        if (!final_status.is_ok()) {
            std::cerr << "Generation failed with error: " << final_status.message << "\n";
            return 1;
        }

        // Process Timing Metrics
        iter.prompt_tokens = runtime_metrics.prompt_tokens;
        iter.completion_tokens = runtime_metrics.completion_tokens;
        iter.queue_wait_time_ms = runtime_metrics.queue_wait_time_ms;
        iter.prefill_time_ms = runtime_metrics.prefill_time_ms;
        iter.prefill_tokens_per_sec = runtime_metrics.prefill_tokens_per_second;
        iter.first_token_sampling_time_ms = runtime_metrics.first_token_sampling_time_ms;
        iter.decode_time_ms = runtime_metrics.decode_time_ms;
        iter.decode_tokens_per_sec = runtime_metrics.decode_tokens_per_second;
        iter.decode_compute_time_ms = runtime_metrics.decode_compute_time_ms;
        iter.pure_decode_tokens_per_sec = runtime_metrics.pure_decode_tokens_per_second;
        iter.ttft_pure_ms = runtime_metrics.pure_inference_ttft_ms > 0.0 ? runtime_metrics.pure_inference_ttft_ms : runtime_metrics.time_to_first_token_ms;
        iter.ttft_e2e_ms = runtime_metrics.end_to_end_ttft_ms > 0.0 ? runtime_metrics.end_to_end_ttft_ms : std::chrono::duration<double, std::milli>(token_arrival_times.empty() ? t_complete - t_enqueue : token_arrival_times.front() - t_enqueue).count();
        iter.total_duration_ms = std::chrono::duration<double, std::milli>(t_complete - t_enqueue).count();
        iter.peak_rss_mb = runtime_metrics.peak_memory_mb;

        if (!token_arrival_times.empty()) {
            // Per-token decode latencies for tokens 2..N
            std::vector<double> latencies;
            for (size_t i = 1; i < token_arrival_times.size(); ++i) {
                double tok_lat = std::chrono::duration<double, std::milli>(token_arrival_times[i] - token_arrival_times[i - 1]).count();
                latencies.push_back(tok_lat);
            }

            if (!latencies.empty()) {
                iter.token_latencies_ms = latencies;
                std::vector<double> sorted = latencies;
                std::sort(sorted.begin(), sorted.end());

                iter.min_latency_ms = sorted.front();
                iter.max_latency_ms = sorted.back();
                iter.p50_latency_ms = compute_percentile(sorted, 50.0);
                iter.p90_latency_ms = compute_percentile(sorted, 90.0);
                iter.p95_latency_ms = compute_percentile(sorted, 95.0);
                iter.p99_latency_ms = compute_percentile(sorted, 99.0);

                // Compute inter-token jitter (stddev of latencies)
                double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
                double mean_lat = sum / latencies.size();
                double var = 0.0;
                for (double l : latencies) var += (l - mean_lat) * (l - mean_lat);
                iter.jitter_stddev_ms = std::sqrt(var / latencies.size());
            }
        }

        std::cout << "Done: " << iter.completion_tokens << " tok generated | "
                  << "TTFT: " << std::fixed << std::setprecision(1) << iter.ttft_e2e_ms << " ms | "
                  << "Prefill: " << iter.prefill_tokens_per_sec << " tok/s | "
                  << "Decode: " << iter.decode_tokens_per_sec << " tok/s | "
                  << "Pure Decode: " << iter.pure_decode_tokens_per_sec << " tok/s | "
                  << "P50 Latency: " << iter.p50_latency_ms << " ms | "
                  << "P95 Latency: " << iter.p95_latency_ms << " ms\n";

        all_metrics.push_back(iter);
    }

    // 5. Aggregate Results Across Measured Iterations
    std::vector<double> ttfts, prefill_tps, decode_tps, pure_decode_tps, p50s, p95s, peak_mems;
    for (const auto& m : all_metrics) {
        if (!m.is_warmup) {
            ttfts.push_back(m.ttft_e2e_ms);
            prefill_tps.push_back(m.prefill_tokens_per_sec);
            decode_tps.push_back(m.decode_tokens_per_sec);
            pure_decode_tps.push_back(m.pure_decode_tokens_per_sec);
            p50s.push_back(m.p50_latency_ms);
            p95s.push_back(m.p95_latency_ms);
            peak_mems.push_back(m.peak_rss_mb);
        }
    }

    auto stat_ttft = compute_stats(ttfts);
    auto stat_prefill = compute_stats(prefill_tps);
    auto stat_decode = compute_stats(decode_tps);
    auto stat_pure_decode = compute_stats(pure_decode_tps);
    auto stat_p50 = compute_stats(p50s);
    auto stat_p95 = compute_stats(p95s);
    auto stat_mem = compute_stats(peak_mems);

    std::cout << "\n================================================================================\n"
              << "                        FINAL BENCHMARK SUMMARY                                 \n"
              << "================================================================================\n"
              << "Metric                   | Mean ± StdDev          | Min      | Max      | Noise (CV%)\n"
              << "-------------------------+------------------------+----------+----------+------------\n";

    auto print_row = [](const std::string& name, const AggregatedStats& s, const std::string& unit) {
        std::cout << std::left << std::setw(24) << name << " | "
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2) << s.mean << " ± "
                  << std::setw(6) << s.stddev << " " << std::left << std::setw(6) << unit << " | "
                  << std::right << std::setw(8) << s.min << " | "
                  << std::setw(8) << s.max << " | "
                  << std::setw(9) << s.cv_percent << "%\n";
    };

    print_row("Time to First Token", stat_ttft, "ms");
    print_row("Prefill Throughput", stat_prefill, "tok/s");
    print_row("Decode Throughput", stat_decode, "tok/s");
    print_row("Pure Decode Throughput", stat_pure_decode, "tok/s");
    print_row("Median (P50) Latency", stat_p50, "ms");
    print_row("P95 Tail Latency", stat_p95, "ms");
    print_row("Peak Process RSS", stat_mem, "MB");
    std::cout << "================================================================================\n";

    // 6. Output Machine-Readable JSON if requested
    if (!cfg.output_json_path.empty()) {
        std::ofstream jf(cfg.output_json_path);
        if (jf.is_open()) {
            jf << "{\n"
               << "  \"benchmark_version\": \"0.1.0-dev\",\n"
               << "  \"os\": \"" << get_os_info() << "\",\n"
               << "  \"arch\": \"" << get_system_arch() << "\",\n"
               << "  \"model\": {\n"
               << "    \"path\": \"" << cfg.model_path << "\",\n"
               << "    \"architecture\": \"" << meta.architecture << "\",\n"
               << "    \"file_size_mb\": " << (meta.file_size_bytes / (1024.0 * 1024.0)) << ",\n"
               << "    \"context_length\": " << cfg.context_length << "\n"
               << "  },\n"
               << "  \"configuration\": {\n"
               << "    \"threads\": " << cfg.threads << ",\n"
               << "    \"max_tokens\": " << cfg.max_tokens << ",\n"
               << "    \"temperature\": " << cfg.temperature << ",\n"
               << "    \"warmup_runs\": " << cfg.warmup_iterations << ",\n"
               << "    \"measured_runs\": " << cfg.measurement_iterations << "\n"
               << "  },\n"
               << "  \"model_loading\": {\n"
               << "    \"duration_ms\": " << model_load_ms << ",\n"
               << "    \"rss_delta_mb\": " << model_rss_delta_mb << ",\n"
               << "    \"session_init_ms\": " << session_create_ms << "\n"
               << "  },\n"
               << "  \"summary\": {\n"
               << "    \"ttft_mean_ms\": " << stat_ttft.mean << ",\n"
               << "    \"ttft_p50_ms\": " << stat_ttft.min << ",\n"
               << "    \"prefill_tps_mean\": " << stat_prefill.mean << ",\n"
               << "    \"decode_tps_mean\": " << stat_decode.mean << ",\n"
               << "    \"pure_decode_tps_mean\": " << stat_pure_decode.mean << ",\n"
               << "    \"p50_latency_ms\": " << stat_p50.mean << ",\n"
               << "    \"p95_latency_ms\": " << stat_p95.mean << ",\n"
               << "    \"peak_rss_mb\": " << stat_mem.mean << "\n"
               << "  },\n"
               << "  \"iterations\": [\n";

            for (size_t i = 0; i < all_metrics.size(); ++i) {
                const auto& it = all_metrics[i];
                jf << "    {\n"
                   << "      \"iteration\": " << it.iteration_index << ",\n"
                   << "      \"is_warmup\": " << (it.is_warmup ? "true" : "false") << ",\n"
                   << "      \"prompt_tokens\": " << it.prompt_tokens << ",\n"
                   << "      \"completion_tokens\": " << it.completion_tokens << ",\n"
                   << "      \"queue_wait_time_ms\": " << it.queue_wait_time_ms << ",\n"
                   << "      \"prefill_time_ms\": " << it.prefill_time_ms << ",\n"
                   << "      \"prefill_tps\": " << it.prefill_tokens_per_sec << ",\n"
                   << "      \"first_token_sampling_time_ms\": " << it.first_token_sampling_time_ms << ",\n"
                   << "      \"decode_time_ms\": " << it.decode_time_ms << ",\n"
                   << "      \"decode_tps\": " << it.decode_tokens_per_sec << ",\n"
                   << "      \"decode_compute_time_ms\": " << it.decode_compute_time_ms << ",\n"
                   << "      \"pure_decode_tps\": " << it.pure_decode_tokens_per_sec << ",\n"
                   << "      \"ttft_pure_ms\": " << it.ttft_pure_ms << ",\n"
                   << "      \"ttft_e2e_ms\": " << it.ttft_e2e_ms << ",\n"
                   << "      \"p50_latency_ms\": " << it.p50_latency_ms << ",\n"
                   << "      \"p90_latency_ms\": " << it.p90_latency_ms << ",\n"
                   << "      \"p95_latency_ms\": " << it.p95_latency_ms << ",\n"
                   << "      \"p99_latency_ms\": " << it.p99_latency_ms << ",\n"
                   << "      \"jitter_stddev_ms\": " << it.jitter_stddev_ms << ",\n"
                   << "      \"peak_rss_mb\": " << it.peak_rss_mb << "\n"
                   << "    }" << (i + 1 < all_metrics.size() ? "," : "") << "\n";
            }
            jf << "  ]\n"
               << "}\n";
            std::cout << "Exported machine-readable results to: " << cfg.output_json_path << "\n";
        }
    }

    engine->shutdown();
    return 0;
}

