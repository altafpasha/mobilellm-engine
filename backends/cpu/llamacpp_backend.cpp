#include "backends/cpu/llamacpp_backend.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LlamaCppBackend", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "LlamaCppBackend", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "LlamaCppBackend", __VA_ARGS__)
#else
#define LOGE(...)
#define LOGW(...)
#define LOGI(...)
#endif

#ifdef MOBILELLM_ENABLE_LLAMACPP
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath>
#include <thread>
#include <unistd.h>

#ifdef MOBILELLM_ENABLE_TENSOR_TRACE
// ---------------------------------------------------------------------------------------------
// OPT-IN DEBUG INSTRUMENTATION — compiled out unless -DMOBILELLM_ENABLE_TENSOR_TRACE is defined.
//
// This diagnostic path installs a llama.cpp `cb_eval` callback. When set, ggml_backend_sched
// stops dispatching the whole compute graph in one shot and instead executes it ONE NODE AT A
// TIME with a full `ggml_backend_synchronize()` after every node (see
// ggml_backend_sched_compute_splits in ggml-backend.cpp). For a 28-layer transformer decode
// graph (~200+ nodes) that turns a single graph dispatch into hundreds of dispatches per token
// and, together with a non-polling threadpool, into a sleep/wake storm. It must never be
// reachable from a release build, hence the compile-time gate.
// ---------------------------------------------------------------------------------------------
static FILE* g_tensor_trace_fp = nullptr;

static bool debug_eval_callback(struct ggml_tensor* t, bool ask, void* /*user_data*/) {
    if (ask) return true;
    if (!t || !t->name[0]) return true;

    const char* name = t->name;
    if (!strstr(name, "norm") && !strstr(name, "cur") && !strstr(name, "Cur") &&
        !strstr(name, "kqv") && !strstr(name, "ffn") && !strstr(name, "output") &&
        !strstr(name, "inp") && !strstr(name, "embd") && !strstr(name, "kq")) {
        return true;
    }

    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    bool is_host = (buf == nullptr) || ggml_backend_buffer_is_host(buf);
    std::vector<uint8_t> host_copy;
    const void* data_ptr = t->data;
    if (!is_host && buf != nullptr) {
        size_t nbytes = ggml_nbytes(t);
        host_copy.resize(nbytes);
        ggml_backend_tensor_get(t, host_copy.data(), 0, nbytes);
        data_ptr = host_copy.data();
    }

    if (data_ptr != nullptr && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {
        int64_t n_elements = ggml_nelements(t);
        float min_v = 1e30f;
        float max_v = -1e30f;
        double sum_v = 0.0;
        double sum_sq = 0.0;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

        for (int64_t i = 0; i < n_elements; ++i) {
            float val = (t->type == GGML_TYPE_F32)
                ? ((const float*)data_ptr)[i]
                : ggml_fp16_to_fp32(((const ggml_fp16_t*)data_ptr)[i]);

            if (i == 0) s0 = val;
            else if (i == 1) s1 = val;
            else if (i == 2) s2 = val;
            else if (i == 3) s3 = val;

            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
            sum_v += val;
            sum_sq += (double)val * (double)val;
        }
        double mean_v = n_elements > 0 ? (sum_v / n_elements) : 0.0;
        double l2 = std::sqrt(sum_sq);

        if (g_tensor_trace_fp) {
            fprintf(g_tensor_trace_fp, "%-30s [%4ld,%4ld,%2ld] L2=%14.6f min=%12.6f max=%12.6f mean=%12.6f s=[%.6f,%.6f,%.6f,%.6f]\n",
                    name, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], l2, min_v, max_v, mean_v, s0, s1, s2, s3);
            fflush(g_tensor_trace_fp);
        }
    }
    return true;
}
#endif // MOBILELLM_ENABLE_TENSOR_TRACE

static std::mutex g_llama_log_mutex;
static std::string g_llama_last_error;

// ---------------------------------------------------------------------------------------------
// Runtime facts captured from llama.cpp INFO logs. The dependency does not expose "how many
// layers actually landed on the GPU" or "how many graph splits the scheduler produced" through
// a public API at this revision, and a config value (n_gpu_layers) is intent, not evidence.
// Parsing the documented INFO lines is the only reliable way to verify real offload and to
// count CPU<->GPU graph boundaries on a physical device.
// ---------------------------------------------------------------------------------------------
struct LlamaRuntimeFacts {
    int32_t graph_nodes{-1};
    int32_t graph_nodes_tg{-1};
    int32_t graph_splits_pp{-1};
    int32_t graph_splits_tg{-1};
    int32_t offloaded_layers{-1};
    int32_t total_offloadable_layers{-1};
    std::string vulkan_matmul_path; // "mat_vec" | "mat_mat" | ""
    int32_t vulkan_subgroup_size{-1};
};

static std::mutex g_llama_facts_mutex;
static LlamaRuntimeFacts g_llama_facts;

static void reset_graph_facts() {
    std::lock_guard<std::mutex> lock(g_llama_facts_mutex);
    g_llama_facts.graph_nodes = -1;
    g_llama_facts.graph_nodes_tg = -1;
    g_llama_facts.graph_splits_pp = -1;
    g_llama_facts.graph_splits_tg = -1;
    g_llama_facts.offloaded_layers = -1;
    g_llama_facts.total_offloadable_layers = -1;
    g_llama_facts.vulkan_matmul_path.clear();
    g_llama_facts.vulkan_subgroup_size = -1;
}

static void capture_llama_log_facts(const char* text) {
    if (!text) return;
    std::lock_guard<std::mutex> lock(g_llama_facts_mutex);
    const char* p = nullptr;

    if ((p = strstr(text, "graph splits = ")) != nullptr) {
        int a = -1, b = -1, c = -1;
        if (sscanf(p, "graph splits = %d (with bs=%d), %d (with bs=1)", &a, &b, &c) == 3) {
            g_llama_facts.graph_splits_pp = a;
            g_llama_facts.graph_splits_tg = c;
        } else if (sscanf(p, "graph splits = %d", &a) == 1) {
            g_llama_facts.graph_splits_pp = a;
            g_llama_facts.graph_splits_tg = a;
        }
    }
    if ((p = strstr(text, "graph nodes")) != nullptr) {
        int a = -1, b = -1, c = -1;
        if (sscanf(p, "graph nodes  = %d (with bs=%d), %d (with bs=1)", &a, &b, &c) == 3) {
            g_llama_facts.graph_nodes = a;
            g_llama_facts.graph_nodes_tg = c;
        } else if (sscanf(p, "graph nodes  = %d", &a) == 1) {
            g_llama_facts.graph_nodes = a;
            g_llama_facts.graph_nodes_tg = a;
        }
    }
    if ((p = strstr(text, "layers to GPU")) != nullptr) {
        int off = -1, total = -1;
        const char* q = strstr(text, "offloaded ");
        if (q && sscanf(q, "offloaded %d/%d layers to GPU", &off, &total) == 2) {
            g_llama_facts.offloaded_layers = off;
            g_llama_facts.total_offloadable_layers = total;
        }
    }
    // Emitted by the local ggml-vulkan patch so the selected batch-1 matmul path is observable
    // from telemetry instead of inferred from the build configuration.
    if ((p = strstr(text, "[mobilellm] vulkan batch-1 matmul path = ")) != nullptr) {
        const char* v = p + strlen("[mobilellm] vulkan batch-1 matmul path = ");
        if (strncmp(v, "mat_vec", 7) == 0) {
            g_llama_facts.vulkan_matmul_path = "mat_vec";
        } else if (strncmp(v, "mat_mat", 7) == 0) {
            g_llama_facts.vulkan_matmul_path = "mat_mat";
        }
        int sg = -1;
        const char* s = strstr(v, "subgroup_size=");
        if (s && sscanf(s, "subgroup_size=%d", &sg) == 1) {
            g_llama_facts.vulkan_subgroup_size = sg;
        }
    }
}

static void llama_log_callback_capture(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!text) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lock(g_llama_log_mutex);
        std::string s(text);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (!s.empty()) {
            if (!g_llama_last_error.empty()) {
                g_llama_last_error += " | ";
            }
            g_llama_last_error += s;
        }
    }
    // Extract offload/split facts from informational output (cheap substring search).
    if (level == GGML_LOG_LEVEL_INFO || level == GGML_LOG_LEVEL_DEBUG) {
        capture_llama_log_facts(text);
    }
#if defined(__ANDROID__)
    int android_level = ANDROID_LOG_INFO;
    if (level == GGML_LOG_LEVEL_ERROR) {
        android_level = ANDROID_LOG_ERROR;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        android_level = ANDROID_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_DEBUG) {
        android_level = ANDROID_LOG_DEBUG;
    }
    __android_log_print(android_level, "llama.cpp", "%s", text);
#else
    std::fprintf(stderr, "[llama.cpp] %s", text);
#endif
}

// Classify a Vulkan device description into a coarse vendor bucket for telemetry.
static std::string classify_gpu_vendor(const std::string& device) {
    auto has = [&](const char* s) { return device.find(s) != std::string::npos; };
    if (has("Adreno") || has("Qualcomm")) return "Qualcomm";
    if (has("Mali") || has("Immortalis")) return "ARM";
    if (has("Apple")) return "Apple";
    if (has("Intel")) return "Intel";
    if (has("NVIDIA") || has("GeForce")) return "NVIDIA";
    if (has("AMD") || has("Radeon")) return "AMD";
    return {};
}

namespace mobilellm {

LlamaCppBackend::LlamaCppBackend() = default;

LlamaCppBackend::~LlamaCppBackend() {
    shutdown();
}

bool LlamaCppBackend::is_gpu_supported() noexcept {
#ifdef MOBILELLM_ENABLE_LLAMACPP
    return llama_supports_gpu_offload();
#else
    return false;
#endif
}

std::string LlamaCppBackend::gpu_device_description() const noexcept {
    return probe_gpu_device_description();
}

std::string LlamaCppBackend::probe_gpu_device_description() noexcept {
#ifdef MOBILELLM_ENABLE_LLAMACPP
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        const char* desc = ggml_backend_dev_description(dev);
        if (desc) return std::string(desc);
    }
#endif
    return {};
}

Status LlamaCppBackend::initialize(const BackendConfig& /*config*/) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        llama_log_set(llama_log_callback_capture, nullptr);
        llama_backend_init();
        initialized_ = true;
    }
    return Status::ok();
}

Status LlamaCppBackend::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, sess] : sessions_) {
        if (sess) {
            std::lock_guard<std::mutex> s_lock(sess->session_mutex);
            if (sess->raw_ctx) {
                // Wait for outstanding GPU operations before destroying dependent resources
                llama_synchronize(sess->raw_ctx);
                if (sess->threadpool) {
                    llama_detach_threadpool(sess->raw_ctx);
                    ggml_threadpool_free(sess->threadpool);
                    sess->threadpool = nullptr;
                }
                llama_free(sess->raw_ctx);
                sess->raw_ctx = nullptr;
            } else if (sess->threadpool) {
                ggml_threadpool_free(sess->threadpool);
                sess->threadpool = nullptr;
            }
        }
    }
    sessions_.clear();

    for (auto& [_, mdl] : models_) {
        if (mdl && mdl->raw_model) {
            llama_free_model(mdl->raw_model);
            mdl->raw_model = nullptr;
        }
    }
    models_.clear();

    if (initialized_) {
        llama_backend_free();
        initialized_ = false;
    }
    return Status::ok();
}

Result<BackendModelHandle> LlamaCppBackend::load_model(const ModelLoadConfig& config) noexcept {
    if (config.model_path.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Model path cannot be empty");
    }

    initialize(BackendConfig{});

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = config.use_mmap;
    mparams.use_mlock = config.use_mlock;

    int32_t target_gpu_layers = 0;
    std::string fallback_reason;

    if (config.preferred_backend == BackendType::AUTO) {
        // Runtime performance and thermal safety guard:
        // Probe GPU device description and vendor.
        bool gpu_avail = is_gpu_supported();
        std::string gpu_desc = probe_gpu_device_description();
        std::string gpu_vend = classify_gpu_vendor(gpu_desc);
        bool is_qualcomm_gpu = (gpu_vend == "Qualcomm" || gpu_desc.find("Adreno") != std::string::npos || gpu_desc.find("Qualcomm") != std::string::npos);

        // Check whether Qualcomm mul_mat_vec is explicitly enabled via environment variable or compile define
        const char* allow_env = getenv("GGML_VULKAN_ALLOW_QUALCOMM_MUL_MAT_VEC");
        bool allow_qualcomm_vec = (allow_env && (strcmp(allow_env, "1") == 0 || strcmp(allow_env, "true") == 0 || strcmp(allow_env, "TRUE") == 0));
#if defined(GGML_VULKAN_ALLOW_QUALCOMM_MUL_MAT_VEC)
        allow_qualcomm_vec = true;
#endif

        if (gpu_avail && is_qualcomm_gpu && !allow_qualcomm_vec) {
            // Qualcomm Adreno GPU (0x5143) hits mul_mm.comp fallback (~2 tok/s decode, severe heating).
            // Automatically fall back to CPU_NEON (4-6 threads) for performance and thermal safety.
            current_backend_.store(BackendType::CPU_NEON);
            target_gpu_layers = 0;
            uint32_t safe_threads = config.n_threads > 0 ? std::clamp(config.n_threads, 4u, 6u) : 4u;
            fallback_reason = "Qualcomm Adreno GPU (0x5143) hits mul_mm.comp fallback (~2 tok/s decode); AUTO routed to CPU_NEON (" +
                              std::to_string(safe_threads) + " threads) for thermal safety and 57x faster decode";
            LOGW("[MobileLLM Safety Guard] %s", fallback_reason.c_str());
        } else if (gpu_avail) {
            target_gpu_layers = (config.n_gpu_layers >= 0) ? config.n_gpu_layers : 99;
            current_backend_.store(BackendType::VULKAN);
            LOGI("AUTO backend selected Vulkan GPU: n_gpu_layers=%d", target_gpu_layers);
        } else {
            current_backend_.store(BackendType::CPU_NEON);
            target_gpu_layers = 0;
            LOGI("AUTO backend selected CPU NEON (no GPU available)");
        }
    } else if (config.preferred_backend == BackendType::VULKAN) {
        if (!is_gpu_supported()) {
            LOGW("Vulkan GPU backend requested, but no GPU device is supported on this platform. Falling back safely to CPU NEON.");
            current_backend_.store(BackendType::CPU_NEON);
            target_gpu_layers = 0;
            fallback_reason = "Explicit Vulkan requested but no Vulkan GPU supported; falling back to CPU_NEON";
        } else {
            target_gpu_layers = (config.n_gpu_layers >= 0) ? config.n_gpu_layers : 99;
            current_backend_.store(BackendType::VULKAN);
            LOGI("Vulkan GPU backend requested and verified: config.n_gpu_layers=%d -> setting n_gpu_layers=%d",
                 config.n_gpu_layers, target_gpu_layers);
        }
    } else {
        // CPU NEON strictly targets CPU: zero GPU layers offloaded
        target_gpu_layers = 0;
        current_backend_.store(BackendType::CPU_NEON);
        LOGI("CPU NEON backend requested: setting n_gpu_layers=0");
    }
    mparams.n_gpu_layers = target_gpu_layers;

    // Detect if model is located on emulated FUSE storage (e.g. /sdcard/ or /storage/emulated/).
    // On Android 11+ (API 30-37), mmap on FUSE storage triggers userspace daemon page-fault round-trips
    // during tensor access, collapsing inference throughput by up to 8x.
    // Forcing sequential read into RAM avoids this bottleneck completely.
    if (mparams.use_mmap) {
        if (config.model_path.find("/sdcard/") != std::string::npos ||
            config.model_path.find("/storage/emulated/") != std::string::npos ||
            config.model_path.rfind("/sdcard", 0) == 0 ||
            config.model_path.rfind("/storage/emulated", 0) == 0) {
            LOGI("Detected emulated FUSE storage path for '%s'. Disabling use_mmap to avoid FUSE page fault latency.",
                 config.model_path.c_str());
            mparams.use_mmap = false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_llama_log_mutex);
        g_llama_last_error.clear();
    }
    reset_graph_facts();

    const auto t_load_start = std::chrono::steady_clock::now();
    llama_model* raw = llama_load_model_from_file(config.model_path.c_str(), mparams);
    if (!raw && mparams.use_mmap) {
        LOGW("Failed to load model with mmap, retrying with use_mmap=false...");
        mparams.use_mmap = false;
        raw = llama_load_model_from_file(config.model_path.c_str(), mparams);
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_load_start).count();
    if (!raw) {
        std::string err_detail;
        {
            std::lock_guard<std::mutex> lock(g_llama_log_mutex);
            err_detail = g_llama_last_error;
        }
        std::string err_msg = "Failed to load model from: " + config.model_path;
        if (!err_detail.empty()) {
            err_msg += " (" + err_detail + ")";
        }
        return Status::error(StatusCode::FILE_NOT_FOUND, err_msg);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BackendModelHandle handle = next_model_id_++;
    auto loaded = std::make_shared<LoadedModel>();
    loaded->handle = handle;
    loaded->path = config.model_path;
    loaded->raw_model = raw;
    loaded->context_length = config.context_length;
    loaded->n_vocab = static_cast<uint32_t>(llama_n_vocab(raw));
    uint32_t model_threads = config.n_threads > 0 ? config.n_threads : 4;
    if (!fallback_reason.empty() && fallback_reason.find("AUTO routed to CPU_NEON") != std::string::npos) {
        model_threads = std::clamp(model_threads, 4u, 6u);
    }
    loaded->n_threads = model_threads;
    loaded->fallback_reason = fallback_reason;
    loaded->requested_gpu_layers = target_gpu_layers;
    int32_t total_model_layers = llama_n_layer(raw);
    int32_t act_layers = 0;
    if (target_gpu_layers > 0 && is_gpu_supported()) {
        act_layers = std::min(target_gpu_layers, total_model_layers + 1);
    }
    loaded->actual_gpu_layers = act_layers;
    loaded->backend_type = current_backend_.load();
    loaded->load_ms = load_ms;

    // Enumerate real GPU devices through the ggml backend registry. `actual_gpu_layers` above is
    // derived from the requested value; it is *not* evidence that tensors landed on the GPU.
    // The description/offload counters below are captured from the backend itself.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        const char* desc = ggml_backend_dev_description(dev);
        if (desc) loaded->gpu_device = desc;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) loaded->gpu_driver = ggml_backend_reg_name(reg);
        loaded->gpu_vendor = classify_gpu_vendor(loaded->gpu_device);
        break;
    }
    {
        std::lock_guard<std::mutex> lock(g_llama_facts_mutex);
        loaded->model_layers = llama_n_layer(raw);
        loaded->offloaded_layers = g_llama_facts.offloaded_layers;
        loaded->total_offloadable_layers = g_llama_facts.total_offloadable_layers;
    }

    models_[handle] = loaded;
    LOGI("Model loaded: handle=%llu, vocab=%u, total_layers=%d, requested_gpu_layers=%d, actual_gpu_layers=%d, threads=%u, load_ms=%.1f",
         (unsigned long long)handle, loaded->n_vocab, total_model_layers,
         target_gpu_layers, act_layers, loaded->n_threads, load_ms);
    LOGI("GPU evidence: device='%s' vendor='%s' backend='%s' offloaded=%d/%d",
         loaded->gpu_device.c_str(), loaded->gpu_vendor.c_str(), loaded->gpu_driver.c_str(),
         loaded->offloaded_layers, loaded->total_offloadable_layers);
    return handle;
}

Status LlamaCppBackend::unload_model(BackendModelHandle model) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(model);
    if (it == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model handle not found");
    }

    // Free any active sessions tied to this model
    for (auto s_it = sessions_.begin(); s_it != sessions_.end();) {
        if (s_it->second->model_handle == model) {
            std::lock_guard<std::mutex> s_lock(s_it->second->session_mutex);
            if (s_it->second->raw_ctx) {
                // Wait for outstanding GPU operations before destroying dependent resources
                llama_synchronize(s_it->second->raw_ctx);
                if (s_it->second->threadpool) {
                    llama_detach_threadpool(s_it->second->raw_ctx);
                    ggml_threadpool_free(s_it->second->threadpool);
                    s_it->second->threadpool = nullptr;
                }
                llama_free(s_it->second->raw_ctx);
                s_it->second->raw_ctx = nullptr;
            } else if (s_it->second->threadpool) {
                ggml_threadpool_free(s_it->second->threadpool);
                s_it->second->threadpool = nullptr;
            }
            s_it = sessions_.erase(s_it);
        } else {
            ++s_it;
        }
    }

    if (it->second->raw_model) {
        llama_free_model(it->second->raw_model);
    }
    models_.erase(it);
    return Status::ok();
}

Result<BackendSessionHandle> LlamaCppBackend::create_session(
    BackendModelHandle model,
    const SessionConfig& config
) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(model);
    if (it == models_.end() || !it->second->raw_model) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model is not loaded");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.max_context_tokens;
    uint32_t threads = it->second->n_threads > 0 ? it->second->n_threads : 4;
    cparams.n_threads = static_cast<int32_t>(threads);
    cparams.n_threads_batch = static_cast<int32_t>(threads);

    // 1. Explicitly configure cparams.n_batch (e.g., 512, capped at n_ctx) and cparams.n_ubatch
    // (e.g., 512 on CPU / 256 on Vulkan) instead of relying on implicit defaults.
    uint32_t req_n_batch = config.n_batch > 0 ? config.n_batch : 512;
    const char* env_batch = getenv("MOBILELLM_N_BATCH");
    if (env_batch && *env_batch) {
        int parsed = std::atoi(env_batch);
        if (parsed > 0) {
            req_n_batch = static_cast<uint32_t>(parsed);
        }
    }
    cparams.n_batch = std::min(req_n_batch, cparams.n_ctx);

    uint32_t req_n_ubatch = config.n_ubatch;
    const char* env_ubatch = getenv("MOBILELLM_N_UBATCH");
    if (env_ubatch && *env_ubatch) {
        int parsed = std::atoi(env_ubatch);
        if (parsed > 0) {
            req_n_ubatch = static_cast<uint32_t>(parsed);
        }
    }
    if (req_n_ubatch == 0) {
        req_n_ubatch = (it->second->backend_type == BackendType::VULKAN) ? 256 : 512;
    }
    cparams.n_ubatch = std::min(req_n_ubatch, static_cast<uint32_t>(cparams.n_batch));

    // 2. Make cparams.flash_attn configurable (instead of unconditionally forcing it off for all Vulkan devices)
    // so we can benchmark FlashAttention on Vulkan once llama.cpp is updated, while keeping the safe default
    // if graph splits increase.
    bool use_flash_attn = false;
    if (config.flash_attn == 1) {
        use_flash_attn = true;
    } else if (config.flash_attn == 0) {
        use_flash_attn = false;
    } else {
        const char* fa_env = getenv("MOBILELLM_VULKAN_FLASH_ATTN");
        if (!fa_env) fa_env = getenv("GGML_VULKAN_FLASH_ATTN");
        bool allow_vk_fa = (fa_env && (strcmp(fa_env, "1") == 0 || strcmp(fa_env, "true") == 0 || strcmp(fa_env, "TRUE") == 0));
        if (it->second->backend_type == BackendType::CPU_NEON) {
            use_flash_attn = true;
        } else {
            use_flash_attn = allow_vk_fa;
        }
    }
    cparams.flash_attn = use_flash_attn;
    cparams.no_perf = false;

#ifdef MOBILELLM_ENABLE_TENSOR_TRACE
    // See the warning above the debug_eval_callback definition: enabling cb_eval collapses the
    // graph to per-node execution and is a severe throughput hazard. It is only reachable when
    // the build explicitly opts in AND a sentinel file is present (for the Vulkan differential
    // validation tooling).
    if (access("/tmp/mobilellm_tensor_dump", F_OK) == 0 ||
        access("/data/local/tmp/mobilellm_tensor_dump", F_OK) == 0 ||
        access("/data/user/0/ai.codesec.nexa.debug/files/mobilellm_tensor_dump", F_OK) == 0 ||
        access("/data/user/0/ai.codesec.nexa/files/mobilellm_tensor_dump", F_OK) == 0) {
        cparams.cb_eval = debug_eval_callback;
        cparams.cb_eval_user_data = nullptr;

        if (!g_tensor_trace_fp) {
            g_tensor_trace_fp = fopen("/data/user/0/ai.codesec.nexa.debug/files/tensor_trace.txt", "a");
            if (!g_tensor_trace_fp) {
                g_tensor_trace_fp = fopen("/data/user/0/ai.codesec.nexa/files/tensor_trace.txt", "a");
            }
            if (!g_tensor_trace_fp) {
                g_tensor_trace_fp = fopen("/tmp/tensor_trace.txt", "a");
            }
        }
        if (g_tensor_trace_fp) {
            fprintf(g_tensor_trace_fp, "\n=== SESSION START: backend=%d, gpu_layers=%d ===\n",
                    (int)it->second->backend_type, it->second->actual_gpu_layers);
            fflush(g_tensor_trace_fp);
        }
        LOGI("Enabling differential tensor evaluation callback (mobilellm_tensor_dump active)");
    }
#endif // MOBILELLM_ENABLE_TENSOR_TRACE

    LOGI("Creating session: ctx=%u, n_batch=%u, n_ubatch=%u, threads=%u, flash_attn=%s, prefix_caching=%s",
         cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, threads, use_flash_attn ? "true" : "false",
         config.enable_prefix_caching ? "true" : "false");

    reset_graph_facts();
    llama_context* ctx = llama_new_context_with_model(it->second->raw_model, cparams);
    if (!ctx) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "Failed to create llama context (out of memory)");
    }

    // Create persistent threadpool for this session to eliminate per-token disposable threadpool overhead.
    // Without an attached threadpool, ggml-cpu instantiates and destroys worker pthreads on EVERY single token,
    // resulting in catastrophic kernel syscall storms, loss of cache locality, and an 84% performance collapse.
    // Keep the upstream default hybrid-polling level (ggml_threadpool_params_default => poll = 50).
    // Overriding it to 0 disables polling entirely, so every graph dispatch forces the worker
    // threads to park on a futex and be woken again. On big.LITTLE mobile SoCs that sleep/wake
    // latency (compounded by any per-node graph dispatch) dominates short batch-1 decode steps.
    struct ggml_threadpool_params ttp = ggml_threadpool_params_default(static_cast<int32_t>(threads));
    struct ggml_threadpool* tpool = ggml_threadpool_new(&ttp);
    if (tpool) {
        llama_attach_threadpool(ctx, tpool, nullptr);
    } else {
        LOGW("Failed to allocate ggml_threadpool (%u threads); continuing with default", threads);
    }

    BackendSessionHandle s_handle = next_session_id_++;
    auto sess = std::make_shared<SessionContext>();
    sess->handle = s_handle;
    sess->model_handle = model;
    sess->raw_ctx = ctx;
    sess->threadpool = tpool;
    sess->sampler = std::make_unique<Sampler>(42 + static_cast<uint64_t>(s_handle));
    sess->enable_prefix_caching = config.enable_prefix_caching;
    sess->n_batch = static_cast<uint32_t>(cparams.n_batch);
    sess->n_ubatch = static_cast<uint32_t>(cparams.n_ubatch);
    sess->n_threads = threads;
    sess->model_load_ms = it->second->load_ms;
    sess->kv_capacity_tokens = static_cast<uint32_t>(llama_n_ctx(ctx));
    sess->gpu_device = it->second->gpu_device;
    sess->gpu_vendor = it->second->gpu_vendor;
    sess->gpu_driver = it->second->gpu_driver;
    sess->model_layers = it->second->model_layers;
    sess->offloaded_layers = it->second->offloaded_layers;
    sess->total_offloadable_layers = it->second->total_offloadable_layers;
    sess->fallback_reason = it->second->fallback_reason;
    if (!sess->fallback_reason.empty() && sess->fallback_reason.find("AUTO routed to CPU_NEON") != std::string::npos) {
        sess->n_threads = std::clamp(sess->n_threads, 4u, 6u);
    }
    {
        std::lock_guard<std::mutex> lock(g_llama_facts_mutex);
        sess->graph_nodes = g_llama_facts.graph_nodes;
        sess->graph_splits_pp = g_llama_facts.graph_splits_pp;
        sess->graph_splits_tg = g_llama_facts.graph_splits_tg;
        sess->vulkan_matmul_path = g_llama_facts.vulkan_matmul_path;
    }

    // PERF-03: Pre-allocate reusable logits buffer once based on model vocabulary
    try {
        sess->reusable_logits.resize(it->second->n_vocab, 0.0f);
    } catch (const std::bad_alloc&) {
        if (tpool) {
            llama_detach_threadpool(ctx);
            ggml_threadpool_free(tpool);
        }
        llama_free(ctx);
        return Status::error(StatusCode::OUT_OF_MEMORY, "Failed to allocate reusable logits buffer");
    }

    sessions_[s_handle] = sess;
    return s_handle;
}

Status LlamaCppBackend::destroy_session(BackendSessionHandle session) noexcept {
    std::shared_ptr<SessionContext> sess;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
        }
        sess = it->second;
        sessions_.erase(it);
    }

    if (sess) {
        std::lock_guard<std::mutex> s_lock(sess->session_mutex);
        sess->reusable_logits.clear();
        sess->reusable_logits.shrink_to_fit();
        if (sess->raw_ctx) {
            llama_synchronize(sess->raw_ctx);
            if (sess->threadpool) {
                llama_detach_threadpool(sess->raw_ctx);
                ggml_threadpool_free(sess->threadpool);
                sess->threadpool = nullptr;
            }
            llama_free(sess->raw_ctx);
            sess->raw_ctx = nullptr;
        } else if (sess->threadpool) {
            ggml_threadpool_free(sess->threadpool);
            sess->threadpool = nullptr;
        }
    }
    return Status::ok();
}

Status LlamaCppBackend::evaluate_prompt(
    BackendSessionHandle session,
    const std::string& prompt,
    std::vector<int32_t>& out_tokens,
    PromptEvalInfo* out_info
) noexcept {
    if (out_info) {
        *out_info = PromptEvalInfo{};
    }

    std::shared_ptr<SessionContext> sess;
    std::shared_ptr<LoadedModel> mdl;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto s_it = sessions_.find(session);
        if (s_it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
        }
        sess = s_it->second;

        auto m_it = models_.find(sess->model_handle);
        if (m_it == models_.end()) {
            return Status::error(StatusCode::MODEL_NOT_LOADED, "Associated model not found");
        }
        mdl = m_it->second;
    }

    if (prompt.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Prompt cannot be empty");
    }

    // Timing for the tokenizer only. This is reported separately from graph compute because a
    // large part of "prefill" latency on mobile is host-side tokenization of chat templates.
    const auto t_tokenize_start = std::chrono::steady_clock::now();

    // Determine if BOS should be added
    bool add_bos = llama_add_bos_token(mdl->raw_model);
    if (prompt.rfind("<|", 0) == 0) {
        add_bos = false;
    }

    // Tokenize prompt with parse_special = true to recognize ChatML control tokens
    const int32_t max_tokens = static_cast<int32_t>(prompt.size() + 16);
    std::vector<llama_token> tokens(max_tokens);
    int32_t n_tokens = llama_tokenize(
        mdl->raw_model,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        max_tokens,
        add_bos,
        true  // parse_special = true
    );

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            mdl->raw_model,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_bos,
            true  // parse_special = true
        );
    }

    if (n_tokens <= 0) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Failed to tokenize prompt");
    }
    tokens.resize(n_tokens);

    const double tokenize_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_tokenize_start).count();

    if (out_info) {
        out_info->prompt_tokens = static_cast<uint32_t>(n_tokens);
        out_info->tokenize_ms = tokenize_ms;
    }

    // Evaluate prompt via llama_decode under per-session lock
    std::lock_guard<std::mutex> s_lock(sess->session_mutex);

    const llama_perf_context_data perf_before = llama_perf_context(sess->raw_ctx);

    uint32_t evaluated_tokens = 0;
    uint32_t cached_tokens = 0;
    int32_t eval_calls = 0;

    // A helper that decodes an explicit token range with explicit positions chunked by n_batch.
    // Positions must be explicit: after removing KV cells the implicit "continue from last position"
    // behaviour of llama_batch_get_one() depends on internal cache bookkeeping that we do not want to rely on.
    // Prompt evaluation is chunked into slices of at most n_batch tokens (as required by llama.cpp's
    // input tensor allocations and GGML_ASSERT(n_tokens_all <= cparams.n_batch)).
    // Logits (batch.logits[i] = 1) are requested only for the very last token of the final chunk so we
    // do not compute or transfer unnecessary logits on intermediate chunks.
    auto decode_range = [&](size_t begin, size_t end) -> bool {
        if (end <= begin) return true;

        const size_t batch_limit = (sess->n_batch > 0) ? sess->n_batch
                                 : (llama_n_batch(sess->raw_ctx) > 0 ? llama_n_batch(sess->raw_ctx) : 512);

        size_t chunk_start = begin;
        while (chunk_start < end) {
            const size_t chunk_end = std::min(chunk_start + batch_limit, end);
            const int32_t chunk_len = static_cast<int32_t>(chunk_end - chunk_start);
            const bool is_last_chunk = (chunk_end == end);

            std::vector<llama_token> batch_tokens(tokens.begin() + static_cast<ptrdiff_t>(chunk_start),
                                                  tokens.begin() + static_cast<ptrdiff_t>(chunk_end));
            std::vector<llama_pos> batch_pos(chunk_len);
            std::vector<int8_t> batch_logits(chunk_len, 0);

            for (int32_t i = 0; i < chunk_len; ++i) {
                batch_pos[i] = static_cast<llama_pos>(chunk_start + static_cast<size_t>(i));
            }

            if (is_last_chunk) {
                batch_logits[chunk_len - 1] = 1;
            }

            llama_batch batch = {
                chunk_len,
                batch_tokens.data(),
                nullptr,
                batch_pos.data(),
                nullptr,
                nullptr,
                batch_logits.data()
            };

            int ret = llama_decode(sess->raw_ctx, batch);
            if (ret != 0) {
                LOGE("llama_decode failed on chunk [%zu, %zu) with code %d", chunk_start, chunk_end, ret);
                return false;
            }

            eval_calls++;
            chunk_start = chunk_end;
        }

        return true;
    };

    // Calculate common prefix with existing session history. Prefix caching is opt-out via
    // SessionConfig::enable_prefix_caching; when disabled every prompt is evaluated from scratch
    // so that KV state is fully deterministic regardless of what was generated before.
    size_t common_prefix = 0;
    if (sess->enable_prefix_caching) {
        while (common_prefix < tokens.size() &&
               common_prefix < sess->history_tokens.size() &&
               tokens[common_prefix] == sess->history_tokens[common_prefix]) {
            common_prefix++;
        }
    }

    if (common_prefix == tokens.size() && !tokens.empty() && sess->enable_prefix_caching) {
        // ---------------------------------------------------------------------------------
        // Full prefix-cache hit: the ENTIRE prompt is already resident in the KV cache.
        //
        // The previous implementation skipped llama_decode() entirely here. That is wrong:
        // llama_get_logits_ith(-1) then returns whatever logits tensor the last decode produced
        // - normally the logits of the *last generated token* of the previous turn, or, on a
        // session that generated nothing, stale data from an older prompt. Sampling from those
        // logits makes an identical prompt produce different output (reproduced: 3 different
        // continuations for the same prompt) and makes prefill_time_ms ~= 0, which in turn
        // produced absurd prefill rates (235k-301k tok/s) from the division in the scheduler.
        //
        // Fix: rewind one token and re-decode the last prompt token. This is one batch-1 decode,
        // i.e. the same cost as a single decode step, and guarantees logits correspond to the
        // prompt's final token.
        // ---------------------------------------------------------------------------------
        const size_t rewind = tokens.size() - 1;
        if (sess->history_tokens.size() > rewind) {
            llama_kv_cache_seq_rm(sess->raw_ctx, 0, static_cast<llama_pos>(rewind), -1);
        }
        if (!decode_range(rewind, tokens.size())) {
            return Status::error(StatusCode::BACKEND_EXECUTION_ERROR,
                                 "llama_decode failed refreshing logits for cached prompt");
        }
        evaluated_tokens = 1;
        cached_tokens = static_cast<uint32_t>(rewind);
    } else if (common_prefix > 0) {
        // Rebuilt conversation prompt or continuation sharing a prefix with history.
        // Drop KV cells past the divergence point and decode only the new suffix.
        if (sess->history_tokens.size() > common_prefix) {
            llama_kv_cache_seq_rm(sess->raw_ctx, 0, static_cast<llama_pos>(common_prefix), -1);
        }
        if (tokens.size() > common_prefix) {
            if (!decode_range(common_prefix, tokens.size())) {
                return Status::error(StatusCode::BACKEND_EXECUTION_ERROR,
                                     "llama_decode failed on prompt suffix");
            }
            evaluated_tokens = static_cast<uint32_t>(tokens.size() - common_prefix);
        }
        cached_tokens = static_cast<uint32_t>(common_prefix);
    } else {
        // Independent prompt or context reset: clear the KV cache to eliminate contamination.
        llama_kv_cache_clear(sess->raw_ctx);
        if (!decode_range(0, tokens.size())) {
            return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "llama_decode failed on prompt");
        }
        evaluated_tokens = static_cast<uint32_t>(tokens.size());
        cached_tokens = 0;
    }

    const llama_perf_context_data perf_after = llama_perf_context(sess->raw_ctx);
    const double backend_eval_ms =
        (perf_after.t_p_eval_ms - perf_before.t_p_eval_ms) +
        (perf_after.t_eval_ms - perf_before.t_eval_ms);

    out_tokens.assign(tokens.begin(), tokens.end());
    sess->history_tokens = out_tokens;
    sess->last_n_tokens = n_tokens;

    if (out_info) {
        out_info->prompt_tokens = static_cast<uint32_t>(n_tokens);
        out_info->prefix_cached_tokens = cached_tokens;
        out_info->evaluated_tokens = evaluated_tokens;
        out_info->prefix_cache_full_hit =
            sess->enable_prefix_caching && out_info->prompt_tokens > 0 &&
            cached_tokens > 0 && cached_tokens + 1 >= out_info->prompt_tokens;
        out_info->backend_eval_ms = backend_eval_ms;
        out_info->eval_calls = eval_calls;
        out_info->tokenize_ms = tokenize_ms;
    }

    return Status::ok();
}

Result<int32_t> LlamaCppBackend::sample_next_token(
    BackendSessionHandle session,
    const SamplingParams& params,
    TokenTiming* out_timing
) noexcept {
    if (out_timing) {
        *out_timing = TokenTiming{};
    }

    std::shared_ptr<SessionContext> sess;
    std::shared_ptr<LoadedModel> mdl;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto s_it = sessions_.find(session);
        if (s_it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
        }
        sess = s_it->second;

        auto m_it = models_.find(sess->model_handle);
        if (m_it == models_.end()) {
            return Status::error(StatusCode::MODEL_NOT_LOADED, "Associated model not found");
        }
        mdl = m_it->second;
    }

    std::lock_guard<std::mutex> s_lock(sess->session_mutex);

    // llama_get_logits_ith() implicitly calls llama_synchronize(): on Vulkan this is where the host
    // blocks until the GPU finishes the *previous* decode. Measuring this separately is the only
    // way to tell "GPU is slow" apart from "host dispatch is slow" in the 2 tok/s report.
    const auto t_wait_start = std::chrono::steady_clock::now();
    float* raw_logits = llama_get_logits_ith(sess->raw_ctx, -1);
    const auto t_wait_end = std::chrono::steady_clock::now();
    if (!raw_logits) {
        LOGE("llama_get_logits_ith(-1) returned NULL!");
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "Failed to obtain logits from context");
    }

    // PERF-03: Reuse pre-allocated session logits buffer; resize safely only if vocabulary metadata changed
    if (sess->reusable_logits.size() != mdl->n_vocab) {
        try {
            sess->reusable_logits.resize(mdl->n_vocab, 0.0f);
        } catch (const std::bad_alloc&) {
            return Status::error(StatusCode::OUT_OF_MEMORY, "Failed to resize reusable logits buffer");
        }
    }

    std::memcpy(sess->reusable_logits.data(), raw_logits, mdl->n_vocab * sizeof(float));

    int32_t token = sess->sampler ? sess->sampler->sample(sess->reusable_logits, sess->history_tokens, params)
                                  : 0;

    const auto t_sample_end = std::chrono::steady_clock::now();

    // Step forward by decoding the sampled token. On Vulkan this call builds + submits the compute
    // graph and normally returns before the GPU has finished; the wait is paid on the next
    // llama_get_logits_ith() above. Do not label this interval "GPU compute".
    const auto t_decode_start = std::chrono::steady_clock::now();
    llama_token l_tok = static_cast<llama_token>(token);
    llama_batch batch = llama_batch_get_one(&l_tok, 1);
    if (llama_decode(sess->raw_ctx, batch) != 0) {
        LOGE("llama_decode failed for next token %d!", token);
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "llama_decode failed for next token");
    }
    const auto t_decode_end = std::chrono::steady_clock::now();

    sess->history_tokens.push_back(token);

    if (out_timing) {
        out_timing->logits_wait_ms =
            std::chrono::duration<double, std::milli>(t_wait_end - t_wait_start).count();
        out_timing->sample_ms =
            std::chrono::duration<double, std::milli>(t_sample_end - t_wait_end).count();
        out_timing->decode_dispatch_ms =
            std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    }

    return token;
}

std::string LlamaCppBackend::decode_token(BackendSessionHandle session, int32_t token_id) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto s_it = sessions_.find(session);
    if (s_it == sessions_.end()) return "";

    auto m_it = models_.find(s_it->second->model_handle);
    if (m_it == models_.end() || !m_it->second->raw_model) return "";

    char buf[128];
    int32_t n = llama_token_to_piece(
        m_it->second->raw_model,
        static_cast<llama_token>(token_id),
        buf,
        sizeof(buf),
        0,
        false
    );

    if (n < 0) {
        std::vector<char> big_buf(-n + 1);
        n = llama_token_to_piece(
            m_it->second->raw_model,
            static_cast<llama_token>(token_id),
            big_buf.data(),
            static_cast<int32_t>(big_buf.size()),
            0,
            false
        );
        return (n > 0) ? std::string(big_buf.data(), n) : "";
    }

    return (n > 0) ? std::string(buf, n) : "";
}

bool LlamaCppBackend::is_eos_token(BackendSessionHandle session, int32_t token_id) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto s_it = sessions_.find(session);
    if (s_it == sessions_.end()) return false;

    auto m_it = models_.find(s_it->second->model_handle);
    if (m_it == models_.end() || !m_it->second->raw_model) return false;

    return llama_token_is_eog(m_it->second->raw_model, static_cast<llama_token>(token_id));
}

GenerationDiagnostics LlamaCppBackend::describe_session(BackendSessionHandle session) const noexcept {
    GenerationDiagnostics d;

    std::shared_ptr<SessionContext> sess;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end() || !it->second) {
            d.error = "session_not_found";
            return d;
        }
        sess = it->second;
    }

    std::lock_guard<std::mutex> s_lock(sess->session_mutex);

    d.backend_selected = (current_backend_.load() == BackendType::VULKAN) ? "vulkan" : "cpu_neon";
    d.backend_requested = d.backend_selected;
    d.gpu_device = sess->gpu_device;
    d.gpu_vendor = sess->gpu_vendor;
    d.gpu_driver = sess->gpu_driver;
    d.vulkan_gemv_path = sess->vulkan_matmul_path;
    d.fallback_reason = sess->fallback_reason;
    if (!sess->fallback_reason.empty() && sess->fallback_reason.find("AUTO routed to CPU_NEON") != std::string::npos) {
        d.backend_requested = "auto";
        d.backend_selected = "cpu_neon";
    }

    d.model_layers = sess->model_layers;
    d.gpu_layers_requested = sess->offloaded_layers >= 0 ? sess->offloaded_layers : 0;
    d.gpu_layers_actual = sess->offloaded_layers;
    d.gpu_offload_verified = sess->offloaded_layers > 0;
    d.graph_nodes = sess->graph_nodes;
    d.graph_splits_pp = sess->graph_splits_pp;
    d.graph_splits_tg = sess->graph_splits_tg;

    d.model_load_ms = sess->model_load_ms;
    d.context_capacity = sess->kv_capacity_tokens;
    d.cpu_threads = sess->n_threads;
    d.cpu_cores = std::thread::hardware_concurrency();

    if (sess->raw_ctx) {
        const int32_t used = llama_get_kv_cache_used_cells(sess->raw_ctx);
        d.kv_used_tokens = used > 0 ? static_cast<uint32_t>(used) : 0;
    }

    return d;
}

SessionContextTelemetry LlamaCppBackend::get_session_telemetry(BackendSessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto s_it = sessions_.find(session);
    if (s_it == sessions_.end() || !s_it->second || !s_it->second->raw_ctx) {
        return SessionContextTelemetry{};
    }

    SessionContextTelemetry t;
    llama_context* ctx = s_it->second->raw_ctx;
    t.context_capacity = llama_n_ctx(ctx);
    int32_t used = llama_get_kv_cache_used_cells(ctx);
    t.used_tokens = used >= 0 ? static_cast<uint32_t>(used) : 0;
    t.remaining_capacity = (t.context_capacity > t.used_tokens) ? (t.context_capacity - t.used_tokens) : 0;
    t.is_available = true;
    return t;
}

Status LlamaCppBackend::set_threads(BackendSessionHandle session, uint32_t n_threads) noexcept {
    std::shared_ptr<SessionContext> sess;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto s_it = sessions_.find(session);
        if (s_it == sessions_.end() || !s_it->second || !s_it->second->raw_ctx) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
        }
        sess = s_it->second;
    }

    std::lock_guard<std::mutex> s_lock(sess->session_mutex);
    int32_t threads = static_cast<int32_t>(n_threads);
    if (threads <= 0) threads = 1;
    llama_set_n_threads(sess->raw_ctx, threads, threads);

    // Reconfigure persistent threadpool for updated thread count
    if (sess->threadpool) {
        llama_detach_threadpool(sess->raw_ctx);
        ggml_threadpool_free(sess->threadpool);
        sess->threadpool = nullptr;
    }
    // Retain the default hybrid-polling level (poll = 50); see create_session() for rationale.
    struct ggml_threadpool_params ttp = ggml_threadpool_params_default(threads);
    sess->threadpool = ggml_threadpool_new(&ttp);
    if (sess->threadpool) {
        llama_attach_threadpool(sess->raw_ctx, sess->threadpool, nullptr);
    }
    return Status::ok();
}

Status LlamaCppBackend::reset_session(BackendSessionHandle session) noexcept {
    std::shared_ptr<SessionContext> sess;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
        }
        sess = it->second;
    }

    if (sess) {
        std::lock_guard<std::mutex> s_lock(sess->session_mutex);
        if (sess->raw_ctx) {
            llama_kv_cache_clear(sess->raw_ctx);
        }
        sess->history_tokens.clear();
        sess->last_n_tokens = 0;
        if (sess->sampler) {
            sess->sampler = std::make_unique<Sampler>(42 + static_cast<uint64_t>(session));
        }
    }
    return Status::ok();
}

Result<std::vector<float>> LlamaCppBackend::get_last_logits(BackendSessionHandle session) const noexcept {
    std::shared_ptr<SessionContext> sess;
    std::shared_ptr<LoadedModel> mdl;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
        }
        sess = it->second;

        auto m_it = models_.find(sess->model_handle);
        if (m_it == models_.end()) {
            return Status::error(StatusCode::MODEL_NOT_LOADED, "Associated model not found");
        }
        mdl = m_it->second;
    }

    std::lock_guard<std::mutex> s_lock(sess->session_mutex);
    if (!sess->raw_ctx) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Session context is null");
    }

    float* raw_logits = llama_get_logits_ith(sess->raw_ctx, -1);
    if (!raw_logits) {
        if (!sess->reusable_logits.empty()) {
            return sess->reusable_logits;
        }
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "No logits available in session context");
    }

    std::vector<float> logits(mdl->n_vocab);
    std::memcpy(logits.data(), raw_logits, mdl->n_vocab * sizeof(float));
    return logits;
}

} // namespace mobilellm

#else

// Stubs when llama.cpp is not linked
namespace mobilellm {

LlamaCppBackend::LlamaCppBackend() = default;
LlamaCppBackend::~LlamaCppBackend() = default;

Status LlamaCppBackend::initialize(const BackendConfig&) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend is not compiled into this build");
}

Status LlamaCppBackend::shutdown() noexcept { return Status::ok(); }

Result<BackendModelHandle> LlamaCppBackend::load_model(const ModelLoadConfig&) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Status LlamaCppBackend::unload_model(BackendModelHandle) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Result<BackendSessionHandle> LlamaCppBackend::create_session(BackendModelHandle, const SessionConfig&) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Status LlamaCppBackend::destroy_session(BackendSessionHandle) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Status LlamaCppBackend::evaluate_prompt(BackendSessionHandle, const std::string&, std::vector<int32_t>&, PromptEvalInfo*) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Result<int32_t> LlamaCppBackend::sample_next_token(BackendSessionHandle, const SamplingParams&, TokenTiming*) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

GenerationDiagnostics LlamaCppBackend::describe_session(BackendSessionHandle) const noexcept {
    return GenerationDiagnostics{};
}

std::string LlamaCppBackend::gpu_device_description() const noexcept { return {}; }

std::string LlamaCppBackend::decode_token(BackendSessionHandle, int32_t) noexcept { return ""; }
bool LlamaCppBackend::is_eos_token(BackendSessionHandle, int32_t) const noexcept { return false; }

SessionContextTelemetry LlamaCppBackend::get_session_telemetry(BackendSessionHandle) const noexcept {
    return SessionContextTelemetry{};
}

Status LlamaCppBackend::set_threads(BackendSessionHandle, uint32_t) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Status LlamaCppBackend::reset_session(BackendSessionHandle) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Result<std::vector<float>> LlamaCppBackend::get_last_logits(BackendSessionHandle) const noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

} // namespace mobilellm

#endif

