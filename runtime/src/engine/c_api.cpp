#include "mobilellm/c_api.h"
#include "mobilellm/engine.h"
#include "mobilellm/model_loader.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <sstream>

namespace {

std::mutex g_engine_mutex;
std::unordered_map<mllm_engine_t, std::unique_ptr<mobilellm::IEngine>> g_engines;
uint64_t g_next_engine_id = 1;

// Thread-local error message storage ensuring valid pointer lifetime
// across C-ABI boundaries without heap leaking or cross-thread data races.
thread_local std::string tls_last_error_message;

const char* default_status_message(int32_t code) {
    switch (static_cast<mobilellm::StatusCode>(code)) {
        case mobilellm::StatusCode::OK: return "";
        case mobilellm::StatusCode::INVALID_ARGUMENT: return "Invalid argument";
        case mobilellm::StatusCode::FILE_NOT_FOUND: return "File not found";
        case mobilellm::StatusCode::UNSUPPORTED_FORMAT: return "Unsupported model format";
        case mobilellm::StatusCode::OUT_OF_MEMORY: return "Out of memory";
        case mobilellm::StatusCode::BACKEND_INITIALIZATION_FAILED: return "Backend initialization failed";
        case mobilellm::StatusCode::BACKEND_EXECUTION_ERROR: return "Backend execution error";
        case mobilellm::StatusCode::SESSION_NOT_FOUND: return "Session not found";
        case mobilellm::StatusCode::MODEL_NOT_LOADED: return "Model not loaded";
        case mobilellm::StatusCode::CANCELLED: return "Operation cancelled";
        case mobilellm::StatusCode::CONTEXT_EXCEEDED: return "Context window exceeded";
        case mobilellm::StatusCode::TIMEOUT: return "Operation timed out";
        case mobilellm::StatusCode::NOT_IMPLEMENTED: return "Not implemented";
        case mobilellm::StatusCode::INTERNAL_ERROR: return "Internal error";
        default: return "Unknown status code";
    }
}

mllm_status_t make_c_status(int32_t code, const std::string& msg) {
    mllm_status_t c_st;
    c_st.code = code;
    if (code == 0) {
        c_st.message = "";
    } else {
        if (!msg.empty()) {
            tls_last_error_message = msg;
            c_st.message = tls_last_error_message.c_str();
        } else {
            c_st.message = default_status_message(code);
        }
    }
    return c_st;
}

mllm_status_t to_c_status(const mobilellm::Status& st) {
    return make_c_status(static_cast<int32_t>(st.code), st.message);
}

mllm_status_t c_error(int32_t code, const char* msg) {
    return make_c_status(code, msg ? std::string(msg) : std::string());
}

mobilellm::IEngine* get_engine(mllm_engine_t id) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(id);
    if (it != g_engines.end()) {
        return it->second.get();
    }
    return nullptr;
}

} // namespace

extern "C" {

mllm_status_t mllm_engine_create(mllm_engine_t* out_engine) {
    if (!out_engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "out_engine pointer is null");
    }

    auto engine = mobilellm::create_engine();
    mobilellm::EngineConfig cfg;
    mobilellm::Status init_st = engine->initialize(cfg);
    if (!init_st.is_ok()) {
        return to_c_status(init_st);
    }

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    mllm_engine_t id = g_next_engine_id++;
    g_engines[id] = std::move(engine);
    *out_engine = id;

    return c_error(0, "");
}

mllm_status_t mllm_engine_destroy(mllm_engine_t engine_id) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    auto it = g_engines.find(engine_id);
    if (it == g_engines.end()) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }
    it->second->shutdown();
    g_engines.erase(it);
    return c_error(0, "");
}

mllm_status_t mllm_model_load(mllm_engine_t engine_id, const mllm_model_config_t* config, mllm_model_t* out_model) {
    if (!config || !out_model) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Config or out_model pointer is null");
    }

    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }

    mobilellm::ModelLoadConfig cpp_cfg;
    if (config->model_path) {
        cpp_cfg.model_path = config->model_path;
    }
    cpp_cfg.preferred_backend = static_cast<mobilellm::BackendType>(config->backend);
    cpp_cfg.context_length = config->context_length > 0 ? config->context_length : 2048;
    cpp_cfg.n_threads = config->n_threads > 0 ? config->n_threads : 4;
    cpp_cfg.use_mmap = config->use_mmap;
    cpp_cfg.use_mlock = config->use_mlock;
    cpp_cfg.memory_budget_mb = config->memory_budget_mb;

    auto res = engine->load_model(cpp_cfg);
    if (!res.is_ok()) {
        return to_c_status(res.status());
    }

    *out_model = res.value();
    return c_error(0, "");
}

mllm_status_t mllm_model_unload(mllm_engine_t engine_id, mllm_model_t model) {
    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }
    return to_c_status(engine->unload_model(model));
}

mllm_status_t mllm_model_inspect(const char* model_path, char* json_buffer, size_t max_len) {
    if (!model_path || !json_buffer || max_len == 0) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid null arguments");
    }

    auto res = mobilellm::ModelLoader::inspect_file(model_path);
    if (!res.is_ok()) {
        return to_c_status(res.status());
    }

    const auto& m = res.value();
    std::ostringstream ss;
    ss << "{"
       << "\"architecture\":\"" << m.architecture << "\","
       << "\"quantization\":\"" << m.quantization << "\","
       << "\"context_length\":" << m.context_length << ","
       << "\"embedding_length\":" << m.embedding_length << ","
       << "\"block_count\":" << m.block_count << ","
       << "\"head_count\":" << m.head_count << ","
       << "\"tensor_count\":" << m.tensor_count << ","
       << "\"file_size_bytes\":" << m.file_size_bytes << ","
       << "\"valid\":true"
       << "}";

    std::string json = ss.str();
    strncpy(json_buffer, json.c_str(), max_len - 1);
    json_buffer[max_len - 1] = '\0';
    return c_error(0, "");
}

mllm_status_t mllm_session_create(
    mllm_engine_t engine_id,
    mllm_model_t model,
    const mllm_session_config_t* config,
    mllm_session_t* out_session
) {
    if (!out_session) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "out_session pointer is null");
    }

    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }

    mobilellm::SessionConfig cpp_cfg;
    if (config) {
        cpp_cfg.max_context_tokens = config->max_context_tokens > 0 ? config->max_context_tokens : 2048;
        cpp_cfg.enable_prefix_caching = config->enable_prefix_caching;
    }

    auto res = engine->create_session(model, cpp_cfg);
    if (!res.is_ok()) {
        return to_c_status(res.status());
    }

    *out_session = res.value();
    return c_error(0, "");
}

mllm_status_t mllm_session_destroy(mllm_engine_t engine_id, mllm_session_t session) {
    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }
    return to_c_status(engine->destroy_session(session));
}

mllm_status_t mllm_generate(
    mllm_engine_t engine_id,
    mllm_session_t session,
    const char* prompt,
    const mllm_sampling_params_t* sampling,
    mllm_token_callback_fn token_cb,
    mllm_complete_callback_fn complete_cb,
    void* user_data,
    mllm_gen_handle_t* out_handle
) {
    if (!prompt) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Prompt is null");
    }

    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }

    mobilellm::GenerationRequest req;
    req.prompt = prompt;
    if (sampling) {
        req.sampling.temperature = sampling->temperature;
        req.sampling.top_p = sampling->top_p;
        req.sampling.top_k = sampling->top_k;
        req.sampling.min_p = sampling->min_p;
        req.sampling.repetition_penalty = sampling->repetition_penalty;
        req.sampling.max_tokens = sampling->max_tokens;
    }

    auto t_cb = [token_cb, user_data](const mobilellm::TokenChunk& chunk) -> bool {
        if (!token_cb) return true;
        mllm_token_chunk_t c_chunk;
        c_chunk.text = chunk.text.c_str();
        c_chunk.token_id = chunk.token_id;
        c_chunk.is_special = chunk.is_special;
        c_chunk.logprob = chunk.logprob;
        return token_cb(&c_chunk, user_data);
    };

    auto c_cb = [complete_cb, user_data](const mobilellm::Status& status, const mobilellm::GenerationMetrics& metrics) {
        if (!complete_cb) return;
        mllm_status_t c_st = to_c_status(status);
        mllm_metrics_t c_metrics;
        c_metrics.prompt_tokens = metrics.prompt_tokens;
        c_metrics.completion_tokens = metrics.completion_tokens;
        c_metrics.ttft_ms = metrics.time_to_first_token_ms;
        c_metrics.tps = metrics.decode_tokens_per_second;
        c_metrics.peak_rss_mb = metrics.peak_memory_mb;
        complete_cb(c_st, &c_metrics, user_data);
    };

    auto res = engine->generate_async(session, req, t_cb, c_cb);
    if (!res.is_ok()) {
        return to_c_status(res.status());
    }

    if (out_handle) {
        *out_handle = res.value();
    }
    return c_error(0, "");
}

mllm_status_t mllm_cancel(mllm_engine_t engine_id, mllm_gen_handle_t handle) {
    auto* engine = get_engine(engine_id);
    if (!engine) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Invalid engine handle");
    }
    return to_c_status(engine->cancel(handle));
}

mllm_status_t mllm_get_version(char* buffer, size_t max_len) {
    if (!buffer || max_len == 0) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Buffer is null or zero length");
    }
    const char* ver = "0.1.0-dev";
    strncpy(buffer, ver, max_len - 1);
    buffer[max_len - 1] = '\0';
    return c_error(0, "");
}

mllm_status_t mllm_detect_hardware(char* json_buffer, size_t max_len) {
    if (!json_buffer || max_len == 0) {
        return c_error(static_cast<int32_t>(mobilellm::StatusCode::INVALID_ARGUMENT), "Buffer is null or zero length");
    }
    const char* info = "{\"device\":\"Host\",\"arch\":\"ARM64\",\"neon\":true,\"vulkan\":false,\"npu\":false}";
    strncpy(json_buffer, info, max_len - 1);
    json_buffer[max_len - 1] = '\0';
    return c_error(0, "");
}

const char* mllm_get_last_error(void) {
    return tls_last_error_message.c_str();
}

} // extern "C"

