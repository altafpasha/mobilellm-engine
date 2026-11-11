#include "backends/cpu/llamacpp_backend.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LlamaCppBackend", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "LlamaCppBackend", __VA_ARGS__)
#else
#define LOGE(...)
#define LOGI(...)
#endif

#ifdef MOBILELLM_ENABLE_LLAMACPP
#include "llama.h"
#include <vector>
#include <string>

namespace mobilellm {

LlamaCppBackend::LlamaCppBackend() : sampler_(42) {}

LlamaCppBackend::~LlamaCppBackend() {
    shutdown();
}

Status LlamaCppBackend::initialize(const BackendConfig& /*config*/) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        llama_backend_init();
        initialized_ = true;
    }
    return Status::ok();
}

Status LlamaCppBackend::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, sess] : sessions_) {
        if (sess && sess->raw_ctx) {
            llama_free(sess->raw_ctx);
            sess->raw_ctx = nullptr;
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

    llama_model* raw = llama_load_model_from_file(config.model_path.c_str(), mparams);
    if (!raw) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Failed to load model from: " + config.model_path);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BackendModelHandle handle = next_model_id_++;
    auto loaded = std::make_shared<LoadedModel>();
    loaded->handle = handle;
    loaded->path = config.model_path;
    loaded->raw_model = raw;
    loaded->context_length = config.context_length;
    loaded->n_vocab = static_cast<uint32_t>(llama_n_vocab(raw));

    models_[handle] = loaded;
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
            if (s_it->second->raw_ctx) {
                llama_free(s_it->second->raw_ctx);
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
    cparams.no_perf = false;

    llama_context* ctx = llama_new_context_with_model(it->second->raw_model, cparams);
    if (!ctx) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "Failed to create llama context (out of memory)");
    }

    BackendSessionHandle s_handle = next_session_id_++;
    auto sess = std::make_shared<SessionContext>();
    sess->handle = s_handle;
    sess->model_handle = model;
    sess->raw_ctx = ctx;

    sessions_[s_handle] = sess;
    return s_handle;
}

Status LlamaCppBackend::destroy_session(BackendSessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }

    if (it->second->raw_ctx) {
        llama_free(it->second->raw_ctx);
    }
    sessions_.erase(it);
    return Status::ok();
}

Status LlamaCppBackend::evaluate_prompt(
    BackendSessionHandle session,
    const std::string& prompt,
    std::vector<int32_t>& out_tokens
) noexcept {
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

    // Tokenize prompt
    const int32_t max_tokens = static_cast<int32_t>(prompt.size() + 16);
    std::vector<llama_token> tokens(max_tokens);
    int32_t n_tokens = llama_tokenize(
        mdl->raw_model,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        max_tokens,
        true,  // add_special (BOS)
        false  // parse_special
    );

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(
            mdl->raw_model,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            false
        );
    }

    if (n_tokens <= 0) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Failed to tokenize prompt");
    }
    tokens.resize(n_tokens);

    // Evaluate prompt via llama_decode
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(sess->raw_ctx, batch) != 0) {
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "llama_decode failed on prompt");
    }

    out_tokens.assign(tokens.begin(), tokens.end());
    sess->history_tokens = out_tokens;
    sess->last_n_tokens = n_tokens;
    return Status::ok();
}

Result<int32_t> LlamaCppBackend::sample_next_token(
    BackendSessionHandle session,
    const SamplingParams& params
) noexcept {
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

    float* raw_logits = llama_get_logits_ith(sess->raw_ctx, -1);
    if (!raw_logits) {
        LOGE("llama_get_logits_ith(-1) returned NULL!");
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "Failed to obtain logits from context");
    }

    std::vector<float> logits(raw_logits, raw_logits + mdl->n_vocab);
    int32_t token = sampler_.sample(logits, sess->history_tokens, params);

    // Step forward by decoding the sampled token
    llama_token l_tok = static_cast<llama_token>(token);
    llama_batch batch = llama_batch_get_one(&l_tok, 1);
    if (llama_decode(sess->raw_ctx, batch) != 0) {
        LOGE("llama_decode failed for next token %d!", token);
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "llama_decode failed for next token");
    }

    sess->history_tokens.push_back(token);
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

} // namespace mobilellm

#else

// Stubs when llama.cpp is not linked
namespace mobilellm {

LlamaCppBackend::LlamaCppBackend() : sampler_(42) {}
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

Status LlamaCppBackend::evaluate_prompt(BackendSessionHandle, const std::string&, std::vector<int32_t>&) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

Result<int32_t> LlamaCppBackend::sample_next_token(BackendSessionHandle, const SamplingParams&) noexcept {
    return Status::error(StatusCode::NOT_IMPLEMENTED, "llama.cpp backend not compiled");
}

std::string LlamaCppBackend::decode_token(BackendSessionHandle, int32_t) noexcept { return ""; }
bool LlamaCppBackend::is_eos_token(BackendSessionHandle, int32_t) const noexcept { return false; }

} // namespace mobilellm

#endif

