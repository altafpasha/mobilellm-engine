#include <gtest/gtest.h>
#include "mobilellm/model_loader.h"
#include "mobilellm/engine.h"
#include "backends/cpu/llamacpp_backend.h"
#include <llama.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>

using namespace mobilellm;

static std::string resolve_model_path(const std::string& rel_path) {
    if (std::filesystem::exists(rel_path)) return rel_path;
    if (std::filesystem::exists("../" + rel_path)) return "../" + rel_path;
    if (std::filesystem::exists("../../" + rel_path)) return "../../" + rel_path;
    return rel_path;
}

static const std::string RAW_QWEN_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
static const std::string RAW_SMOLLM_PATH = "models/smollm2-135m-instruct-q4_k_m.gguf";

class InferenceCorrectnessTest : public ::testing::Test {
protected:
    std::string qwen_path;
    std::string smollm_path;

    void SetUp() override {
        qwen_path = resolve_model_path(RAW_QWEN_PATH);
        smollm_path = resolve_model_path(RAW_SMOLLM_PATH);

        if (!std::filesystem::exists(qwen_path)) {
            GTEST_SKIP() << "Qwen2.5 model not found at " << qwen_path << ", skipping physical model tests.";
        }
        llama_backend_init();
    }

    void TearDown() override {
        llama_backend_free();
    }
};

static std::string run_inference_prompt(
    LlamaCppBackend& backend,
    BackendSessionHandle s_handle,
    const std::string& full_prompt,
    int max_tokens = 32
) {
    std::vector<int32_t> prompt_tokens;
    Status eval_st = backend.evaluate_prompt(s_handle, full_prompt, prompt_tokens);
    if (!eval_st.is_ok()) return "EVAL_ERROR: " + eval_st.message;

    SamplingParams sparams;
    sparams.temperature = 0.0f; // Greedy
    sparams.repetition_penalty = 1.0f;
    sparams.max_tokens = max_tokens;

    std::string out_text;
    for (int step = 0; step < max_tokens; ++step) {
        auto tok_res = backend.sample_next_token(s_handle, sparams);
        if (!tok_res.is_ok()) break;
        int32_t tok_id = tok_res.value();
        if (backend.is_eos_token(s_handle, tok_id)) break;
        out_text += backend.decode_token(s_handle, tok_id);
    }
    return out_text;
}

// ============================================================================
// Phase 1 & 2: Model Inspection and GGUF Metadata Validation
// ============================================================================

TEST_F(InferenceCorrectnessTest, Phase1_Phase2_GGUFMetadataValidation) {
    auto meta_res = ModelLoader::inspect_file(qwen_path);
    ASSERT_TRUE(meta_res.is_ok()) << "Failed to inspect Qwen2.5 GGUF file: " << meta_res.status().message;
    const auto& meta = meta_res.value();

    EXPECT_EQ(meta.architecture, "qwen2");
    EXPECT_EQ(meta.gguf_version, 3);
    EXPECT_EQ(meta.block_count, 28);
    EXPECT_EQ(meta.head_count, 12);
    EXPECT_EQ(meta.head_count_kv, 2);
    EXPECT_EQ(meta.embedding_length, 1536);
    EXPECT_EQ(meta.quantization, "Q4_K_M");
}

// ============================================================================
// Workstream 1: Session Reuse Regression
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream1_SessionReuseRegression) {
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 2048;
    m_cfg.n_threads = 4;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.n_gpu_layers = 0;

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok()) << "Model load failed: " << m_res.status().message;
    BackendModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 2048;
    auto s_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    BackendSessionHandle s_reuse = s_res.value();

    std::string prompt_a = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n";
    std::string prompt_b = "<|im_start|>user\nWhat is the capital of France? Answer with just the city name.<|im_end|>\n<|im_start|>assistant\n";

    // Repeat session reuse cycle 3 times
    for (int iter = 0; iter < 3; ++iter) {
        // Step 1: Run Prompt A on reused session
        std::string reply_a = run_inference_prompt(backend, s_reuse, prompt_a, 16);
        EXPECT_NE(reply_a.find("4"), std::string::npos)
            << "[Iter " << iter << "] Prompt A failed: " << reply_a;

        // Step 2: Run unrelated Prompt B on SAME reused session
        std::string reply_b_reused = run_inference_prompt(backend, s_reuse, prompt_b, 16);
        EXPECT_NE(reply_b_reused.find("Paris"), std::string::npos)
            << "[Iter " << iter << "] Prompt B on reused session failed to contain 'Paris': " << reply_b_reused;

        // Step 3: Run Prompt B on a fresh session
        auto s_fresh_res = backend.create_session(m_handle, s_cfg);
        ASSERT_TRUE(s_fresh_res.is_ok());
        BackendSessionHandle s_fresh = s_fresh_res.value();

        std::string reply_b_fresh = run_inference_prompt(backend, s_fresh, prompt_b, 16);
        backend.destroy_session(s_fresh);

        // Step 4: Compare reused vs fresh execution
        EXPECT_EQ(reply_b_reused, reply_b_fresh)
            << "[Iter " << iter << "] Mismatch between reused and fresh session outputs!";

        // Step 5: Verify no contamination from Prompt A in Prompt B
        EXPECT_EQ(reply_b_reused.find("2 + 2"), std::string::npos);
        EXPECT_EQ(reply_b_reused.find("4"), std::string::npos);
    }

    backend.destroy_session(s_reuse);
    backend.unload_model(m_handle);
    backend.shutdown();
}

// ============================================================================
// Workstream 2: Multi-Turn Semantics
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream2_MultiTurnSemantics) {
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 2048;
    m_cfg.n_threads = 4;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.n_gpu_layers = 0;

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    BackendModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 2048;
    auto s_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    BackendSessionHandle session = s_res.value();

    // 1. Behavior: Continue existing conversation (KV cache preservation / Prefix caching)
    std::string turn1_prompt = "<|im_start|>user\nMy favorite secret code is PHOENIX99.<|im_end|>\n<|im_start|>assistant\nUnderstood, I will remember PHOENIX99.<|im_end|>\n";
    std::vector<int32_t> turn1_tokens;
    Status st1 = backend.evaluate_prompt(session, turn1_prompt, turn1_tokens);
    ASSERT_TRUE(st1.is_ok());

    std::string turn2_prompt = turn1_prompt + "<|im_start|>user\nWhat is my secret code?<|im_end|>\n<|im_start|>assistant\n";
    std::string reply_turn2 = run_inference_prompt(backend, session, turn2_prompt, 20);
    std::cout << "[Workstream 2: Turn 2 Reply] " << reply_turn2 << std::endl;
    EXPECT_NE(reply_turn2.find("PHOENIX99"), std::string::npos)
        << "Conversation continuation failed to retain context across turns!";

    // 2. Behavior: Reset conversation (explicit context-reset operation)
    Status reset_st = backend.reset_session(session);
    ASSERT_TRUE(reset_st.is_ok()) << "reset_session failed: " << reset_st.message;

    // After reset, query secret code again without prior context
    std::string post_reset_prompt = "<|im_start|>user\nWhat is my secret code?<|im_end|>\n<|im_start|>assistant\n";
    std::string reply_post_reset = run_inference_prompt(backend, session, post_reset_prompt, 20);
    std::cout << "[Workstream 2: Post-Reset Reply] " << reply_post_reset << std::endl;
    EXPECT_EQ(reply_post_reset.find("PHOENIX99"), std::string::npos)
        << "Reset failed: session retained memory of PHOENIX99 after reset_session()!";

    // 3. Behavior: Rebuild full conversation prompt (common prefix caching with new suffix)
    std::string rebuilt_prompt1 = "<|im_start|>user\nName the chemical formula for water.<|im_end|>\n<|im_start|>assistant\nH2O.<|im_end|>\n";
    std::string rebuilt_prompt2 = rebuilt_prompt1 + "<|im_start|>user\nAnd for table salt?<|im_end|>\n<|im_start|>assistant\n";
    std::string reply_salt = run_inference_prompt(backend, session, rebuilt_prompt2, 20);
    std::cout << "[Workstream 2: Salt Reply] " << reply_salt << std::endl;
    EXPECT_NE(reply_salt.find("NaCl"), std::string::npos);

    // 4. Behavior: Cancel and retry generation
    std::string cancel_prompt = "<|im_start|>user\nCount from 1 to 10.<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> cancel_toks;
    ASSERT_TRUE(backend.evaluate_prompt(session, cancel_prompt, cancel_toks).is_ok());
    // Sample only 2 tokens then cancel
    SamplingParams sparams;
    sparams.temperature = 0.0f;
    sparams.max_tokens = 2;
    backend.sample_next_token(session, sparams);
    backend.sample_next_token(session, sparams);
    // Cancellation / Retry on same session
    ASSERT_TRUE(backend.reset_session(session).is_ok());
    std::string retry_reply = run_inference_prompt(backend, session, "<|im_start|>user\nWhat is 3 + 3?<|im_end|>\n<|im_start|>assistant\n", 10);
    EXPECT_NE(retry_reply.find("6"), std::string::npos) << "Retry failed after cancellation: " << retry_reply;

    backend.destroy_session(session);
    backend.unload_model(m_handle);
    backend.shutdown();
}

// ============================================================================
// Workstream 3: Tokenizer Correctness
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream3_TokenizerCorrectness) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model* raw_model = llama_load_model_from_file(qwen_path.c_str(), mparams);
    ASSERT_NE(raw_model, nullptr);

    // 1. Qwen2.5 ChatML special tokens
    std::string special_text = "<|im_start|>system\n<|im_end|>\n<|endoftext|>";
    std::vector<llama_token> tokens(special_text.size() + 16);
    int32_t n_tokens = llama_tokenize(
        raw_model, special_text.c_str(), static_cast<int32_t>(special_text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        false, true // parse_special = true
    );
    ASSERT_GT(n_tokens, 0);
    tokens.resize(n_tokens);

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "  special_text tok[" << i << "] = " << tokens[i] << std::endl;
    }

    EXPECT_EQ(tokens[0], 151644); // <|im_start|>
    // Search for <|im_end|> and <|endoftext|>
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), 151644), tokens.end());
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), 151645), tokens.end());
    EXPECT_NE(std::find(tokens.begin(), tokens.end(), 151643), tokens.end());

    // 2. Check BOS configuration
    bool has_bos = llama_add_bos_token(raw_model);
    std::cout << "[Tokenizer Test] Model llama_add_bos_token: " << has_bos << std::endl;

    // 3. EOG tokens check via LlamaCppBackend
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());
    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 512;
    m_cfg.n_threads = 2;
    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    SessionConfig s_cfg;
    auto s_res = backend.create_session(m_res.value(), s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    BackendSessionHandle sess = s_res.value();

    EXPECT_TRUE(backend.is_eos_token(sess, 151645)); // <|im_end|>
    EXPECT_TRUE(backend.is_eos_token(sess, 151643)); // <|endoftext|>
    EXPECT_FALSE(backend.is_eos_token(sess, 151644)); // <|im_start|> is not EOS
    EXPECT_FALSE(backend.is_eos_token(sess, 100)); // arbitrary text token is not EOS

    // 4. Unicode & Chinese Roundtrip
    std::string chinese_sample = "你好，世界！人工智能模型正在进行推断测试。";
    std::vector<llama_token> c_tokens(chinese_sample.size() * 2);
    int32_t n_c = llama_tokenize(
        raw_model, chinese_sample.c_str(), static_cast<int32_t>(chinese_sample.size()),
        c_tokens.data(), static_cast<int32_t>(c_tokens.size()),
        false, true
    );
    ASSERT_GT(n_c, 0);
    c_tokens.resize(n_c);

    std::string reconstructed;
    for (auto tok : c_tokens) {
        reconstructed += backend.decode_token(sess, tok);
    }
    EXPECT_EQ(chinese_sample, reconstructed) << "Chinese roundtrip tokenization mismatch!";

    // 5. Empty and Malformed Prompts
    std::vector<int32_t> empty_out;
    Status empty_st = backend.evaluate_prompt(sess, "", empty_out);
    EXPECT_EQ(empty_st.code, StatusCode::INVALID_ARGUMENT)
        << "Empty prompt must be rejected with INVALID_ARGUMENT";

    std::vector<int32_t> malformed_out;
    Status malformed_st = backend.evaluate_prompt(sess, "<|im_start|> incomplete user query <|", malformed_out);
    EXPECT_TRUE(malformed_st.is_ok()) << "Malformed prompt should be parsed gracefully without crashing";

    backend.destroy_session(sess);
    backend.unload_model(m_res.value());
    backend.shutdown();
    llama_free_model(raw_model);
}

// ============================================================================
// Workstream 4: Numerical Correctness & Deterministic Greedy Inference
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream4_NumericalCorrectness) {
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 512;
    m_cfg.n_threads = 4;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.n_gpu_layers = 0;

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    BackendModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    auto s1_res = backend.create_session(m_handle, s_cfg);
    auto s2_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s1_res.is_ok());
    ASSERT_TRUE(s2_res.is_ok());

    std::string prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int32_t> toks1, toks2;

    ASSERT_TRUE(backend.evaluate_prompt(s1_res.value(), prompt, toks1).is_ok());
    ASSERT_TRUE(backend.evaluate_prompt(s2_res.value(), prompt, toks2).is_ok());

    auto l1_res = backend.get_last_logits(s1_res.value());
    auto l2_res = backend.get_last_logits(s2_res.value());
    ASSERT_TRUE(l1_res.is_ok()) << "get_last_logits failed for session 1";
    ASSERT_TRUE(l2_res.is_ok()) << "get_last_logits failed for session 2";

    const auto& l1 = l1_res.value();
    const auto& l2 = l2_res.value();
    ASSERT_EQ(l1.size(), l2.size());
    ASSERT_GT(l1.size(), 100000u) << "Logits size must equal vocabulary size (~152064)";

    // 1. Logit comparison: Max absolute error & Relative error
    float max_abs_error = 0.0f;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < l1.size(); ++i) {
        float diff = std::abs(l1[i] - l2[i]);
        if (diff > max_abs_error) max_abs_error = diff;
        sum_sq_diff += diff * diff;
        sum_sq_ref += l1[i] * l1[i];
    }
    double rel_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-9);

    std::cout << "[Numerical Correctness] Vocab Size:       " << l1.size() << std::endl;
    std::cout << "[Numerical Correctness] Max Absolute Err: " << max_abs_error << std::endl;
    std::cout << "[Numerical Correctness] Relative Error:   " << rel_error << std::endl;

    // Deterministic CPU inference must yield bitwise or near-identical logits (diff < 1e-5)
    EXPECT_LT(max_abs_error, 1e-5f) << "Deterministic CPU logits diverged!";
    EXPECT_LT(rel_error, 1e-6) << "Deterministic CPU relative error too high!";

    // 2. Argmax Top-1 token agreement
    auto it1 = std::max_element(l1.begin(), l1.end());
    auto it2 = std::max_element(l2.begin(), l2.end());
    int32_t top1_s1 = static_cast<int32_t>(std::distance(l1.begin(), it1));
    int32_t top1_s2 = static_cast<int32_t>(std::distance(l2.begin(), it2));
    EXPECT_EQ(top1_s1, top1_s2);

    std::string top1_piece = backend.decode_token(s1_res.value(), top1_s1);
    std::cout << "[Numerical Correctness] Top-1 Token ID: " << top1_s1 << " ('" << top1_piece << "')" << std::endl;
    EXPECT_EQ(top1_s1, top1_s2);

    // Verify greedy sampling continues deterministically and produces correct answer containing "4"
    SamplingParams sparams;
    sparams.temperature = 0.0f;
    sparams.max_tokens = 8;
    std::string gen_seq = top1_piece;
    for (int step = 0; step < 8; ++step) {
        auto tok_res = backend.sample_next_token(s1_res.value(), sparams);
        if (!tok_res.is_ok()) break;
        int32_t tid = tok_res.value();
        if (backend.is_eos_token(s1_res.value(), tid)) break;
        gen_seq += backend.decode_token(s1_res.value(), tid);
    }
    std::cout << "[Numerical Correctness] Greedy sequence: \"" << gen_seq << "\"" << std::endl;
    EXPECT_NE(gen_seq.find("4"), std::string::npos);

    // 3. Top-k token agreement (k = 5)
    std::vector<size_t> indices1(l1.size()), indices2(l2.size());
    for (size_t i = 0; i < l1.size(); ++i) { indices1[i] = i; indices2[i] = i; }
    std::partial_sort(indices1.begin(), indices1.begin() + 5, indices1.end(),
                      [&l1](size_t a, size_t b) { return l1[a] > l1[b]; });
    std::partial_sort(indices2.begin(), indices2.begin() + 5, indices2.end(),
                      [&l2](size_t a, size_t b) { return l2[a] > l2[b]; });

    int top_k_matches = 0;
    for (int k = 0; k < 5; ++k) {
        if (indices1[k] == indices2[k]) top_k_matches++;
    }
    EXPECT_EQ(top_k_matches, 5) << "Top-5 tokens must agree 100% across deterministic CPU sessions";

    backend.destroy_session(s1_res.value());
    backend.destroy_session(s2_res.value());
    backend.unload_model(m_handle);
    backend.shutdown();
}

// ============================================================================
// Workstream 5: Cross-Backend Regression & Fallback Verification
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream5_CrossBackendRegression) {
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());

    // Test fallback when Vulkan backend is preferred or layers requested on CPU-only system
    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 512;
    m_cfg.n_threads = 4;
    m_cfg.preferred_backend = BackendType::VULKAN;
    m_cfg.n_gpu_layers = 99; // Request all layers on GPU

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok()) << "Model load with Vulkan preference/fallback failed: " << m_res.status().message;
    BackendModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    auto s_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());

    std::string prompt = "<|im_start|>user\nWhat is 5 * 5?<|im_end|>\n<|im_start|>assistant\n";
    std::string reply = run_inference_prompt(backend, s_res.value(), prompt, 16);
    std::cout << "[Cross-Backend Reply] " << reply << std::endl;
    EXPECT_NE(reply.find("25"), std::string::npos);

    backend.destroy_session(s_res.value());
    backend.unload_model(m_handle);
    backend.shutdown();
}

// ============================================================================
// Workstream 6: Resource and Concurrency Safety
// ============================================================================

TEST_F(InferenceCorrectnessTest, Workstream6_ResourceConcurrencySafety) {
    LlamaCppBackend backend;
    BackendConfig b_cfg;
    ASSERT_TRUE(backend.initialize(b_cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = qwen_path;
    m_cfg.context_length = 512;
    m_cfg.n_threads = 2;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.n_gpu_layers = 0;

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    BackendModelHandle m_handle = m_res.value();

    // 1. Rapid session lifecycle (create, eval, sample, destroy x 5)
    for (int i = 0; i < 5; ++i) {
        SessionConfig s_cfg;
        auto s_res = backend.create_session(m_handle, s_cfg);
        ASSERT_TRUE(s_res.is_ok());
        BackendSessionHandle sess = s_res.value();

        std::vector<int32_t> toks;
        ASSERT_TRUE(backend.evaluate_prompt(sess, "Hello world", toks).is_ok());
        SamplingParams sparams;
        sparams.temperature = 0.0f;
        sparams.max_tokens = 2;
        backend.sample_next_token(sess, sparams);

        backend.destroy_session(sess);
    }

    // 2. Concurrency Safety: Multi-threaded generation on distinct sessions of the same model
    SessionConfig s_cfg;
    auto s1 = backend.create_session(m_handle, s_cfg);
    auto s2 = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s1.is_ok() && s2.is_ok());

    auto run_thread = [&](BackendSessionHandle sess, const std::string& query, const std::string& expected) {
        std::string prompt = "<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n";
        std::string out = run_inference_prompt(backend, sess, prompt, 16);
        return out.find(expected) != std::string::npos;
    };

    auto f1 = std::async(std::launch::async, run_thread, s1.value(), "What is 10 + 10?", "20");
    auto f2 = std::async(std::launch::async, run_thread, s2.value(), "What is 7 * 7?", "49");

    bool ok1 = f1.get();
    bool ok2 = f2.get();

    EXPECT_TRUE(ok1) << "Concurrent thread 1 inference failed";
    EXPECT_TRUE(ok2) << "Concurrent thread 2 inference failed";

    backend.destroy_session(s1.value());
    backend.destroy_session(s2.value());

    // 3. Low-memory budget enforcement
    ModelLoadConfig low_mem_cfg = m_cfg;
    low_mem_cfg.memory_budget_mb = 1.0f; // Insufficient for 1GB model
    auto low_mem_res = backend.load_model(low_mem_cfg);
    // Either rejects with OUT_OF_MEMORY or logs warning and handles safely
    if (!low_mem_res.is_ok()) {
        EXPECT_EQ(low_mem_res.status().code, StatusCode::OUT_OF_MEMORY);
    } else {
        backend.unload_model(low_mem_res.value());
    }

    backend.unload_model(m_handle);
    backend.shutdown();
}

// ============================================================================
// Chunked Prompt Evaluation Test (Prompt Longer than n_batch)
// ============================================================================

TEST_F(InferenceCorrectnessTest, PromptLongerThanNBatch_ChunkedEvaluation) {
    if (!std::filesystem::exists(smollm_path)) {
        GTEST_SKIP() << "SmolLM2 model not found at " << smollm_path;
    }

    LlamaCppBackend backend;
    ASSERT_TRUE(backend.initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = smollm_path;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.context_length = 2048;
    m_cfg.n_threads = 4;

    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok()) << "Failed to load SmolLM2 model: " << m_res.status().message;
    BackendModelHandle m_handle = m_res.value();

    // Explicitly configure n_batch = 256 on a 2048-token context session
    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 2048;
    s_cfg.n_batch = 256;
    s_cfg.n_ubatch = 256;
    s_cfg.enable_prefix_caching = false;

    auto s_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok()) << "Failed to create session with n_batch=256: " << s_res.status().message;
    BackendSessionHandle s_handle = s_res.value();

    // Construct a prompt of at least 1024 tokens (4x larger than n_batch = 256)
    std::string repeat_unit = "The quick brown fox jumps over the lazy dog in the sunny summer afternoon. ";
    std::string long_prompt = "<|im_start|>user\n";
    while (long_prompt.size() < 6000) {
        long_prompt += repeat_unit;
    }
    long_prompt += "\nSummarize the text in one word.<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int32_t> out_tokens;
    PromptEvalInfo out_info{};
    Status eval_st = backend.evaluate_prompt(s_handle, long_prompt, out_tokens, &out_info);

    ASSERT_TRUE(eval_st.is_ok()) << "evaluate_prompt failed on prompt longer than n_batch: " << eval_st.message;
    EXPECT_GE(out_tokens.size(), 1024u) << "Expected prompt tokens to be at least 1024, got " << out_tokens.size();
    EXPECT_GE(out_info.eval_calls, 4) << "Expected at least 4 chunk decode calls for 1024 tokens with n_batch=256, got " << out_info.eval_calls;
    EXPECT_EQ(out_info.prompt_tokens, out_tokens.size());

    // Verify that logits for the final token are valid and non-NaN
    auto logits_res = backend.get_last_logits(s_handle);
    ASSERT_TRUE(logits_res.is_ok());
    const auto& logits = logits_res.value();
    ASSERT_FALSE(logits.empty());
    bool has_nan = false;
    for (float v : logits) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan) << "Logits contain NaN or Inf values after chunked prompt evaluation!";

    // Verify that subsequent token generation succeeds
    SamplingParams sparams;
    sparams.temperature = 0.0f; // Greedy
    sparams.max_tokens = 4;
    auto tok_res = backend.sample_next_token(s_handle, sparams);
    ASSERT_TRUE(tok_res.is_ok()) << "sample_next_token failed after chunked prompt evaluation: " << tok_res.status().message;
    EXPECT_GE(tok_res.value(), 0);

    EXPECT_TRUE(backend.destroy_session(s_handle).is_ok());
    EXPECT_TRUE(backend.unload_model(m_handle).is_ok());
    EXPECT_TRUE(backend.shutdown().is_ok());
}
