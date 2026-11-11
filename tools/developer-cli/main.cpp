#include "mobilellm/engine.h"
#include "mobilellm/model_loader.h"
#include <iostream>
#include <string>
#include <vector>
#include <future>
#include <chrono>

void print_usage() {
    std::cout << "MobileLLM CLI — Developer Tooling\n"
              << "Usage:\n"
              << "  mobilellm-cli version\n"
              << "  mobilellm-cli info --model <path-to-gguf>\n"
              << "  mobilellm-cli run --model <path-to-gguf> --prompt <text> [--max-tokens 128] [--temp 0.7]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    std::string command = argv[1];

    if (command == "version") {
        std::cout << "MobileLLM Engine v0.1.0-dev (Phase 1 Baseline)\n";
        return 0;
    }

    if (command == "info") {
        std::string model_path;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            }
        }
        if (model_path.empty()) {
            std::cerr << "Error: --model argument required\n";
            return 1;
        }

        auto res = mobilellm::ModelLoader::inspect_file(model_path);
        if (!res.is_ok()) {
            std::cerr << "Validation Error: " << res.status().message << "\n";
            return 1;
        }

        const auto& meta = res.value();
        std::cout << "GGUF Model Information:\n"
                  << "  File: " << meta.filepath << "\n"
                  << "  GGUF Version: " << meta.gguf_version << "\n"
                  << "  Architecture: " << meta.architecture << "\n"
                  << "  Tensors: " << meta.tensor_count << "\n"
                  << "  Context Length: " << meta.context_length << "\n"
                  << "  Size: " << (meta.file_size_bytes / (1024 * 1024)) << " MB\n";
        return 0;
    }

    if (command == "run") {
        std::string model_path;
        std::string prompt = "Hello! Who are you?";
        uint32_t max_tokens = 64;
        float temperature = 0.7f;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                prompt = argv[++i];
            } else if (arg == "--max-tokens" && i + 1 < argc) {
                max_tokens = std::stoul(argv[++i]);
            } else if (arg == "--temp" && i + 1 < argc) {
                temperature = std::stof(argv[++i]);
            }
        }

        if (model_path.empty()) {
            std::cerr << "Error: --model argument required\n";
            return 1;
        }

        auto engine = mobilellm::create_engine();
        mobilellm::EngineConfig cfg;
        engine->initialize(cfg);

        mobilellm::ModelLoadConfig m_cfg;
        m_cfg.model_path = model_path;
        auto m_res = engine->load_model(m_cfg);
        if (!m_res.is_ok()) {
            std::cerr << "Failed to load model: " << m_res.status().message << "\n";
            return 1;
        }

        mobilellm::SessionConfig s_cfg;
        auto s_res = engine->create_session(m_res.value(), s_cfg);
        if (!s_res.is_ok()) {
            std::cerr << "Failed to create session: " << s_res.status().message << "\n";
            return 1;
        }

        mobilellm::GenerationRequest req;
        req.prompt = prompt;
        req.sampling.max_tokens = max_tokens;
        req.sampling.temperature = temperature;

        std::cout << "\n[Prompt]: " << prompt << "\n[Response]: " << std::flush;

        std::promise<void> done_promise;
        auto done_future = done_promise.get_future();

        mobilellm::GenerationMetrics final_metrics;
        mobilellm::Status final_status = mobilellm::Status::ok();

        auto token_cb = [](const mobilellm::TokenChunk& chunk) -> bool {
            std::cout << chunk.text << std::flush;
            return true;
        };

        auto complete_cb = [&](const mobilellm::Status& st, const mobilellm::GenerationMetrics& m) {
            final_status = st;
            final_metrics = m;
            done_promise.set_value();
        };

        auto gen_res = engine->generate_async(s_res.value(), req, token_cb, complete_cb);
        if (!gen_res.is_ok()) {
            std::cerr << "Generation error: " << gen_res.status().message << "\n";
            return 1;
        }

        done_future.wait();
        std::cout << "\n\n--- Performance Metrics ---\n"
                  << "Prompt tokens: " << final_metrics.prompt_tokens << "\n"
                  << "Completion tokens: " << final_metrics.completion_tokens << "\n"
                  << "Time to First Token (TTFT): " << final_metrics.time_to_first_token_ms << " ms\n"
                  << "Tokens per second: " << final_metrics.decode_tokens_per_second << " tok/s\n";

        engine->shutdown();
        return final_status.is_ok() ? 0 : 1;
    }

    print_usage();
    return 1;
}

