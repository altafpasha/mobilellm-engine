#include <gtest/gtest.h>
#include "runtime/src/sampling/sampler.h"
#include <cmath>
#include <future>
#include <thread>
#include <vector>

using namespace mobilellm;

TEST(SamplingTest, GreedySampling) {
    Sampler sampler(42);
    std::vector<float> logits = {0.1f, 5.0f, 0.2f, 1.0f};
    std::vector<int32_t> history;

    SamplingParams params;
    params.temperature = 0.0f; // Greedy

    int32_t token = sampler.sample(logits, history, params);
    EXPECT_EQ(token, 1); // index 1 has highest logit (5.0f)
}

TEST(SamplingTest, RepetitionPenalty) {
    std::vector<float> logits = {2.0f, 4.0f, 1.0f};
    std::vector<int32_t> history = {1}; // Token 1 was recently generated

    Sampler::apply_repetition_penalty(logits, history, 2.0f);
    // Logit 1 was 4.0f, with penalty 2.0f it becomes 4.0 / 2.0 = 2.0f
    EXPECT_NEAR(logits[1], 2.0f, 1e-4f);
}

TEST(SamplingTest, SoftmaxProbabilitySumToOne) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    Sampler::softmax(logits);

    float sum = 0.0f;
    for (float p : logits) {
        sum += p;
        EXPECT_GT(p, 0.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SamplingTest, TopKFiltering) {
    std::vector<std::pair<float, int32_t>> pairs = {
        {0.5f, 1}, {0.3f, 2}, {0.1f, 3}, {0.05f, 4}, {0.05f, 5}
    };
    Sampler::apply_top_k(pairs, 2);
    EXPECT_EQ(pairs.size(), 2);
    EXPECT_EQ(pairs[0].second, 1);
    EXPECT_EQ(pairs[1].second, 2);
}

TEST(SamplingTest, TopPFiltering) {
    std::vector<std::pair<float, int32_t>> pairs = {
        {0.6f, 1}, {0.25f, 2}, {0.1f, 3}, {0.05f, 4}
    };
    Sampler::apply_top_p(pairs, 0.8f);
    // 0.6 + 0.25 = 0.85 >= 0.8 -> keep top 2
    EXPECT_EQ(pairs.size(), 2);
}

TEST(SamplingTest, DeterministicSeededGeneration) {
    constexpr uint64_t SEED = 987654321ULL;
    Sampler s1(SEED);
    Sampler s2(SEED);

    SamplingParams params;
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.top_k = 10;

    std::vector<int32_t> history1;
    std::vector<int32_t> history2;

    for (int step = 0; step < 20; ++step) {
        std::vector<float> logits1 = {0.2f, 1.5f, 0.8f, 2.1f, 0.05f, 1.8f};
        std::vector<float> logits2 = logits1;

        int32_t tok1 = s1.sample(logits1, history1, params);
        int32_t tok2 = s2.sample(logits2, history2, params);

        EXPECT_EQ(tok1, tok2) << "Discrepancy at step " << step;
        history1.push_back(tok1);
        history2.push_back(tok2);
    }
}

TEST(SamplingTest, ConcurrentSamplingAcrossIndependentSamplers) {
    constexpr uint64_t BASE_SEED = 12345ULL;
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_STEPS = 50;

    // Precompute expected sequence for BASE_SEED
    std::vector<int32_t> expected_seq;
    {
        Sampler reference(BASE_SEED);
        SamplingParams params;
        params.temperature = 0.7f;
        std::vector<int32_t> hist;
        for (int i = 0; i < NUM_STEPS; ++i) {
            std::vector<float> logits = {0.5f, 2.0f, 1.2f, 0.8f, 0.3f};
            int32_t t = reference.sample(logits, hist, params);
            hist.push_back(t);
            expected_seq.push_back(t);
        }
    }

    // Launch concurrent samplers each with BASE_SEED
    std::vector<std::future<std::vector<int32_t>>> futures;
    for (int t = 0; t < NUM_THREADS; ++t) {
        futures.push_back(std::async(std::launch::async, []() {
            Sampler s(BASE_SEED);
            SamplingParams params;
            params.temperature = 0.7f;
            std::vector<int32_t> hist;
            std::vector<int32_t> seq;
            for (int i = 0; i < NUM_STEPS; ++i) {
                std::vector<float> logits = {0.5f, 2.0f, 1.2f, 0.8f, 0.3f};
                int32_t tok = s.sample(logits, hist, params);
                hist.push_back(tok);
                seq.push_back(tok);
            }
            return seq;
        }));
    }

    for (int t = 0; t < NUM_THREADS; ++t) {
        auto seq = futures[t].get();
        EXPECT_EQ(seq, expected_seq) << "Thread " << t << " sequence diverged!";
    }
}

