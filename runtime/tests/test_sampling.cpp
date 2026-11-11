#include <gtest/gtest.h>
#include "runtime/src/sampling/sampler.h"
#include <cmath>

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

