#pragma once

#include "mobilellm/types.h"
#include <vector>
#include <random>

namespace mobilellm {

class Sampler {
public:
    explicit Sampler(uint64_t seed = 42);

    void set_seed(uint64_t seed) { rng_.seed(seed); }

    /**
     * @brief Samples a single token id from raw logits according to SamplingParams.
     */
    int32_t sample(
        std::vector<float>& logits,
        const std::vector<int32_t>& history,
        const SamplingParams& params
    );

    /**
     * @brief Modifies logits in-place to apply repetition penalty against recently generated tokens.
     */
    static void apply_repetition_penalty(
        std::vector<float>& logits,
        const std::vector<int32_t>& history,
        float penalty
    );

    /**
     * @brief Scales logits by temperature (temperature <= 0 implies greedy).
     */
    static void apply_temperature(std::vector<float>& logits, float temperature);

    /**
     * @brief Calculates softmax probabilities over logits in-place.
     */
    static void softmax(std::vector<float>& logits);

    /**
     * @brief Filters probabilities keeping only the top K highest.
     */
    static void apply_top_k(std::vector<std::pair<float, int32_t>>& prob_token_pairs, int32_t top_k);

    /**
     * @brief Filters probabilities keeping cumulative mass up to top_p.
     */
    static void apply_top_p(std::vector<std::pair<float, int32_t>>& prob_token_pairs, float top_p);

    /**
     * @brief Filters probabilities keeping tokens with p >= min_p * max_p.
     */
    static void apply_min_p(std::vector<std::pair<float, int32_t>>& prob_token_pairs, float min_p);

private:
    std::mt19937_64 rng_;
};

} // namespace mobilellm

