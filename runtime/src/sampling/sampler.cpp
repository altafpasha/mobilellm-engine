#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace mobilellm {

Sampler::Sampler(uint64_t seed) : rng_(seed) {}

void Sampler::apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<int32_t>& history,
    float penalty
) {
    if (penalty <= 1.0f || history.empty()) return;

    std::unordered_set<int32_t> unique_tokens(history.begin(), history.end());
    for (int32_t tok : unique_tokens) {
        if (tok >= 0 && static_cast<size_t>(tok) < logits.size()) {
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

void Sampler::apply_temperature(std::vector<float>& logits, float temperature) {
    if (temperature <= 0.0f) return;
    const float inv_t = 1.0f / temperature;
    for (float& l : logits) {
        l *= inv_t;
    }
}

void Sampler::softmax(std::vector<float>& logits) {
    if (logits.empty()) return;
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float& l : logits) {
        l = std::exp(l - max_l);
        sum += l;
    }
    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (float& l : logits) {
            l *= inv_sum;
        }
    }
}

void Sampler::apply_top_k(std::vector<std::pair<float, int32_t>>& prob_token_pairs, int32_t top_k) {
    if (top_k <= 0 || static_cast<size_t>(top_k) >= prob_token_pairs.size()) return;

    std::partial_sort(
        prob_token_pairs.begin(),
        prob_token_pairs.begin() + top_k,
        prob_token_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    prob_token_pairs.resize(top_k);
}

void Sampler::apply_top_p(std::vector<std::pair<float, int32_t>>& prob_token_pairs, float top_p) {
    if (top_p >= 1.0f || prob_token_pairs.empty()) return;

    std::sort(
        prob_token_pairs.begin(),
        prob_token_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    float cumulative = 0.0f;
    size_t keep = 0;
    for (size_t i = 0; i < prob_token_pairs.size(); ++i) {
        cumulative += prob_token_pairs[i].first;
        keep = i + 1;
        if (cumulative >= top_p) break;
    }
    prob_token_pairs.resize(keep);
}

void Sampler::apply_min_p(std::vector<std::pair<float, int32_t>>& prob_token_pairs, float min_p) {
    if (min_p <= 0.0f || prob_token_pairs.empty()) return;

    float max_p = 0.0f;
    for (const auto& item : prob_token_pairs) {
        if (item.first > max_p) max_p = item.first;
    }

    const float cutoff = max_p * min_p;
    prob_token_pairs.erase(
        std::remove_if(
            prob_token_pairs.begin(),
            prob_token_pairs.end(),
            [cutoff](const auto& item) { return item.first < cutoff; }
        ),
        prob_token_pairs.end()
    );
}

int32_t Sampler::sample(
    std::vector<float>& logits,
    const std::vector<int32_t>& history,
    const SamplingParams& params
) {
    if (logits.empty()) return 0;

    // Apply repetition penalty
    apply_repetition_penalty(logits, history, params.repetition_penalty);

    // Greedy sampling shortcut if temperature <= 0.0
    if (params.temperature <= 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return static_cast<int32_t>(std::distance(logits.begin(), max_it));
    }

    // Apply temperature
    apply_temperature(logits, params.temperature);

    // Convert to probabilities
    softmax(logits);

    // Build pairs of (prob, token_id)
    std::vector<std::pair<float, int32_t>> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > 1e-6f) {
            candidates.emplace_back(logits[i], static_cast<int32_t>(i));
        }
    }

    if (candidates.empty()) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return static_cast<int32_t>(std::distance(logits.begin(), max_it));
    }

    // Apply filters
    apply_top_k(candidates, params.top_k);
    apply_top_p(candidates, params.top_p);
    apply_min_p(candidates, params.min_p);

    if (candidates.empty()) {
        return 0;
    }

    // Renormalize probabilities
    float sum_p = 0.0f;
    for (const auto& c : candidates) {
        sum_p += c.first;
    }

    // Multinomial random selection
    std::uniform_real_distribution<float> dist(0.0f, sum_p);
    float r = dist(rng_);
    float acc = 0.0f;
    for (const auto& c : candidates) {
        acc += c.first;
        if (acc >= r) {
            return c.second;
        }
    }

    return candidates.back().second;
}

} // namespace mobilellm

