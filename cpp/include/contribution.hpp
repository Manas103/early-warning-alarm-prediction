// Per-channel "contribution" score shown to the operator alongside a
// prediction. This is deliberately NOT a gradient-based attribution (that
// would need autodiff support the hand-rolled forward pass does not have).
// It is a magnitude-of-deviation-from-baseline ranking: for each channel,
// the mean absolute z-score (using the same training-set mean/std the model
// itself was normalized with) over the scored window. Channels sitting far
// from their training baseline for sustained periods rank highest. This is
// an honest, cheap proxy for "which tags looked unusual", not a rigorous
// measure of which channels actually drove the network's decision; the
// README documents that tradeoff explicitly.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "model.hpp"

namespace ewap {

struct ChannelContribution {
    int channel_index;
    float mean_abs_zscore;
};

inline std::array<float, kNChannels> compute_contribution_scores(
    const std::vector<float>& window_raw, const Normalization& norm) {
    std::array<float, kNChannels> scores{};
    for (int c = 0; c < kNChannels; ++c) {
        float sum = 0.0f;
        for (int t = 0; t < kWindowMinutes; ++t) {
            float raw = window_raw[t * kNChannels + c];
            float z = (raw - norm.mean[c]) / norm.std[c];
            sum += std::fabs(z);
        }
        scores[c] = sum / static_cast<float>(kWindowMinutes);
    }
    return scores;
}

// Returns the top_k channel indices sorted by descending contribution score.
inline std::vector<ChannelContribution> top_k_contributors(
    const std::array<float, kNChannels>& scores, int top_k) {
    std::vector<ChannelContribution> all;
    all.reserve(kNChannels);
    for (int c = 0; c < kNChannels; ++c) all.push_back({c, scores[c]});
    std::sort(all.begin(), all.end(), [](const ChannelContribution& a, const ChannelContribution& b) {
        return a.mean_abs_zscore > b.mean_abs_zscore;
    });
    if (static_cast<int>(all.size()) > top_k) all.resize(top_k);
    return all;
}

}  // namespace ewap
