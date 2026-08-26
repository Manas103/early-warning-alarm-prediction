// Hand-rolled main()-based check suite for the C++ side (no GoogleTest
// dependency needed for a suite this small). Run via ctest or directly;
// returns nonzero and prints which check failed on any failure.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "channel_names.hpp"
#include "contribution.hpp"
#include "model.hpp"

using namespace ewap;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool cond, const std::string& msg) {
    g_checks++;
    if (!cond) {
        g_failures++;
        std::printf("  FAIL: %s\n", msg.c_str());
    } else {
        std::printf("  ok:   %s\n", msg.c_str());
    }
}

std::string weights_path, norm_path;

void test_normalization_load() {
    std::printf("test_normalization_load\n");
    Normalization norm = Normalization::load(norm_path);
    // std must never be exactly zero (division safety, guaranteed by
    // features.py::channel_stats clamping to 1e-6)
    bool all_positive = true;
    for (int c = 0; c < kNChannels; ++c) {
        if (norm.std[c] <= 0.0f) all_positive = false;
    }
    check(all_positive, "all 24 channel std values are positive");
}

void test_model_load_shapes() {
    std::printf("test_model_load_shapes\n");
    // load() itself throws on any size mismatch; reaching here means the
    // weights blob size matched the expected architecture exactly.
    bool threw = false;
    try {
        EarlyWarningModel::load(weights_path);
    } catch (...) {
        threw = true;
    }
    check(!threw, "weights.bin loads without a size-mismatch exception");
}

void test_forward_is_deterministic() {
    std::printf("test_forward_is_deterministic\n");
    EarlyWarningModel model = EarlyWarningModel::load(weights_path);
    Normalization norm = Normalization::load(norm_path);
    std::vector<float> window(kWindowMinutes * kNChannels, 0.0f);
    for (int t = 0; t < kWindowMinutes; ++t) {
        for (int c = 0; c < kNChannels; ++c) {
            window[t * kNChannels + c] = norm.mean[c] + 0.1f * static_cast<float>((t + c) % 5);
        }
    }
    float p1 = model.forward(window, norm);
    float p2 = model.forward(window, norm);
    check(p1 == p2, "same window scored twice gives bit-identical probability");
    check(p1 >= 0.0f && p1 <= 1.0f, "probability is in [0, 1]");
}

void test_forward_reacts_to_a_large_deviation() {
    std::printf("test_forward_reacts_to_a_large_deviation\n");
    EarlyWarningModel model = EarlyWarningModel::load(weights_path);
    Normalization norm = Normalization::load(norm_path);

    std::vector<float> baseline(kWindowMinutes * kNChannels);
    for (int t = 0; t < kWindowMinutes; ++t)
        for (int c = 0; c < kNChannels; ++c)
            baseline[t * kNChannels + c] = norm.mean[c];
    float p_baseline = model.forward(baseline, norm);

    // ramp channel 0 up by 8 std over the window, mimicking a real upset
    std::vector<float> upset = baseline;
    for (int t = 0; t < kWindowMinutes; ++t) {
        float frac = static_cast<float>(t) / static_cast<float>(kWindowMinutes - 1);
        upset[t * kNChannels + 0] = norm.mean[0] + frac * 8.0f * norm.std[0];
    }
    float p_upset = model.forward(upset, norm);

    check(p_baseline >= 0.0f && p_baseline <= 1.0f, "baseline-window probability in [0,1]");
    // this is a smoke check, not a claim about calibration: a model trained
    // on ramps should not be perfectly insensitive to a textbook ramp.
    check(std::fabs(p_upset - p_baseline) > 1e-6f,
          "a synthetic ramp window produces a different probability than a flat baseline window");
}

void test_contribution_ranks_the_deviating_channel_first() {
    std::printf("test_contribution_ranks_the_deviating_channel_first\n");
    Normalization norm = Normalization::load(norm_path);
    std::vector<float> window(kWindowMinutes * kNChannels);
    for (int t = 0; t < kWindowMinutes; ++t)
        for (int c = 0; c < kNChannels; ++c)
            window[t * kNChannels + c] = norm.mean[c];
    // push channel 5 far from baseline, every other channel stays exactly
    // at the training mean (zero z-score)
    for (int t = 0; t < kWindowMinutes; ++t) {
        window[t * kNChannels + 5] = norm.mean[5] + 10.0f * norm.std[5];
    }
    auto scores = compute_contribution_scores(window, norm);
    auto top = top_k_contributors(scores, 3);
    check(top[0].channel_index == 5,
          "the one deliberately-deviated channel ranks as the top contributor");
    check(top[0].mean_abs_zscore > top[1].mean_abs_zscore,
          "top contributor's score strictly exceeds the runner-up's");
}

void test_channel_names_are_unique_and_ordered() {
    std::printf("test_channel_names_are_unique_and_ordered\n");
    auto names = channel_names();
    check(names[0] == "pressure_01", "first channel name is pressure_01");
    check(names[6] == "temperature_01", "7th channel name is temperature_01");
    check(names[12] == "flow_01", "13th channel name is flow_01");
    check(names[18] == "vibration_01", "19th channel name is vibration_01");
    bool all_unique = true;
    for (int i = 0; i < kNChannels; ++i)
        for (int j = i + 1; j < kNChannels; ++j)
            if (names[i] == names[j]) all_unique = false;
    check(all_unique, "all 24 channel names are unique");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_model <weights.bin> <norm.txt>\n");
        return 2;
    }
    weights_path = argv[1];
    norm_path = argv[2];

    test_normalization_load();
    test_model_load_shapes();
    test_forward_is_deterministic();
    test_forward_reacts_to_a_large_deviation();
    test_contribution_ranks_the_deviating_channel_first();
    test_channel_names_are_unique_and_ordered();

    std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
