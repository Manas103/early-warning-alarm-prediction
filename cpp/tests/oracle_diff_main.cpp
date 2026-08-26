// Reference-oracle diff: replays the exact windows PyTorch scored (dumped
// by python/train.py::export_oracle_reference) through the hand-rolled C++
// forward pass and compares logits/probabilities to PyTorch's own output on
// the same input, to a stated float tolerance. This is the test that
// actually proves the C++ port is correct, not merely plausible.
//
// Usage: oracle_diff <weights.bin> <norm.txt> <oracle_windows.bin> <oracle_targets.txt>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "model.hpp"

using namespace ewap;

namespace {
constexpr float kLogitTolerance = 1e-3f;
constexpr float kProbTolerance = 1e-4f;
}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: oracle_diff <weights.bin> <norm.txt> <oracle_windows.bin> <oracle_targets.txt>\n";
        return 2;
    }
    EarlyWarningModel model = EarlyWarningModel::load(argv[1]);
    Normalization norm = Normalization::load(argv[2]);

    std::ifstream windows_f(argv[3], std::ios::binary);
    if (!windows_f) { std::cerr << "cannot open " << argv[3] << "\n"; return 2; }
    std::ifstream targets_f(argv[4]);
    if (!targets_f) { std::cerr << "cannot open " << argv[4] << "\n"; return 2; }

    int n;
    targets_f >> n;
    std::cout << "oracle_diff: comparing " << n << " reference windows\n";

    float max_logit_diff = 0.0f;
    float max_prob_diff = 0.0f;
    int failures = 0;

    for (int i = 0; i < n; ++i) {
        std::vector<float> window(kWindowMinutes * kNChannels);
        windows_f.read(reinterpret_cast<char*>(window.data()), window.size() * sizeof(float));
        if (!windows_f) { std::cerr << "oracle_windows.bin truncated at sample " << i << "\n"; return 2; }

        float expected_logit, expected_prob;
        targets_f >> expected_logit >> expected_prob;

        float actual_logit = model.forward_logit(window, norm);
        float actual_prob = 1.0f / (1.0f + std::exp(-actual_logit));

        float logit_diff = std::fabs(actual_logit - expected_logit);
        float prob_diff = std::fabs(actual_prob - expected_prob);
        max_logit_diff = std::max(max_logit_diff, logit_diff);
        max_prob_diff = std::max(max_prob_diff, prob_diff);

        bool ok = (logit_diff <= kLogitTolerance) && (prob_diff <= kProbTolerance);
        if (!ok) failures++;
        std::printf("  sample %2d: pytorch_logit=%.6f cpp_logit=%.6f diff=%.6g | "
                    "pytorch_prob=%.6f cpp_prob=%.6f diff=%.6g %s\n",
                    i, expected_logit, actual_logit, logit_diff,
                    expected_prob, actual_prob, prob_diff, ok ? "OK" : "FAIL");
    }

    std::printf("\nmax_logit_diff=%.6g (tolerance %.6g)\n", max_logit_diff, kLogitTolerance);
    std::printf("max_prob_diff=%.6g (tolerance %.6g)\n", max_prob_diff, kProbTolerance);

    if (failures > 0) {
        std::printf("ORACLE DIFF FAILED: %d/%d samples exceeded tolerance\n", failures, n);
        return 1;
    }
    std::printf("ORACLE DIFF PASSED: all %d samples within tolerance\n", n);
    return 0;
}
