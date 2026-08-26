// Hand-rolled forward pass for the early-warning CNN, ported minute-for-line
// from python/model.py::EarlyWarningCNN. Keep the two in lock-step; the
// architecture constants below MUST match python/model.py exactly or
// oracle_diff will (correctly) fail.
//
// weights.bin layout (flat float32, no header), written by
// python/train.py::export_weights, in this exact order:
//   conv1.weight (16,24,5)  conv1.bias (16)
//   conv2.weight (8,16,5)   conv2.bias (8)
//   fc.weight (1,8)         fc.bias (1)
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ewap {

constexpr int kWindowMinutes = 60;
constexpr int kNChannels = 24;
constexpr int kConv1Out = 16;
constexpr int kConv1K = 5;
constexpr int kConv1Len = kWindowMinutes - kConv1K + 1;   // 56
constexpr int kConv2Out = 8;
constexpr int kConv2K = 5;
constexpr int kConv2Len = kConv1Len - kConv2K + 1;        // 52

struct Normalization {
    std::array<float, kNChannels> mean{};
    std::array<float, kNChannels> std{};

    static Normalization load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open norm file: " + path);
        int n;
        f >> n;
        if (n != kNChannels) {
            throw std::runtime_error("norm file channel count mismatch: expected " +
                                      std::to_string(kNChannels) + " got " + std::to_string(n));
        }
        Normalization out;
        for (int i = 0; i < n; ++i) {
            f >> out.mean[i] >> out.std[i];
        }
        if (!f) throw std::runtime_error("norm file truncated: " + path);
        return out;
    }
};

class EarlyWarningModel {
public:
    static EarlyWarningModel load(const std::string& weights_path) {
        std::ifstream f(weights_path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open weights file: " + weights_path);
        f.seekg(0, std::ios::end);
        std::streamsize bytes = f.tellg();
        f.seekg(0, std::ios::beg);

        const size_t n_conv1_w = kConv1Out * kNChannels * kConv1K;
        const size_t n_conv1_b = kConv1Out;
        const size_t n_conv2_w = kConv2Out * kConv1Out * kConv2K;
        const size_t n_conv2_b = kConv2Out;
        const size_t n_fc_w = kConv2Out;
        const size_t n_fc_b = 1;
        const size_t total = n_conv1_w + n_conv1_b + n_conv2_w + n_conv2_b + n_fc_w + n_fc_b;

        if (static_cast<size_t>(bytes) != total * sizeof(float)) {
            throw std::runtime_error("weights file size mismatch: expected " +
                                      std::to_string(total * sizeof(float)) + " bytes, got " +
                                      std::to_string(bytes));
        }

        std::vector<float> blob(total);
        f.read(reinterpret_cast<char*>(blob.data()), bytes);
        if (!f) throw std::runtime_error("failed reading weights file: " + weights_path);

        EarlyWarningModel m;
        size_t off = 0;
        m.conv1_w_.assign(blob.begin() + off, blob.begin() + off + n_conv1_w); off += n_conv1_w;
        m.conv1_b_.assign(blob.begin() + off, blob.begin() + off + n_conv1_b); off += n_conv1_b;
        m.conv2_w_.assign(blob.begin() + off, blob.begin() + off + n_conv2_w); off += n_conv2_w;
        m.conv2_b_.assign(blob.begin() + off, blob.begin() + off + n_conv2_b); off += n_conv2_b;
        m.fc_w_.assign(blob.begin() + off, blob.begin() + off + n_fc_w); off += n_fc_w;
        m.fc_b_ = blob[off]; off += n_fc_b;
        return m;
    }

    // window: kWindowMinutes rows x kNChannels cols, row-major, RAW (not yet
    // normalized) values. Normalization is applied here using `norm`.
    // Returns the sigmoid probability.
    float forward(const std::vector<float>& window_raw, const Normalization& norm) const {
        return sigmoid(forward_logit(window_raw, norm));
    }

    // Same as forward() but returns the pre-sigmoid logit, which is what the
    // oracle diff compares against PyTorch's raw logit output (more numeric
    // headroom than comparing post-sigmoid probabilities near 0 or 1).
    float forward_logit(const std::vector<float>& window_raw, const Normalization& norm) const {
        // 1. normalize + transpose to (channel, time)
        std::vector<float> x(kNChannels * kWindowMinutes);
        for (int t = 0; t < kWindowMinutes; ++t) {
            for (int c = 0; c < kNChannels; ++c) {
                float raw = window_raw[t * kNChannels + c];
                x[c * kWindowMinutes + t] = (raw - norm.mean[c]) / norm.std[c];
            }
        }

        // 2. conv1: 24 -> 16, kernel 5, no padding, ReLU
        std::vector<float> h1(kConv1Out * kConv1Len);
        for (int o = 0; o < kConv1Out; ++o) {
            for (int t = 0; t < kConv1Len; ++t) {
                float acc = conv1_b_[o];
                for (int ci = 0; ci < kNChannels; ++ci) {
                    for (int k = 0; k < kConv1K; ++k) {
                        float w = conv1_w_[(o * kNChannels + ci) * kConv1K + k];
                        float v = x[ci * kWindowMinutes + t + k];
                        acc += w * v;
                    }
                }
                h1[o * kConv1Len + t] = acc > 0.0f ? acc : 0.0f;
            }
        }

        // 3. conv2: 16 -> 8, kernel 5, no padding, ReLU
        std::vector<float> h2(kConv2Out * kConv2Len);
        for (int o = 0; o < kConv2Out; ++o) {
            for (int t = 0; t < kConv2Len; ++t) {
                float acc = conv2_b_[o];
                for (int ci = 0; ci < kConv1Out; ++ci) {
                    for (int k = 0; k < kConv2K; ++k) {
                        float w = conv2_w_[(o * kConv1Out + ci) * kConv2K + k];
                        float v = h1[ci * kConv1Len + t + k];
                        acc += w * v;
                    }
                }
                h2[o * kConv2Len + t] = acc > 0.0f ? acc : 0.0f;
            }
        }

        // 4. global average pool over time -> (8,)
        std::array<float, kConv2Out> pooled{};
        for (int o = 0; o < kConv2Out; ++o) {
            float sum = 0.0f;
            for (int t = 0; t < kConv2Len; ++t) sum += h2[o * kConv2Len + t];
            pooled[o] = sum / static_cast<float>(kConv2Len);
        }

        // 5. fc: 8 -> 1
        float logit = fc_b_;
        for (int o = 0; o < kConv2Out; ++o) logit += fc_w_[o] * pooled[o];
        return logit;
    }

private:
    static float sigmoid(float logit) { return 1.0f / (1.0f + std::exp(-logit)); }

    std::vector<float> conv1_w_, conv1_b_, conv2_w_, conv2_b_, fc_w_;
    float fc_b_ = 0.0f;
};

}  // namespace ewap
