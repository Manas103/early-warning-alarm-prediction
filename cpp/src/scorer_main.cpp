// Edge scorer: loads the exported CNN weights, scores every minute of one
// simulated run with no network access of any kind (reads local files only,
// writes a local JSON file), and applies a debounce rule (K consecutive
// minutes at or above the probability threshold) before calling a
// prediction "confirmed". This is the binary the Dockerfile containerizes.
//
// Usage:
//   edge_scorer <weights.bin> <norm.txt> <run.bin> <T> <threshold> <debounce_k> <out.json> <run_id>
//
// run.bin: T rows x 24 channels, row-major float32, raw (unnormalized) values.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "channel_names.hpp"
#include "contribution.hpp"
#include "model.hpp"

using namespace ewap;

namespace {

std::vector<float> load_run(const std::string& path, int T) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open run file: " + path);
    std::vector<float> data(static_cast<size_t>(T) * kNChannels);
    f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!f) throw std::runtime_error("run file shorter than expected: " + path);
    return data;
}

std::string json_escape(const std::string& s) { return s; }  // channel names are plain ascii

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: edge_scorer <weights.bin> <norm.txt> <run.bin> <T> <threshold> "
                     "<debounce_k> <out.json> <run_id>\n";
        return 2;
    }
    const std::string weights_path = argv[1];
    const std::string norm_path = argv[2];
    const std::string run_path = argv[3];
    const int T = std::stoi(argv[4]);
    const float threshold = std::stof(argv[5]);
    const int debounce_k = std::stoi(argv[6]);
    const std::string out_path = argv[7];
    const std::string run_id = argv[8];

    EarlyWarningModel model = EarlyWarningModel::load(weights_path);
    Normalization norm = Normalization::load(norm_path);
    std::vector<float> run = load_run(run_path, T);
    auto names = channel_names();

    const auto t_start = std::chrono::steady_clock::now();

    std::vector<float> probabilities;                 // one per scored minute
    std::vector<int> scored_minutes;                   // minute index (0-based) for each entry
    std::vector<std::vector<ChannelContribution>> event_contribs;
    std::vector<int> event_minutes;
    std::vector<float> event_probs;

    int consecutive = 0;
    int confirmed_minute = -1;

    for (int t = kWindowMinutes - 1; t < T; ++t) {
        std::vector<float> window(kWindowMinutes * kNChannels);
        for (int wt = 0; wt < kWindowMinutes; ++wt) {
            int src_t = t - kWindowMinutes + 1 + wt;
            for (int c = 0; c < kNChannels; ++c) {
                window[wt * kNChannels + c] = run[static_cast<size_t>(src_t) * kNChannels + c];
            }
        }
        float p = model.forward(window, norm);
        probabilities.push_back(p);
        scored_minutes.push_back(t);

        bool positive = p >= threshold;
        if (positive) {
            consecutive += 1;
            auto scores = compute_contribution_scores(window, norm);
            event_contribs.push_back(top_k_contributors(scores, 4));
            event_minutes.push_back(t);
            event_probs.push_back(p);
            if (consecutive >= debounce_k && confirmed_minute < 0) {
                confirmed_minute = t;
            }
        } else {
            consecutive = 0;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("cannot open output file: " + out_path);
    out.precision(6);
    out << "{\n";
    out << "  \"run_id\": \"" << json_escape(run_id) << "\",\n";
    out << "  \"T\": " << T << ",\n";
    out << "  \"window_minutes\": " << kWindowMinutes << ",\n";
    out << "  \"threshold\": " << threshold << ",\n";
    out << "  \"debounce_k\": " << debounce_k << ",\n";
    out << "  \"confirmed_prediction_minute\": " << (confirmed_minute >= 0 ? std::to_string(confirmed_minute) : "null") << ",\n";
    out << "  \"scoring_wall_ms\": " << elapsed_ms << ",\n";

    out << "  \"probabilities\": [";
    for (size_t i = 0; i < probabilities.size(); ++i) {
        out << probabilities[i];
        if (i + 1 < probabilities.size()) out << ", ";
    }
    out << "],\n";
    out << "  \"scored_minutes\": [";
    for (size_t i = 0; i < scored_minutes.size(); ++i) {
        out << scored_minutes[i];
        if (i + 1 < scored_minutes.size()) out << ", ";
    }
    out << "],\n";

    out << "  \"events\": [\n";
    for (size_t i = 0; i < event_minutes.size(); ++i) {
        out << "    {\"minute\": " << event_minutes[i]
            << ", \"probability\": " << event_probs[i]
            << ", \"contributors\": [";
        const auto& contribs = event_contribs[i];
        for (size_t j = 0; j < contribs.size(); ++j) {
            out << "{\"channel\": \"" << names[contribs[j].channel_index] << "\", \"score\": "
                << contribs[j].mean_abs_zscore << "}";
            if (j + 1 < contribs.size()) out << ", ";
        }
        out << "]}";
        if (i + 1 < event_minutes.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::cout << "run_id=" << run_id << " scored " << probabilities.size()
              << " minutes, confirmed_prediction_minute="
              << (confirmed_minute >= 0 ? std::to_string(confirmed_minute) : "none")
              << " (" << elapsed_ms << " ms)\n";
    return 0;
}
