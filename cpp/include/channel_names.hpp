// Channel naming, generated the same way as python/simulate.py::CHANNEL_NAMES:
// chamber-major layout, 4 simulated process chambers x 6 physical
// quantities (chamber pressure, RF forward power, RF reflected power, gas
// flow, optical emission intensity, endpoint signal) = 24 channels, channel
// index = chamber_idx * 6 + quantity_idx. Kept as a function rather than a
// literal array so there is exactly one place that encodes the chamber
// count / quantity order.
#pragma once

#include <array>
#include <cstdio>
#include <string>

#include "model.hpp"

namespace ewap {

inline std::array<std::string, kNChannels> channel_names() {
    static const char* quantities[6] = {
        "pressure", "rf_forward", "rf_reflected", "gas_flow", "oes_intensity", "endpoint"
    };
    static const int n_chambers = 4;
    static const int n_quantities = 6;
    std::array<std::string, kNChannels> names{};
    int idx = 0;
    for (int c = 1; c <= n_chambers; ++c) {
        for (int q = 0; q < n_quantities; ++q) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "chamber%d_%s", c, quantities[q]);
            names[idx++] = buf;
        }
    }
    return names;
}

}  // namespace ewap
