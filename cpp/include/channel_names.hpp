// Channel naming, generated the same way as python/simulate.py::CHANNEL_NAMES
// (6 pressure, 6 temperature, 6 flow, 6 vibration, 1-indexed within group).
// Kept as a function rather than a literal array so there is exactly one
// place that encodes the group order/sizes.
#pragma once

#include <array>
#include <cstdio>
#include <string>

#include "model.hpp"

namespace ewap {

inline std::array<std::string, kNChannels> channel_names() {
    static const char* groups[4] = {"pressure", "temperature", "flow", "vibration"};
    std::array<std::string, kNChannels> names{};
    int idx = 0;
    for (int g = 0; g < 4; ++g) {
        for (int i = 1; i <= 6; ++i) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s_%02d", groups[g], i);
            names[idx++] = buf;
        }
    }
    return names;
}

}  // namespace ewap
