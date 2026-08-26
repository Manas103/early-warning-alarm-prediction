"""The early-warning classifier: a small 1D-CNN over a sliding window of the
24 raw (normalized) historian channels.

Architecture (fixed, also hand-ported to cpp/src/model.cpp -- keep the two
in lock-step if you ever change this file):

    input:  (W=60 minutes, C=24 channels), per-channel z-scored
    conv1d: 24 -> 16, kernel=5, stride=1, no padding, + ReLU   -> (16, 56)
    conv1d: 16 -> 8,  kernel=5, stride=1, no padding, + ReLU   -> (8, 52)
    global average pool over time                              -> (8,)
    fc:     8 -> 1                                              -> logit
    sigmoid -> P(upset ramp in progress in this window)

No padding is used deliberately so the receptive field and output length
are trivial to hand-verify in C++ without any padding-index bookkeeping.
"""
from __future__ import annotations

import torch
import torch.nn as nn

WINDOW_MINUTES = 60
N_CHANNELS = 24
CONV1_OUT = 16
CONV1_K = 5
CONV2_OUT = 8
CONV2_K = 5


class EarlyWarningCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(N_CHANNELS, CONV1_OUT, kernel_size=CONV1_K)
        self.conv2 = nn.Conv1d(CONV1_OUT, CONV2_OUT, kernel_size=CONV2_K)
        self.fc = nn.Linear(CONV2_OUT, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, C) -> (batch, C, W) for conv1d
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)          # global average pool over time -> (batch, CONV2_OUT)
        logit = self.fc(x).squeeze(-1)
        return logit               # caller applies sigmoid (BCEWithLogitsLoss during training)
