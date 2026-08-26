import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from model import CONV1_K, CONV1_OUT, CONV2_K, CONV2_OUT, N_CHANNELS, WINDOW_MINUTES, EarlyWarningCNN  # noqa: E402


def test_forward_output_shape():
    model = EarlyWarningCNN()
    x = torch.randn(4, WINDOW_MINUTES, N_CHANNELS)
    logits = model(x)
    assert logits.shape == (4,)


def test_forward_is_deterministic_in_eval_mode():
    model = EarlyWarningCNN()
    model.eval()
    x = torch.randn(2, WINDOW_MINUTES, N_CHANNELS)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_parameter_shapes_match_documented_layout():
    model = EarlyWarningCNN()
    sd = model.state_dict()
    assert tuple(sd["conv1.weight"].shape) == (CONV1_OUT, N_CHANNELS, CONV1_K)
    assert tuple(sd["conv1.bias"].shape) == (CONV1_OUT,)
    assert tuple(sd["conv2.weight"].shape) == (CONV2_OUT, CONV1_OUT, CONV2_K)
    assert tuple(sd["conv2.bias"].shape) == (CONV2_OUT,)
    assert tuple(sd["fc.weight"].shape) == (1, CONV2_OUT)
    assert tuple(sd["fc.bias"].shape) == (1,)


def test_conv_output_lengths_match_cpp_constants():
    # cpp/include/model.hpp hardcodes kConv1Len=56, kConv2Len=52; if these
    # ever drift the C++ port silently breaks, so pin them here too
    conv1_len = WINDOW_MINUTES - CONV1_K + 1
    conv2_len = conv1_len - CONV2_K + 1
    assert conv1_len == 56
    assert conv2_len == 52
