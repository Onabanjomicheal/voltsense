"""Tests for src/models/models.py."""
import pytest
import torch

from src.models.models import build_model


def test_lstm_forward_shape():
    model = build_model("lstm", input_size=4, hidden_size=16, num_layers=2)
    x = torch.randn(4, 20, 4)
    out = model(x)
    assert out.shape == (4, 1)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("not_a_real_model")
