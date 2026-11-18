import torch.nn as nn

from src.models.lstm_model import SoCLSTM
from src.models.gru_model import SoCGRU
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention

MODEL_REGISTRY = {
    "lstm": SoCLSTM,
    "gru": SoCGRU,
    "cnn_bilstm_attention": CNNBiLSTMAttention,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)