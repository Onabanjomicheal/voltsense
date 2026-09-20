"""LSTM model for SoC estimation. Takes a window of shape [batch, seq_len, n_features]
and outputs a single SoC prediction per window: [batch, 1].

input_size defaults to 4, matching FEATURE_COLUMNS in src/data/dataset.py
(Voltage, Current, Temperature, Power). CC_Capacity is intentionally excluded --
see dataset.py module docstring for why.
"""
import torch.nn as nn


class SoCLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=94, num_layers=4, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


MODEL_REGISTRY = {
    "lstm": SoCLSTM,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
