"""
Inference wrapper. Loads the exact model architecture + hyperparameters that were
actually trained (from the checkpoint), plus the persisted scaler -- so serving-time
feature engineering and normalization match training exactly.

Predictions are clipped to [0, 1] here (sensible for a user-facing SoC value) --
the training pipeline's raw evaluate() output is intentionally NOT clipped, so
MSE/MAE reflect true model error. This is a deliberate difference, not a bug --
see tests/test_serving.py for the regression test that accounts for it correctly.
"""
import numpy as np
import pandas as pd
import torch

from src.data.dataset import engineer_features, load_scaler
from src.models.models import build_model


class SoCPredictor:
    def __init__(self, model_path: str, scaler_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model_name = checkpoint["model_name"]
        self.sequence_length = checkpoint["sequence_length"]
        self.feature_columns = checkpoint["feature_columns"]

        self.model = build_model(self.model_name, **checkpoint["model_kwargs"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = load_scaler(scaler_path)

    def predict_from_readings(self, readings: list) -> float:
        if len(readings) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} readings (got {len(readings)}). "
                f"The model requires {self.sequence_length} seconds of history to predict."
            )

        df = pd.DataFrame(readings).rename(columns={
            "time_s": "Time [s]",
            "voltage_v": "Voltage [V]",
            "current_a": "Current [A]",
            "temperature_c": "Temperature [degC]",
        })

        df = engineer_features(df)

        window_df = df.tail(self.sequence_length)
        feats = self.scaler.transform(window_df[self.feature_columns].values)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x).squeeze().item()

        return float(np.clip(pred, 0.0, 1.0))
