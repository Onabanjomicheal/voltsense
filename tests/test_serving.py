"""Tests for src/serving/ -- includes the critical regression test proving
serving-time predictions exactly match the training pipeline's own evaluation,
and the test that catches the CC_Capacity-style bug if ever reintroduced."""
import joblib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SoCWindowDataset, fit_scaler_on_train, list_drive_cycle_files, load_files
from src.models.models import build_model
from src.serving.predictor import SoCPredictor


@pytest.fixture
def trained_checkpoint(tmp_path, synthetic_processed_dir):
    files = list_drive_cycle_files(synthetic_processed_dir, ["25degC"])
    df = load_files(files, validate=False)
    scaler = fit_scaler_on_train(df)

    seq_len = 10
    ds = SoCWindowDataset(df, scaler, sequence_length=seq_len)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = build_model("lstm", input_size=4, hidden_size=8, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(2):
        for x, y in loader:
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

    model_path = tmp_path / "model.pth"
    scaler_path = tmp_path / "scaler.joblib"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "lstm",
        "model_kwargs": {"input_size": 4, "hidden_size": 8, "num_layers": 1},
        "sequence_length": seq_len,
        "feature_columns": ["Voltage [V]", "Current [A]", "Temperature [degC]", "Power [W]"],
    }, model_path)
    joblib.dump(scaler, scaler_path)

    return str(model_path), str(scaler_path), df, scaler, seq_len, model


def test_predictor_output_is_valid_probability_range(trained_checkpoint, sample_readings):
    model_path, scaler_path, *_ = trained_checkpoint
    predictor = SoCPredictor(model_path, scaler_path)
    soc = predictor.predict_from_readings(sample_readings)
    assert 0.0 <= soc <= 1.0


def test_predictor_rejects_too_few_readings(trained_checkpoint):
    model_path, scaler_path, *_ = trained_checkpoint
    predictor = SoCPredictor(model_path, scaler_path)
    with pytest.raises(ValueError, match="at least"):
        predictor.predict_from_readings([{"time_s": 0, "voltage_v": 4.0, "current_a": -0.5, "temperature_c": 25.0}])


def test_serving_matches_training_pipeline_exactly(trained_checkpoint):
    model_path, scaler_path, df, scaler, seq_len, model = trained_checkpoint

    one_file_name = df["__file__"].iloc[0]
    one_file_df = df[df["__file__"] == one_file_name].reset_index(drop=True)

    ds = SoCWindowDataset(one_file_df, scaler, sequence_length=seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.eval()
    pipeline_preds = []
    with torch.no_grad():
        for x, y in loader:
            pipeline_preds.append(model(x).squeeze(-1).item())

    predictor = SoCPredictor(model_path, scaler_path)
    readings = [
        {"time_s": row["Time [s]"], "voltage_v": row["Voltage [V]"],
         "current_a": row["Current [A]"], "temperature_c": row["Temperature [degC]"]}
        for _, row in one_file_df.iterrows()
    ]
    serving_preds = [
        predictor.predict_from_readings(readings[:end])
        for end in range(seq_len, len(readings) + 1)
    ]

    assert len(pipeline_preds) == len(serving_preds)
    # serving clips to [0, 1]; training's raw evaluate() output is intentionally
    # unclipped for honest MSE/MAE. Apply the same clipping before comparing.
    pipeline_preds_clipped = np.clip(np.array(pipeline_preds), 0.0, 1.0)
    max_diff = np.max(np.abs(pipeline_preds_clipped - np.array(serving_preds)))
    assert max_diff < 1e-5, f"Serving predictions diverge from training pipeline! max_diff={max_diff}"


def test_predictor_responds_to_voltage_changes(trained_checkpoint):
    """Regression test for the CC_Capacity bug: confirms predictions actually
    change in a sensible direction as voltage changes."""
    model_path, scaler_path, *_ = trained_checkpoint
    predictor = SoCPredictor(model_path, scaler_path)

    high_v = [{"time_s": float(i), "voltage_v": 4.1, "current_a": -0.5, "temperature_c": 25.0} for i in range(10)]
    low_v = [{"time_s": float(i), "voltage_v": 2.8, "current_a": -0.5, "temperature_c": 25.0} for i in range(10)]

    high_soc = predictor.predict_from_readings(high_v)
    low_soc = predictor.predict_from_readings(low_v)

    assert high_soc > low_soc, (
        f"High voltage predicted LOWER SoC ({high_soc}) than low voltage ({low_soc}) -- "
        f"this is the exact failure mode caused by the CC_Capacity feature bug."
    )
