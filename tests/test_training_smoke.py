"""End-to-end smoke test: confirms the full training pipeline runs without error
on synthetic data. Does NOT check for good metrics -- just that nothing crashes."""
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    FEATURE_COLUMNS,
    SoCWindowDataset,
    SplitConfig,
    fit_scaler_on_train,
    list_drive_cycle_files,
    load_files,
    save_scaler,
    split_files,
)
from src.models.models import build_model
from src.training.train import evaluate, train_model


def test_full_training_pipeline_runs_end_to_end(tmp_path, synthetic_processed_dir):
    files = list_drive_cycle_files(synthetic_processed_dir, ["25degC", "0degC"])
    assert len(files) > 0

    split_cfg = SplitConfig(strategy="random_by_file", test_size=0.3, val_size=0.15, random_state=24)
    train_files, val_files, test_files = split_files(files, split_cfg)

    train_df = load_files(train_files, validate=False)
    val_df = load_files(val_files, validate=False)
    test_df = load_files(test_files, validate=False)

    scaler = fit_scaler_on_train(train_df)
    scaler_path = tmp_path / "scaler.joblib"
    save_scaler(scaler, str(scaler_path))

    seq_len = 10
    train_ds = SoCWindowDataset(train_df, scaler, sequence_length=seq_len)
    val_ds = SoCWindowDataset(val_df, scaler, sequence_length=seq_len)
    test_ds = SoCWindowDataset(test_df, scaler, sequence_length=seq_len)
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cpu")
    model = build_model("lstm", input_size=4, hidden_size=8, num_layers=1).to(device)

    model, best_val, history = train_model(
        model, train_loader, val_loader, len(train_ds), device, lr=0.01, epochs=2, patience=5,
    )
    assert len(history["train_loss"]) == 2

    metrics, _, _ = evaluate(model, test_loader, device)
    test_mse, test_mae = metrics["mse"], metrics["mae"]
    assert test_mse >= 0 and test_mae >= 0

    model_path = tmp_path / "model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": "lstm",
        "model_kwargs": {"input_size": 4, "hidden_size": 8, "num_layers": 1},
        "sequence_length": seq_len,
        "feature_columns": FEATURE_COLUMNS,
    }, model_path)
    assert os.path.exists(model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    reloaded_model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["feature_columns"] == FEATURE_COLUMNS
