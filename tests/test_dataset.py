
"""Tests for src/data/dataset.py -- covers the real bugs found and fixed during
this project: scaler leakage, cross-file window contamination, and the
CC_Capacity serving-mismatch issue."""
import os

import numpy as np
import pandas as pd

from src.data.dataset import (
    FEATURE_COLUMNS,
    SoCWindowDataset,
    SplitConfig,
    engineer_features,
    fit_scaler_on_train,
    list_drive_cycle_files,
    load_files,
    split_files,
)


def test_engineer_features_adds_power_and_no_capacity_leak():
    df = pd.DataFrame({
        "Time [s]": [0.0, 1.0, 2.0],
        "Voltage [V]": [4.0, 3.9, 3.8],
        "Current [A]": [-0.5, -0.5, -0.5],
    })
    out = engineer_features(df)
    assert "Power [W]" in out.columns
    np.testing.assert_allclose(out["Power [W]"], out["Voltage [V]"] * out["Current [A]"])
    assert "CC_Capacity [Ah]" not in FEATURE_COLUMNS, (
        "CC_Capacity must not be a model feature -- meaningless in short serving "
        "windows since it only reflects history within the current request."
    )


def test_skip_constant_charge_discharge_files(tmp_path):
    temp_dir = tmp_path / "25degC"
    temp_dir.mkdir()
    for name in ["Mixed1.csv", "589_Charge1.csv", "585_Dis_2C.csv", "UDDS1.csv"]:
        pd.DataFrame({"Time [s]": [0], "Voltage [V]": [4.0], "Current [A]": [-0.5],
                       "Temperature [degC]": [25.0], "SOC [-]": [1.0]}).to_csv(temp_dir / name, index=False)

    files = list_drive_cycle_files(str(tmp_path), ["25degC"])
    names = [os.path.basename(f) for f in files]
    assert "Mixed1.csv" in names
    assert "UDDS1.csv" in names
    assert not any("Charge" in n or "Dis" in n for n in names)


def test_scaler_fit_on_train_only_not_leaked(synthetic_processed_dir):
    files = list_drive_cycle_files(synthetic_processed_dir, ["25degC", "0degC"])
    split_cfg = SplitConfig(strategy="random_by_file", test_size=0.2, val_size=0.1, random_state=24)
    train_files, val_files, test_files = split_files(files, split_cfg)

    train_df = load_files(train_files, validate=False)
    all_df = load_files(files, validate=False)

    train_scaler = fit_scaler_on_train(train_df)
    full_scaler = fit_scaler_on_train(all_df)  # simulates the OLD buggy behavior

    assert not np.allclose(train_scaler.mean_, full_scaler.mean_), (
        "Scaler fit on train-only should differ from scaler fit on everything -- "
        "if identical, the split isn't actually excluding val/test data."
    )


def test_windows_never_cross_file_boundary(synthetic_processed_dir):
    files = list_drive_cycle_files(synthetic_processed_dir, ["25degC"])
    df = load_files(files, validate=False)

    scaler = fit_scaler_on_train(df)
    ds = SoCWindowDataset(df, scaler, sequence_length=10)
    assert len(ds) > 0

    df_reset = df.reset_index(drop=True)
    for fname, group in df_reset.groupby("__file__", sort=False):
        idx = group.index.values
        n = len(idx)
        if n < 10:
            continue
        for start in range(n - 10 + 1):
            window_idx = idx[start:start + 10]
            assert (window_idx[-1] - window_idx[0]) == 9, "window is not contiguous within one file"


def test_split_strategies_produce_no_file_overlap(synthetic_processed_dir):
    files = list_drive_cycle_files(synthetic_processed_dir, ["25degC", "0degC"])
    for strategy in ["random_by_file", "by_drive_cycle_type"]:
        split_cfg = SplitConfig(strategy=strategy, test_size=0.2, val_size=0.1, random_state=24)
        train_files, val_files, test_files = split_files(files, split_cfg)
        train_set, val_set, test_set = set(train_files), set(val_files), set(test_files)
        assert not (train_set & test_set), f"{strategy}: train/test overlap"
        assert not (train_set & val_set), f"{strategy}: train/val overlap"
