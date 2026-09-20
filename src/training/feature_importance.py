"""
Permutation feature importance for the trained SoC LSTM.

Method: for each feature, shuffle that feature's values across all test windows
(breaking its relationship with the label while preserving its marginal
distribution and every other feature's values), re-run inference, and measure
how much test MSE degrades. A larger MSE increase means the model relies on
that feature more heavily.

This is a real, standard, model-agnostic importance method (Breiman 2001) --
not a proxy like weight magnitude, which doesn't account for feature scale or
interactions.

Usage:
    python -m src.training.feature_importance --config configs/lstm.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.data.dataset import (
    FEATURE_COLUMNS,
    SoCWindowDataset,
    SplitConfig,
    list_drive_cycle_files,
    load_files,
    load_scaler,
    split_files,
)
from src.models.models import build_model
from src.utils.logging_config import get_logger

logger = get_logger("feature_importance")


def compute_permutation_importance(model, dataset, device, n_repeats=5, seed=42):
    """Returns {feature_name: {'importance_mean': float, 'importance_std': float}}."""
    rng = np.random.RandomState(seed)

    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    baseline_preds, baseline_trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).squeeze(-1).cpu().numpy()
            baseline_preds.extend(out.tolist())
            baseline_trues.extend(y.numpy().tolist())
    baseline_mse = mean_squared_error(baseline_trues, baseline_preds)
    logger.info(f"Baseline test MSE (unpermuted): {baseline_mse:.6f}")

    # Stack all windows into one array for easy per-feature shuffling: [n_samples, seq_len, n_features]
    all_windows = np.stack([w for w, _ in dataset.samples])
    all_labels = np.array([label for _, label in dataset.samples])

    results = {}
    for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
        degradations = []
        for repeat in range(n_repeats):
            permuted = all_windows.copy()
            # shuffle this feature's values across the SAMPLE axis, independently
            # at each timestep, breaking the feature-label relationship while
            # preserving the feature's own marginal distribution
            for t in range(permuted.shape[1]):
                perm_idx = rng.permutation(permuted.shape[0])
                permuted[:, t, feat_idx] = permuted[perm_idx, t, feat_idx]

            x = torch.tensor(permuted, dtype=torch.float32)
            preds = []
            with torch.no_grad():
                for i in range(0, len(x), 128):
                    batch = x[i:i + 128].to(device)
                    preds.extend(model(batch).squeeze(-1).cpu().numpy().tolist())

            permuted_mse = mean_squared_error(all_labels, preds)
            degradations.append(permuted_mse - baseline_mse)

        results[feat_name] = {
            "importance_mean": float(np.mean(degradations)),
            "importance_std": float(np.std(degradations)),
        }
        logger.info(f"  {feat_name}: MSE increase = {results[feat_name]['importance_mean']:.6f} "
                    f"(+/- {results[feat_name]['importance_std']:.6f})")

    return results, baseline_mse


def main(config_path: str, output_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(cfg["output"]["artifact_dir"], "lstm_model.pth")
    scaler_path = os.path.join(cfg["output"]["artifact_dir"], "scaler.joblib")
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = load_scaler(scaler_path)
    seq_len = checkpoint["sequence_length"]

    data_dir = cfg["data"]["dir"]
    temps = cfg["data"]["temperatures"]
    files = list_drive_cycle_files(data_dir, temps)
    split_cfg = SplitConfig(**cfg["data"]["split"])
    _, _, test_files = split_files(files, split_cfg)
    test_df = load_files(test_files)
    test_ds = SoCWindowDataset(test_df, scaler, sequence_length=seq_len)

    logger.info(f"Computing permutation importance on {len(test_ds)} test windows...")
    results, baseline_mse = compute_permutation_importance(model, test_ds, device)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["importance_mean"], reverse=True)
    logger.info("Feature ranking (most to least important):")
    for rank, (feat, vals) in enumerate(ranked, 1):
        logger.info(f"  {rank}. {feat}: {vals['importance_mean']:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "baseline_test_mse": baseline_mse,
            "method": "permutation_importance",
            "n_repeats": 5,
            "features": results,
            "ranking": [feat for feat, _ in ranked],
        }, f, indent=2)
    logger.info(f"Saved feature importance report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lstm.yaml")
    parser.add_argument("--output", default="reports/feature_importance.json")
    args = parser.parse_args()
    main(args.config, args.output)
