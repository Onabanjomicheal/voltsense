import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import SplitConfig, engineer_features, list_drive_cycle_files, load_scaler, split_files
from src.data.validation import validate_processed_file
from src.models.models import build_model
from src.training.plot_predictions import predict_full_file
from src.utils.logging_config import get_logger

logger = get_logger("check_cycle_bias")

CYCLE_TAGS = ("Mixed", "UDDS", "HWFET", "LA92", "US06")


def cycle_type_of(path: str) -> str:
    name = os.path.basename(path)
    for tag in CYCLE_TAGS:
        if tag in name:
            return tag
    return "Unknown"


def main(config_path: str, target_temp: str, model_path: str = None, scaler_path: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["data"]["dir"]
    temps = cfg["data"]["temperatures"]
    files = list_drive_cycle_files(data_dir, temps)

    split_cfg = SplitConfig(**cfg["data"]["split"])
    train_files, val_files, test_files = split_files(files, split_cfg)

    artifact_dir = cfg["output"]["artifact_dir"]
    model_path = model_path or os.path.join(artifact_dir, "model.pth")
    scaler_path = scaler_path or os.path.join(artifact_dir, "scaler.joblib")

    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(cfg["model"]["name"], **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    sequence_length = checkpoint["sequence_length"]
    scaler = load_scaler(scaler_path)

    val_at_temp = [f for f in val_files if target_temp in os.path.normpath(f).split(os.sep)]
    train_at_temp = [f for f in train_files if target_temp in os.path.normpath(f).split(os.sep)]

    logger.info(f"{target_temp}: {len(val_at_temp)} val file(s), {len(train_at_temp)} train file(s) "
                f"(test files excluded from this check).")

    def run_group(file_list, label):
        results = []
        for path in file_list:
            df = pd.read_csv(path)
            validate_processed_file(df, file_path=path)
            df = engineer_features(df)
            _, preds, actual = predict_full_file(model, scaler, df, sequence_length, device)
            if preds is None:
                continue
            bias = float(np.mean(preds - actual))
            mae = float(np.mean(np.abs(preds - actual)))
            results.append({
                "file": os.path.basename(path),
                "cycle_type": cycle_type_of(path),
                "bias": bias,
                "mae": mae,
                "provenance": label,
            })
        return results

    val_results = run_group(val_at_temp, "val")
    train_results = run_group(train_at_temp, "train (not a fair bias test -- model saw this)")

    logger.info(f"\n--- {target_temp} VAL results (meaningful for the bias question) ---")
    by_cycle = defaultdict(list)
    for r in val_results:
        by_cycle[r["cycle_type"]].append(r)
        logger.info(f"  [{r['cycle_type']}] {r['file']}: bias={r['bias']:+.4f}  mae={r['mae']:.4f}")

    if len(by_cycle) <= 1:
        logger.warning(
            f"Only {len(by_cycle)} distinct cycle type(s) present in {target_temp}'s val split "
            f"({list(by_cycle.keys())}). Cannot compare UDDS vs. Mixed bias from val data alone "
            f"at this temperature -- the split didn't happen to include both types here."
        )

    if train_results:
        logger.info(f"\n--- {target_temp} TRAIN results (context only, NOT a fair bias test) ---")
        for r in train_results:
            logger.info(f"  [{r['cycle_type']}] {r['file']}: bias={r['bias']:+.4f}  mae={r['mae']:.4f}")

    logger.info(f"\n--- Summary: mean signed bias by cycle type at {target_temp} ---")
    for cycle, group in by_cycle.items():
        mean_bias = np.mean([g["bias"] for g in group])
        logger.info(f"  VAL [{cycle}]: mean bias = {mean_bias:+.4f}  (n={len(group)} file(s))")

    return val_results, train_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--temp", default="n10degC", help="Temperature folder name to check")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--scaler-path", default=None)
    args = parser.parse_args()
    main(args.config, args.temp, args.model_path, args.scaler_path)
