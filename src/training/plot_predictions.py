import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from src.data.dataset import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    SplitConfig,
    engineer_features,
    list_drive_cycle_files,
    load_scaler,
    split_files,
)
from src.data.validation import validate_processed_file
from src.models.models import build_model
from src.utils.logging_config import get_logger

logger = get_logger("plot_predictions")

MODELS = {
    "LSTM": "lstm",
    "GRU": "gru",
    "CNN-BiLSTM-Attention": "cnn_bilstm_attention",
}
TEMPERATURE_ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
TEMPERATURE_TITLES = {
    "n20degC": "−20 °C", "n10degC": "−10 °C", "0degC": "0 °C",
    "10degC": "10 °C", "25degC": "25 °C", "40degC": "40 °C",
}
MODEL_COLORS = {"LSTM": "#2a78d6", "GRU": "#eb6834", "CNN-BiLSTM-Attention": "#1baf7a"}
REFERENCE_COLOR = "#0b0b0b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
REFERENCE_LABEL = "Reference SOC (Coulomb counting)"
PANEL_LETTERS = "abcdef"


def pick_files(test_files, temperatures, cycle):
    chosen = {}
    for temp in temperatures:
        candidates = [f for f in sorted(test_files) if temp in os.path.normpath(f).split(os.sep)]
        preferred = [f for f in candidates if cycle.lower() in os.path.basename(f).lower()]
        if preferred:
            chosen[temp] = preferred[0]
        elif candidates:
            chosen[temp] = candidates[0]
            logger.warning(f"No {cycle} test file at {temp}; using {os.path.basename(candidates[0])}.")
        else:
            logger.warning(f"No test file found for temperature {temp}; skipping.")
    return chosen


def cycle_name(path):
    name = os.path.basename(path).upper()
    for key in ("UDDS", "US06", "HWFET", "LA92"):
        if key in name:
            return key
    return os.path.splitext(os.path.basename(path))[0]


def load_run(model_name, run_dir, device):
    checkpoint = torch.load(os.path.join(run_dir, "model.pth"), map_location=device)
    model = build_model(model_name, **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = load_scaler(os.path.join(run_dir, "scaler.joblib"))
    return model, scaler, checkpoint["sequence_length"]


def predict_full_file(model, scaler, df, sequence_length, device):
    feats = scaler.transform(df[FEATURE_COLUMNS].values)
    n = len(df)
    if n < sequence_length:
        return None, None, None
    windows = np.stack([feats[i:i + sequence_length] for i in range(n - sequence_length + 1)])
    x = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(x).cpu().numpy().reshape(-1)
    actual = df[LABEL_COLUMN].values[sequence_length - 1:]
    time_s = df["Time [s]"].values[sequence_length - 1:]
    return time_s, preds, actual


def style_axis(ax):
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def zoom_window(time_s, actual, preds_by_model):
    errors = np.mean([np.abs(p - actual) for p in preds_by_model.values()], axis=0)
    smooth = np.convolve(errors, np.ones(60) / 60, mode="same")
    center = time_s[int(np.argmax(smooth))]
    half = max(75.0, 0.04 * (time_s[-1] - time_s[0]))
    lo = max(time_s[0], center - half)
    hi = min(time_s[-1], center + half)
    return lo, hi


def add_inset(ax, time_s, series, lo, hi):
    mask = (time_s >= lo) & (time_s <= hi)
    values = np.concatenate([s[mask] for s, _, _ in series])
    pad = 0.10 * (values.max() - values.min() + 1e-9)
    y0, y1 = values.min() - pad, values.max() + pad
    inset = ax.inset_axes([0.08, 0.10, 0.36, 0.34])
    for s, color, lw in series:
        inset.plot(time_s[mask], s[mask], color=color, linewidth=lw)
    inset.set_xlim(lo, hi)
    inset.set_ylim(y0, y1)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    inset.tick_params(labelsize=7, colors=INK_SECONDARY, length=2, pad=1)
    for spine in inset.spines.values():
        spine.set_color(INK_SECONDARY)
        spine.set_linewidth(0.7)
    inset.set_facecolor("white")
    try:
        ax.indicate_inset([lo, y0, hi - lo, y1 - y0], inset, edgecolor=INK_SECONDARY, linewidth=0.7, alpha=0.9)
    except AttributeError:
        ax.add_patch(Rectangle((lo, y0), hi - lo, y1 - y0, fill=False, edgecolor=INK_SECONDARY,
                               linewidth=0.7, linestyle="--"))


def save_figure(fig, out_base):
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {out_base}.png and .pdf")


def chart_soc(r, temp, out_base):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    t = r["time"]
    style_axis(ax)
    ax.plot(t, r["actual"] * 100, color=REFERENCE_COLOR, linewidth=1.6, label=REFERENCE_LABEL)
    for name, preds in r["preds"].items():
        ax.plot(t, preds * 100, color=MODEL_COLORS[name], linewidth=1.0, label=name)
    lo, hi = zoom_window(t, r["actual"] * 100, {k: v * 100 for k, v in r["preds"].items()})
    series = [(r["actual"] * 100, REFERENCE_COLOR, 1.4)] + \
             [(p * 100, MODEL_COLORS[n], 0.9) for n, p in r["preds"].items()]
    add_inset(ax, t, series, lo, hi)
    ax.set_xlabel("Time (s)", fontsize=10, color=INK)
    ax.set_ylabel("State of charge (%)", fontsize=10, color=INK)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, out_base)


def chart_error(r, temp, out_base):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style_axis(ax)
    ymax = 0.0
    for name, preds in r["preds"].items():
        err = np.abs(preds - r["actual"]) * 100
        ymax = max(ymax, err.max())
        ax.plot(r["time"], err, color=MODEL_COLORS[name], linewidth=0.9, label=name)
    ax.set_xlabel("Time (s)", fontsize=10, color=INK)
    ax.set_ylabel("Absolute SOC error (%)", fontsize=10, color=INK)
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, out_base)


def main(config_path, protocol, seed, cycles, output_dir):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["data"]["dir"]
    temps_cfg = cfg["data"]["temperatures"]
    files = list_drive_cycle_files(data_dir, temps_cfg)
    split_cfg = SplitConfig(**cfg["data"]["split"])
    _, _, test_files = split_files(files, split_cfg)
    if not test_files:
        raise RuntimeError("No test files found -- check data.dir / data.split in the config.")

    temps = [t for t in TEMPERATURE_ORDER if t in temps_cfg]

    runs = {}
    for display, model_name in MODELS.items():
        run_dir = os.path.join("models", "runs", protocol, f"{model_name}_seed{seed}")
        runs[display] = load_run(model_name, run_dir, device)
        logger.info(f"Loaded {display} from {run_dir}")

    soc_dir = os.path.join(output_dir, "soc")
    err_dir = os.path.join(output_dir, "error")
    os.makedirs(soc_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)

    for cycle in cycles:
        for temp in temps:
            matches = [f for f in sorted(test_files)
                       if temp in os.path.normpath(f).split(os.sep) and cycle.lower() in os.path.basename(f).lower()]
            if not matches:
                logger.warning(f"No {cycle} test file at {temp}; skipping.")
                continue
            path = matches[0]
            df = pd.read_csv(path)
            validate_processed_file(df, file_path=path)
            df = engineer_features(df)

            preds_by_model = {}
            time_s = actual = None
            for display, (model, scaler, seq_len) in runs.items():
                t, preds, act = predict_full_file(model, scaler, df, seq_len, device)
                if preds is None:
                    break
                if time_s is None:
                    time_s, actual = t, act
                preds_by_model[display] = preds
            if len(preds_by_model) != len(runs):
                logger.warning(f"{path} too short; skipping.")
                continue

            r = {"time": time_s, "actual": actual, "preds": preds_by_model, "cycle": cycle}
            for display, preds in preds_by_model.items():
                err = np.abs(preds - actual) * 100
                logger.info(f"[{cycle} {temp}] {os.path.basename(path)} {display}: "
                            f"MAE={err.mean():.3f}% RMSE={np.sqrt((err ** 2).mean()):.3f}% max={err.max():.3f}%")
            chart_soc(r, temp, os.path.join(soc_dir, f"soc_{cycle}_{temp}"))
            chart_error(r, temp, os.path.join(err_dir, f"error_{cycle}_{temp}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated/cycle_holdout_gru_seed42.yaml")
    parser.add_argument("--protocol", default="cycle_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", nargs="+", default=["UDDS", "US06"])
    parser.add_argument("--output-dir", default="reports/figures/predictions")
    args = parser.parse_args()
    main(args.config, args.protocol, args.seed, args.cycles, args.output_dir)
