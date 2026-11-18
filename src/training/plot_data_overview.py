import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("plot_data_overview")

TEMPERATURE_TITLES = {
    "n20degC": "−20 °C", "n10degC": "−10 °C", "0degC": "0 °C",
    "10degC": "10 °C", "25degC": "25 °C", "40degC": "40 °C",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
QUANTITIES = [("Voltage", "Voltage (V)"), ("Current", "Current (A)"), ("Temp", "Cell temperature (°C)")]


def find_column(df, key):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    exact = [c for c in numeric if c.lower().startswith(key.lower())]
    if exact:
        return exact[0]
    for col in numeric:
        if key.lower() in col.lower():
            return col
    raise KeyError(f"No numeric column containing '{key}' in {list(df.columns)}")


def find_file(data_dir, temp, cycle):
    pattern = os.path.join(data_dir, temp, f"*{cycle}*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern}")
    return matches[0]


def main(data_dir, temps, cycle, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for color, temp in zip(COLORS, temps):
        path = find_file(data_dir, temp, cycle)
        df = pd.read_csv(path)
        time_col = find_column(df, "Time")
        t = df[time_col].values - df[time_col].values[0]
        logger.info(f"[{temp}] {os.path.basename(path)}: {len(df)} samples")
        for ax, (key, _) in zip(axes, QUANTITIES):
            ax.plot(t, df[find_column(df, key)].values, color=color, linewidth=0.9,
                    label=TEMPERATURE_TITLES[temp])

    for i, (ax, (_, ylabel)) in enumerate(zip(axes, QUANTITIES)):
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    axes[-1].set_xlabel("Time (s)", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5), frameon=False, fontsize=9,
               title="Ambient temperature", title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "fig_data_overview")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {base}.png and .pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--temps", nargs="+", default=["n20degC", "0degC", "25degC"])
    parser.add_argument("--cycle", default="US06")
    parser.add_argument("--output-dir", default="reports/figures/data")
    args = parser.parse_args()
    main(args.data_dir, args.temps, args.cycle, args.output_dir)
