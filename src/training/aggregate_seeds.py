import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger("aggregate_seeds")

DISPLAY_NAMES = {"lstm": "LSTM", "gru": "GRU", "cnn_bilstm_attention": "CNN-BiLSTM-Attention"}
TEMPERATURE_ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
LATEX_LABELS = {"n20degC": "$-20$", "n10degC": "$-10$", "0degC": "0",
                "10degC": "10", "25degC": "25", "40degC": "40"}
METRICS = ["mse", "mae", "rmse", "r2", "max_error"]
HIGHER_IS_BETTER = {"r2"}
PERCENT_METRICS = {"mae", "rmse", "max_error"}


def load_runs(protocol, models):
    runs = {}
    for model in models:
        pattern = os.path.join("models", "runs", protocol, f"{model}_seed*", "manifest.json")
        manifests = []
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                manifests.append(json.load(f))
        if manifests:
            runs[model] = manifests
    return runs


def overall_stats(runs):
    rows = []
    for model, manifests in runs.items():
        row = {"model": DISPLAY_NAMES.get(model, model), "n_seeds": len(manifests)}
        for metric in METRICS:
            values = np.array([m[f"test_{metric}"] for m in manifests], dtype=float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_temperature_stats(runs):
    rows = []
    for model, manifests in runs.items():
        for temp in TEMPERATURE_ORDER:
            entries = [m["per_temperature"][temp] for m in manifests if temp in m["per_temperature"]]
            if not entries:
                continue
            row = {"model": DISPLAY_NAMES.get(model, model), "temperature": temp, "n_seeds": len(entries)}
            for metric in METRICS:
                values = np.array([e[metric] for e in entries], dtype=float)
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def cell(mean, std, metric):
    if metric == "r2":
        return f"{mean:.4f} $\\pm$ {std:.4f}"
    scale = 100.0 if metric in PERCENT_METRICS else 1.0
    return f"{mean * scale:.3f} $\\pm$ {std * scale:.3f}"


def latex_overall(df):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Overall test performance on held-out cycles (UDDS and US06), "
        "mean $\\pm$ standard deviation over three seeds.}",
        "\\label{tab:overall}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & MAE (\\%) & RMSE (\\%) & R$^2$ \\\\",
        "\\midrule",
    ]
    best = {m: (df[f"{m}_mean"].idxmax() if m in HIGHER_IS_BETTER else df[f"{m}_mean"].idxmin())
            for m in ["mae", "rmse", "r2"]}
    for idx, row in df.iterrows():
        cells = [row["model"]]
        for m in ["mae", "rmse", "r2"]:
            text = cell(row[f"{m}_mean"], row[f"{m}_std"], m)
            cells.append(f"\\textbf{{{text}}}" if best[m] == idx else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_per_temperature(df):
    models = list(df["model"].unique())
    columns = "l" + "rrr" * len(models)
    header_top = "Temp. (\\textdegree C) & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models)))
    header_sub = " & " + " & ".join(["MAE (\\%) & RMSE (\\%) & R$^2$"] * len(models)) + " \\\\"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test MAE, RMSE and R$^2$ per ambient temperature on held-out cycles, "
        "mean $\\pm$ standard deviation over three seeds.}",
        "\\label{tab:per_temperature}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header_top,
        cmid,
        header_sub,
        "\\midrule",
    ]
    for temp in TEMPERATURE_ORDER:
        sub = df[df["temperature"] == temp].set_index("model")
        if sub.empty:
            continue
        best = {m: (sub[f"{m}_mean"].idxmax() if m in HIGHER_IS_BETTER else sub[f"{m}_mean"].idxmin())
                for m in ["mae", "rmse", "r2"]}
        cells = [LATEX_LABELS[temp]]
        for model in models:
            for m in ["mae", "rmse", "r2"]:
                text = cell(sub.loc[model, f"{m}_mean"], sub.loc[model, f"{m}_std"], m)
                cells.append(f"\\textbf{{{text}}}" if model == best[m] else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def latex_single_metric(df, metric):
    models = list(df["model"].unique())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Test {metric.upper()} (\\%) per ambient temperature on held-out cycles, "
        "mean $\\pm$ standard deviation over three seeds.}",
        f"\\label{{tab:{metric}_temperature}}",
        "\\begin{tabular}{l" + "r" * len(models) + "}",
        "\\toprule",
        "Temp. (\\textdegree C) & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for temp in TEMPERATURE_ORDER:
        sub = df[df["temperature"] == temp].set_index("model")
        if sub.empty:
            continue
        best = sub[f"{metric}_mean"].idxmin()
        cells = [LATEX_LABELS[temp]]
        for model in models:
            text = cell(sub.loc[model, f"{metric}_mean"], sub.loc[model, f"{metric}_std"], metric)
            cells.append(f"\\textbf{{{text}}}" if model == best else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main(protocol, models, out_tables):
    os.makedirs(out_tables, exist_ok=True)

    runs = load_runs(protocol, models)
    if not runs:
        raise FileNotFoundError(f"No runs found under models/runs/{protocol}/")

    overall = overall_stats(runs)
    per_temp = per_temperature_stats(runs)

    overall.to_csv(os.path.join(out_tables, f"seeds_{protocol}_overall.csv"), index=False, float_format="%.6f")
    per_temp.to_csv(os.path.join(out_tables, f"seeds_{protocol}_per_temperature.csv"), index=False, float_format="%.6f")

    with open(os.path.join(out_tables, f"seeds_{protocol}_overall.tex"), "w", encoding="utf-8") as f:
        f.write(latex_overall(overall))
    with open(os.path.join(out_tables, f"seeds_{protocol}_per_temperature.tex"), "w", encoding="utf-8") as f:
        f.write(latex_per_temperature(per_temp))
    for metric in ["mae", "rmse"]:
        with open(os.path.join(out_tables, f"seeds_{protocol}_{metric}_per_temperature.tex"), "w",
                  encoding="utf-8") as f:
            f.write(latex_single_metric(per_temp, metric))

    display = overall.copy()
    for m in PERCENT_METRICS:
        display[f"{m}_mean"] *= 100
        display[f"{m}_std"] *= 100
    logger.info("Overall (MAE, RMSE, max_error in % SOC):\n" +
                display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    logger.info(f"Tables written to {out_tables}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "cnn_bilstm_attention"])
    parser.add_argument("--out-tables", default="reports/tables")
    args = parser.parse_args()
    main(args.protocol, args.models, args.out_tables)
