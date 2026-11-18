import argparse
import os
import statistics
import time

import pandas as pd
import torch

from src.models.models import build_model
from src.utils.logging_config import get_logger

logger = get_logger("compare")

DEFAULT_RUNS = {
    "LSTM": ("lstm", "models/runs/cycle_holdout/lstm_seed42"),
    "GRU": ("gru", "models/runs/cycle_holdout/gru_seed42"),
    "CNN-BiLSTM-Attention": ("cnn_bilstm_attention", "models/runs/cycle_holdout/cnn_bilstm_attention_seed42"),
}


def load_model(model_name, run_dir):
    path = os.path.join(run_dir, "model.pth")
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(model_name, **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["sequence_length"], os.path.getsize(path) / (1024 * 1024)


def time_inference(model, sequence_length, n_features, warmup, repeats):
    x = torch.randn(1, sequence_length, n_features)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def main(runs, out_tables, n_features, warmup, repeats, threads):
    os.makedirs(out_tables, exist_ok=True)
    torch.set_num_threads(threads)

    rows = []
    for display, (model_name, run_dir) in runs.items():
        model, sequence_length, size_mb = load_model(model_name, run_dir)
        params = sum(p.numel() for p in model.parameters())
        latency = time_inference(model, sequence_length, n_features, warmup, repeats)
        rows.append({"model": display, "parameters": params, "size_mb": size_mb, "inference_ms": latency})
        logger.info(f"{display}: {params:,} parameters, {size_mb:.3f} MB, {latency:.3f} ms per window")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_tables, "model_size.csv"), index=False, float_format="%.4f")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model size and estimation time per input window "
        f"(CPU, {threads} thread{'s' if threads != 1 else ''}, batch size 1).}}",
        "\\label{tab:model_size}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Parameters & Size (MB) & Time (ms) \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['model']} & {int(r['parameters']):,} & {r['size_mb']:.2f} & {r['inference_ms']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(out_tables, "model_size.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    logger.info(f"Tables written to {out_tables}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="cycle_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-tables", default="reports/tables")
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    runs = {d: (n, os.path.join("models", "runs", args.protocol, f"{n}_seed{args.seed}"))
            for d, (n, _) in DEFAULT_RUNS.items()}
    main(runs, args.out_tables, args.n_features, args.warmup, args.repeats, args.threads)
