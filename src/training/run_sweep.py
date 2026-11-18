import argparse
import copy

import yaml

from src.training.train import main as train_main


def run_sweep(base_config_path: str, sequence_lengths: list, model_tag: str):
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    results = {}

    for seq_len in sequence_lengths:
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["sequence_length"] = seq_len
        cfg["output"]["artifact_dir"] = f"models/{model_tag}_seq{seq_len}"

        generated_path = f"configs/{model_tag}_seq{seq_len}.yaml"
        with open(generated_path, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n=== Running {model_tag}, sequence_length={seq_len} ===")
        metrics = train_main(generated_path)
        results[seq_len] = metrics

    print(f"\n=== Sweep summary: {model_tag} ===")
    print(f"{'seq_len':>8} {'mae':>10} {'rmse':>10} {'r2':>8} {'max_error':>10}")
    for seq_len, m in results.items():
        print(f"{seq_len:>8} {m['mae']:>10.6f} {m['rmse']:>10.6f} {m['r2']:>8.4f} {m['max_error']:>10.4f}")

    best_seq_len = min(results, key=lambda k: results[k]["mae"])
    print(f"\nBest sequence_length by test MAE: {best_seq_len}")

    return results, best_seq_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, required=True)
    parser.add_argument("--model-tag", required=True)
    args = parser.parse_args()
    run_sweep(args.config, args.sequence_lengths, args.model_tag)
