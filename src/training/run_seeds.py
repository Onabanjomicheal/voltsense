import argparse
import copy
import os

import yaml

from src.training.train import main as train_main


def run_seeds(base_config_path, seeds, protocol, train_cycles, test_cycles):
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    model_tag = base_cfg["model"]["name"]

    if test_cycles:
        base_cfg["data"]["split"]["strategy"] = "by_drive_cycle_type"
        base_cfg["data"]["split"]["train_cycle_keywords"] = train_cycles
        base_cfg["data"]["split"]["test_cycle_keywords"] = test_cycles

    os.makedirs("configs/generated", exist_ok=True)

    for seed in seeds:
        run_dir = f"models/runs/{protocol}/{model_tag}_seed{seed}"
        if os.path.exists(os.path.join(run_dir, "manifest.json")):
            print(f"Skipping {model_tag} seed {seed}: already trained at {run_dir}")
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        cfg["output"]["artifact_dir"] = run_dir

        generated_path = f"configs/generated/{protocol}_{model_tag}_seed{seed}.yaml"
        with open(generated_path, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n=== {model_tag} | protocol={protocol} | seed={seed} ===")
        train_main(generated_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--train-cycles", nargs="+", default=None)
    parser.add_argument("--test-cycles", nargs="+", default=None)
    args = parser.parse_args()
    run_seeds(args.config, args.seeds, args.protocol, args.train_cycles, args.test_cycles)
