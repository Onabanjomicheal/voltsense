import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.data.dataset import (
    FEATURE_COLUMNS,
    SoCWindowDataset,
    SplitConfig,
    build_cycle_balanced_sampler,
    fit_scaler_on_train,
    list_drive_cycle_files,
    load_files,
    save_scaler,
    split_files,
)
from src.models.models import build_model
from src.utils.logging_config import get_logger

logger = get_logger("training")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).cpu().numpy().reshape(-1)
            preds.extend(out.tolist())
            trues.extend(y.numpy().reshape(-1).tolist())
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = mse ** 0.5
    r2 = r2_score(trues, preds) if len(set(trues)) > 1 else float("nan")
    max_abs_error = max_error(trues, preds)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "max_error": max_abs_error}, preds, trues


def per_temperature_eval(model, df, scaler, cfg, device):
    results = {}
    for temp in cfg["data"]["temperatures"]:
        sub = df[df["__temp__"] == temp]
        if sub.empty:
            continue
        ds = SoCWindowDataset(sub, scaler, sequence_length=cfg["model"]["sequence_length"])
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"])
        metrics, _, _ = evaluate(model, loader, device)
        results[temp] = {**metrics, "n_samples": len(ds)}
    return results


def load_all_with_temp_tag(data_dir, temp_folders):
    frames = []
    for temp in temp_folders:
        files = list_drive_cycle_files(data_dir, [temp])
        df = load_files(files)
        if not df.empty:
            df["__temp__"] = temp
            frames.append(df)
    import pandas as pd
    return pd.concat(frames, ignore_index=True) if frames else None


def train_model(model, train_loader, val_loader, train_ds_len, device, lr, epochs, patience,
                 mlflow_run=None, log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None
    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_rmse": [], "val_r2": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / max(train_ds_len, 1)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_mse = val_metrics["mse"]
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])
        logger.info(f"{log_prefix}epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} "
                    f"val_mse={val_mse:.6f} val_mae={val_metrics['mae']:.6f} "
                    f"val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.4f} "
                    f"val_max_err={val_metrics['max_error']:.4f}")

        if mlflow_run:
            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )

        if val_mse < best_val:
            best_val = val_mse
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"{log_prefix}Early stopping triggered at epoch {epoch+1} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, history


def run_optuna_search(cfg, train_ds, val_ds, train_loader, val_loader, seq_len, device):
    import optuna

    opt_cfg = cfg["optuna"]

    def objective(trial):
        kwargs = {
            "input_size": cfg["model"]["kwargs"].get("input_size", 4),
            "hidden_size": trial.suggest_int("hidden_size", *opt_cfg["hidden_size_range"]),
            "num_layers": trial.suggest_int("num_layers", *opt_cfg["num_layers_range"]),
        }
        lr = trial.suggest_float("learning_rate", *opt_cfg["lr_range"], log=True)

        model = build_model(cfg["model"]["name"], **kwargs).to(device)
        _, best_val, _ = train_model(
            model, train_loader, val_loader, len(train_ds), device,
            lr=lr, epochs=opt_cfg["epochs"], patience=opt_cfg.get("patience", 3),
            log_prefix="[optuna trial] "
        )
        return best_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=opt_cfg["n_trials"])
    logger.info(f"[optuna] Best params: {study.best_trial.params}")
    return study.best_trial.params


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["data"]["dir"]
    temps = cfg["data"]["temperatures"]
    files = list_drive_cycle_files(data_dir, temps)
    if not files:
        raise FileNotFoundError(
            f"No drive-cycle CSVs found under {data_dir} for temps {temps}. "
            "Run the data_processing pipeline (dvc repro) first."
        )

    split_cfg = SplitConfig(**cfg["data"]["split"])
    train_files, val_files, test_files = split_files(files, split_cfg)

    train_df = load_files(train_files)
    val_df = load_files(val_files)
    test_df = load_files(test_files)

    scaler = fit_scaler_on_train(train_df)
    os.makedirs(cfg["output"]["artifact_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["output"]["artifact_dir"], "scaler.joblib")
    save_scaler(scaler, scaler_path)

    seq_len = cfg["model"]["sequence_length"]
    train_ds = SoCWindowDataset(train_df, scaler, sequence_length=seq_len)
    val_ds = SoCWindowDataset(val_df, scaler, sequence_length=seq_len)
    test_ds = SoCWindowDataset(test_df, scaler, sequence_length=seq_len)

    bs = cfg["training"]["batch_size"]

    balance_cycles = cfg["training"].get("balance_by_cycle_type", False)
    if balance_cycles:
        sampler = build_cycle_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler)
        logger.info(f"Cycle-type balancing ENABLED. Training window counts by type: "
                    f"{dict(Counter(train_ds.cycle_types))}")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        logger.info("Cycle-type balancing disabled -- using plain shuffle.")

    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    model_kwargs = dict(cfg["model"].get("kwargs", {}))

    final_lr = cfg["training"]["lr"]
    if cfg.get("optuna", {}).get("enabled", False):
        best_params = run_optuna_search(cfg, train_ds, val_ds, train_loader, val_loader, seq_len, device)
        model_kwargs["hidden_size"] = best_params["hidden_size"]
        model_kwargs["num_layers"] = best_params["num_layers"]
        final_lr = best_params["learning_rate"]
        logger.info(f"Using Optuna-discovered hyperparameters for final training: {model_kwargs}, lr={final_lr}")
    else:
        logger.info(f"Optuna disabled -- using config hyperparameters: {model_kwargs}, lr={final_lr}")

    model = build_model(cfg["model"]["name"], **model_kwargs).to(device)

    patience = cfg["training"].get("early_stopping_patience", 5)

    if MLFLOW_AVAILABLE and cfg.get("mlflow", {}).get("enabled", False):
        tracking_uri = cfg["mlflow"].get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg["mlflow"].get("experiment_name", "battery_soc"))
        run_ctx = mlflow.start_run(run_name="lstm")
    else:
        run_ctx = None

    try:
        if run_ctx:
            mlflow.log_params({**cfg["model"], **cfg["training"], "final_lr": final_lr})

        model, best_val, history = train_model(
            model, train_loader, val_loader, len(train_ds), device,
            lr=final_lr, epochs=cfg["training"]["epochs"], patience=patience,
            mlflow_run=run_ctx, log_prefix=f"[{cfg['model']['name']}] "
        )

        test_metrics, _, _ = evaluate(model, test_loader, device)
        test_mse, test_mae = test_metrics["mse"], test_metrics["mae"]
        logger.info(f"[{cfg['model']['name']}] FINAL test_mse={test_mse:.6f} test_mae={test_mae:.6f} "
                    f"test_rmse={test_metrics['rmse']:.6f} test_r2={test_metrics['r2']:.4f} "
                    f"test_max_err={test_metrics['max_error']:.4f}")

        temp_df = load_all_with_temp_tag(data_dir, temps)
        per_temp = {}
        if temp_df is not None:
            test_file_set = {os.path.normpath(f) for f in test_files}
            temp_test_df = temp_df[temp_df["__file__"].isin(test_file_set)]
            per_temp = per_temperature_eval(model, temp_test_df, scaler, cfg, device)
            for temp, metrics in per_temp.items():
                logger.info(f"  [{temp}] mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
                            f"rmse={metrics['rmse']:.6f} r2={metrics['r2']:.4f} "
                            f"max_err={metrics['max_error']:.4f} n={metrics['n_samples']}")

        model_path = os.path.join(cfg["output"]["artifact_dir"], "model.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": cfg["model"]["name"],
            "model_kwargs": model_kwargs,
            "sequence_length": seq_len,
            "feature_columns": FEATURE_COLUMNS,
        }, model_path)
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved scaler to {scaler_path}")

        mlflow_run_id = run_ctx.info.run_id if run_ctx else None

        manifest = {
            "model_name": cfg["model"]["name"],
            "mlflow_run_id": mlflow_run_id,
            "balance_by_cycle_type": balance_cycles,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_max_error": test_metrics["max_error"],
            "per_temperature": per_temp,
            "hyperparameters": model_kwargs,
            "final_learning_rate": final_lr,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = os.path.join(cfg["output"]["artifact_dir"], "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved traceability manifest to {manifest_path}")

        if run_ctx:
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            for temp, metrics in per_temp.items():
                mlflow.log_metrics(
                    {f"test_{k}_{temp}": v for k, v in metrics.items() if k != "n_samples"}
                )
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)
            mlflow.log_artifact(manifest_path)

            registry_cfg = cfg.get("mlflow", {}).get("registry", {})
            if registry_cfg.get("enabled", False):
                registered_name = registry_cfg.get("model_name", "battery_soc_lstm")
                client = mlflow.MlflowClient()
                try:
                    client.get_registered_model(registered_name)
                except Exception:
                    client.create_registered_model(registered_name)

                mv = client.create_model_version(
                    name=registered_name,
                    source=f"runs:/{mlflow_run_id}/{os.path.basename(model_path)}",
                    run_id=mlflow_run_id,
                )
                logger.info(f"Registered model version {mv.version} for '{registered_name}'")

                should_promote = True
                try:
                    current_prod = client.get_model_version_by_alias(registered_name, "production")
                    prod_run = client.get_run(current_prod.run_id)
                    prod_mse = prod_run.data.metrics.get("test_mse", float("inf"))
                    should_promote = test_mse < prod_mse
                    logger.info(f"Current production test_mse={prod_mse:.6f}, "
                                f"new model test_mse={test_mse:.6f}, promote={should_promote}")
                except Exception:
                    logger.info("No existing 'production' alias found -- this will be the first.")

                if should_promote:
                    client.set_registered_model_alias(registered_name, "production", mv.version)
                    logger.info(f"Set 'production' alias to version {mv.version}.")
                else:
                    logger.info(f"Version {mv.version} registered but NOT promoted "
                                f"(did not beat current production model).")
    finally:
        if run_ctx:
            mlflow.end_run()

    return {**test_metrics, "per_temperature": per_temp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
