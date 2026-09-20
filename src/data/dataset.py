"""
Data loading, feature engineering, windowing, and splitting for the Battery SoC project.

Expected raw file schema (matches the real dataset):
    Columns: 'Time [s]', 'Voltage [V]', 'Current [A]', 'Temperature [degC]', 'SOC [-]'
    Directory layout: data/processed/<temp_folder>/<filename>.csv
    e.g. data/processed/25degC/Mixed1.csv

Files containing 'Charge' or 'Dis' in the name are skipped (constant charge/discharge
files, not dynamic drive cycles) -- matches the original notebook's behavior.

IMPORTANT: CC_Capacity (cumulative Coulomb-counted capacity) is deliberately NOT a
model feature. It's a cumulative sum computed from the start of whatever data is
given -- meaningful across a full training file, but meaningless in short serving
windows since every request starts fresh with no true history. Including it caused
serving predictions to barely respond to voltage at all (see reports/model_card.md).
Only instantaneous features (Voltage, Current, Temperature, Power) are used.
"""
import glob
import os
from dataclasses import dataclass, field
from typing import List, Literal

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = ["Voltage [V]", "Current [A]", "Temperature [degC]", "Power [W]"]
LABEL_COLUMN = "SOC [-]"
SKIP_SUBSTRINGS = ("Charge", "Dis")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Power [W] (instantaneous, safe for serving) and CC_Capacity [Ah]
    (kept for reference/analysis but NOT used as a model feature -- see module
    docstring)."""
    df = df.copy()
    df["Power [W]"] = df["Voltage [V]"] * df["Current [A]"]
    dt = df["Time [s]"].diff().fillna(0.0)
    df["CC_Capacity [Ah]"] = (df["Current [A]"] * dt / 3600.0).cumsum()
    return df


def list_drive_cycle_files(data_dir: str, temp_folders: List[str]) -> List[str]:
    """Return full paths to dynamic drive-cycle CSVs across the given temperature folders."""
    files = []
    for temp in temp_folders:
        for path in sorted(glob.glob(os.path.join(data_dir, temp, "*.csv"))):
            name = os.path.basename(path)
            if any(s in name for s in SKIP_SUBSTRINGS):
                continue
            files.append(path)
    return files


@dataclass
class SplitConfig:
    strategy: Literal["random_by_file", "by_drive_cycle_type"] = "random_by_file"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 24
    train_cycle_keywords: List[str] = field(default_factory=lambda: ["Mixed"])
    test_cycle_keywords: List[str] = field(default_factory=lambda: ["UDDS", "HWFET", "LA92", "US06"])


def split_files(files: List[str], cfg: SplitConfig):
    if cfg.strategy == "random_by_file":
        train_files, rest = train_test_split(
            files, test_size=cfg.test_size + cfg.val_size, random_state=cfg.random_state
        )
        rel_val = cfg.val_size / (cfg.test_size + cfg.val_size) if (cfg.test_size + cfg.val_size) > 0 else 0
        val_files, test_files = train_test_split(rest, test_size=1 - rel_val, random_state=cfg.random_state)
        return train_files, val_files, test_files

    elif cfg.strategy == "by_drive_cycle_type":
        train_files = [f for f in files if any(k in os.path.basename(f) for k in cfg.train_cycle_keywords)]
        test_pool = [f for f in files if any(k in os.path.basename(f) for k in cfg.test_cycle_keywords)]
        if len(test_pool) < 2:
            val_files, test_files = test_pool, test_pool
        else:
            val_files, test_files = train_test_split(test_pool, test_size=0.5, random_state=cfg.random_state)
        return train_files, val_files, test_files

    raise ValueError(f"Unknown split strategy: {cfg.strategy}")


def load_files(files: List[str], validate: bool = True) -> pd.DataFrame:
    from src.data.validation import validate_processed_file

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if validate:
            validate_processed_file(df, file_path=path)
        df = engineer_features(df)
        df["__file__"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [LABEL_COLUMN, "__file__"])
    return pd.concat(frames, ignore_index=True)


class SoCWindowDataset(Dataset):
    """
    Sliding-window dataset over one or more battery log files. Windows are built
    PER FILE (grouped by __file__) so a window can never silently splice together
    the end of one drive-cycle file and the start of an unrelated one.
    """

    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.samples = []

        df = df.reset_index(drop=True)
        feats = scaler.transform(df[FEATURE_COLUMNS].values)
        labels = df[LABEL_COLUMN].values

        for _, file_df in df.groupby("__file__", sort=False):
            idx = file_df.index.values
            f = feats[idx]
            y = labels[idx]
            n = len(idx)
            if n < sequence_length:
                continue
            for start in range(0, n - sequence_length + 1):
                window = f[start:start + sequence_length]
                label = y[start + sequence_length - 1]
                self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        window, label = self.samples[i]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def fit_scaler_on_train(train_df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on TRAIN ONLY. Fixes the original leakage bug (original
    notebook fit the scaler on the full dataset before splitting)."""
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)
    return scaler


def save_scaler(scaler: StandardScaler, path: str):
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)
