import glob
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Literal

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, WeightedRandomSampler

FEATURE_COLUMNS = ["Voltage [V]", "Current [A]", "Temperature [degC]", "Power [W]"]
LABEL_COLUMN = "SOC [-]"

DRIVE_CYCLE_TAGS = ("Mixed", "UDDS", "HWFET", "LA92", "US06")


def is_drive_cycle(name: str) -> bool:
    return any(tag in name for tag in DRIVE_CYCLE_TAGS)


def cycle_type_of_file(path: str) -> str:
    name = os.path.basename(path)
    for tag in DRIVE_CYCLE_TAGS:
        if tag in name:
            return tag
    return "Unknown"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Power [W]"] = df["Voltage [V]"] * df["Current [A]"]
    dt = df["Time [s]"].diff().fillna(0.0)
    df["CC_Capacity [Ah]"] = (df["Current [A]"] * dt / 3600.0).cumsum()
    return df


def list_drive_cycle_files(data_dir: str, temp_folders: List[str]) -> List[str]:
    files = []
    for temp in temp_folders:
        for path in sorted(glob.glob(os.path.join(data_dir, temp, "*.csv"))):
            name = os.path.basename(path)
            if not is_drive_cycle(name):
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
        train_pool = [f for f in files if any(k in os.path.basename(f) for k in cfg.train_cycle_keywords)]
        test_files = [f for f in files if any(k in os.path.basename(f) for k in cfg.test_cycle_keywords)]
        train_files, val_files = train_test_split(
            train_pool, test_size=cfg.val_size, random_state=cfg.random_state
        )
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
        df["__file__"] = os.path.normpath(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [LABEL_COLUMN, "__file__"])
    return pd.concat(frames, ignore_index=True)


class SoCWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.samples = []
        self.cycle_types = []

        df = df.reset_index(drop=True)
        feats = scaler.transform(df[FEATURE_COLUMNS].values)
        labels = df[LABEL_COLUMN].values

        for file_name, file_df in df.groupby("__file__", sort=False):
            idx = file_df.index.values
            f = feats[idx]
            y = labels[idx]
            n = len(idx)
            if n < sequence_length:
                continue
            cycle_type = cycle_type_of_file(file_name)
            for start in range(0, n - sequence_length + 1):
                window = f[start:start + sequence_length]
                label = y[start + sequence_length - 1]
                self.samples.append((window, label))
                self.cycle_types.append(cycle_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        window, label = self.samples[i]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def fit_scaler_on_train(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)
    return scaler


def save_scaler(scaler: StandardScaler, path: str):
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


def build_cycle_balanced_sampler(dataset: SoCWindowDataset) -> WeightedRandomSampler:
    counts = Counter(dataset.cycle_types)
    weights = [1.0 / counts[ct] for ct in dataset.cycle_types]
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)