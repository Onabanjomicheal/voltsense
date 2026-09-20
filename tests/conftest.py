"""Shared pytest fixtures: synthetic multi-file battery data matching the real
processed CSV schema, used across all tests so no test depends on the real
dataset being present."""
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_processed_dir(tmp_path):
    data_dir = tmp_path / "processed"
    names = ["Mixed1", "Mixed2", "Mixed3", "UDDS1", "UDDS2", "US061", "US062", "HWFET1", "LA921", "LA922"]

    rng = np.random.RandomState(0)
    for temp in ["25degC", "0degC"]:
        temp_dir = data_dir / temp
        os.makedirs(temp_dir, exist_ok=True)
        for name in names:
            n = 60
            t = np.arange(n, dtype=float)
            soc = np.clip(1.0 - t / n + rng.normal(0, 0.01, n), 0, 1)
            df = pd.DataFrame({
                "Time [s]": t,
                "Voltage [V]": 3.0 + 1.2 * soc + rng.normal(0, 0.01, n),
                "Current [A]": rng.normal(-0.5, 0.1, n),
                "Temperature [degC]": 25.0,
                "SOC [-]": soc,
            })
            df.to_csv(temp_dir / f"{name}.csv", index=False)

    return str(data_dir)


@pytest.fixture
def sample_readings():
    return [
        {
            "time_s": float(i),
            "voltage_v": 4.0 - i * 0.01,
            "current_a": -0.45,
            "temperature_c": 25.0,
        }
        for i in range(20)
    ]
