"""Generates fake but schema-correct battery CSVs for local testing/smoke-checks.
NOT for real training -- use this only to verify the pipeline runs before pointing
it at the real Mendeley-derived dataset (place that in data/processed/)."""
import os

import numpy as np
import pandas as pd

np.random.seed(0)

OUT_DIR = "data/processed"
TEMPS = ["25degC", "0degC", "n10degC", "10degC", "40degC", "n20degC"]
CYCLE_NAMES = ["Mixed1", "Mixed2", "UDDS1", "HWFET1", "LA921", "US061"]
N_ROWS = 300


def make_file(path, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(N_ROWS, dtype=float)
    soc = np.clip(1.0 - t / N_ROWS + rng.normal(0, 0.01, N_ROWS), 0, 1)
    voltage = 3.0 + 1.2 * soc + rng.normal(0, 0.02, N_ROWS)
    current = rng.normal(-1.0, 0.5, N_ROWS)
    temp_c = rng.normal(25, 1.0, N_ROWS)
    df = pd.DataFrame({
        "Time [s]": t, "Voltage [V]": voltage, "Current [A]": current,
        "Temperature [degC]": temp_c, "SOC [-]": soc,
    })
    df.to_csv(path, index=False)


def main():
    for temp in TEMPS:
        d = os.path.join(OUT_DIR, temp)
        os.makedirs(d, exist_ok=True)
        for cycle in CYCLE_NAMES:
            make_file(os.path.join(d, f"{cycle}.csv"), seed=hash((temp, cycle)) % (2**31))
    print(f"Synthetic dataset written under {OUT_DIR}")


if __name__ == "__main__":
    main()
