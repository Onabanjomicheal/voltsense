"""
DVC Stage 2: data/interim -> data/processed (reference SoC generation).

Scope: DRIVE-CYCLE files only (Mixed, UDDS, HWFET, LA92, US06). The slow 60 s files
(Charge, Discharge, Cap, C20) and HPPC are NOT written to data/processed. C20 is used
only as the reference for the starting SoC and the capacities.

Method
------
1. Starting SoC: pseudo-OCV (pOCV). The first voltage of each file is mapped to SoC
   through a voltage->SoC curve built from the slow C20 test, using the charge or
   discharge curve depending on the first non-zero current direction. This is NOT a
   true equilibrium OCV (the cell is usually already under load), so treat the
   starting SoC as an estimate.
2. Every later step is Coulomb counting at the NATIVE rate (10 Hz), normalised by the
   C20 charge/discharge capacity and clipped to [0, 1]. Rest keeps SoC constant.
3. Only after integration is the file decimated to 1 Hz (one raw row kept per
   rounded second). Decimation, not averaging, so the rows look like instantaneous
   1 Hz sensor readings, which is what the deployed API will receive.

The resulting "SOC [-]" is a Coulomb-counted REFERENCE SoC, not lab ground truth.
"""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

TIME_COL = "Time [min]"
TIME_COL_S = "Time [s]"
VOLTAGE_COL = "Voltage [V]"
CURRENT_COL = "Current [A]"
TEMPERATURE_COL = "Temperature [degC]"
CAPACITY_COL = "Capacity [Ah]"
SOC_COL = "SOC [-]"

# Current below this magnitude counts as rest (A).
ZERO_CURRENT_EPS = 1e-3

# Files whose name contains one of these tags are in scope. Add "PausCycl" here if
# you decide to include it.
DRIVE_CYCLE_TAGS = ("Mixed", "UDDS", "HWFET", "LA92", "US06")

# Expected native sampling of drive-cycle files (s) and how far the median may stray.
DRIVE_DT_RANGE_S = (0.08, 0.12)

# Any time gap larger than this makes Coulomb counting unreliable -> fail loudly.
MAX_GAP_S = 5.0

# Known-bad files, as (temperature folder, file stem). Confirmed against the data
# quality report:
#   n10degC/602_Mixed4  : timestamp frozen for ~4,160 rows, then jumps ~416 s.
#   25degC/551_HWFET    : current is 0 A for all 6,002 rows (confirmed directly);
#                         file runs ~600 s vs ~3,300-5,600 s for HWFET elsewhere.
#   40degC/562_Mixed4-8 and 562_PausCycl: truncated to ~780-1,440 s vs
#                         ~6,000-8,000 s for Mixed elsewhere at other temperatures.
EXCLUDE_FILES = {
    ("n10degC", "602_Mixed4"),
    ("25degC", "551_HWFET"),
    ("40degC", "562_Mixed4"),
    ("40degC", "562_Mixed5"),
    ("40degC", "562_Mixed6"),
    ("40degC", "562_Mixed7"),
    ("40degC", "562_Mixed8"),
    ("40degC", "562_PausCycl"),
}

MIN_CURVE_POINTS = 50


def is_drive_cycle(name: str) -> bool:
    return any(tag in name for tag in DRIVE_CYCLE_TAGS)


def _build_curve(segment: pd.DataFrame, discharge: bool):
    """Voltage->SoC interpolator and total capacity (Ah) from one C20 segment."""
    if len(segment) < MIN_CURVE_POINTS:
        raise ValueError(f"C20 segment too short ({len(segment)} rows)")
    capacity = (segment[CAPACITY_COL] - segment[CAPACITY_COL].iloc[0]).abs()
    total = float(capacity.max())
    if not total > 0:
        raise ValueError("C20 segment has non-positive capacity")

    soc = 1.0 - capacity / total if discharge else capacity / total
    curve = pd.DataFrame({VOLTAGE_COL: segment[VOLTAGE_COL].to_numpy(), SOC_COL: soc.to_numpy()})
    curve = curve.dropna()
    # Repeated voltages break interpolation: average them, then sort.
    curve = curve.groupby(VOLTAGE_COL, as_index=False)[SOC_COL].mean().sort_values(VOLTAGE_COL)
    if len(curve) < MIN_CURVE_POINTS:
        raise ValueError(f"Too few unique voltage points ({len(curve)})")

    # Outside the measured voltage range, hold the end values instead of extrapolating
    # (linear extrapolation can leave [0, 1]). The caller logs when this happens.
    fn = interp1d(
        curve[VOLTAGE_COL].to_numpy(),
        curve[SOC_COL].to_numpy(),
        bounds_error=False,
        fill_value=(float(curve[SOC_COL].iloc[0]), float(curve[SOC_COL].iloc[-1])),
    )
    return fn, total


def load_c20_reference(c20_parsed_file_path: str):
    """Return (charge_fn, discharge_fn, max_charge_capacity, max_discharge_capacity)."""
    df = pd.read_csv(c20_parsed_file_path)
    for col in (VOLTAGE_COL, CURRENT_COL, CAPACITY_COL):
        if col not in df.columns:
            raise ValueError(f"C20 file missing column {col!r}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[VOLTAGE_COL, CURRENT_COL, CAPACITY_COL])

    current = df[CURRENT_COL]
    discharge_fn, max_dis = _build_curve(df[current < -ZERO_CURRENT_EPS], discharge=True)
    charge_fn, max_chg = _build_curve(df[current > ZERO_CURRENT_EPS], discharge=False)
    return charge_fn, discharge_fn, max_chg, max_dis



# Gaps up to this size between a file's starting voltage and the C20 curve's
# measured range are treated as measurement noise, not a real extrapolation.
# Confirmed on real data: C20 charge max = 4.19995 V, a drive-cycle start of
# 4.18632 V (a ~14 mV gap) is normal end-of-charge behavior, not a data problem.
VOLTAGE_TOLERANCE = 0.05


def get_initial_soc(df: pd.DataFrame, charge_soc_fn, discharge_soc_fn) -> float:
    active = df[df[CURRENT_COL].abs() > ZERO_CURRENT_EPS]
    if active.empty:
        raise ValueError("File has no non-zero current")
    fn = discharge_soc_fn if active[CURRENT_COL].iloc[0] < 0 else charge_soc_fn
    v0 = float(df[VOLTAGE_COL].iloc[0])
    lo, hi = float(fn.x.min()), float(fn.x.max())
    if v0 < lo - VOLTAGE_TOLERANCE or v0 > hi + VOLTAGE_TOLERANCE:
        logging.warning(
            f"Initial voltage {v0:.4f} V is more than {VOLTAGE_TOLERANCE} V "
            f"outside the C20 range [{lo:.4f}, {hi:.4f}]; using end value"
        )
    return float(np.clip(fn(v0), 0.0, 1.0))


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.sort_values(TIME_COL_S)
    df = df.drop_duplicates(subset=TIME_COL_S, keep="first")
    return df.reset_index(drop=True)


def validate_drive_sampling(t: np.ndarray) -> None:
    dt = np.diff(t)
    if dt.size == 0:
        raise ValueError("Too few rows")
    if dt.min() <= 0:
        raise ValueError("Non-increasing timestamps after cleaning")
    median = float(np.median(dt))
    if not DRIVE_DT_RANGE_S[0] <= median <= DRIVE_DT_RANGE_S[1]:
        raise ValueError(f"Median dt {median:.3f} s is not a ~0.1 s drive-cycle file")
    if dt.max() > MAX_GAP_S:
        raise ValueError(f"Time gap of {dt.max():.1f} s; Coulomb counting unreliable")


def estimate_soc_for_file(df: pd.DataFrame, charge_soc_fn, discharge_soc_fn,
                          max_charge_capacity: float, max_discharge_capacity: float) -> pd.DataFrame:
    df = clean_time_series(df)
    t = df[TIME_COL_S].to_numpy(dtype=float)
    current = df[CURRENT_COL].to_numpy(dtype=float)
    validate_drive_sampling(t)

    dq = current[1:] * np.diff(t) / 3600.0  # Ah moved in each step, signed

    soc = np.empty(len(df))
    soc[0] = get_initial_soc(df, charge_soc_fn, discharge_soc_fn)
    for k in range(1, len(df)):
        i = current[k]
        if i > ZERO_CURRENT_EPS:
            soc[k] = min(1.0, soc[k - 1] + dq[k - 1] / max_charge_capacity)
        elif i < -ZERO_CURRENT_EPS:
            soc[k] = max(0.0, soc[k - 1] + dq[k - 1] / max_discharge_capacity)
        else:
            soc[k] = soc[k - 1]

    df[SOC_COL] = soc
    df["Cumulative_Capacity_Ah"] = np.concatenate([[0.0], np.cumsum(dq)])
    return df


def decimate_to_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first raw row in each rounded second. Not true resampling."""
    df = df.copy()
    df["Rounded_Time"] = df[TIME_COL_S].round().astype(int)
    return df.drop_duplicates(subset="Rounded_Time").reset_index(drop=True)


def check_processed(df: pd.DataFrame) -> None:
    if df[SOC_COL].isna().any():
        raise ValueError("NaN in SOC")
    if df[SOC_COL].min() < 0 or df[SOC_COL].max() > 1:
        raise ValueError("SOC outside [0, 1]")


def generate_and_save_soc_plot(df: pd.DataFrame, save_file_path: str, fig_title: str = ""):
    fig, axs = plt.subplots(5, 1, figsize=(10, 12.5))
    axs[0].plot(df[TIME_COL], df[VOLTAGE_COL], label="Voltage")
    axs[1].plot(df[TIME_COL], df[CURRENT_COL], label="Current")
    axs[2].plot(df[TIME_COL], df[TEMPERATURE_COL], label="Temperature")
    axs[3].plot(df[TIME_COL], df[CAPACITY_COL], label="Capacity")
    axs[4].plot(df[TIME_COL], df[SOC_COL], label="SoC")
    for ax in axs:
        ax.legend()
        ax.set_xlabel(TIME_COL)
    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_file_path, format="pdf")
    plt.close(fig)


def estimate_soc_all(interim_data_directory: str, processed_data_directory: str, processed_plots_directory: str):
    """DVC stage entrypoint: data/interim -> data/processed, drive cycles only."""
    temperatures = [d for d in os.listdir(interim_data_directory) if "degC" in d]
    n_ok, n_failed = 0, 0

    for T in temperatures:
        interim_dir = os.path.join(interim_data_directory, T)
        processed_dir = os.path.join(processed_data_directory, T)
        processed_plots_dir = os.path.join(processed_plots_directory, T)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(processed_plots_dir, exist_ok=True)

        c20_file = next((f for f in os.listdir(interim_dir) if "C20" in f), None)
        if not c20_file:
            logging.warning(f"No C20 reference file found in {interim_dir}, skipping temperature {T}.")
            continue

        try:
            charge_fn, discharge_fn, max_chg_cap, max_dis_cap = load_c20_reference(
                os.path.join(interim_dir, c20_file)
            )
        except Exception:
            logging.error(f"C20 reference invalid for {T}", exc_info=True)
            continue
        logging.info(f"{T}: C20 capacity charge={max_chg_cap:.3f} Ah, discharge={max_dis_cap:.3f} Ah")

        for f in sorted(os.listdir(interim_dir)):
            if not f.endswith(".csv"):
                continue
            name = f.replace("_parsed.csv", "")
            if not is_drive_cycle(name):
                continue
            if (T, name) in EXCLUDE_FILES:
                logging.warning(f"Skipping known-bad file {name} @ {T}")
                continue
            try:
                df = pd.read_csv(os.path.join(interim_dir, f))
                df = estimate_soc_for_file(df, charge_fn, discharge_fn, max_chg_cap, max_dis_cap)
                df = decimate_to_1hz(df)
                check_processed(df)
                df.to_csv(os.path.join(processed_dir, f"{name}_processed.csv"), index=False)
                generate_and_save_soc_plot(
                    df,
                    os.path.join(processed_plots_dir, f"{name}_processed_plot.pdf"),
                    fig_title=f"{name} SOC @ {T}",
                )
                logging.info(f"Estimated SoC for {name} @ {T}")
                n_ok += 1
            except Exception:
                logging.error(f"Error estimating SoC for {name} @ {T}", exc_info=True)
                n_failed += 1

    logging.info(f"Done: {n_ok} files written, {n_failed} failed")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--processed_plots_dir", default="reports/figures/processed")
    args = parser.parse_args()
    estimate_soc_all(args.interim_dir, args.processed_dir, args.processed_plots_dir)
