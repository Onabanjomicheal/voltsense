"""
DVC Stage 2: data/interim -> data/processed (SoC estimation).

SoC estimation method: pOCV (pseudo Open Circuit Voltage) is used ONCE per file to
estimate the STARTING SoC only, based on initial voltage and initial charge/discharge
direction, using a voltage->SoC lookup curve built from a slow C20 reference test.
Every timestep after that is pure Coulomb counting: SoC drifts by integrating current
over time, normalized against max charge/discharge capacity from the C20 file,
clamped to [0, 1]. "1Hz resampling" = round timestamps to nearest second, drop
duplicates.
"""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

TIME_COL = "Time [min]"
TIME_COL_S = "Time [s]"
VOLTAGE_COL = "Voltage [V]"
CURRENT_COL = "Current [A]"
TEMPERATURE_COL = "Temperature [degC]"
CAPACITY_COL = "Capacity [Ah]"
SOC_COL = "SOC [-]"


def get_pocv_soc_interp_fn(c20_parsed_file_path: str):
    df = pd.read_csv(c20_parsed_file_path)

    df_discharge = df[df[CURRENT_COL] < 0].copy()
    df_discharge[CAPACITY_COL] = df_discharge[CAPACITY_COL] - df_discharge[CAPACITY_COL].iloc[0]
    df_discharge[SOC_COL] = 1 - abs(df_discharge[CAPACITY_COL] / df_discharge[CAPACITY_COL].iloc[-1])
    max_v_dis = df_discharge[VOLTAGE_COL].max()
    df_discharge = df_discharge[df_discharge[VOLTAGE_COL] <= max_v_dis]
    discharge_interp = interp1d(df_discharge[VOLTAGE_COL], df_discharge[SOC_COL],
                                 bounds_error=False, fill_value="extrapolate")

    df_charge = df[df[CURRENT_COL] > 0].copy()
    df_charge[CAPACITY_COL] = df_charge[CAPACITY_COL] - df_charge[CAPACITY_COL].iloc[0]
    df_charge[SOC_COL] = abs(df_charge[CAPACITY_COL]) / df_charge[CAPACITY_COL].iloc[-1]
    max_v_chg = df_charge[VOLTAGE_COL].max()
    df_charge = df_charge[df_charge[VOLTAGE_COL] <= max_v_chg]
    charge_interp = interp1d(df_charge[VOLTAGE_COL], df_charge[SOC_COL],
                              bounds_error=False, fill_value="extrapolate")

    return charge_interp, discharge_interp


def get_max_capacities(c20_parsed_file_path: str):
    df = pd.read_csv(c20_parsed_file_path)
    charge_start_index = df[df[CURRENT_COL] > 0].index[0]
    df_discharge = df.iloc[:charge_start_index]
    df_charge = df.iloc[charge_start_index:]
    max_discharge_capacity = df_discharge[CAPACITY_COL].max() - df_discharge[CAPACITY_COL].min()
    max_charge_capacity = df_charge[CAPACITY_COL].max() - df_charge[CAPACITY_COL].min()
    return max_charge_capacity, max_discharge_capacity


def get_initial_soc(df: pd.DataFrame, charge_soc_fn, discharge_soc_fn) -> float:
    initial_voltage = df[VOLTAGE_COL].iloc[0]
    first_non_zero_index = df[df[CURRENT_COL] != 0].index[0]
    first_non_zero_current = df[CURRENT_COL].iloc[first_non_zero_index]
    if first_non_zero_current < 0:
        return float(discharge_soc_fn(initial_voltage))
    return float(charge_soc_fn(initial_voltage))


def estimate_soc_for_file(df: pd.DataFrame, charge_soc_fn, discharge_soc_fn,
                           max_charge_capacity: float, max_discharge_capacity: float) -> pd.DataFrame:
    df = df.copy()
    initial_soc = get_initial_soc(df, charge_soc_fn, discharge_soc_fn)
    df[SOC_COL] = initial_soc
    df["Cumulative_Capacity_Ah"] = 0.0

    last_known_soc = initial_soc
    for index in df.index[1:]:
        time_diff = (df.at[index, TIME_COL_S] - df.at[index - 1, TIME_COL_S]) / 3600
        cumulative_capacity_change = df.at[index, CURRENT_COL] * time_diff
        df.at[index, "Cumulative_Capacity_Ah"] = df.at[index - 1, "Cumulative_Capacity_Ah"] + cumulative_capacity_change

        current = df.at[index, CURRENT_COL]
        if current != 0:
            if current < 0:
                soc = last_known_soc - (abs(cumulative_capacity_change) / max_discharge_capacity)
            else:
                soc = last_known_soc + (cumulative_capacity_change / max_charge_capacity)
            soc = max(0.0, min(soc, 1.0))
            last_known_soc = soc
        else:
            soc = last_known_soc
        df.at[index, SOC_COL] = soc

    return df


def resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rounded_Time"] = df[TIME_COL_S].round().astype(int)
    return df.drop_duplicates(subset="Rounded_Time")


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
    """DVC stage entrypoint: data/interim -> data/processed, for every temperature folder."""
    temperatures = [d for d in os.listdir(interim_data_directory) if "degC" in d]

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

        c20_path = os.path.join(interim_dir, c20_file)
        charge_fn, discharge_fn = get_pocv_soc_interp_fn(c20_path)
        max_chg_cap, max_dis_cap = get_max_capacities(c20_path)

        for f in os.listdir(interim_dir):
            if "C20" in f or not f.endswith(".csv"):
                continue
            name = f.replace("_parsed.csv", "")
            try:
                df = pd.read_csv(os.path.join(interim_dir, f))
                df = estimate_soc_for_file(df, charge_fn, discharge_fn, max_chg_cap, max_dis_cap)
                df = resample_1hz(df)
                df.to_csv(os.path.join(processed_dir, f"{name}_processed.csv"), index=False)
                generate_and_save_soc_plot(
                    df,
                    os.path.join(processed_plots_dir, f"{name}_processed_plot.pdf"),
                    fig_title=f"{name} SOC @ {T}",
                )
                logging.info(f"Estimated SoC for {name} @ {T}")
            except Exception as e:
                logging.error(f"Error estimating SoC for {name} @ {T}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--processed_plots_dir", default="reports/figures/processed")
    args = parser.parse_args()
    estimate_soc_all(args.interim_dir, args.processed_dir, args.processed_plots_dir)
