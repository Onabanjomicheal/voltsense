"""
DVC Stage 1: data/raw -> data/interim.

Ported from the original data_processing/data_parsing.py (author: Arpan Biswas),
split out as its own stage so DVC can detect when only this step's inputs change.
"""
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TIMESTAMP_COL = "Timestamp"
TIME_COL = "Time [min]"
TIME_COL_S = "Time [s]"
VOLTAGE_COL = "Voltage [V]"
CURRENT_COL = "Current [A]"
TEMPERATURE_COL = "Temperature [degC]"
CAPACITY_COL = "Capacity [Ah]"


def parse_raw_data(file_path: str) -> pd.DataFrame:
    with open(file_path) as f:
        lines = f.readlines()

    column_index = lines.index(next(filter(lambda l: "Time Stamp" in l, lines)))
    column_line = lines[column_index].split(",")
    data_lines = [l.split(",") for l in lines[column_index + 2:]]

    abs_timestamp_data = []
    timestamp_data_seconds = []
    for l in data_lines:
        abs_timestamp_data.append(pd.Timestamp(l[column_line.index("Time Stamp")]))
        timestamp_str = [float(s) for s in l[column_line.index("Prog Time")].split(":")]
        timestamp_seconds = timestamp_str[0] * 3600 + timestamp_str[1] * 60 + timestamp_str[2]
        timestamp_data_seconds.append(timestamp_seconds)

    df = pd.DataFrame({
        TIMESTAMP_COL: abs_timestamp_data,
        TIME_COL: [(t - timestamp_data_seconds[0]) / 60 for t in timestamp_data_seconds],
        TIME_COL_S: [(t - timestamp_data_seconds[0]) for t in timestamp_data_seconds],
        VOLTAGE_COL: [float(l[column_line.index("Voltage")]) for l in data_lines],
        CURRENT_COL: [float(l[column_line.index("Current")]) for l in data_lines],
        TEMPERATURE_COL: [float(l[column_line.index("Temperature")]) for l in data_lines],
        CAPACITY_COL: [float(l[column_line.index("Capacity")]) for l in data_lines],
    })
    return df


def generate_and_save_plot(data_df: pd.DataFrame, save_file_path: str, fig_title: str = "") -> None:
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(data_df[TIME_COL], data_df[VOLTAGE_COL], label="Voltage")
    axs[1].plot(data_df[TIME_COL], data_df[CURRENT_COL], label="Current")
    axs[2].plot(data_df[TIME_COL], data_df[TEMPERATURE_COL], label="Temperature")
    axs[3].plot(data_df[TIME_COL], data_df[CAPACITY_COL], label="Capacity")
    for ax in axs:
        ax.legend()
        ax.set_xlabel(TIME_COL)
        ax.set_ylabel("Value")
    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_file_path, format="pdf")
    plt.close(fig)


def parse_all_raw_files(raw_data_directory: str, interim_data_directory: str, interim_plots_directory: str):
    """DVC stage entrypoint: data/raw -> data/interim, for every temperature folder and CSV.

    Every file is still parsed to data/interim, but only the FIRST file found in each
    temperature folder gets a diagnostic plot -> one representative PDF per dataset,
    not one per file.
    """
    temperatures = [d for d in os.listdir(raw_data_directory) if "degC" in d]
    os.makedirs(interim_plots_directory, exist_ok=True)

    for T in temperatures:
        raw_dir = os.path.join(raw_data_directory, T)
        interim_dir = os.path.join(interim_data_directory, T)
        os.makedirs(interim_dir, exist_ok=True)

        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        plotted_this_dataset = False

        for csv_file in csv_files:
            name = csv_file[:-4]
            try:
                df = parse_raw_data(os.path.join(raw_dir, csv_file))
                df.to_csv(os.path.join(interim_dir, f"{name}_parsed.csv"), index=False)
                logging.info(f"Parsed {name} @ {T}")

                if not plotted_this_dataset:
                    generate_and_save_plot(
                        df,
                        os.path.join(interim_plots_directory, f"{T}_parsed_plot.pdf"),
                        fig_title=f"{name} Parsed @ {T} (sample file)",
                    )
                    plotted_this_dataset = True
            except Exception as e:
                logging.error(f"Error parsing {name} @ {T}: {e}")

        if not plotted_this_dataset:
            logging.warning(f"No files parsed for {T}; skipping dataset plot")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--interim_plots_dir", default="reports/figures/interim")
    args = parser.parse_args()
    parse_all_raw_files(args.raw_dir, args.interim_dir, args.interim_plots_dir)
