"""
Per-file data quality report for battery CSVs (data/interim or data/processed).

Purpose: decide things with evidence before changing the pipeline. In particular
it reports the ACTUAL sampling interval (dt) per file and per file type, so we
know whether calibration files, charge/discharge files and drive cycles share
one sampling rate before deciding to keep or remove any resampling step.

Usage:
    python -m src.data_processing.quality_checks --data_dir data/interim
    python -m src.data_processing.quality_checks --data_dir data/processed \
        --output reports/data_quality_processed.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TIME_COL_S = "Time [s]"
CURRENT_COL = "Current [A]"
SOC_COL = "SOC [-]"
REQUIRED_COLS = ["Time [s]", "Voltage [V]", "Current [A]", "Temperature [degC]"]
REST_EPS = 1e-8  # |current| below this counts as rest


def file_category(filename: str) -> str:
    """Group files by test type using the same naming the dataset uses."""
    n = filename.upper()
    if "C20" in n:
        return "C20"
    if "HPPC" in n:
        return "HPPC"
    if "CAP" in n:
        return "Cap"
    if "CHARGE" in n:
        return "Charge"
    if "DIS" in n:
        return "Discharge"
    for key in ("MIXED", "UDDS", "HWFET", "LA92", "US06"):
        if key in n:
            return f"DriveCycle-{key.title() if key == 'MIXED' else key}"
    return "Other"


def analyze_file(path: str) -> dict:
    df = pd.read_csv(path)
    report = {"rows": int(len(df))}

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        report["error"] = f"missing columns: {missing}"
        return report

    t = pd.to_numeric(df[TIME_COL_S], errors="coerce")
    dt = t.diff().dropna()

    report.update({
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(t.duplicated().sum()),
        "is_sorted": bool(t.is_monotonic_increasing),
        "negative_dt": int((dt < 0).sum()),
        "zero_dt": int((dt == 0).sum()),
        "nan_cells": int(df[REQUIRED_COLS].isna().sum().sum()),
        "dt_median": float(dt.median()) if len(dt) else None,
        "dt_p05": float(dt.quantile(0.05)) if len(dt) else None,
        "dt_p95": float(dt.quantile(0.95)) if len(dt) else None,
        "dt_max": float(dt.max()) if len(dt) else None,
        "duration_s": float(t.max() - t.min()) if len(t) else None,
    })

    if SOC_COL in df.columns:
        soc = pd.to_numeric(df[SOC_COL], errors="coerce")
        cur = pd.to_numeric(df[CURRENT_COL], errors="coerce")
        rest = cur.abs() <= REST_EPS
        rest_pair = rest & rest.shift(fill_value=False)
        rest_changes = soc.diff()[rest_pair].abs()
        report.update({
            "soc_min": float(soc.min()) if soc.notna().any() else None,
            "soc_max": float(soc.max()) if soc.notna().any() else None,
            "soc_nan": int(soc.isna().sum()),
            "soc_out_of_range": int(((soc < 0) | (soc > 1)).sum()),
            "rest_soc_max_change": float(rest_changes.max()) if len(rest_changes) else 0.0,
        })

    return report


def summarize(files: dict) -> tuple:
    """Aggregate per-file reports by test type and by temperature folder."""
    by_category, by_temperature = {}, {}
    for rel, rep in files.items():
        temp = rel.split(os.sep)[0]
        cat = file_category(os.path.basename(rel))
        by_temperature.setdefault(temp, {"files": 0, "rows": 0})
        by_temperature[temp]["files"] += 1
        by_temperature[temp]["rows"] += rep.get("rows", 0)

        c = by_category.setdefault(cat, {"files": 0, "dt_medians": []})
        c["files"] += 1
        if rep.get("dt_median") is not None:
            c["dt_medians"].append(rep["dt_median"])

    for cat, c in by_category.items():
        meds = c.pop("dt_medians")
        c["dt_median_of_files"] = float(np.median(meds)) if meds else None
        c["dt_min_across_files"] = float(min(meds)) if meds else None
        c["dt_max_across_files"] = float(max(meds)) if meds else None
    return by_category, by_temperature


def count_issues(files: dict) -> dict:
    keys = ["duplicate_rows", "duplicate_timestamps", "negative_dt", "zero_dt",
            "nan_cells", "soc_nan", "soc_out_of_range"]
    totals = {k: 0 for k in keys}
    unsorted = 0
    for rep in files.values():
        for k in keys:
            totals[k] += rep.get(k, 0) or 0
        if rep.get("is_sorted") is False:
            unsorted += 1
    totals["unsorted_files"] = unsorted
    return totals


def print_summary(by_category: dict, by_temperature: dict, issues: dict, n_files: int):
    print(f"\nFiles analyzed: {n_files}\n")
    print("Files per temperature folder")
    for temp, v in sorted(by_temperature.items()):
        print(f"  {temp:>8}: {v['files']:>4} files, {v['rows']:>10,} rows")

    print("\nSampling interval (dt, seconds) by file type -- median across files")
    print(f"  {'type':<18}{'files':>6}{'median':>12}{'min':>12}{'max':>12}")
    for cat, c in sorted(by_category.items()):
        def fmt(x):
            return f"{x:.3f}" if x is not None else "n/a"
        print(f"  {cat:<18}{c['files']:>6}{fmt(c['dt_median_of_files']):>12}"
              f"{fmt(c['dt_min_across_files']):>12}{fmt(c['dt_max_across_files']):>12}")

    medians = [c["dt_median_of_files"] for c in by_category.values() if c["dt_median_of_files"]]
    if medians and max(medians) / min(medians) > 2:
        print("\nWARNING: sampling interval differs by more than 2x between file types.")
        print("  Mixing them in one model changes what 'sequence_length' means per file.")

    print("\nIssue counts (should all be 0 for clean data)")
    for k, v in issues.items():
        print(f"  {k:<22}{v}")


def main(data_dir: str, output: str, fail_on_issues: bool):
    files = {}
    for temp in sorted(os.listdir(data_dir)):
        temp_dir = os.path.join(data_dir, temp)
        if not os.path.isdir(temp_dir):
            continue
        for name in sorted(os.listdir(temp_dir)):
            if name.endswith(".csv"):
                rel = os.path.join(temp, name)
                files[rel] = analyze_file(os.path.join(temp_dir, name))

    if not files:
        print(f"No CSV files found under {data_dir}")
        sys.exit(1)

    by_category, by_temperature = summarize(files)
    issues = count_issues(files)
    print_summary(by_category, by_temperature, issues, len(files))

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump({"data_dir": data_dir, "issues": issues, "by_category": by_category,
                   "by_temperature": by_temperature, "files": files}, f, indent=2)
    print(f"\nFull per-file report written to {output}")

    if fail_on_issues and any(v for v in issues.values()):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/interim")
    parser.add_argument("--output", default="reports/data_quality.json")
    parser.add_argument("--fail_on_issues", action="store_true",
                        help="exit non-zero if any issue count is non-zero (for CI/DVC)")
    args = parser.parse_args()
    main(args.data_dir, args.output, args.fail_on_issues)
