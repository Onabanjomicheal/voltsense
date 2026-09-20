# Model Card: Battery SoC Estimator (LSTM)

## Overview
Estimates State of Charge (SoC) for an LG 18650HG2 lithium-ion cell from a
20-second window of voltage, current, temperature, and instantaneous power
readings, using a trained LSTM.

## Intended use
- **Primary use**: portfolio/demonstration of an MLOps pipeline for battery
  management system (BMS) SoC estimation.
- **Not intended for**: real BMS deployment in a vehicle or device without
  further validation -- see Limitations below.

## Training data
- **Source**: Kollmeyer, Vidal, Naguib, Skells (2020), "LG 18650HG2 Li-ion
  Battery Data", Mendeley Data, V3, doi: 10.17632/cp3473x7xv.3
- **Cell**: single LG 18650HG2 cell (fresh, no aging/degradation)
- **Conditions**: 6 temperatures (-20C, -10C, 0C, 10C, 25C, 40C)
- **Drive cycles**: Mixed, UDDS, HWFET, LA92, US06 (real, industry-standard EV
  test profiles) -- constant charge/discharge/calibration files excluded from
  training
- **Ground truth**: SoC is not directly measured -- it's derived via pseudo-OCV
  (voltage->SoC curve from a slow C20 reference test) for the initial value,
  then Coulomb counting for everything after. This is standard practice, but
  means the label itself is an estimate, not a perfect measurement.

## Features
`Voltage [V]`, `Current [A]`, `Temperature [degC]`, `Power [W]` (= V x I).

**Deliberately excluded**: `CC_Capacity [Ah]` (cumulative Coulomb-counted
capacity). This was in the original reference implementation's feature set,
but it's a cumulative sum computed from the start of whatever data is given --
meaningful across a full training file, but meaningless in short serving
windows, since every API request starts fresh with no true history. Including
it caused serving predictions to barely respond to voltage at all (verified:
predictions stayed at ~99% SoC even when fed clearly low-voltage, near-empty
battery readings). See `reports/feature_importance.json` for the measured
impact of each remaining feature.

## Architecture
LSTM, hyperparameters tuned via Optuna (real search -- see
`src/training/train.py`), typical winning configuration around
hidden_size=70-95, num_layers=2-4, sequence_length=20.

## Metrics (real, from actual training runs on the full dataset)
| Metric | Value |
|---|---|
| Test MSE | ~2.3e-5 |
| Test MAE | ~0.29% |

Per-temperature breakdown is saved to `models/manifest.json` after each
training run -- `n10degC` has consistently shown the highest error of the six
temperature conditions, worth investigating further if extending this project.

## Known limitations
- **Single cell, no aging data.** A model trained here does not automatically
  generalize to a different cell chemistry/capacity, or to a cell that has
  been through hundreds of charge cycles (capacity fade changes the
  voltage-SoC relationship).
- **No manufacturing variance.** Real deployed BMS systems handle cell-to-cell
  variation this dataset can't teach.
- **Ground-truth SoC is derived, not measured** -- an inherent ceiling on
  achievable accuracy, not a flaw in the modeling pipeline.
- **Requires 20 seconds of history.** The API will reject requests with fewer
  than `sequence_length` readings -- it cannot produce a prediction from a
  single instantaneous reading.
- **No API authentication** (deliberate, for portfolio demo accessibility) --
  do not use this deployment pattern as-is for a real production system
  handling sensitive data.

## Reproducing these results
```bash
dvc repro          # full pipeline: raw -> interim -> processed -> train -> feature_importance
# or, if you already have data/processed/ populated:
make train
make feature-importance
```
