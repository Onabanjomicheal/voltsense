"""Tests for src/data/validation.py."""
import pandas as pd
import pytest

from src.data.validation import validate_processed_file


def test_valid_data_passes():
    df = pd.DataFrame({
        "Time [s]": [0.0, 1.0, 2.0],
        "Voltage [V]": [4.0, 3.9, 3.8],
        "Current [A]": [-0.5, -0.5, -0.5],
        "Temperature [degC]": [25.0, 25.1, 25.0],
        "SOC [-]": [1.0, 0.99, 0.98],
    })
    validate_processed_file(df)  # should not raise


def test_out_of_range_voltage_rejected():
    df = pd.DataFrame({
        "Time [s]": [0.0],
        "Voltage [V]": [10.0],  # way outside 1.5-4.5V physical range
        "Current [A]": [-0.5],
        "Temperature [degC]": [25.0],
        "SOC [-]": [1.0],
    })
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_processed_file(df)


def test_missing_column_rejected():
    df = pd.DataFrame({
        "Time [s]": [0.0],
        "Voltage [V]": [4.0],
        # Current [A] missing
        "Temperature [degC]": [25.0],
    })
    with pytest.raises(ValueError):
        validate_processed_file(df)
