"""
Pandera schema validation for raw sensor readings, applied before data reaches
training. Catches malformed data (missing columns, out-of-physical-range values,
wrong types) early, with a clear error, instead of letting it silently corrupt
training or fail deep inside the pipeline with a confusing error.

Value ranges are set generously around the real LG 18650HG2 cell's physical
limits (datasheet: 2.0-4.2V operating range) with margin for sensor noise.
"""
import pandera as pa
from pandera import Column, DataFrameSchema, Check

RAW_READING_SCHEMA = DataFrameSchema(
    {
        "Time [s]": Column(float, Check.ge(0), nullable=False),
        "Voltage [V]": Column(float, Check.in_range(1.5, 4.5), nullable=False,
                               description="LG HG2 datasheet range is 2.0-4.2V; "
                                           "1.5-4.5 allows margin for sensor noise."),
        "Current [A]": Column(float, Check.in_range(-30.0, 30.0), nullable=False,
                               description="LG HG2 max continuous discharge is ~20A; "
                                           "30A margin allows for brief pulses."),
        "Temperature [degC]": Column(float, Check.in_range(-40.0, 80.0), nullable=False),
        "SOC [-]": Column(float, Check.in_range(0.0, 1.0), nullable=True),
        # nullable=True on SOC: raw/interim data may not have SOC yet (added by
        # the estimate_soc pipeline stage), only processed data requires it.
    },
    strict=False,  # allow extra columns (e.g. __file__ tag added during loading)
)


def validate_processed_file(df, file_path: str = "<unknown>"):
    """Validate a single processed CSV's DataFrame against the schema.
    Raises pandera.errors.SchemaError with a clear message if invalid."""
    try:
        RAW_READING_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(
            f"Data validation failed for {file_path}:\n{e.failure_cases.to_string()}"
        ) from e
