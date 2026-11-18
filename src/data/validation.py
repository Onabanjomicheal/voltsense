"""
Pandera schema validation for battery sensor readings, applied before data reaches
training. Catches malformed data (missing columns, out-of-physical-range values,
wrong types) early, with a clear error, instead of letting it silently corrupt
training or fail deep inside the pipeline with a confusing error.

Value ranges are set generously around the real LG 18650HG2 cell's physical
limits (datasheet: 2.0-4.2V operating range) with margin for sensor noise.

Two schemas, deliberately different strictness:

- RawDataSchema: for data/interim files, BEFORE SoC estimation has run. SOC may
  be absent entirely, or present with generous margin (-0.01 to 1.05) to allow
  for the Coulomb-counting integration drift/noise that estimate_soc_for_file
  clips at the end but that intermediate values may briefly exceed.

- ProcessedDataSchema: for data/processed files, AFTER estimate_soc_for_file has
  run and clipped every value to [0, 1]. SOC must be present, non-null, and
  strictly within [0, 1] -- no margin. A processed file that fails this either
  has a bug in soc_estimation.py or was never actually processed, and either way
  should fail loudly here rather than train on a bad reference SOC silently.
"""
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

RawDataSchema = DataFrameSchema(
    {
        "Time [s]": Column(float, Check.ge(0), nullable=False),
        "Voltage [V]": Column(
            float,
            Check.in_range(1.5, 4.5),
            nullable=False,
            description="LG HG2 datasheet range is 2.0-4.2V; "
                        "1.5-4.5 allows margin for sensor noise.",
        ),
        "Current [A]": Column(
            float,
            Check.in_range(-30.0, 30.0),
            nullable=False,
            description="LG HG2 max continuous discharge is ~20A; "
                        "30A margin allows for brief pulses.",
        ),
        "Temperature [degC]": Column(float, Check.in_range(-40.0, 80.0), nullable=False),
        "SOC [-]": Column(
            float,
            Check.in_range(-0.01, 1.05),
            nullable=True,
            required=False,
            description="Raw/interim data may not have SOC yet (added by the "
                        "estimate_soc pipeline stage). When present, allows margin "
                        "up to 1.05 (105%) for Coulomb-counting integration drift.",
        ),
    },
    strict=False,  # allow extra columns (e.g. __file__ tag added during loading)
)

ProcessedDataSchema = DataFrameSchema(
    {
        "Time [s]": Column(float, Check.ge(0), nullable=False),
        "Voltage [V]": Column(
            float,
            Check.in_range(1.5, 4.5),
            nullable=False,
            description="LG HG2 datasheet range is 2.0-4.2V; "
                        "1.5-4.5 allows margin for sensor noise.",
        ),
        "Current [A]": Column(
            float,
            Check.in_range(-30.0, 30.0),
            nullable=False,
            description="LG HG2 max continuous discharge is ~20A; "
                        "30A margin allows for brief pulses.",
        ),
        "Temperature [degC]": Column(float, Check.in_range(-40.0, 80.0), nullable=False),
        "SOC [-]": Column(
            float,
            Check.in_range(0.0, 1.0),
            nullable=False,
            required=True,
            description="Processed data must have a Coulomb-counted reference SOC, "
                        "clipped to exactly [0, 1] by estimate_soc_for_file. No "
                        "margin here -- a value outside this range means "
                        "soc_estimation.py has a bug, or this file was never "
                        "actually processed.",
        ),
    },
    strict=False,  # allow extra columns (e.g. __file__, Power [W], Cumulative_Capacity_Ah)
)


def validate_raw_file(df, file_path: str = "<unknown>"):
    """Validate a single raw/interim CSV's DataFrame against RawDataSchema.
    Raises ValueError with a clear message if invalid."""
    try:
        RawDataSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(
            f"Raw data validation failed for {file_path}:\n{e.failure_cases.to_string()}"
        ) from e


def validate_processed_file(df, file_path: str = "<unknown>"):
    """Validate a single processed CSV's DataFrame against ProcessedDataSchema.
    Raises ValueError with a clear message if invalid."""
    try:
        ProcessedDataSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(
            f"Processed data validation failed for {file_path}:\n{e.failure_cases.to_string()}"
        ) from e