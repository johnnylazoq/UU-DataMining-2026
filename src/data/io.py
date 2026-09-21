"""Input and output helpers for CampusGather datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_bluetooth_data(filepath: str | Path) -> pd.DataFrame:
    """Load the Bluetooth file using its documented schema."""
    return pd.read_csv(
        filepath,
        skiprows=1,
        names=["timestamp", "user_a", "user_b", "rssi"],
        header=None,
    )


def load_facebook_data(filepath: str | Path) -> pd.DataFrame:
    """Load anonymized Facebook friendship links."""
    return pd.read_csv(filepath, skiprows=1, names=["user_a", "user_b"], header=None)


def load_gender_data(filepath: str | Path) -> pd.DataFrame:
    """Load participant gender metadata."""
    return pd.read_csv(filepath, skiprows=1, names=["user_id", "is_female"], header=None)


def load_telecom_data(
    filepath: str | Path,
    log_type: Literal["calls", "sms"] = "sms",
) -> pd.DataFrame:
    """Load calls or SMS records and flag missed calls when applicable."""
    dataframe = pd.read_csv(filepath)
    if log_type == "calls":
        dataframe["is_missed_call"] = dataframe["duration"] == -1
    return dataframe


def write_processed_data(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Write a processed dataframe as Parquet and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / filename
    dataframe.to_parquet(destination, engine="pyarrow")
    return destination


def export_secondary_datasets(
    config: dict,
    raw_data_path: str | Path,
) -> dict[str, Path]:
    """Load and persist the non-Bluetooth datasets."""
    raw_dir = Path(raw_data_path)
    output_dir = config["paths"]["processed_data_dir"]
    datasets = {
        "fb_friends": (
            load_facebook_data(raw_dir / "fb_friends.csv"),
            "fb_friends.parquet",
        ),
        "genders": (
            load_gender_data(raw_dir / "genders.csv"),
            "genders.parquet",
        ),
        "calls": (
            load_telecom_data(raw_dir / "calls.csv", log_type="calls"),
            "calls.parquet",
        ),
        "sms": (
            load_telecom_data(raw_dir / "sms.csv"),
            "sms.parquet",
        ),
    }
    return {
        name: write_processed_data(dataframe, output_dir, filename)
        for name, (dataframe, filename) in datasets.items()
    }
