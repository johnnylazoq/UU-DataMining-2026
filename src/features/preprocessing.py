"""Cleaning and normalization for raw CampusGather data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.io import load_bluetooth_data, write_processed_data


def clean_bluetooth_data(
    filepath: str | Path,
    config: dict,
) -> pd.DataFrame:
    """Remove Bluetooth sentinels, apply the baseline RSSI filter, and slot time."""
    dataframe = load_bluetooth_data(filepath)
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["timestamp", "user_a", "user_b", "rssi"]
    )

    threshold = min(config["bluetooth_parameters"]["rssi_thresholds"])
    contacts = dataframe[
        (dataframe["user_a"] >= 0)
        & (dataframe["user_b"] >= 0)
        & (dataframe["rssi"] < 0)
        & (dataframe["rssi"] >= threshold)
    ].copy()
    coverage = dataframe[dataframe["user_b"] < 0].copy()

    slot_duration = config["time_parameters"]["slot_duration_sec"]
    contacts["slot_id"] = (contacts["timestamp"] // slot_duration).astype("int64")

    output_dir = config["paths"]["processed_data_dir"]
    write_processed_data(contacts, output_dir, "contacts.parquet")
    write_processed_data(coverage, output_dir, "coverage.parquet")
    return contacts
