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
    # Coerce every column to numeric and drop rows whose core fields fail to parse.
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["timestamp", "user_a", "user_b", "rssi"]
    )

    # Read the active threshold explicitly. Previously this was
    # min(rssi_thresholds), which silently pinned every run to -90 and meant the
    # -80 / -85 entries in the config were never used.
    threshold = config["bluetooth_parameters"]["rssi_threshold"]

    # rssi < 0 is a validity guard, not an analytical choice: per
    # bt_symmetric.README an RSSI of 0 only ever accompanies the user_b = -1
    # empty-scan sentinel, so among real participant pairs this drops just 4
    # physically impossible positive readings out of 5,474,289 rows.
    contacts = dataframe[
        (dataframe["user_a"] >= 0)
        & (dataframe["user_b"] >= 0)
        & (dataframe["rssi"] < 0)
        & (dataframe["rssi"] >= threshold)
    ].copy()
    # Sentinel rows (-1 empty scan, -2 non-study device) kept separately as evidence a phone was scanning.
    coverage = dataframe[dataframe["user_b"] < 0].copy()

    # Bin each contact into a fixed-width time slot (the 5-minute scan interval).
    slot_duration = config["time_parameters"]["slot_duration_sec"]
    contacts["slot_id"] = (contacts["timestamp"] // slot_duration).astype("int64")

    output_dir = config["paths"]["processed_data_dir"]
    write_processed_data(contacts, output_dir, "contacts.parquet")
    write_processed_data(coverage, output_dir, "coverage.parquet")
    return contacts
