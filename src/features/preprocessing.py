"""Turn the EDA-prepared Bluetooth rows into slotted contacts and coverage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.io import load_prepared_bluetooth_data, write_processed_data


def clean_bluetooth_data(
    filepath: str | Path,
    config: dict,
) -> pd.DataFrame:
    """Apply the baseline RSSI filter to prepared rows and slot time.

    `filepath` is `bt_symmetric_prepared.parquet`, written by
    `notebooks/eda_bt_symmetric.ipynb`. That notebook owns numeric parsing and
    the sentinel categories (`is_valid_contact`, `is_empty_scan`,
    `is_external_device`), so they are read here instead of re-derived.
    """
    dataframe = load_prepared_bluetooth_data(filepath)

    # Read the active threshold explicitly. Previously this was
    # min(rssi_thresholds), which silently pinned every run to -90 and meant the
    # -80 / -85 entries in the config were never used.
    threshold = config["bluetooth_parameters"]["rssi_threshold"]

    # rssi < 0 is a validity guard, not an analytical choice: per
    # bt_symmetric.README an RSSI of 0 only ever accompanies the user_b = -1
    # empty-scan sentinel, so among real participant pairs this drops just 4
    # physically impossible positive readings out of 5,474,289 rows.
    contacts = dataframe[
        dataframe["is_valid_contact"]
        & (dataframe["rssi"] < 0)
        & (dataframe["rssi"] >= threshold)
    ].copy()
    # Sentinel rows (-1 empty scan, -2 non-study device) kept separately as evidence a phone was scanning.
    coverage = dataframe[
        dataframe["is_empty_scan"] | dataframe["is_external_device"]
    ].copy()

    # Keep the output schemas unchanged; the flags are input-side bookkeeping.
    flag_columns = ["is_empty_scan", "is_external_device", "is_valid_contact"]
    contacts = contacts.drop(columns=flag_columns)
    coverage = coverage.drop(columns=flag_columns)

    # Bin each contact into a fixed-width time slot (the 5-minute scan interval).
    slot_duration = config["time_parameters"]["slot_duration_sec"]
    contacts["slot_id"] = (contacts["timestamp"] // slot_duration).astype("int64")

    output_dir = config["paths"]["processed_data_dir"]
    write_processed_data(contacts, output_dir, "contacts.parquet")
    write_processed_data(coverage, output_dir, "coverage.parquet")
    return contacts
