"""Tests for Bluetooth preprocessing."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.preprocessing import clean_bluetooth_data
from src.pipeline import validate_input_files


def _write_prepared(path: Path) -> Path:
    """Write a tiny stand-in for bt_symmetric_prepared.parquet."""
    rows = pd.DataFrame(
        {
            "timestamp": [0, 300, 600, 900, 1200],
            "user_a": [1, 2, 3, 4, 6],
            "user_b": [2, -1, -2, 5, 7],
            "rssi": [-80, 0, -70, 0, -95],
        }
    )
    rows["is_empty_scan"] = (rows["user_b"] == -1) & (rows["rssi"] == 0)
    rows["is_external_device"] = rows["user_b"] == -2
    rows["is_valid_contact"] = (rows["user_a"] >= 0) & (rows["user_b"] >= 0)
    # EDA-only column the pipeline must ignore.
    rows["rssi_scaled"] = 0.0
    rows.to_parquet(path)
    return path


def test_clean_bluetooth_data_filters_sentinels_and_creates_slots(
    tmp_path: Path,
) -> None:
    source = _write_prepared(tmp_path / "bt_symmetric_prepared.parquet")
    config = {
        "bluetooth_parameters": {"rssi_threshold": -85},
        "time_parameters": {"slot_duration_sec": 300},
        "paths": {"processed_data_dir": str(tmp_path / "processed")},
    }

    contacts = clean_bluetooth_data(source, config)

    # Kept: 1-2 at -80. Dropped: sentinels, 4-5 at rssi 0, 6-7 below threshold.
    assert contacts[["user_a", "user_b"]].values.tolist() == [[1, 2]]
    assert contacts["slot_id"].tolist() == [0]
    assert list(contacts.columns) == ["timestamp", "user_a", "user_b", "rssi", "slot_id"]
    coverage = pd.read_parquet(tmp_path / "processed" / "coverage.parquet")
    assert coverage["user_b"].tolist() == [-1, -2]
    assert list(coverage.columns) == ["timestamp", "user_a", "user_b", "rssi"]


def test_clean_bluetooth_data_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "bt_symmetric_prepared.parquet"
    pd.DataFrame({"timestamp": [0], "user_a": [1], "user_b": [2], "rssi": [-80]}).to_parquet(source)
    config = {
        "bluetooth_parameters": {"rssi_threshold": -85},
        "time_parameters": {"slot_duration_sec": 300},
        "paths": {"processed_data_dir": str(tmp_path)},
    }

    with pytest.raises(ValueError, match="is_valid_contact"):
        clean_bluetooth_data(source, config)


def test_validate_input_files_points_to_notebook_when_prepared_file_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="eda_bt_symmetric.ipynb"):
        validate_input_files(tmp_path, tmp_path, skip_secondary=True)
