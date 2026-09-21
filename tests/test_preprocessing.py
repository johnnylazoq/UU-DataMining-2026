"""Tests for Bluetooth preprocessing."""

from pathlib import Path

import pandas as pd

from src.preprocessing import clean_bluetooth_data


def test_clean_bluetooth_data_filters_sentinels_and_creates_slots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "bt_symmetric.csv"
    source.write_text(
        "# timestamp,user_a,user_b,rssi\n"
        "0,1,2,-80\n"
        "300,2,-1,0\n"
        "600,3,-2,-70\n"
        "900,4,5,0\n",
        encoding="utf-8",
    )
    config = {
        "bluetooth_parameters": {"rssi_thresholds": [-85, -90]},
        "time_parameters": {"slot_duration_sec": 300},
        "paths": {"processed_data_dir": str(tmp_path / "processed")},
    }

    contacts = clean_bluetooth_data(source, config)

    assert contacts[["user_a", "user_b"]].values.tolist() == [[1, 2]]
    assert contacts["slot_id"].tolist() == [0]
    assert (tmp_path / "processed" / "contacts.parquet").is_file()
    assert (tmp_path / "processed" / "coverage.parquet").is_file()
