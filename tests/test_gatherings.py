"""Tests for gathering extraction."""

from pathlib import Path

import pandas as pd

from src.gatherings import extract_gatherings


def test_extract_gatherings_finds_connected_component(tmp_path: Path) -> None:
    contacts = pd.DataFrame(
        [
            {"slot_id": 0, "user_a": 1, "user_b": 2, "timestamp": 0, "rssi": -80},
            {"slot_id": 0, "user_a": 2, "user_b": 3, "timestamp": 1, "rssi": -81},
            {"slot_id": 1, "user_a": 4, "user_b": 5, "timestamp": 2, "rssi": -80},
        ]
    )
    config = {
        "clustering_parameters": {"min_gathering_size": 3},
        "paths": {"processed_data_dir": str(tmp_path)},
    }

    gatherings = extract_gatherings(contacts, config)

    assert len(gatherings) == 1
    assert gatherings.iloc[0]["participant_set"] == [1, 2, 3]
    assert (tmp_path / "gatherings_v1.parquet").is_file()
