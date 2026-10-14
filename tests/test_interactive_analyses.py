"""Acceptance checks: page parameters cross into Python and change its computation."""
import importlib.util
import json

import pandas as pd
import pytest

if not all(importlib.util.find_spec(name) for name in ("networkx", "pyarrow", "sklearn")):
    pytest.skip("interactive pipeline dependencies are installed in the app image", allow_module_level=True)

from server import analysis_compute
from server.analysis_registry import config_snapshot, registry, validate_parameters


def _pipeline_inputs(tmp_path, monkeypatch):
    config = config_snapshot()
    config["paths"]["raw_data_dir"] = "raw"
    config["bluetooth_parameters"]["rssi_threshold"] = -90
    config["clustering_parameters"]["min_gathering_size"] = 3
    config["episode_parameters"]["jaccard_threshold"] = 0.7
    config["episode_parameters"]["max_gap_slots"] = 1
    config["feature_parameters"]["communication_lookback_hours"] = 24
    prepared = pd.DataFrame([
        # Slots 0 and 1 have Jaccard 0.5; slot 2 repeats slot 1.
        {"timestamp": 1, "user_a": 1, "user_b": 2, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 2, "user_a": 2, "user_b": 3, "rssi": -88, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 301, "user_a": 1, "user_b": 2, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 302, "user_a": 2, "user_b": 4, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 601, "user_a": 1, "user_b": 2, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 602, "user_a": 2, "user_b": 4, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        # A group beginning after 1h but before 2h verifies lookback eligibility.
        {"timestamp": 3601, "user_a": 5, "user_b": 6, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 3602, "user_a": 6, "user_b": 7, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 7201, "user_a": 1, "user_b": 2, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
        {"timestamp": 7202, "user_a": 2, "user_b": 3, "rssi": -82, "is_empty_scan": False, "is_external_device": False, "is_valid_contact": True},
    ])
    monkeypatch.setattr(analysis_compute, "ROOT", tmp_path)
    monkeypatch.setattr(analysis_compute, "config_snapshot", lambda: json.loads(json.dumps(config)))
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: prepared.copy())
    import src.features.gatherings as gathering_module
    monkeypatch.setattr(gathering_module, "write_processed_data", lambda *_args, **_kwargs: None)
    import src.data.io as io
    monkeypatch.setattr(io, "load_facebook_data", lambda *_args, **_kwargs: pd.DataFrame({"user_a": [1, 2, 3, 5], "user_b": [2, 3, 4, 6]}))
    def telecom(_path, log_type="sms"):
        if log_type == "calls":
            return pd.DataFrame({"timestamp": [2000], "caller": [1], "callee": [2], "duration": [30]})
        return pd.DataFrame({"timestamp": [], "sender": [], "recipient": []})
    monkeypatch.setattr(io, "load_telecom_data", telecom)
    return config


def test_bluetooth_page_rssi_control_changes_filtered_contacts(tmp_path, monkeypatch):
    _pipeline_inputs(tmp_path, monkeypatch)
    inclusive = analysis_compute.build_lineage({"rssi_threshold": -90}, tmp_path / "run-a")
    strict = analysis_compute.build_lineage({"rssi_threshold": -85}, tmp_path / "run-b")
    assert len(inclusive["contacts"]) > len(strict["contacts"])


def test_bluetooth_page_lookback_control_changes_communication_aggregate(tmp_path, monkeypatch):
    _pipeline_inputs(tmp_path, monkeypatch)
    one = analysis_compute.compute_analysis("bluetooth", {"rssi_threshold": -90, "communication_lookback_hours": 1}, tmp_path / "one")
    two = analysis_compute.compute_analysis("bluetooth", {"rssi_threshold": -90, "communication_lookback_hours": 2}, tmp_path / "two")
    assert one["metrics"]["calls_in_episode_windows"] < two["metrics"]["calls_in_episode_windows"]


def test_gathering_page_minimum_size_control_changes_extraction(tmp_path, monkeypatch):
    _pipeline_inputs(tmp_path, monkeypatch)
    three = analysis_compute.build_lineage({"rssi_threshold": -90, "min_gathering_size": 3}, tmp_path / "run-a")
    four = analysis_compute.build_lineage({"rssi_threshold": -90, "min_gathering_size": 4}, tmp_path / "run-b")
    assert len(three["gatherings"]) > len(four["gatherings"])


def test_episode_page_jaccard_control_changes_linking(tmp_path, monkeypatch):
    _pipeline_inputs(tmp_path, monkeypatch)
    permissive = analysis_compute.build_lineage({"jaccard_threshold": 0.4, "max_gap_slots": 1}, tmp_path / "run-a")
    strict = analysis_compute.build_lineage({"jaccard_threshold": 0.7, "max_gap_slots": 1}, tmp_path / "run-b")
    assert len(permissive["episodes"]) < len(strict["episodes"])


def test_episode_page_gap_control_changes_linking(tmp_path, monkeypatch):
    config = _pipeline_inputs(tmp_path, monkeypatch)
    # Restrict the fixture to otherwise identical gatherings in slots 0 and 2.
    from src.features.episodes import build_episodes
    gatherings = pd.DataFrame({"slot_id": [0, 2], "gathering_id": [0, 1], "participant_set": [[1, 2], [1, 2]], "start_time": [1, 601], "end_time": [1, 601], "median_rssi": [-80, -80]})
    config["episode_parameters"]["jaccard_threshold"] = 0.7
    config["episode_parameters"]["max_gap_slots"] = 0
    no_gap = build_episodes(gatherings, config)
    config["episode_parameters"]["max_gap_slots"] = 1
    one_gap = build_episodes(gatherings, config)
    assert len(one_gap) < len(no_gap)


def test_clustering_page_lookback_control_changes_eligible_features(tmp_path, monkeypatch):
    _pipeline_inputs(tmp_path, monkeypatch)
    one_hour = analysis_compute.build_lineage({"communication_lookback_hours": 1}, tmp_path / "run-a")
    two_hours = analysis_compute.build_lineage({"communication_lookback_hours": 2}, tmp_path / "run-b")
    assert one_hour["features"]["has_full_communication_window"].sum() > two_hours["features"]["has_full_communication_window"].sum()


def test_clustering_page_k_control_changes_number_of_fitted_clusters(tmp_path, monkeypatch):
    config = config_snapshot()
    count = 30
    features = pd.DataFrame({
        "episode_id": range(count), "has_full_communication_window": [True] * count,
        "log_median_size": [float(i // 10) + (i % 3) * 0.01 for i in range(count)],
        "log_span_minutes": [float(i // 10) + (i % 5) * 0.02 for i in range(count)],
        "gap_fraction": [float(i % 4) / 10 for i in range(count)],
        "membership_expansion_ratio": [1 + float(i % 7) / 10 for i in range(count)],
        "median_rssi": [-90 + float(i % 11) for i in range(count)],
        "median_size": [3 + float(i % 8) for i in range(count)],
        "span_minutes": [5 + float(i % 12) for i in range(count)],
        "internal_fb_density": [float(i % 10) / 10 for i in range(count)],
        "calls_events_pre": [i % 5 for i in range(count)], "sms_events_pre": [i % 4 for i in range(count)],
        "n_unique_participants": [3 + i % 5 for i in range(count)],
        "calls_events_per_pair_pre": [float(i % 5) / 4 for i in range(count)],
        "sms_events_per_pair_pre": [float(i % 4) / 3 for i in range(count)],
    })
    lineage = {"config": config, "contacts": pd.DataFrame(), "gatherings": pd.DataFrame(), "episodes": pd.DataFrame(), "features": features, "provenance": {}}
    monkeypatch.setattr(analysis_compute, "build_lineage", lambda *_args, **_kwargs: lineage)
    common = {"representation": "structural", "seed": 42, "n_init": 3, "silhouette_sample_size": 20}
    two = analysis_compute.compute_analysis("clustering", {**common, "k": 2}, tmp_path / "two")
    three = analysis_compute.compute_analysis("clustering", {**common, "k": 3}, tmp_path / "three")
    assert len(two["metrics"]) > 0
    assert len(pd.read_csv(tmp_path / "two" / "profiles.csv")) == 2
    assert len(pd.read_csv(tmp_path / "three" / "profiles.csv")) == 3


def _episode_fixture():
    records = []
    transactions = [[1, 2, 3]] * 250 + [[1, 2]] * 100 + [[6, 7]] * 200
    for index, members in enumerate(transactions):
        records.append({"episode_id": index, "participant_set": members, "start_time": index * 3600, "end_time": index * 3600 + 300, "start_hour": index % 24, "night_share": 0.0, "n_slots": 1, "n_gap_slots": 0})
    return pd.DataFrame(records)


@pytest.mark.parametrize("analysis_id", ["cohorts", "associations"])
def test_itemset_pages_minimum_support_changes_computed_result(tmp_path, monkeypatch, analysis_id):
    config = config_snapshot()
    monkeypatch.setattr(analysis_compute, "build_lineage", lambda *_args, **_kwargs: {"config": config, "contacts": pd.DataFrame(), "gatherings": pd.DataFrame(), "episodes": _episode_fixture(), "features": pd.DataFrame(), "provenance": {}})
    low = analysis_compute.compute_analysis(analysis_id, {"min_support": 200}, tmp_path / "low")
    high = analysis_compute.compute_analysis(analysis_id, {"min_support": 201}, tmp_path / "high")
    assert low["metrics"]["frequent_itemsets"] > high["metrics"]["frequent_itemsets"]
    assert low["metrics"]["closed_itemsets" if analysis_id == "cohorts" else "rules"] > high["metrics"]["closed_itemsets" if analysis_id == "cohorts" else "rules"]


def test_registry_offers_no_unimplemented_anomaly_detector():
    assert "anomalies" not in registry()
    with pytest.raises(ValueError, match="Unknown analysis"):
        validate_parameters("anomalies", {})
