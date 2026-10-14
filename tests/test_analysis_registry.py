"""Tests for app-run parameter contracts and run isolation."""
import pytest

from server.analysis_registry import registry, validate_parameters
from server import worker


def test_registry_defaults_are_loaded_from_project_config():
    definition = registry()["clustering"]
    assert definition["defaults"]["k"] == 5
    assert definition["defaults"]["seed"] == 42
    assert definition["defaults"]["n_init"] == 20


def test_each_implemented_report_has_validated_backend_defaults():
    definitions = registry()
    for analysis_id in ("bluetooth", "gatherings", "episodes", "clustering", "cohorts", "associations"):
        assert validate_parameters(analysis_id, {}) == definitions[analysis_id]["defaults"]
    assert definitions["cohorts"]["defaults"]["min_support"] == 200
    assert definitions["associations"]["defaults"]["min_support"] == 200
    assert validate_parameters("cohorts", {"min_support": 201})["min_support"] == 201
    with pytest.raises(ValueError):
        validate_parameters("associations", {"min_support": 199})
    assert "anomalies" not in definitions


def test_parameter_validation_rejects_unknown_and_out_of_range_values():
    defaults = registry()["clustering"]["defaults"]
    assert validate_parameters("clustering", {}) == defaults
    with pytest.raises(ValueError):
        validate_parameters("clustering", {"k": 99})
    with pytest.raises(ValueError):
        validate_parameters("clustering", {"shell": "echo nope"})
    with pytest.raises(ValueError):
        validate_parameters("not-registered", {})


def test_failed_jobs_keep_separate_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "RUNS", tmp_path)
    first = worker.execute_analysis("a" * 32, "not-registered", {})
    second = worker.execute_analysis("b" * 32, "not-registered", {})
    assert first["status"] == second["status"] == "failed"
    assert (tmp_path / ("a" * 32) / "metadata.json").is_file()
    assert (tmp_path / ("b" * 32) / "metadata.json").is_file()
    assert first["run_id"] != second["run_id"]


def test_cache_key_changes_when_parameters_or_input_fingerprint_changes():
    original = worker.cache_key("clustering", "v1", "input-a", {"k": 5})
    assert original != worker.cache_key("clustering", "v1", "input-a", {"k": 4})
    assert original != worker.cache_key("clustering", "v1", "input-b", {"k": 5})
    assert original != worker.cache_key("clustering", "v2", "input-a", {"k": 5})


def test_lineage_cache_identity_omits_fit_parameters_but_tracks_upstream_settings():
    upstream = {"rssi_threshold": -90, "min_gathering_size": 3, "jaccard_threshold": 0.7, "max_gap_slots": 1, "communication_lookback_hours": 24}
    first = worker.cache_key("lineage", "lineage-v1", {"prepared": "abc"}, upstream)
    same_lineage = worker.cache_key("lineage", "lineage-v1", {"prepared": "abc"}, dict(upstream))
    changed = worker.cache_key("lineage", "lineage-v1", {"prepared": "abc"}, {**upstream, "communication_lookback_hours": 48})
    assert first == same_lineage
    assert first != changed


def test_worker_executes_and_writes_run_scoped_artifacts(tmp_path, monkeypatch):
    import json
    from server import analysis_compute, analysis_registry
    project_config = analysis_registry.config_snapshot()
    root = tmp_path / "project"
    raw = root / "data" / "raw" / "7267433"
    prepared = root / "data" / "processed" / "bt_symmetric_prepared.parquet"
    raw.mkdir(parents=True)
    prepared.parent.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "config" / "config.yaml").write_text("test: true")
    prepared.write_bytes(b"prepared")
    for name in ("fb_friends.csv", "calls.csv", "sms.csv"):
        (raw / name).write_text("input")
    run_root = tmp_path / "runs"
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "RUNS", run_root)
    monkeypatch.setattr(analysis_registry, "ROOT", root)
    project_config["paths"]["raw_data_dir"] = "data/raw/7267433"
    project_config["paths"]["processed_data_dir"] = "data/processed"
    monkeypatch.setattr(analysis_registry, "config_snapshot", lambda: project_config)
    def compute(analysis_id, params, run_dir, **_kwargs):
        (run_dir / "summary.json").write_text(json.dumps({"ok": True}))
        (run_dir / "result.csv").write_text("value\n1\n")
        return {"metrics": {"ok": 1}, "artifacts": ["summary.json", "result.csv"], "provenance": {}, "charts": []}
    monkeypatch.setattr(analysis_compute, "compute_analysis", compute)
    params = {"rssi_threshold": -90}
    result = worker.execute_analysis("c" * 32, "bluetooth", params, fresh=True)
    assert result["status"] == "succeeded"
    assert {"result.csv", "summary.json"} == set(result["artifacts"])
    assert (run_root / ("c" * 32) / "result.csv").is_file()
    assert result["parameters"] == params
