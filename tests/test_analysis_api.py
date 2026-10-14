"""API contract checks; skipped when optional service dependencies are absent."""
import importlib.util
import json

import pytest

for dependency in ("fastapi", "redis", "rq", "httpx"):
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed", allow_module_level=True)

from fastapi.testclient import TestClient
from server import api


class FakeQueue:
    name = "campusgather"
    count = 0

    def enqueue(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeRedis:
    def lock(self, *_args, **_kwargs):
        return FakeLock()


def test_job_submission_returns_promptly_with_allowlisted_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNS", tmp_path)
    monkeypatch.setattr(api, "queue", FakeQueue())
    monkeypatch.setattr(api, "redis", FakeRedis())
    monkeypatch.setattr(api, "StartedJobRegistry", lambda *args, **kwargs: type("Registry", (), {"count": 0})())
    client = TestClient(api.app)
    response = client.post("/api/analyses/clustering/runs", json={"parameters": {"k": 4}})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    saved = json.loads((tmp_path / run_id / "metadata.json").read_text())
    assert saved["status"] == "queued"
    assert saved["parameters"]["k"] == 4


def test_artifact_route_only_serves_declared_files(tmp_path, monkeypatch):
    run_id = "a" * 32
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "safe.csv").write_text("x\n1\n")
    (run_dir / "metadata.json").write_text(json.dumps({"run_id": run_id, "artifacts": ["safe.csv"]}))
    monkeypatch.setattr(api, "RUNS", tmp_path)
    client = TestClient(api.app)
    assert client.get(f"/api/runs/{run_id}/artifacts/safe.csv").status_code == 200
    assert client.get(f"/api/runs/{run_id}/artifacts/metadata.json").status_code == 404


def test_saved_rssi_section_reads_checked_csv_without_recomputation(tmp_path, monkeypatch):
    import pandas as pd
    tables = tmp_path / "outputs" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([{"rssi_threshold": -90, "theta": 0.7, "gap": 1, "contacts": 1603916, "gatherings": 98194, "episodes": 22530}]).to_csv(tables / "rssi_sensitivity_episodes.csv", index=False)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    response = TestClient(api.app).get("/api/exploration/rssi")
    assert response.status_code == 200
    assert response.json()["status"] == "saved precomputed table"
    assert response.json()["rows"][0]["episodes"] == 22530


def test_saved_cohort_selection_changes_episode_drilldown(tmp_path, monkeypatch):
    import pandas as pd
    tables = tmp_path / "outputs" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame([
        {"members": "101 202", "support": 2, "days_seen": 2, "weeks_seen": 1, "support_first_half": 2, "support_second_half": 0, "night_episode_share": 0.5, "sessions_1h": 2, "episodes_version": "episodes_v1", "minsup": 200, "jaccard_threshold": 0.7, "max_gap_slots": 1, "rssi_threshold": -90},
        {"members": "303 404", "support": 1, "days_seen": 1, "weeks_seen": 1, "support_first_half": 1, "support_second_half": 0, "night_episode_share": 0.0, "sessions_1h": 1, "episodes_version": "episodes_v1", "minsup": 200, "jaccard_threshold": 0.7, "max_gap_slots": 1, "rssi_threshold": -90},
    ]).to_csv(tables / "recurring_cohorts.csv", index=False)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "episodes_v1.parquet").touch()
    episodes = pd.DataFrame([
        {"episode_id": 1, "participant_set": [101, 202, 999], "start_time": 100, "end_time": 100, "start_hour": 0, "night_share": 1.0},
        {"episode_id": 2, "participant_set": [101, 202], "start_time": 100000, "end_time": 100000, "start_hour": 12, "night_share": 0.0},
        {"episode_id": 3, "participant_set": [303, 404], "start_time": 200000, "end_time": 200000, "start_hour": 13, "night_share": 0.0},
    ])
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api.pd, "read_parquet", lambda *_args, **_kwargs: episodes.copy())
    client = TestClient(api.app)
    first = client.get("/api/exploration/cohort-drilldown", params={"cohort": "101 202"}).json()
    second = client.get("/api/exploration/cohort-drilldown", params={"cohort": "303 404"}).json()
    assert first["verified_episode_rows"] == 2
    assert second["verified_episode_rows"] == 1
    assert first["charts"][1]["y"] != second["charts"][1]["y"]
