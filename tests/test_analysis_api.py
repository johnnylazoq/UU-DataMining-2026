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
