"""FastAPI job submission and read-only analysis artifact API."""
from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from redis import Redis
from rq import Queue
from rq.registry import StartedJobRegistry

from server.analysis_registry import ROOT, config_snapshot, registry, validate_parameters
from server.worker import RUNS, execute_analysis
from server.comparison import compare_assignments
import pandas as pd

app = FastAPI(title="CampusGather Analysis API", version="1.0")
logger = logging.getLogger(__name__)
redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
queue = Queue("campusgather", connection=redis, default_timeout=1800)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: dict[str, Any] = {}
    fresh: bool = False


def read_metadata(run_id: str) -> dict[str, Any]:
    if not run_id.isalnum() or len(run_id) != 32:
        raise HTTPException(404, "Run not found")
    path = RUNS / run_id / "metadata.json"
    if not path.is_file():
        raise HTTPException(404, "Run not found")
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {"ok": True, "queue": redis.ping()}


@app.get("/api/registry")
def get_registry():
    definitions = registry()
    return {"analyses": list(definitions.values()), "config": {"bluetooth": config_snapshot()["bluetooth_parameters"], "episodes": config_snapshot()["episode_parameters"], "features": config_snapshot()["feature_parameters"]}}


@app.post("/api/analyses/{analysis_id}/runs", status_code=202)
def submit_run(analysis_id: str, submission: Submission):
    try:
        params = validate_parameters(analysis_id, submission.parameters)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    run_id = uuid.uuid4().hex
    run_dir = RUNS / run_id
    now = datetime.now(timezone.utc).isoformat()
    metadata = {"run_id": run_id, "analysis_id": analysis_id, "parameters": params, "status": "queued", "stage": "queued", "logs": ["Job accepted and waiting for a worker."], "artifacts": [], "started_at": now, "ended_at": None, "fresh": submission.fresh}
    try:
        with redis.lock("campusgather:submission-capacity", timeout=5, blocking_timeout=3):
            active = queue.count + StartedJobRegistry(queue.name, connection=redis).count
            if active >= 8:
                raise HTTPException(429, "Analysis queue is full; wait for a worker to finish before submitting another run")
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            queue.enqueue(execute_analysis, run_id, analysis_id, params, submission.fresh, job_id=run_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Background queue submission failed for run %s", run_id)
        if run_dir.is_dir():
            metadata.update(status="failed", stage="failed", error=f"Queue unavailable: {exc}", ended_at=datetime.now(timezone.utc).isoformat())
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        raise HTTPException(503, "Background queue is unavailable") from exc
    return {"run_id": run_id, "status": "queued"}


@app.get("/api/runs")
def list_runs():
    records = []
    for path in sorted(RUNS.glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return {"runs": records[:100]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    metadata = read_metadata(run_id)
    if metadata.get("status") in {"queued", "running"}:
        job = queue.fetch_job(run_id)
        if job is not None and job.get_status(refresh=True) == "failed":
            metadata.update(status="failed", stage="failed", error="The worker terminated the job or the configured time limit was exceeded.", ended_at=datetime.now(timezone.utc).isoformat())
            (RUNS / run_id / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    metadata = read_metadata(run_id)
    if metadata["status"] != "queued":
        raise HTTPException(409, "Only queued jobs can be cancelled")
    job = queue.fetch_job(run_id)
    if job is None:
        raise HTTPException(409, "Queued job is no longer available")
    job.cancel()
    metadata.update(status="cancelled", stage="cancelled", ended_at=datetime.now(timezone.utc).isoformat())
    (RUNS / run_id / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


@app.get("/api/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str):
    metadata = read_metadata(run_id)
    if name not in metadata.get("artifacts", []):
        raise HTTPException(404, "Artifact not found")
    path = (RUNS / run_id / name).resolve()
    if path.parent != (RUNS / run_id).resolve() or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, filename=name)


@app.post("/api/compare")
def compare_runs(payload: dict[str, str]):
    first_id, second_id = payload.get("first_run_id", ""), payload.get("second_run_id", "")
    first_meta, second_meta = read_metadata(first_id), read_metadata(second_id)
    if first_meta.get("analysis_id") != "clustering" or second_meta.get("analysis_id") != "clustering":
        raise HTTPException(422, "Only clustering runs can be compared")
    if first_meta.get("status") != "succeeded" or second_meta.get("status") != "succeeded":
        raise HTTPException(409, "Both runs must have succeeded")
    first_file, second_file = RUNS / first_id / "assignments.csv", RUNS / second_id / "assignments.csv"
    if not first_file.is_file() or not second_file.is_file():
        raise HTTPException(404, "Run assignments are unavailable")
    try:
        first_assignments, second_assignments = pd.read_csv(first_file), pd.read_csv(second_file)
        first_profiles = pd.read_csv(RUNS / first_id / "profiles.csv").to_dict(orient="records")
        second_profiles = pd.read_csv(RUNS / second_id / "profiles.csv").to_dict(orient="records")
        def cluster_shares(assignments):
            counts = assignments["cluster_id"].value_counts(normalize=True).sort_values(ascending=False)
            return [float(value) for value in counts.values]
        return {"first_run_id": first_id, "second_run_id": second_id, "parameters": {"first": first_meta["parameters"], "second": second_meta["parameters"]}, "metrics": {"first": first_meta.get("metrics", {}), "second": second_meta.get("metrics", {})}, "profiles": {"first": first_profiles, "second": second_profiles}, "cluster_shares": {"first": cluster_shares(first_assignments), "second": cluster_shares(second_assignments)}, **compare_assignments(first_assignments, second_assignments)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/baseline")
def get_baseline():
    tables = ROOT / "outputs" / "tables"
    baseline = {}
    for key, filename in {"validity": "cluster_validity_normalized_without_gap.csv", "stability": "cluster_seed_stability_summary.csv", "cohorts": "recurring_cohorts.csv"}.items():
        path = tables / filename
        if path.is_file():
            with path.open(newline="") as stream:
                records = list(csv.DictReader(stream))[:3000]
            if key == "cohorts":
                records = [{column: value for column, value in record.items() if column != "members"} for record in records]
            baseline[key] = records
    model_path = ROOT / "outputs" / "models" / "gathering_clustering_v1_metadata.json"
    baseline["model_metadata"] = json.loads(model_path.read_text()) if model_path.is_file() else None
    coordinates_path = tables / "cluster_visualization_coordinates.csv"
    if coordinates_path.is_file():
        coordinates = pd.read_csv(coordinates_path, usecols=["PC1", "PC2", "cluster_id"])
        baseline["coordinates"] = coordinates.sample(n=min(3000, len(coordinates)), random_state=42).to_dict(orient="records")
        baseline["coordinates_sampled"] = len(coordinates) > 3000
    baseline["label"] = "Precomputed project outputs; not produced by the selected controls."
    return baseline


@app.get("/api/baseline/artifacts/{name}")
def get_baseline_artifact(name: str):
    allowed = {"cluster_validity_normalized_without_gap.csv", "cluster_seed_stability_summary.csv"}
    if name not in allowed:
        raise HTTPException(404, "Baseline artifact not found")
    path = ROOT / "outputs" / "tables" / name
    if not path.is_file():
        raise HTTPException(404, "Baseline artifact not found")
    return FileResponse(path, filename=name)


@app.get("/api/exploration/{section}")
def get_exploration(section: str):
    """Small aggregate views over saved data; never returns participant records."""
    processed = ROOT / "data" / "processed"
    if section == "bluetooth":
        config = config_snapshot()
        path = processed / "contacts.parquet"
        if not path.is_file():
            raise HTTPException(404, "Processed Bluetooth contacts are unavailable")
        frame = pd.read_parquet(path, columns=["rssi", "slot_id", "user_a", "user_b"])
        bins = [-100, -90, -85, -80, -75, -70, -65, -60, 0]
        rssi = pd.cut(frame["rssi"], bins=bins, right=False).value_counts(sort=False)
        daily = frame.assign(relative_day=(frame["slot_id"] * config_snapshot()["time_parameters"]["slot_duration_sec"] // 86400)).groupby("relative_day").size().sort_index()
        participants = pd.concat([frame["user_a"], frame["user_b"]]).nunique()
        return {"metrics": {"contact_records": int(len(frame)), "participants": int(participants), "median_rssi_dbm": float(frame["rssi"].median()) if len(frame) else None}, "configured_defaults": {"rssi_threshold_dbm": config["bluetooth_parameters"]["rssi_threshold"], "slot_duration_seconds": config["time_parameters"]["slot_duration_sec"]}, "charts": [{"title": "RSSI distribution", "x_title": "RSSI band (dBm)", "y_title": "Contact records", "x": [str(x) for x in rssi.index], "y": [int(v) for v in rssi.values]}, {"title": "Contact activity by relative day", "x_title": "Relative day index", "y_title": "Contact records", "x": [int(x) for x in daily.index], "y": [int(v) for v in daily.values]}], "note": "Relative day indices are derived from the dataset origin and are not calendar dates. Pair-normalized event counts are not probabilities."}
    if section == "gatherings":
        config = config_snapshot()
        path = processed / "gatherings_v1.parquet"
        if not path.is_file():
            raise HTTPException(404, "Gathering output is unavailable")
        frame = pd.read_parquet(path, columns=["participant_set", "median_rssi"])
        sizes = frame["participant_set"].map(len).value_counts().sort_index()
        return {"metrics": {"gatherings": int(len(frame)), "mean_size": float(sizes.index.to_series().repeat(sizes.values).mean()) if len(frame) else None, "median_rssi_dbm": float(frame["median_rssi"].median()) if len(frame) else None}, "configured_defaults": {"rssi_threshold_dbm": config["bluetooth_parameters"]["rssi_threshold"], "min_gathering_size": config["clustering_parameters"]["min_gathering_size"], "slot_duration_seconds": config["time_parameters"]["slot_duration_sec"]}, "charts": [{"title": "Gathering size distribution", "x_title": "Participants in connected component", "y_title": "Gathering slots", "x": [int(x) for x in sizes.index], "y": [int(v) for v in sizes.values]}], "note": "Each record is a connected component within one time slot; it does not imply all-pairs contact."}
    if section == "episodes":
        config = config_snapshot()
        path = processed / "episodes_v1.parquet"
        if not path.is_file():
            raise HTTPException(404, "Episode output is unavailable")
        frame = pd.read_parquet(path, columns=["n_slots", "n_gap_slots", "participant_set", "start_hour"])
        durations = frame["n_slots"].value_counts().sort_index().head(60)
        return {"metrics": {"episodes": int(len(frame)), "median_slots": float(frame["n_slots"].median()) if len(frame) else None, "episodes_with_gaps": int((frame["n_gap_slots"] > 0).sum()), "median_union_size": float(frame["participant_set"].map(len).median()) if len(frame) else None}, "configured_defaults": {"jaccard_threshold": config["episode_parameters"]["jaccard_threshold"], "max_gap_slots": config["episode_parameters"]["max_gap_slots"], "slot_duration_seconds": config["time_parameters"]["slot_duration_sec"]}, "charts": [{"title": "Episode span distribution", "x_title": "Covered five-minute slots (gaps included)", "y_title": "Episodes", "x": [int(x) for x in durations.index], "y": [int(v) for v in durations.values]}], "note": "Membership is the union across an episode and does not imply simultaneous attendance. Time of day in the source follows an unverified relative origin."}
    raise HTTPException(404, "No aggregate exploration is registered for this section")
