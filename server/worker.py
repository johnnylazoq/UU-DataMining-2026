"""Isolated, provenance-recording RQ analysis jobs."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "app_runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        result[name] = digest.hexdigest()
    return result


def cache_key(analysis_id: str, analysis_version: str, fingerprints: dict[str, str] | str, parameters: dict[str, Any]) -> str:
    identity = {"analysis": analysis_id, "version": analysis_version, "inputs": fingerprints, "parameters": parameters}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _write(run_dir: Path, metadata: dict[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))


def execute_analysis(run_id: str, analysis_id: str, params: dict[str, Any], fresh: bool = False) -> dict[str, Any]:
    """RQ entry point: validate an allowlisted computation and isolate every output."""
    from server.analysis_registry import ROOT as PROJECT_ROOT, validate_parameters

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"run_id": run_id, "analysis_id": analysis_id, "parameters": params, "status": "running", "stage": "validating", "logs": [], "artifacts": [], "started_at": _now(), "input_fingerprints": {}, "git_revision": None, "software": {"python": platform.python_version()}, "random_seed": params.get("seed")}
    _write(run_dir, metadata)
    try:
        params = validate_parameters(analysis_id, params)
        metadata["parameters"] = params
        metadata["stage"] = "checking_inputs"
        from server.analysis_registry import config_snapshot
        config = config_snapshot()
        raw_dir = Path(config["paths"]["raw_data_dir"])
        if not raw_dir.is_absolute():
            raw_dir = PROJECT_ROOT / raw_dir
        processed_dir = Path(config["paths"]["processed_data_dir"])
        if not processed_dir.is_absolute():
            processed_dir = PROJECT_ROOT / processed_dir
        paths = [PROJECT_ROOT / "config" / "config.yaml", processed_dir / "bt_symmetric_prepared.parquet", raw_dir / "fb_friends.csv", raw_dir / "calls.csv", raw_dir / "sms.csv"]
        missing = []
        for path in paths:
            if not path.is_file():
                try:
                    missing.append(str(path.relative_to(PROJECT_ROOT)))
                except ValueError:
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"Required analysis input(s) unavailable: {', '.join(missing)}")
        fingerprints = _fingerprint(paths)
        metadata["input_fingerprints"] = fingerprints
        try:
            metadata["git_revision"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            metadata["git_revision"] = None
        identity = cache_key(analysis_id, "interactive-v2", fingerprints, params)
        metadata["cache_key"] = identity
        upstream = {
            "rssi_threshold": params.get("rssi_threshold", config["bluetooth_parameters"]["rssi_threshold"]),
            "min_gathering_size": params.get("min_gathering_size", config["clustering_parameters"]["min_gathering_size"]),
            "jaccard_threshold": params.get("jaccard_threshold", config["episode_parameters"]["jaccard_threshold"]),
            "max_gap_slots": params.get("max_gap_slots", config["episode_parameters"]["max_gap_slots"]),
            "communication_lookback_hours": params.get("communication_lookback_hours", config["feature_parameters"]["communication_lookback_hours"]),
        }
        lineage_identity = cache_key("lineage", "lineage-v1", fingerprints, upstream)
        if not fresh:
            for prior_path in sorted(RUNS.glob("*/metadata.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                if prior_path.parent == run_dir:
                    continue
                try:
                    prior = json.loads(prior_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if prior.get("status") == "succeeded" and prior.get("cache_key") == identity:
                    artifacts = prior.get("artifacts", [])
                    for name in artifacts:
                        source = prior_path.parent / name
                        if source.is_file() and source.parent.resolve() == prior_path.parent.resolve():
                            shutil.copy2(source, run_dir / name)
                    if all((run_dir / name).is_file() for name in artifacts):
                        metadata.update({"status": "succeeded", "stage": "complete", "cache_hit": True, "lineage_cache_hit": prior.get("lineage_cache_hit", False), "artifacts": artifacts, "metrics": prior.get("metrics"), "provenance": prior.get("provenance"), "logs": [f"Reused exact matching result from run {prior.get('run_id')}."]})
                        return metadata
        metadata["stage"] = "recomputing_upstream_pipeline"
        metadata["logs"].append("Preparing the contacts → gatherings → episodes → communication-feature lineage for this parameter snapshot.")
        _write(run_dir, metadata)
        from server.analysis_compute import compute_analysis

        result = compute_analysis(analysis_id, params, run_dir, lineage_cache_key=lineage_identity)
        metadata.update({"status": "succeeded", "stage": "complete", "cache_hit": False, "metrics": result["metrics"], "artifacts": result["artifacts"], "provenance": result["provenance"]})
        metadata["logs"].append(f"Completed {analysis_id} using {len(fingerprints)} fingerprinted upstream inputs.")
        metadata["lineage_cache_hit"] = result.get("lineage_cache_hit", False)
    except Exception as exc:
        metadata.update({"status": "failed", "stage": "failed", "error": str(exc)})
        metadata["logs"].append(traceback.format_exc(limit=8))
    finally:
        shutil.rmtree(run_dir / "work", ignore_errors=True)
        metadata["ended_at"] = _now()
        _write(run_dir, metadata)
    return metadata
