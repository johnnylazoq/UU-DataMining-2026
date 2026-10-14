"""Allowlisted analysis definitions and config-derived parameter validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def config_snapshot() -> dict[str, Any]:
    return load_config(ROOT / "config" / "config.yaml")


def registry() -> dict[str, dict[str, Any]]:
    config = config_snapshot()
    cluster = config["archetype_clustering_parameters"]
    rssi_values = sorted(set([*map(int, config["bluetooth_parameters"].get("rssi_thresholds", [])), int(config["bluetooth_parameters"]["rssi_threshold"])]), reverse=True)
    common = {
        "rssi_threshold": int(config["bluetooth_parameters"]["rssi_threshold"]),
        "min_gathering_size": int(config["clustering_parameters"]["min_gathering_size"]),
        "jaccard_threshold": float(config["episode_parameters"]["jaccard_threshold"]),
        "max_gap_slots": int(config["episode_parameters"]["max_gap_slots"]),
        "communication_lookback_hours": int(config["feature_parameters"]["communication_lookback_hours"]),
    }
    features = {
        "structural": ["log_median_size", "log_span_minutes", "gap_fraction", "membership_expansion_ratio", "median_rssi"],
        "social_communication": ["log_median_size", "log_span_minutes", "gap_fraction", "membership_expansion_ratio", "median_rssi", "internal_fb_density", "log_calls_events_pre", "log_sms_events_pre"],
        "social_pair_normalized_without_gap": ["log_median_size", "log_span_minutes", "membership_expansion_ratio", "median_rssi", "internal_fb_density", "log_calls_events_per_pair_pre", "log_sms_events_per_pair_pre"],
    }
    episode_fields = ["rssi_threshold", "min_gathering_size", "jaccard_threshold", "max_gap_slots"]
    upstream_fields = episode_fields + ["communication_lookback_hours"]

    def definition(identifier: str, title: str, description: str, fields: list[str], extras: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = {key: common[key] for key in fields if key in common}
        defaults.update(extras or {})
        schemas: dict[str, dict[str, Any]] = {
            "rssi_threshold": {"type": "enum", "values": rssi_values, "label": "RSSI threshold (dBm)"},
            "min_gathering_size": {"type": "integer", "minimum": 2, "label": "Minimum gathering size"},
            "jaccard_threshold": {"type": "enum", "values": [0.6, 0.7, 0.8], "label": "Jaccard threshold (θ)"},
            "max_gap_slots": {"type": "enum", "values": [0, 1, 2], "label": "Maximum missing slots (g)"},
            "communication_lookback_hours": {"type": "integer", "minimum": 1, "label": "Communication lookback (hours)"},
            "min_support": {"type": "integer", "minimum": 100, "maximum": 400, "label": "Minimum episode support"},
            "representation": {"type": "enum", "values": list(features), "label": "Feature representation"},
            "k": {"type": "enum", "values": [int(k) for k in cluster["k_values"]], "label": "Cluster count (k)"},
            "seed": {"type": "enum", "values": sorted(set([int(cluster["random_seed"]), *map(int, config["cluster_stability_parameters"]["seeds"])])), "label": "Random seed"},
            "n_init": {"type": "integer", "minimum": 1, "maximum": 100, "label": "K-Means initializations"},
            "silhouette_sample_size": {"type": "integer", "minimum": 100, "maximum": 10000, "label": "Silhouette sample size"},
        }
        constraints: dict[str, Any] = {key: schemas[key] for key in defaults}
        if identifier == "clustering":
            constraints.update({
                "representations": list(features), "k_values": [int(k) for k in cluster["k_values"]],
                "seeds": sorted(set([int(cluster["random_seed"]), *map(int, config["cluster_stability_parameters"]["seeds"])])),
            })
        return {
            "id": identifier, "title": title, "description": description, "defaults": defaults,
            "constraints": constraints,
            "parameter_schema": {key: schemas[key] for key in defaults},
            **({"features": features} if identifier == "clustering" else {}),
            "required_inputs": [], "dependencies": ["prepared Bluetooth contacts", "configured upstream stages"],
            "callable": "server.worker.execute_analysis", "result_schema": f"{identifier}-v1",
        }

    return {
        "clustering": definition("clustering", "Episode clustering", "Fit K-Means after rebuilding the complete upstream lineage for these settings.", upstream_fields + ["representation", "k", "seed", "n_init", "silhouette_sample_size"], {
            "representation": "social_pair_normalized_without_gap", "k": int(cluster["descriptive_k"]), "seed": int(cluster["random_seed"]), "n_init": int(cluster["n_init"]), "silhouette_sample_size": int(cluster["silhouette_sample_size"]),
        }),
        "bluetooth": definition("bluetooth", "Bluetooth & communication", "Re-filter Bluetooth contacts and recompute episode-window communication summaries.", ["rssi_threshold", "communication_lookback_hours"]),
        "gatherings": definition("gatherings", "Gathering extraction", "Re-extract connected components and downstream episodes from prepared contacts.", ["rssi_threshold", "min_gathering_size"]),
        "episodes": definition("episodes", "Episode linking", "Re-link gatherings and rebuild downstream episode features.", ["rssi_threshold", "min_gathering_size", "jaccard_threshold", "max_gap_slots"]),
        "cohorts": definition("cohorts", "Recurring cohorts", "Mine closed participant itemsets from recomputed episodes.", episode_fields + ["min_support"], {"min_support": 200}),
        "associations": definition("associations", "Association rules", "Mine association rules from recomputed episodes.", episode_fields + ["min_support"], {"min_support": 200}),
    }


def validate_parameters(analysis_id: str, params: dict[str, Any]) -> dict[str, Any]:
    definitions = registry()
    if analysis_id not in definitions:
        raise ValueError("Unknown analysis ID")
    definition = definitions[analysis_id]
    defaults = definition["defaults"]
    values = {**defaults, **params}
    if set(values) != set(defaults):
        raise ValueError("Unsupported parameter supplied")
    for name, schema in definition["parameter_schema"].items():
        value = values[name]
        if schema["type"] == "enum":
            if value not in schema["values"]:
                raise ValueError(f"{name} must be one of the configured values")
            continue
        if schema["type"] == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be an integer")
        if schema["type"] == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{name} must be numeric")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{name} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{name} must be at most {schema['maximum']}")
    return values
