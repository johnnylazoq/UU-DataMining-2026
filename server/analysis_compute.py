"""Run-scoped implementations shared by the interactive analysis jobs."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from server.analysis_registry import ROOT, config_snapshot
from src.features.episodes import build_episodes
from src.features.gatherings import extract_gatherings
from src.features.communication import add_pair_normalized_communication, communication_summary, prepare_events


def build_lineage(params: dict[str, Any], work_dir: Path, lineage_cache_key: str | None = None) -> dict[str, Any]:
    """Recompute contacts -> gatherings -> episodes -> features using one config snapshot."""
    from src.data.io import load_facebook_data, load_telecom_data, PREPARED_BLUETOOTH_FILENAME

    config = config_snapshot()
    config["bluetooth_parameters"]["rssi_threshold"] = params.get("rssi_threshold", config["bluetooth_parameters"]["rssi_threshold"])
    config["clustering_parameters"]["min_gathering_size"] = params.get("min_gathering_size", config["clustering_parameters"]["min_gathering_size"])
    config["episode_parameters"]["jaccard_threshold"] = params.get("jaccard_threshold", config["episode_parameters"]["jaccard_threshold"])
    config["episode_parameters"]["max_gap_slots"] = params.get("max_gap_slots", config["episode_parameters"]["max_gap_slots"])
    config["feature_parameters"]["communication_lookback_hours"] = params.get("communication_lookback_hours", config["feature_parameters"]["communication_lookback_hours"])
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(config["paths"]["processed_data_dir"])
    if not prepared_dir.is_absolute():
        prepared_dir = ROOT / prepared_dir
    config["paths"]["processed_data_dir"] = str(work_dir)

    prepared_path = prepared_dir / PREPARED_BLUETOOTH_FILENAME
    cache_dir = ROOT / "outputs" / "app_runs" / ".lineage_cache" / lineage_cache_key if lineage_cache_key else None
    cached_names = ["contacts", "gatherings", "episodes", "features"]
    if cache_dir and (cache_dir / "manifest.json").is_file() and all((cache_dir / f"{name}.parquet").is_file() for name in cached_names):
        try:
            manifest = json.loads((cache_dir / "manifest.json").read_text())
            if manifest.get("cache_key") == lineage_cache_key:
                cached = {name: pd.read_parquet(cache_dir / f"{name}.parquet") for name in cached_names}
                provenance = manifest["provenance"]
                for column, value in provenance.items():
                    cached["features"][column] = value
                return {"config": config, **cached, "provenance": provenance, "lineage_cache_hit": True}
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    prepared = pd.read_parquet(prepared_path, columns=["timestamp", "user_a", "user_b", "rssi", "is_empty_scan", "is_external_device", "is_valid_contact"])
    contacts = prepared.loc[prepared["is_valid_contact"] & (prepared["rssi"] < 0) & (prepared["rssi"] >= config["bluetooth_parameters"]["rssi_threshold"])].copy()
    contacts["slot_id"] = (contacts["timestamp"] // config["time_parameters"]["slot_duration_sec"]).astype("int64")
    contacts = contacts.drop(columns=["is_empty_scan", "is_external_device", "is_valid_contact"])
    gatherings = extract_gatherings(contacts, config)
    episodes = build_episodes(gatherings, config)
    friends_path = ROOT / config["paths"]["raw_data_dir"] / "fb_friends.csv"
    calls_path = ROOT / config["paths"]["raw_data_dir"] / "calls.csv"
    sms_path = ROOT / config["paths"]["raw_data_dir"] / "sms.csv"
    friends = load_facebook_data(friends_path)
    calls = load_telecom_data(calls_path, log_type="calls")
    sms = load_telecom_data(sms_path, log_type="sms")

    features = episodes.copy()
    features["n_unique_participants"] = features["participant_set"].map(lambda values: len(set(values)))
    features["n_observed_slots"] = features["n_slots"] - features["n_gap_slots"]
    slot_seconds = config["time_parameters"]["slot_duration_sec"]
    features["span_minutes"] = features["n_slots"] * slot_seconds / 60
    features["observed_minutes"] = features["n_observed_slots"] * slot_seconds / 60
    features["gap_fraction"] = features["n_gap_slots"] / features["n_slots"]
    features["membership_expansion_ratio"] = features["n_unique_participants"] / features["median_size"]
    features["relative_day_index"] = (features["start_time"] // 86400).astype(int)
    features["log_median_size"] = np.log1p(features["median_size"])
    features["log_span_minutes"] = np.log1p(features["span_minutes"])
    features["input_version"] = "episodes_v1"
    features["feature_version"] = "social_communication_v1"

    valid_friends = friends[["user_a", "user_b"]].dropna().astype(int)
    pairs = {tuple(sorted((a, b))) for a, b in valid_friends.itertuples(index=False, name=None) if a >= 0 and b >= 0 and a != b}
    neighbors: dict[int, set[int]] = {}
    for a, b in pairs:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    edges, density = [], []
    for values in features["participant_set"]:
        members = set(map(int, values))
        possible = len(members) * (len(members) - 1) // 2
        n_edges = sum(len(neighbors.get(member, set()) & members) for member in members) // 2
        edges.append(n_edges)
        density.append(n_edges / possible if possible else 0.0)
    features["internal_fb_edges"] = edges
    features["internal_fb_density"] = density

    lookback_seconds = int(config["feature_parameters"]["communication_lookback_hours"] * 3600)
    durations = pd.to_numeric(calls["duration"], errors="raise")
    completed = prepare_events(calls.loc[durations >= 0], "caller", "callee")
    missed = prepare_events(calls.loc[durations == -1], "caller", "callee")
    sms_events = prepare_events(sms, "sender", "recipient")
    for event_table in [
        communication_summary(completed, features, "caller", "callee", "calls", lookback_seconds),
        communication_summary(missed, features, "caller", "callee", "missed_calls", lookback_seconds),
        communication_summary(sms_events, features, "sender", "recipient", "sms", lookback_seconds),
    ]:
        features = features.merge(event_table, on="episode_id", how="left", validate="one_to_one")
    features["prehistory_available_hours"] = features["start_time"].clip(lower=0, upper=lookback_seconds) / 3600
    features["has_full_communication_window"] = features["start_time"] >= lookback_seconds
    features["communication_lookback_hours"] = config["feature_parameters"]["communication_lookback_hours"]
    for prefix in ["calls", "missed_calls", "sms"]:
        features[f"log_{prefix}_events_pre"] = np.log1p(features[f"{prefix}_events_pre"])
    possible_pairs = features["n_unique_participants"] * (features["n_unique_participants"] - 1) / 2
    if (possible_pairs <= 0).any():
        raise ValueError("An episode has fewer than two distinct participants")
    features["n_possible_member_pairs"] = possible_pairs.astype(int)
    for prefix in ["calls", "sms"]:
        features[f"{prefix}_events_per_pair_pre"] = features[f"{prefix}_events_pre"] / possible_pairs
        features[f"log_{prefix}_events_per_pair_pre"] = np.log1p(features[f"{prefix}_events_per_pair_pre"])

    provenance = {
        "rssi_threshold": config["bluetooth_parameters"]["rssi_threshold"],
        "min_gathering_size": config["clustering_parameters"]["min_gathering_size"],
        "episode_jaccard_threshold": config["episode_parameters"]["jaccard_threshold"],
        "episode_max_gap_slots": config["episode_parameters"]["max_gap_slots"],
        "communication_lookback_hours": config["feature_parameters"]["communication_lookback_hours"],
    }
    for column, value in provenance.items():
        features[column] = value
    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in [("contacts", contacts), ("gatherings", gatherings), ("episodes", episodes), ("features", features)]:
                frame.to_parquet(cache_dir / f"{name}.parquet", index=False)
            (cache_dir / "manifest.json").write_text(json.dumps({"cache_key": lineage_cache_key, "provenance": provenance}, indent=2))
        except (OSError, ImportError):
            pass
    return {"config": config, "contacts": contacts, "gatherings": gatherings, "episodes": episodes, "features": features, "provenance": provenance, "lineage_cache_hit": False}


def compute_analysis(analysis_id: str, params: dict[str, Any], run_dir: Path, lineage_cache_key: str | None = None) -> dict[str, Any]:
    """Compute one registered report and write only aggregate/run-scoped artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    from server.analysis_registry import registry

    lineage = build_lineage(params, run_dir / "work", lineage_cache_key)
    contacts, gatherings, episodes, features = (lineage[key] for key in ("contacts", "gatherings", "episodes", "features"))
    config = lineage["config"]
    metrics: dict[str, Any]
    charts: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}
    if analysis_id == "bluetooth":
        rssi_bins = pd.cut(contacts["rssi"], bins=[-100, -90, -85, -80, -75, -70, -65, -60, 0], right=False).value_counts(sort=False)
        day = (contacts["slot_id"] * config["time_parameters"]["slot_duration_sec"] // 86400).astype(int)
        daily = day.value_counts().sort_index()
        comm_daily = features.assign(relative_day=(features["start_time"] // 86400).astype(int)).groupby("relative_day")[["calls_events_pre", "sms_events_pre"]].sum() if len(features) else pd.DataFrame(columns=["calls_events_pre", "sms_events_pre"])
        metrics = {"contact_records": int(len(contacts)), "participants": int(pd.concat([contacts["user_a"], contacts["user_b"]]).nunique()), "median_rssi_dbm": float(contacts["rssi"].median()) if len(contacts) else None, "episodes": int(len(features)), "calls_in_episode_windows": int(features["calls_events_pre"].sum()), "sms_in_episode_windows": int(features["sms_events_pre"].sum())}
        charts = [{"title": "RSSI distribution", "x_title": "RSSI band (dBm)", "y_title": "Contact records", "x": [str(x) for x in rssi_bins.index], "y": [int(v) for v in rssi_bins.values]}, {"title": "Contact activity by relative day", "x_title": "Relative day index", "y_title": "Contact records", "x": [int(x) for x in daily.index], "y": [int(v) for v in daily.values]}, {"title": "Communication events in episode lookback windows", "x_title": "Episode start relative day", "y_title": "Events counted across episode windows", "x": [int(x) for x in comm_daily.index], "y": [int(v) for v in (comm_daily["calls_events_pre"] + comm_daily["sms_events_pre"]).tolist()]}]
        tables["contacts_by_relative_day"] = daily.rename_axis("relative_day").rename("contact_records").reset_index()
        tables["communication_by_episode_day"] = comm_daily.reset_index()
    elif analysis_id == "gatherings":
        sizes = gatherings["participant_set"].map(len).value_counts().sort_index()
        distribution = sizes.rename_axis("participants").rename("gathering_slots").reset_index()
        metrics = {"gatherings": int(len(gatherings)), "mean_size": float(np.average(sizes.index, weights=sizes.values)) if len(sizes) else None, "median_rssi_dbm": float(gatherings["median_rssi"].median()) if len(gatherings) else None}
        charts = [{"title": "Gathering size distribution", "x_title": "Participants in connected component", "y_title": "Gathering slots", "x": distribution["participants"].astype(int).tolist(), "y": distribution["gathering_slots"].astype(int).tolist()}]
        tables["gathering_size_distribution"] = distribution
    elif analysis_id == "episodes":
        spans = episodes["n_slots"].value_counts().sort_index()
        distribution = spans.rename_axis("covered_slots").rename("episodes").reset_index()
        from src.rssi_sensitivity import summarise
        sensitivity = summarise(params["rssi_threshold"], contacts, gatherings, episodes)
        metrics = {"episodes": int(len(episodes)), "median_slots": float(episodes["n_slots"].median()) if len(episodes) else None, "episodes_with_gaps": int((episodes["n_gap_slots"] > 0).sum()), "median_union_size": float(episodes["participant_set"].map(len).median()) if len(episodes) else None, "contacts": sensitivity["contacts"], "gatherings": sensitivity["gatherings"], "one_slot_share_pct": sensitivity["one_slot_share_pct"], "drift_pct": sensitivity["drift_pct"]}
        if params["rssi_threshold"] == -90 and params["min_gathering_size"] == 3:
            expected = {(0.7, 1): (22530, 54.9, 6.3), (0.6, 1): (18269, 51.2, 17.3), (0.8, 1): (29411, 60.7, 1.0)}.get((params["jaccard_threshold"], params["max_gap_slots"]))
            if expected:
                observed = (metrics["episodes"], metrics["one_slot_share_pct"], metrics["drift_pct"])
                if observed[0] != expected[0] or abs(observed[1] - expected[1]) > 0.15 or abs(observed[2] - expected[2]) > 0.15:
                    raise ValueError(f"Episode check failed for θ={params['jaccard_threshold']}, g={params['max_gap_slots']}: got {observed}, expected {expected}")
        charts = [{"title": "Episode span distribution", "x_title": "Covered five-minute slots (gaps included)", "y_title": "Episodes", "x": distribution["covered_slots"].astype(int).tolist(), "y": distribution["episodes"].astype(int).tolist()}]
        tables["episode_span_distribution"] = distribution
    elif analysis_id == "clustering":
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
        definition = registry()["clustering"]
        eligible = features.loc[features["has_full_communication_window"]].copy().reset_index(drop=True)
        representation = params["representation"]
        columns = definition["features"][representation]
        if representation == "social_pair_normalized_without_gap":
            eligible = add_pair_normalized_communication(eligible)
        matrix = eligible[columns].astype(float)
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Feature values must all be finite")
        matrix = matrix.loc[:, matrix.nunique() > 1]
        if params["k"] >= len(eligible):
            raise ValueError("k must be smaller than eligible episodes")
        scaled = StandardScaler().fit_transform(matrix)
        model = KMeans(n_clusters=params["k"], n_init=params["n_init"], random_state=params["seed"])
        labels = model.fit_predict(scaled)
        sizes = pd.Series(labels).value_counts().sort_index()
        if len(sizes) < 2:
            raise ValueError("K-Means produced a degenerate one-cluster fit")
        sample_size = min(params["silhouette_sample_size"], len(eligible))
        metrics = {"n_episodes": len(eligible), "n_excluded_incomplete_history": len(features) - len(eligible), "silhouette_sampled": float(silhouette_score(scaled, labels, sample_size=sample_size, random_state=params["seed"])), "silhouette_sample_size": sample_size, "davies_bouldin": float(davies_bouldin_score(scaled, labels)), "inertia": float(model.inertia_), "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max())}
        tables["assignments"] = pd.DataFrame({"episode_id": eligible["episode_id"], "cluster_id": labels})
        profile_columns = [c for c in ["median_size", "span_minutes", "gap_fraction", "membership_expansion_ratio", "median_rssi", "internal_fb_density", "calls_events_pre", "sms_events_pre", "calls_events_per_pair_pre", "sms_events_per_pair_pre"] if c in eligible]
        tables["profiles"] = eligible.assign(cluster_id=labels).groupby("cluster_id")[profile_columns].median().reset_index()
        coordinates = PCA(n_components=2).fit_transform(scaled)
        tables["coordinates"] = pd.DataFrame({"episode_id": eligible["episode_id"], "pc1": coordinates[:, 0], "pc2": coordinates[:, 1], "cluster_id": labels})
        charts = [{"title": "Run-specific PCA view", "x_title": "Principal component 1", "y_title": "Principal component 2", "x": tables["coordinates"]["pc1"].tolist(), "y": tables["coordinates"]["pc2"].tolist()}]
    elif analysis_id in {"cohorts", "associations"}:
        from src.features.transactions import build_transactions, to_one_hot
        from src.models.itemsets import apriori, find_closed, find_maximal, levels_to_counts, mlxtend_to_counts
        transactions = build_transactions(episodes)
        min_support = params["min_support"]
        started = perf_counter()
        levels, level_table = apriori(transactions, min_support)
        apriori_seconds = perf_counter() - started
        supports = levels_to_counts(levels)
        closed, maximal = find_closed(levels), find_maximal(levels)
        n_total = len(transactions)
        itemset_rows = [{"members": ", ".join(map(str, itemset)), "size": len(itemset), "support": count, "closed": itemset in closed, "maximal": itemset in maximal} for itemset, count in supports.items()]
        tables["frequent_itemsets"] = pd.DataFrame(itemset_rows, columns=["members", "size", "support", "closed", "maximal"])
        tables["apriori_levels"] = level_table
        if analysis_id == "cohorts":
            from mlxtend.frequent_patterns import apriori as mlx_apriori
            started = perf_counter()
            frequent = mlx_apriori(to_one_hot(transactions), min_support=(min_support - 0.5) / n_total, use_colnames=True, low_memory=True)
            mlx_counts = mlxtend_to_counts(frequent, n_total)
            mlx_seconds = perf_counter() - started
            if mlx_counts != supports:
                raise ValueError("The repository Apriori and mlxtend results differ; refusing to display unchecked itemsets")
            canonical_episode_input = n_total == 22530 and params.get("rssi_threshold") == -90 and params.get("min_gathering_size") == 3 and params.get("jaccard_threshold") == 0.7 and params.get("max_gap_slots") == 1
            if min_support == 200 and canonical_episode_input:
                observed = (len(supports), sum(len(level) for level in levels[:1]), sum(len(level) for level in levels[1:]), len(levels), len(closed), len(maximal))
                if observed != (736, 393, 343, 6, 735, 413):
                    raise ValueError(f"Default cohort check failed: got {observed}, expected (736, 393, 343, 6, 735, 413)")
            if min_support == 100 and canonical_episode_input and (len(supports), len(levels)) != (9812, 10):
                raise ValueError(f"Minimum-support check failed at 100: got {(len(supports), len(levels))}, expected (9812, 10)")
            metrics = {"episodes": n_total, "frequent_itemsets": len(supports), "single_itemsets": sum(len(level) for level in levels[:1]), "groups_of_2_plus": sum(len(level) for level in levels[1:]), "longest_itemset": len(levels), "closed_itemsets": len(closed), "maximal_itemsets": len(maximal), "apriori_runtime_seconds": apriori_seconds, "mlxtend_runtime_seconds": mlx_seconds, "min_support": min_support}
            charts = [{"title": "Frequent itemsets by size", "x_title": "Participants per itemset", "y_title": "Frequent itemsets", "x": level_table["k"].astype(int).tolist(), "y": level_table["frequent"].astype(int).tolist()}]
        else:
            from src.models.rules import make_rules
            rules = make_rules(supports, n_total)
            if len(rules):
                rules["antecedent_size"] = rules["antecedent"].map(len)
                rules["consequent_size"] = rules["consequent"].map(len)
                rules["antecedent"] = rules["antecedent"].map(lambda group: ", ".join(map(str, group)))
                rules["consequent"] = rules["consequent"].map(lambda group: ", ".join(map(str, group)))
            tables["rules"] = rules
            toy = [frozenset(items) for items in ({1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3}, {2, 4, 5}, {1, 5, 6})]
            toy_levels, _ = apriori(toy, 2)
            toy_rules = make_rules(levels_to_counts(toy_levels), len(toy))
            if len(toy_rules) != 14:
                raise ValueError(f"Toy rule check failed: got {len(toy_rules)} rules, expected 14")
            toy_lookup = {(row.antecedent, row.consequent): row for row in toy_rules.itertuples()}
            if abs(toy_lookup[((1,), (2,))].confidence - 0.75) > 1e-12 or abs(toy_lookup[((1,), (2,))].lift - 0.9375) > 1e-12 or abs(toy_lookup[((1, 2), (3,))].confidence - 1.0) > 1e-12 or abs(toy_lookup[((1, 2), (3,))].lift - (5 / 3)) > 1e-12:
                raise ValueError("Toy confidence/lift checks failed; refusing to show rule results")
            toy_rules["antecedent"] = toy_rules["antecedent"].map(lambda group: ", ".join(map(str, group)))
            toy_rules["consequent"] = toy_rules["consequent"].map(lambda group: ", ".join(map(str, group)))
            tables["toy_rules"] = toy_rules
            metrics = {"episodes": n_total, "frequent_itemsets": len(supports), "rules": len(rules), "conf_at_least_0_8": int((rules["confidence"] >= 0.8).sum()), "lift_at_least_50": int((rules["lift"] >= 50).sum()), "both_filters": int(((rules["confidence"] >= 0.8) & (rules["lift"] >= 50)).sum()), "min_support": min_support}
            canonical_episode_input = n_total == 22530 and params.get("rssi_threshold") == -90 and params.get("min_gathering_size") == 3 and params.get("jaccard_threshold") == 0.7 and params.get("max_gap_slots") == 1
            if min_support == 200 and canonical_episode_input:
                observed = (metrics["rules"], metrics["conf_at_least_0_8"], metrics["lift_at_least_50"], metrics["both_filters"])
                if observed != (2214, 720, 580, 243):
                    raise ValueError(f"Default rules check failed: got {observed}, expected (2214, 720, 580, 243)")
            charts = [{"title": f"Rules at minimum support {min_support}", "x_title": "Confidence", "y_title": "Lift", "x": rules["confidence"].tolist(), "y": rules["lift"].tolist()}] if len(rules) else []
    else:
        raise ValueError(f"No interactive computation is registered for {analysis_id}")

    for name, frame in tables.items():
        frame.to_csv(run_dir / f"{name}.csv", index=False)
    summary = {"metrics": metrics, "charts": charts, "tables": {name: (frame if name in {"rules", "frequent_itemsets"} else frame.head(500)).to_dict(orient="records") for name, frame in tables.items()}, "profiles": tables.get("profiles", pd.DataFrame()).to_dict(orient="records"), "parameters": params, "provenance": lineage["provenance"], "note": "Timestamps are relative to the source data origin. Component and episode IDs are specific to these run settings."}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"metrics": metrics, "artifacts": ["summary.json", *[f"{name}.csv" for name in tables]], "provenance": lineage["provenance"], "charts": charts, "lineage_cache_hit": lineage.get("lineage_cache_hit", False)}
