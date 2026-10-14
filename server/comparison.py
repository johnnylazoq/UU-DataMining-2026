"""Safe run comparison aligned strictly by episode_id."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compare_assignments(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    required = {"episode_id", "cluster_id"}
    if not required.issubset(first.columns) or not required.issubset(second.columns):
        raise ValueError("Both assignment tables must contain episode_id and cluster_id")
    if first["episode_id"].duplicated().any() or second["episode_id"].duplicated().any():
        raise ValueError("episode_id must be unique within each assignment table")
    aligned = first[["episode_id", "cluster_id"]].merge(second[["episode_id", "cluster_id"]], on="episode_id", how="inner", suffixes=("_first", "_second"), validate="one_to_one")
    return {
        "shared_episodes": int(len(aligned)),
        "only_first": int(first["episode_id"].nunique() - len(aligned)),
        "only_second": int(second["episode_id"].nunique() - len(aligned)),
        "ari_shared_observations": float(adjusted_rand_score(aligned["cluster_id_first"], aligned["cluster_id_second"])) if len(aligned) >= 2 else None,
        "alignment": "episode_id",
        "note": "Cluster IDs are arbitrary; ARI compares partitions over shared episode IDs only.",
    }
