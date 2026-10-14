"""Reusable communication-window and episode feature calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_events(table: pd.DataFrame, left_column: str, right_column: str) -> pd.DataFrame:
    """Validate, filter, and timestamp-sort communication events as in the notebook."""
    events = table[["timestamp", left_column, right_column]].copy()
    for column in events.columns:
        events[column] = pd.to_numeric(events[column], errors="raise")
    if events.isna().any().any() or (events["timestamp"] < 0).any():
        raise ValueError("Communication events require nonmissing values and nonnegative timestamps")
    events = events[(events[left_column] >= 0) & (events[right_column] >= 0) & (events[left_column] != events[right_column])]
    return events.sort_values("timestamp").reset_index(drop=True)


def communication_summary(
    events: pd.DataFrame,
    features: pd.DataFrame,
    left_column: str,
    right_column: str,
    prefix: str,
    lookback_seconds: int,
) -> pd.DataFrame:
    """Count internal events in [start-lookback, start), bounded at time zero.

    Active-pair counts deduplicate event direction while event counts retain
    every event, matching `notebooks/gathering_features.ipynb`.
    """
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    if not events["timestamp"].is_monotonic_increasing:
        raise ValueError("Events must be sorted by timestamp")
    times = events["timestamp"].to_numpy()
    left_ids = events[left_column].to_numpy(dtype=int)
    right_ids = events[right_column].to_numpy(dtype=int)
    records = []
    for episode in features.itertuples(index=False):
        start = int(episode.start_time)
        members = np.asarray(episode.participant_set, dtype=int)
        window_start = max(0, start - lookback_seconds)
        lower = np.searchsorted(times, window_start, side="left")
        upper = np.searchsorted(times, start, side="left")
        a, b = left_ids[lower:upper], right_ids[lower:upper]
        internal = np.isin(a, members) & np.isin(b, members)
        a, b = a[internal], b[internal]
        active_pairs = {tuple(sorted((int(user_a), int(user_b)))) for user_a, user_b in zip(a, b)}
        records.append({"episode_id": episode.episode_id, f"{prefix}_events_pre": len(a), f"{prefix}_active_pairs_pre": len(active_pairs)})
    return pd.DataFrame(records, columns=[
        "episode_id", f"{prefix}_events_pre", f"{prefix}_active_pairs_pre"
    ])


def add_pair_normalized_communication(features: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's log1p(event-count / all-possible-membership-pairs) features."""
    required = {"n_unique_participants", "calls_events_pre", "sms_events_pre"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing pair-normalization inputs: {missing}")
    result = features.copy()
    possible_pairs = result["n_unique_participants"] * (result["n_unique_participants"] - 1) / 2
    if (possible_pairs <= 0).any():
        raise ValueError("Episode membership unions must contain at least two participants")
    for prefix in ("calls", "sms"):
        column = f"{prefix}_events_per_pair_pre"
        result[column] = result[f"{prefix}_events_pre"] / possible_pairs
        result[f"log_{column}"] = np.log1p(result[column])
    return result
