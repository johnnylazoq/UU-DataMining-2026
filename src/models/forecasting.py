"""Forecasting recurring cohorts: predict future gatherings and test stability.

Methods:
1. Cohort recurrence rate: P(cohort appears again | past observations)
2. Temporal stability: does cohort maintain same frequency over time?
3. Next encounter prediction: estimate when cohort appears next
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def compute_cohort_recurrence(
    episodes: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Compute recurrence rates for each cohort.

    For each cohort, compute:
    - first_seen: when cohort first appears
    - last_seen: when cohort last appears
    - total_occurrences: how many times it appears
    - time_span_days: from first to last occurrence
    - recurrence_rate: occurrences per day
    - mean_gap_days: average days between consecutive appearances
    - min_gap_days, max_gap_days: min/max gap between appearances

    Args:
        episodes: DataFrame with start_time and participant_set
        cohorts: DataFrame with members (space-separated IDs)

    Returns:
        DataFrame with recurrence metrics for each cohort
    """
    # Pre-compute participant sets once
    participant_sets = [frozenset(map(int, p)) for p in episodes["participant_set"]]
    start_times = episodes["start_time"].tolist()

    results = []

    for cohort_idx, row in cohorts.iterrows():
        members = frozenset(map(int, row["members"].split()))

        # Find all episodes containing this cohort
        episode_times = [
            start_times[i]
            for i in range(len(participant_sets))
            if members.issubset(participant_sets[i])
        ]

        if len(episode_times) == 0:
            continue

        # Sort by time
        episode_times = sorted(episode_times)
        first_seen = episode_times[0]
        last_seen = episode_times[-1]

        # Convert to datetime for duration calculation
        if isinstance(first_seen, (int, float)):
            # Assume Unix timestamp (seconds since epoch)
            first_dt = datetime.fromtimestamp(first_seen)
            last_dt = datetime.fromtimestamp(last_seen)
        else:
            first_dt = pd.to_datetime(first_seen)
            last_dt = pd.to_datetime(last_seen)

        time_span = (last_dt - first_dt).total_seconds() / 86400  # days
        time_span = max(time_span, 1)  # avoid division by zero

        # Compute gaps between consecutive occurrences
        gaps = []
        for i in range(1, len(episode_times)):
            if isinstance(episode_times[i], (int, float)):
                gap = (episode_times[i] - episode_times[i - 1]) / 86400
            else:
                gap = (pd.to_datetime(episode_times[i]) - pd.to_datetime(episode_times[i - 1])).total_seconds() / 86400
            gaps.append(gap)

        results.append({
            "cohort_idx": cohort_idx,
            "members": row["members"],
            "total_occurrences": len(episode_times),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "time_span_days": time_span,
            "recurrence_rate_per_day": len(episode_times) / time_span,
            "mean_gap_days": np.mean(gaps) if gaps else np.nan,
            "median_gap_days": np.median(gaps) if gaps else np.nan,
            "min_gap_days": np.min(gaps) if gaps else np.nan,
            "max_gap_days": np.max(gaps) if gaps else np.nan,
            "std_gap_days": np.std(gaps) if gaps else np.nan,
        })

    return pd.DataFrame(results).set_index("cohort_idx")


def compute_temporal_stability(
    episodes: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Test if cohort frequency is stable over time (first half vs second half).

    Args:
        episodes: DataFrame with start_time and participant_set
        cohorts: DataFrame with members

    Returns:
        DataFrame with temporal stability metrics
    """
    # Pre-compute participant sets once
    participant_sets = [frozenset(map(int, p)) for p in episodes["participant_set"]]
    start_times = episodes["start_time"].tolist()

    results = []

    for cohort_idx, row in cohorts.iterrows():
        members = frozenset(map(int, row["members"].split()))

        # Find all episodes containing this cohort
        episode_times = [
            start_times[i]
            for i in range(len(participant_sets))
            if members.issubset(participant_sets[i])
        ]

        if len(episode_times) < 2:
            continue

        # Sort by time
        episode_times = sorted(episode_times)
        first_time = episode_times[0]
        last_time = episode_times[-1]

        # Split into first and second half
        midpoint = (first_time + last_time) / 2
        first_half = sum(1 for t in episode_times if t < midpoint)
        second_half = sum(1 for t in episode_times if t >= midpoint)

        # Chi-square test (simple version)
        expected = len(episode_times) / 2
        chi_square = ((first_half - expected) ** 2 + (second_half - expected) ** 2) / expected if expected > 0 else 0

        results.append({
            "cohort_idx": cohort_idx,
            "members": row["members"],
            "first_half_count": first_half,
            "second_half_count": second_half,
            "ratio_first_to_second": first_half / second_half if second_half > 0 else np.inf,
            "chi_square": chi_square,
        })

    return pd.DataFrame(results).set_index("cohort_idx")


def predict_next_encounter(
    episodes: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Predict when each cohort will appear next (based on historical pattern).

    Uses median inter-occurrence gap to estimate next appearance.

    Args:
        episodes: DataFrame with start_time and participant_set
        cohorts: DataFrame with members

    Returns:
        DataFrame with predictions
    """
    # Pre-compute participant sets once
    participant_sets = [frozenset(map(int, p)) for p in episodes["participant_set"]]
    start_times = episodes["start_time"].tolist()

    results = []

    for cohort_idx, row in cohorts.iterrows():
        members = frozenset(map(int, row["members"].split()))

        # Find all episodes containing this cohort
        episode_times = [
            start_times[i]
            for i in range(len(participant_sets))
            if members.issubset(participant_sets[i])
        ]

        if len(episode_times) < 2:
            continue

        # Sort and compute gaps
        episode_times = sorted(episode_times)
        last_time = episode_times[-1]

        gaps = []
        for i in range(1, len(episode_times)):
            if isinstance(episode_times[i], (int, float)):
                gap = (episode_times[i] - episode_times[i - 1]) / 86400
            else:
                gap = (pd.to_datetime(episode_times[i]) - pd.to_datetime(episode_times[i - 1])).total_seconds() / 86400
            gaps.append(gap)

        median_gap = np.median(gaps)

        # Predict next occurrence
        if isinstance(last_time, (int, float)):
            last_dt = datetime.fromtimestamp(last_time)
            next_predicted = last_dt + timedelta(days=median_gap)
        else:
            last_dt = pd.to_datetime(last_time)
            next_predicted = last_dt + timedelta(days=median_gap)

        # Days until next occurrence (from now)
        now = datetime.now()
        days_until = (next_predicted - now).total_seconds() / 86400

        results.append({
            "cohort_idx": cohort_idx,
            "members": row["members"],
            "last_occurrence": last_time,
            "median_gap_days": median_gap,
            "predicted_next": next_predicted,
            "days_until_next": days_until,
            "confidence": 1.0 / (1.0 + np.std(gaps)) if gaps else 0,  # lower std = higher confidence
        })

    return pd.DataFrame(results).set_index("cohort_idx")
