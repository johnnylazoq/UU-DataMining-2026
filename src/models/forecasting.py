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


def merge_episodes_into_sessions(episode_times: list, gap_hours: float = 1.0) -> list[tuple]:
    """Merge consecutive episodes into sessions based on time gap.
    
    Episodes closer together than gap_hours are grouped into one session.
    Returns list of (session_start_time, session_end_time) tuples.
    
    Args:
        episode_times: sorted list of episode start times
        gap_hours: maximum gap (hours) to merge episodes into same session
    
    Returns:
        List of (start_time, end_time) tuples for each session
    """
    if len(episode_times) == 0:
        return []
    
    gap_seconds = gap_hours * 3600
    sessions = []
    session_start = episode_times[0]
    session_end = episode_times[0]
    
    for i in range(1, len(episode_times)):
        current_time = episode_times[i]
        
        # Convert to seconds if needed
        if isinstance(current_time, (int, float)):
            prev_end = session_end
            time_diff = current_time - prev_end
        else:
            prev_end = pd.to_datetime(session_end)
            current_dt = pd.to_datetime(current_time)
            time_diff = (current_dt - prev_end).total_seconds()
        
        if time_diff <= gap_seconds:
            # Extend current session
            session_end = current_time
        else:
            # Start new session
            sessions.append((session_start, session_end))
            session_start = current_time
            session_end = current_time
    
    # Add final session
    sessions.append((session_start, session_end))
    
    return sessions


def compute_cohort_recurrence(
    episodes: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Compute recurrence rates for each cohort using sessions (not episodes).

    For each cohort, compute:
    - first_seen: when cohort first appears
    - last_seen: when cohort last appears
    - total_sessions: number of merged sessions (not episodes)
    - total_episodes: raw episode count (for reference)
    - time_span_days: from first to last occurrence
    - recurrence_rate: sessions per day (not episodes per day)
    - mean_gap_days: average days between consecutive sessions
    - min_gap_days, max_gap_days: min/max gap between sessions

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

        # Merge episodes into sessions (episodes < 1 hour apart)
        sessions = merge_episodes_into_sessions(episode_times, gap_hours=1.0)
        
        # Compute gaps between consecutive sessions
        session_gaps = []
        for i in range(1, len(sessions)):
            session_end_prev = sessions[i - 1][1]
            session_start_curr = sessions[i][0]
            
            if isinstance(session_start_curr, (int, float)):
                gap = (session_start_curr - session_end_prev) / 86400
            else:
                gap = (pd.to_datetime(session_start_curr) - pd.to_datetime(session_end_prev)).total_seconds() / 86400
            session_gaps.append(gap)

        results.append({
            "cohort_idx": cohort_idx,
            "members": row["members"],
            "total_sessions": len(sessions),
            "total_episodes": len(episode_times),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "time_span_days": time_span,
            "recurrence_rate_per_day": len(sessions) / time_span,
            "mean_gap_days": np.mean(session_gaps) if session_gaps else np.nan,
            "median_gap_days": np.median(session_gaps) if session_gaps else np.nan,
            "min_gap_days": np.min(session_gaps) if session_gaps else np.nan,
            "max_gap_days": np.max(session_gaps) if session_gaps else np.nan,
            "std_gap_days": np.std(session_gaps) if session_gaps else np.nan,
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
    """Predict when each cohort will appear next (based on historical session pattern).

    Uses median inter-session gap to estimate next appearance.

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

        # Sort and merge into sessions
        episode_times = sorted(episode_times)
        sessions = merge_episodes_into_sessions(episode_times, gap_hours=1.0)
        
        if len(sessions) < 2:
            continue
        
        last_session_end = sessions[-1][1]

        # Compute gaps between sessions
        session_gaps = []
        for i in range(1, len(sessions)):
            session_end_prev = sessions[i - 1][1]
            session_start_curr = sessions[i][0]
            
            if isinstance(session_start_curr, (int, float)):
                gap = (session_start_curr - session_end_prev) / 86400
            else:
                gap = (pd.to_datetime(session_start_curr) - pd.to_datetime(session_end_prev)).total_seconds() / 86400
            session_gaps.append(gap)

        median_gap = np.median(session_gaps)

        # Predict next occurrence
        if isinstance(last_session_end, (int, float)):
            last_dt = datetime.fromtimestamp(last_session_end)
            next_predicted = last_dt + timedelta(days=median_gap)
        else:
            last_dt = pd.to_datetime(last_session_end)
            next_predicted = last_dt + timedelta(days=median_gap)

        # Days until next occurrence (from now)
        now = datetime.now()
        days_until = (next_predicted - now).total_seconds() / 86400

        results.append({
            "cohort_idx": cohort_idx,
            "members": row["members"],
            "last_session_end": last_session_end,
            "median_gap_days": median_gap,
            "predicted_next": next_predicted,
            "days_until_next": days_until,
            "confidence": 1.0 / (1.0 + np.std(session_gaps)) if session_gaps else 0,
        })

    return pd.DataFrame(results).set_index("cohort_idx")
