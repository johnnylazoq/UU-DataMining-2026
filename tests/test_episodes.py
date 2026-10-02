"""Tests for linking gatherings into episodes."""

import pandas as pd
import pytest

from src.features.episodes import build_episodes, candidate_edges, link_rows

# first 17 rows of gatherings_v1 (slots 0 to 4), linked by hand in the notebook
TOY_SLOTS = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]
TOY_SETS = [
    {12, 244, 454}, {176, 221, 263, 472}, {215, 311, 524},
    {57, 101, 586}, {12, 244, 454}, {221, 263, 472}, {282, 455, 621},
    {12, 244, 454}, {176, 221, 263, 472}, {282, 455, 621},
    {57, 101, 586}, {12, 244, 454}, {176, 221, 263, 472},
    {64, 90, 91}, {57, 101, 586}, {176, 221, 263, 452, 472}, {282, 455, 621},
]
EXPECTED = [(0.6, 0, 8), (0.6, 1, 6), (0.8, 0, 10), (0.8, 1, 7)]

EPISODE_COLUMNS = [
    "episode_id", "first_slot", "last_slot", "start_time", "end_time",
    "n_slots", "n_gap_slots", "gathering_ids", "participant_set",
    "median_size", "median_rssi", "start_hour", "night_share",
]


def make_toy_gatherings():
    rows = []
    for gathering_id, (slot, members) in enumerate(zip(TOY_SLOTS, TOY_SETS)):
        rows.append({
            "slot_id": slot,
            "gathering_id": gathering_id,
            "participant_set": sorted(members),
            "start_time": slot * 300,
            "end_time": slot * 300 + 299,
            "median_rssi": -80.0,
        })
    return pd.DataFrame(rows)


def make_config(theta, gap):
    return {
        "time_parameters": {"slot_duration_sec": 300},
        "episode_parameters": {"jaccard_threshold": theta, "max_gap_slots": gap},
    }


@pytest.mark.parametrize("theta, gap, n_expected", EXPECTED)
def test_link_rows_matches_hand_count(theta, gap, n_expected):
    edges = candidate_edges(TOY_SLOTS, TOY_SETS, 0.4, 3)
    episodes = link_rows(edges, len(TOY_SETS), theta, gap)
    assert len(episodes) == n_expected


@pytest.mark.parametrize("theta, gap, n_expected", EXPECTED)
def test_build_episodes_matches_hand_count(theta, gap, n_expected):
    result = build_episodes(make_toy_gatherings(), make_config(theta, gap))
    assert len(result) == n_expected


def test_build_episodes_columns_and_row_counts():
    result = build_episodes(make_toy_gatherings(), make_config(0.6, 1))
    assert list(result.columns) == EPISODE_COLUMNS
    for _, episode in result.iterrows():
        n_rows = len(episode["gathering_ids"])
        assert episode["n_slots"] - episode["n_gap_slots"] == n_rows


def test_rssi_sensitivity_summary_on_hand_example():
    from src.rssi_sensitivity import summarise

    # Group A drifts: ten members, one person replaced per slot over 5 slots.
    # Each step has J = 9/11 = 0.82, but first row vs last row is 6/14 = 0.43.
    # Group B is a stable trio in all 5 slots (J = 1, no drift).
    rows = []
    for slot in range(5):
        for members in (list(range(1 + slot, 11 + slot)), [101, 102, 103]):
            rows.append({
                "slot_id": slot,
                "gathering_id": len(rows),
                "participant_set": members,
                "start_time": slot * 300,
                "end_time": slot * 300 + 299,
                "median_rssi": -80.0,
            })
    gatherings = pd.DataFrame(rows)

    episodes = build_episodes(gatherings, make_config(0.7, 1))
    summary = summarise(-90, [], gatherings, episodes)

    assert summary["gatherings"] == 10
    assert summary["episodes"] == 2                  # one chain for A, one for B
    assert summary["one_slot_share_pct"] == 0
    assert summary["long_episodes"] == 2             # both last 5 slots
    assert summary["drift_pct"] == 50                # only A drifted
    assert summary["median_gathering_size"] == 6.5   # five rows of 10 and five rows of 3
