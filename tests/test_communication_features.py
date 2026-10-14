import numpy as np
import pandas as pd
import pytest

from src.features.communication import (
    add_pair_normalized_communication,
    communication_summary,
    prepare_events,
)


def test_communication_window_is_left_inclusive_right_exclusive_and_origin_bounded():
    events = pd.DataFrame({
        "timestamp": [0, 3600, 3601, 86400, 86401],
        "caller": [1, 1, 2, 1, 1],
        "callee": [2, 2, 1, 2, 2],
    })
    prepared = prepare_events(events, "caller", "callee")
    episodes = pd.DataFrame({"episode_id": [1, 2], "start_time": [3600, 86400], "participant_set": [[1, 2], [1, 2]]})
    summary = communication_summary(prepared, episodes, "caller", "callee", "calls", 86400)
    assert summary["calls_events_pre"].tolist() == [1, 3]
    assert summary["calls_active_pairs_pre"].tolist() == [1, 1]


def test_pair_normalization_uses_union_pair_denominator_and_log1p():
    frame = pd.DataFrame({"n_unique_participants": [2, 4], "calls_events_pre": [2, 6], "sms_events_pre": [1, 0]})
    result = add_pair_normalized_communication(frame)
    assert result["calls_events_per_pair_pre"].tolist() == [2.0, 1.0]
    assert np.allclose(result["log_sms_events_per_pair_pre"], [np.log1p(1), 0])
    assert "calls_events_per_pair_pre" not in frame


def test_pair_normalization_rejects_single_member_union():
    frame = pd.DataFrame({"n_unique_participants": [1], "calls_events_pre": [0], "sms_events_pre": [0]})
    with pytest.raises(ValueError, match="at least two"):
        add_pair_normalized_communication(frame)
