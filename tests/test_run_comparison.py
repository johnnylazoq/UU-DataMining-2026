import pandas as pd
import pytest

pytest.importorskip("sklearn")
from server.comparison import compare_assignments


def test_comparison_aligns_by_episode_id_and_reports_omissions():
    first = pd.DataFrame({"episode_id": [10, 11, 12], "cluster_id": [0, 0, 1]})
    second = pd.DataFrame({"episode_id": [12, 11, 99], "cluster_id": [7, 7, 8]})
    result = compare_assignments(first, second)
    assert result["shared_episodes"] == 2
    assert result["only_first"] == result["only_second"] == 1
    assert result["ari_shared_observations"] == 1.0
    assert result["alignment"] == "episode_id"


def test_comparison_rejects_duplicate_episode_ids():
    duplicate = pd.DataFrame({"episode_id": [1, 1], "cluster_id": [0, 1]})
    with pytest.raises(ValueError, match="unique"):
        compare_assignments(duplicate, pd.DataFrame({"episode_id": [1], "cluster_id": [0]}))
