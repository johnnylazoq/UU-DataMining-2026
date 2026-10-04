"""Tests for the transactions built from episodes."""

import pandas as pd

from src.features.transactions import (
    build_transactions,
    support_count,
    to_one_hot,
    transaction_stats,
)

# five episodes made by hand: episodes 0 and 2 have the same members
TOY_EPISODES = pd.DataFrame({
    "episode_id": [0, 1, 2, 3, 4],
    "participant_set": [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [2, 4, 5], [1, 5, 6]],
})


# five episodes give five transactions, each the set of that episode's participants
def test_one_transaction_per_episode():
    transactions = build_transactions(TOY_EPISODES)
    assert len(transactions) == 5
    assert transactions[0] == frozenset({1, 2, 3})
    assert transactions[3] == frozenset({2, 4, 5})


# shuffling the input rows gives the same transactions, because build_transactions sorts by episode_id
def test_order_of_input_rows_does_not_matter():
    shuffled = TOY_EPISODES.iloc[[3, 0, 4, 2, 1]]
    assert build_transactions(shuffled) == build_transactions(TOY_EPISODES)


# support is the number of episodes containing the itemset; episodes 0 and 2 have the same members and count twice
def test_support_counts_distinct_episodes():
    transactions = build_transactions(TOY_EPISODES)
    assert support_count(transactions, {1, 2}) == 3         # episodes 0, 1, 2
    assert support_count(transactions, {1, 2, 3}) == 3      # episodes 0, 1, 2 (0 and 2 count twice)
    assert support_count(transactions, {2, 4}) == 2         # episodes 1 and 3
    assert support_count(transactions, {3, 6}) == 0         # never together


# N, number of distinct items, maximum and median length of the toy transactions
def test_stats():
    stats = transaction_stats(build_transactions(TOY_EPISODES))
    assert stats["n_transactions"] == 5
    assert stats["n_items"] == 6
    assert stats["max_length"] == 4
    assert stats["median_length"] == 3


# 5 rows by 6 participant columns, True exactly where the person is in the episode
def test_one_hot_table():
    table = to_one_hot(build_transactions(TOY_EPISODES))
    assert table.shape == (5, 6)
    assert list(table.columns) == [1, 2, 3, 4, 5, 6]
    assert list(table.iloc[3]) == [False, True, False, True, True, False]   # episode 3 = {2, 4, 5}
