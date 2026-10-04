"""Transactions for frequent-pattern mining: one transaction per episode."""

import pandas as pd


def build_transactions(episodes):
    """One transaction per episode: the set of participant ids, in episode_id order."""
    # a fixed row order makes the transaction list reproducible
    episodes = episodes.sort_values("episode_id")
    transactions = []
    for members in episodes["participant_set"]:
        # parquet gives numpy arrays; a frozenset of plain ints is the itemset we mine on
        transactions.append(frozenset(int(person) for person in members))
    return transactions


def support_count(transactions, itemset):
    """sigma(X): how many transactions (distinct episodes) contain every item of X."""
    itemset = frozenset(itemset)
    count = 0
    for transaction in transactions:
        if itemset <= transaction:      # X is a subset of the transaction
            count += 1                  # one per episode, never per slot row (D13)
    return count


def transaction_stats(transactions):
    """Numbers to look at before mining."""
    # the length distribution tells us how low minsup can go before the long episodes explode
    lengths = pd.Series([len(t) for t in transactions])
    all_items = set()
    for transaction in transactions:
        all_items = all_items | transaction     # union of all participants = the item universe
    return {
        "n_transactions": len(transactions),
        "n_items": len(all_items),
        "median_length": lengths.median(),
        "mean_length": lengths.mean(),
        "p95_length": lengths.quantile(0.95),
        "max_length": lengths.max(),
    }


def to_one_hot(transactions):
    """Table of True/False: one row per transaction, one column per participant (for mlxtend)."""
    all_items = set()
    for transaction in transactions:
        all_items = all_items | transaction
    columns = sorted(all_items)                 # fixed column order

    rows = []
    for transaction in transactions:
        # True where the participant is in this episode, False otherwise
        rows.append([person in transaction for person in columns])
    return pd.DataFrame(rows, columns=columns)
