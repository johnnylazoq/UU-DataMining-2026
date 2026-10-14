"""Frequent itemsets: Apriori from scratch, maximal and closed itemsets, and mlxtend conversion."""

from itertools import combinations

import pandas as pd


def count_items(transactions):
    """Level 1: in how many transactions does each single item appear?"""
    counts = {}
    for transaction in transactions:
        for item in transaction:
            counts[(item,)] = counts.get((item,), 0) + 1  # 1-tuple key, same format as later levels; 0 on first sight
    return counts


def make_candidates(frequent_prev, k):
    """Join and prune: build the size-k candidates from the sorted list of frequent (k-1)-itemsets."""
    known = set(frequent_prev)                          # a set makes the prune lookups fast

    # join: same first k-2 items, and the last item of a is smaller than the last item of b
    joined = []
    for a in frequent_prev:
        for b in frequent_prev:
            if a[:-1] == b[:-1] and a[-1] < b[-1]:      # same prefix; "<" builds each candidate only once
                joined.append(a + (b[-1],))             # (1,2) + (1,3) gives (1,2,3)

    # prune: every (k-1)-subset of a candidate must be frequent
    candidates = []
    for candidate in joined:
        subsets = combinations(candidate, k - 1)        # (2,3,4) gives (2,3), (2,4), (3,4)
        if all(subset in known for subset in subsets):  # one infrequent subset kills the candidate
            candidates.append(candidate)
    return joined, candidates


def count_candidates(transactions, candidates, k):
    """One scan: for every transaction, look at its k-subsets and count the ones that are candidates."""
    candidate_set = set(candidates)                      # a set makes the subset lookups fast
    counts = {candidate: 0 for candidate in candidates}  # start at 0, so unseen candidates still appear
    for transaction in transactions:
        if len(transaction) < k:                         # too short to contain a k-itemset
            continue
        for subset in combinations(transaction, k):      # sorted transaction gives sorted subsets, like the candidates
            if subset in candidate_set:                  # once per transaction, so it counts distinct episodes
                counts[subset] += 1
    return counts


def apriori(transactions, minsup):
    """Return the list of frequent itemsets per level ({itemset: support}) and a table of candidate counts."""
    transactions = [tuple(sorted(t)) for t in transactions]                               # sorted tuples: the join compares prefixes

    item_counts = count_items(transactions)                                               # level 1: the first scan counts every item
    frequent = {itemset: n for itemset, n in item_counts.items() if n >= minsup}          # filter: support >= minsup
    levels = [frequent]                                                                   # levels[0] is L1
    rows = [{"k": 1, "joined": len(item_counts), "candidates": len(item_counts), "frequent": len(frequent)}]  # numbers per level for the report

    k = 2
    while len(frequent) > 0:                                                              # stop at an empty level (downward closure)
        keep = set(item for itemset in frequent for item in itemset)                      # items inside some frequent (k-1)-itemset
        transactions = [tuple(item for item in t if item in keep) for t in transactions]  # drop the others: fewer subsets to count

        joined, candidates = make_candidates(sorted(frequent), k)                         # join + prune, no data is read
        if len(candidates) > 0:                                                           # scan only if a candidate survived the prune
            counts = count_candidates(transactions, candidates, k)
        else:
            counts = {}
        frequent = {itemset: n for itemset, n in counts.items() if n >= minsup}           # filter after counting

        rows.append({"k": k, "joined": len(joined), "candidates": len(candidates), "frequent": len(frequent)})  # numbers of this level
        if len(frequent) > 0:                                                             # store non-empty levels only
            levels.append(frequent)                                                       # so len(levels) is the longest itemset size
        k += 1
    return levels, pd.DataFrame(rows)


def find_maximal(levels):
    """Maximal itemsets: frequent itemsets that have no frequent superset."""
    not_maximal = set()                                             # itemsets that have a frequent superset
    for level in levels[1:]:                                        # itemsets of size 2 and up
        for itemset in level:
            for subset in combinations(itemset, len(itemset) - 1):  # its subsets with one item less
                not_maximal.add(subset)                             # they have a frequent superset (this itemset), so they are not maximal
    maximal = set()
    for level in levels:
        for itemset in level:
            if itemset not in not_maximal:                          # no frequent superset found
                maximal.add(itemset)
    return maximal


def find_closed(levels):
    """Closed itemsets: frequent itemsets that have no frequent superset with the same support."""
    not_closed = set()                                              # itemsets that occur in exactly the same episodes as a bigger itemset
    for k in range(1, len(levels)):                                 # levels[k] holds the itemsets of size k + 1
        for itemset, support in levels[k].items():
            for subset in combinations(itemset, len(itemset) - 1):  # its subsets with one item less
                if levels[k - 1][subset] == support:                # the subset has the same support as this itemset
                    not_closed.add(subset)                          # so the bigger itemset hides it: not closed
    closed = set()
    for level in levels:
        for itemset in level:
            if itemset not in not_closed:                           # every frequent superset has a lower support
                closed.add(itemset)
    return closed


def levels_to_counts(levels):
    """Flatten the levels of our Apriori into one dictionary {itemset: support}."""
    counts = {}
    for level in levels:
        counts.update(level)  # add the itemsets of this level
    return counts


def mlxtend_to_counts(result, n_total):
    """Turn an mlxtend result (support as a fraction) into {sorted tuple: support count}."""
    counts = {}
    for itemset, support in zip(result["itemsets"], result["support"]):
        key = tuple(sorted(int(item) for item in itemset))  # same format as our itemsets
        counts[key] = round(support * n_total)              # from a fraction back to a count of episodes
    return counts
