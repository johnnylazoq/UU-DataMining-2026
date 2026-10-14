"""Association rules A -> B from frequent itemsets, with support, confidence and lift."""

from itertools import combinations

import pandas as pd


def make_rules(supports, n_total):
    """All rules A -> B from the frequent itemsets: A and B are non-empty, disjoint, and A + B is frequent."""
    rows = []
    for itemset, support in supports.items():
        if len(itemset) < 2:                                             # a rule needs at least one student on each side
            continue
        for size in range(1, len(itemset)):                              # size of the antecedent A
            for antecedent in combinations(itemset, size):               # every way to choose A inside the itemset
                consequent = tuple(item for item in itemset if item not in antecedent)  # B is the rest of the itemset
                confidence = support / supports[antecedent]              # sigma(A + B) / sigma(A); A is frequent, so it is in supports
                lift = confidence / (supports[consequent] / n_total)     # confidence divided by how common B is overall
                rows.append({
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": support,                                  # episodes that contain A and B
                    "confidence": confidence,
                    "lift": lift,
                })
    return pd.DataFrame(rows)


def rules_to_dict(rules):
    """{(antecedent, consequent): (confidence, lift)}, to compare two rule tables."""
    result = {}
    for row in rules.itertuples():
        result[(row.antecedent, row.consequent)] = (row.confidence, row.lift)
    return result
