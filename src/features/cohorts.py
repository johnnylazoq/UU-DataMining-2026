"""Helpers for the recurring-cohort analysis."""

from itertools import combinations


def count_sessions(rows, max_gap_sec, slot_sec):
    rows = rows.sort_values("start_time")
    sessions = 0
    last_end = None
    for start, end in zip(rows["start_time"], rows["end_time"] + slot_sec):  # end_time is the start of the last slot
        if last_end is None or start - last_end > max_gap_sec:
            sessions += 1                                                  # a gap: a new session starts
            last_end = end
        else:
            last_end = max(last_end, end)                                  # same session goes on
    return sessions


def lift_range(cohort, support, supports, n_total):
    lifts = []
    for size in range(1, len(cohort)):
        for antecedent in combinations(cohort, size):
            consequent = tuple(item for item in cohort if item not in antecedent)
            lifts.append(n_total * support / (supports[antecedent] * supports[consequent]))
    return min(lifts), max(lifts)
