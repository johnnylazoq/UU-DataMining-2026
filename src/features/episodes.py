"""Linking of slot-level gatherings into episodes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def candidate_edges(slots, sets, theta_min, gap_max):
    """Return all (earlier row i, later row j) pairs at most gap_max + 1 slots apart with J >= theta_min."""
    # rows_in_slot[s] = positions of all rows that belong to slot s
    rows_in_slot = {}
    for i, s in enumerate(slots):
        rows_in_slot.setdefault(s, []).append(i)

    edges = []
    for i in range(len(sets)):                          # every row i ...
        for gap in range(0, gap_max + 1):               # ... skipping 0, 1, 2, ... slots
            later_slot = slots[i] + 1 + gap             # slot where we look for a continuation
            for j in rows_in_slot.get(later_slot, []):
                shared = len(sets[i] & sets[j])         # |A ∩ B|
                if shared == 0:
                    continue                            # nobody in common: not a candidate
                jac = shared / len(sets[i] | sets[j])   # |A ∩ B| / |A ∪ B|
                if jac >= theta_min:
                    edges.append((i, j, jac, gap))
    return pd.DataFrame(edges, columns=["i", "j", "jac", "gap"])


def accept_links(edges, theta, g):
    """Choose which candidate links to use; returns {row: next row of the same episode}."""
    # keep only the links allowed by this theta and gap
    allowed = edges[(edges.jac >= theta) & (edges.gap <= g)]

    # best links first: fewest skipped slots, then highest J, then row order
    allowed = allowed.sort_values(["gap", "jac", "i", "j"], ascending=[True, False, True, True])

    successor = {}                                      # successor[i] = j means "i continues as j"
    has_predecessor = set()                             # rows that already have a link pointing at them
    for i, j in zip(allowed.i, allowed.j):
        i_is_free = i not in successor                  # row i has no next row yet
        j_is_free = j not in has_predecessor            # row j has no previous row yet
        if i_is_free and j_is_free:
            successor[int(i)] = int(j)
            has_predecessor.add(int(j))
    return successor


def link_rows(edges, n_rows, theta, g):
    """Turn the accepted links into episodes (lists of row positions in time order)."""
    successor = accept_links(edges, theta, g)
    has_predecessor = set(successor.values())

    episodes = []
    for start in range(n_rows):
        if start in has_predecessor:
            continue                                    # this row continues an earlier one, not a start
        chain = [start]
        while chain[-1] in successor:                   # follow the links until the chain ends
            chain.append(successor[chain[-1]])
        episodes.append(chain)
    return episodes


def build_episodes(gatherings: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Link the rows of gatherings_v1 into episodes and return one row per episode."""
    # the settings come from config.yaml, not from the code
    theta = config["episode_parameters"]["jaccard_threshold"]
    gap = config["episode_parameters"]["max_gap_slots"]
    slot_sec = config["time_parameters"]["slot_duration_sec"]   # seconds per slot (300), used for the hour of day

    # the linking works on row positions, so the order of the rows must be fixed
    gath = gatherings.sort_values(["slot_id", "gathering_id"]).reset_index(drop=True)
    slots = gath["slot_id"].to_numpy()                  # slots[k] = slot of row k
    members = [frozenset(int(u) for u in a) for a in gath["participant_set"]]   # members[k] = participants of row k

    edges = candidate_edges(slots, members, theta, gap)       # all possible links at exactly this theta and gap
    chains = link_rows(edges, len(gath), theta, gap)          # each chain is one episode: row positions in time order

    rows = []                                           # one dictionary per episode, turned into a DataFrame at the end
    for episode_id, chain in enumerate(chains):         # one pass per episode
        first_slot = int(slots[chain[0]])               # slot of the first row
        last_slot = int(slots[chain[-1]])               # slot of the last row
        n_slots = last_slot - first_slot + 1            # slots covered, gaps included

        everyone = set()                                # all members that appear anywhere in the chain
        for k in chain:
            everyone = everyone | members[k]

        sizes = [len(members[k]) for k in chain]        # group size of every row in the chain
        rssi_values = [gath["median_rssi"].iloc[k] for k in chain]   # median RSSI of every row in the chain

        # hour of each covered slot, assuming timestamp 0 = midnight
        covered = np.arange(first_slot, last_slot + 1)
        hours = (covered * slot_sec // 3600) % 24

        rows.append({
            "episode_id": episode_id,                   # running number, changes if theta or gap change
            "first_slot": first_slot,
            "last_slot": last_slot,
            "start_time": gath["start_time"].iloc[chain[0]],       # start of the first row
            "end_time": gath["end_time"].iloc[chain[-1]],          # end of the last row
            "n_slots": n_slots,
            "n_gap_slots": n_slots - len(chain),        # covered slots without a row
            "gathering_ids": [int(gath["gathering_id"].iloc[k]) for k in chain],   # the rows of gatherings_v1 in this episode
            "participant_set": sorted(everyone),        # union of all members, one transaction per episode later
            "median_size": float(np.median(sizes)),     # typical group size over the rows
            "median_rssi": float(np.median(rssi_values)),   # median of the rows' medians (dBm: never average)
            "start_hour": int(hours[0]),                # hour of the first slot (0-23)
            "night_share": float((hours < 7).mean()),   # share of covered slots between 00:00 and 07:00
        })

    # fixed column order, the same as the schema in the notebook
    columns = ["episode_id", "first_slot", "last_slot", "start_time", "end_time",
               "n_slots", "n_gap_slots", "gathering_ids", "participant_set",
               "median_size", "median_rssi", "start_hour", "night_share"]
    return pd.DataFrame(rows, columns=columns)
