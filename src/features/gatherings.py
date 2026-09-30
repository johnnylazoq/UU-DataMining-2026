"""Gathering extraction from slotted Bluetooth contact graphs."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from ..data.io import write_processed_data


def extract_gatherings(clean_contacts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extract connected components meeting the configured minimum size."""
    minimum_size = config["clustering_parameters"]["min_gathering_size"]
    gatherings: list[dict] = []
    # Global running ID, unique across all slots.
    gathering_id = 0

    for slot_id, slot_data in clean_contacts.groupby("slot_id"):
        # One undirected proximity graph per slot: nodes are students, edges are sightings.
        graph = nx.Graph()
        graph.add_edges_from(
            (row.user_a, row.user_b, {"rssi": row.rssi})
            for row in slot_data.itertuples(index=False)
        )

        # A gathering is a connected component: members are linked by a chain of contacts, not all pairwise.
        for participants in nx.connected_components(graph):
            if len(participants) < minimum_size:
                continue
            # The component's own edges, used for its time span and signal summary.
            component_data = slot_data[
                slot_data["user_a"].isin(participants)
                & slot_data["user_b"].isin(participants)
            ]
            gatherings.append(
                {
                    "slot_id": slot_id,
                    "gathering_id": gathering_id,
                    "participant_set": sorted(participants),
                    "start_time": component_data["timestamp"].min(),
                    "end_time": component_data["timestamp"].max(),
                    "median_rssi": component_data["rssi"].median(),
                }
            )
            gathering_id += 1

    result = pd.DataFrame(gatherings)
    # Versioned filename: bump to v2 if the gathering definition changes.
    write_processed_data(
        result,
        config["paths"]["processed_data_dir"],
        "gatherings_v1.parquet",
    )
    return result
