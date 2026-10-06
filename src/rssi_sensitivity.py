"""RSSI sensitivity of gatherings and episodes.

Runs the pipeline at every threshold in config.yaml on a copy of the config whose
output folder is a temporary directory, so the canonical -90 files are never touched.
Run from the project root:  python -m src.rssi_sensitivity
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.io import PREPARED_BLUETOOTH_FILENAME
from .features.episodes import build_episodes
from .features.gatherings import extract_gatherings
from .features.preprocessing import clean_bluetooth_data
from .pipeline import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LONG_EPISODE_SLOTS = 5      # 25 minutes or more, the same cut-off as in episodes_theta_gap.ipynb


def summarise(threshold, contacts, gatherings, episodes) -> dict:
    """One row of the sensitivity table."""
    members = dict(zip(gatherings["gathering_id"], gatherings["participant_set"]))
    long_episodes = episodes[episodes["n_slots"] >= LONG_EPISODE_SLOTS]

    drifted = 0
    for ids in long_episodes["gathering_ids"]:
        first, last = set(members[ids[0]]), set(members[ids[-1]])   # first and last row of the episode
        if len(first & last) / len(first | last) < 0.5:             # end-to-end Jaccard below 0.5 = drift
            drifted += 1

    return {
        "rssi_threshold": threshold,
        "contacts": len(contacts),
        "gatherings": len(gatherings),
        "median_gathering_size": gatherings["participant_set"].apply(len).median(),
        "episodes": len(episodes),
        "one_slot_share_pct": 100 * (episodes["n_slots"] == 1).mean(),
        "long_episodes": len(long_episodes),
        "drift_pct": 100 * drifted / len(long_episodes) if len(long_episodes) else float("nan"),
    }


def main() -> None:
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    processed = resolve_path(config["paths"]["processed_data_dir"], PROJECT_ROOT)
    prepared = processed / PREPARED_BLUETOOTH_FILENAME

    rows = []
    for threshold in config["bluetooth_parameters"]["rssi_thresholds"]:
        with tempfile.TemporaryDirectory() as scratch:
            run_config = copy.deepcopy(config)                     # the real config is never modified
            run_config["bluetooth_parameters"]["rssi_threshold"] = threshold
            run_config["paths"]["processed_data_dir"] = scratch    # every write goes to scratch
            contacts = clean_bluetooth_data(prepared, run_config)
            gatherings = extract_gatherings(contacts, run_config)
            episodes = build_episodes(gatherings, run_config)
        row = summarise(threshold, contacts, gatherings, episodes)
        # say which episode settings produced the row, so a stale file is easy to spot
        row = {"rssi_threshold": row.pop("rssi_threshold"),
               "theta": config["episode_parameters"]["jaccard_threshold"],
               "gap": config["episode_parameters"]["max_gap_slots"],
               **row}
        rows.append(row)
        print(f"done {threshold} dBm")

    table = pd.DataFrame(rows)

    # Reproducibility check: the active threshold must match the canonical files.
    active = config["bluetooth_parameters"]["rssi_threshold"]
    canonical = processed / "episodes_v1.parquet"
    if canonical.is_file():
        n_canonical = len(pd.read_parquet(canonical))
        n_scratch = int(table.loc[table["rssi_threshold"] == active, "episodes"].iloc[0])
        print(f"episodes at {active} dBm: scratch {n_scratch}, canonical {n_canonical}, "
              f"match: {n_scratch == n_canonical}")

    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "rssi_sensitivity_episodes.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
