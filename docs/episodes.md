# Episodes (`episodes_v1`)

Hand-off table from gathering extraction to feature building and itemset mining.

## What an episode is

`gatherings_v1.parquet` has one row per group per 5-minute slot, so one real group
that stays together appears in many rows. An episode links those rows over time.

- Two rows are linked if the Jaccard similarity of their member sets is at least
  `jaccard_threshold` (J = |A ∩ B| / |A ∪ B|) and they are at most `max_gap_slots`
  skipped slots apart.
- Each row gets at most one successor and one predecessor. Candidate links are taken
  in the order: fewest skipped slots, then highest J (greedy).
- Both settings are in `config/config.yaml` under `episode_parameters`
  (0.7 and 1).
- Code: `src/features/episodes.py`. The choice of the two values is documented in
  `notebooks/episodes_theta_gap.ipynb`.

## Schema (13 columns, frozen for v1)

| Column | Meaning |
|---|---|
| `episode_id` | Running number; changes if theta or gap change |
| `first_slot`, `last_slot` | First and last slot of the episode |
| `start_time`, `end_time` | Start of the first row, end of the last row (seconds) |
| `n_slots` | Slots covered from first to last, gaps included |
| `n_gap_slots` | Covered slots without a row (rows = `n_slots` − `n_gap_slots`) |
| `gathering_ids` | List of `gathering_id` values of `gatherings_v1` in this episode |
| `participant_set` | Union of all members (one transaction per episode for itemset mining) |
| `median_size` | Median group size over the rows |
| `median_rssi` | Median of the rows' median RSSI (dBm: never averaged) |
| `start_hour` | Hour of the first slot (0–23) |
| `night_share` | Share of covered slots between 00:00 and 07:00 |

Notes:
- `start_hour` and `night_share` assume timestamp 0 = 00:00. The dataset paper gives
  relative time only, so this is not confirmed. They are flags, no rows are filtered.
- One-slot episodes are kept (`n_slots == 1`); filter them downstream if needed.
- The schema is fixed so that downstream code does not break when the definition is
  tuned. A changed definition gets a new version (`episodes_v2`).

## How to build

Processed files are not in Git (dataset-derived, regenerated from code).

1. Place the raw data in `data/raw/` and run `notebooks/eda_bt_symmetric.ipynb`
   once (it writes `data/processed/bt_symmetric_prepared.parquet`).
2. Run `python main.py --skip-secondary`.

This writes `contacts`, `coverage`, `gatherings_v1` and `episodes_v1` (parquet) to
`data/processed/` at the active threshold (`rssi_threshold: -90`).

## RSSI sensitivity

The threshold -90 dBm is the baseline. To see how gatherings and episodes change at
-80 and -85:

```bash
python -m src.rssi_sensitivity
```

The script runs the pipeline on a copy of the config with a temporary output folder,
so the baseline files are never overwritten. It writes a small summary table to
`outputs/tables/rssi_sensitivity_episodes.csv` (tracked in Git). The columns `theta`
and `gap` show which episode settings produced it; rerun the script if they change.
It also checks that its -90 run gives the same number of episodes as the existing
`episodes_v1.parquet`.
