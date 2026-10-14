# Interactive analysis app

The web application exposes the implemented CampusGather workflows through the
existing API, Redis/RQ worker, per-run metadata, cache, and artifact routes.
Parameter defaults come from `config/config.yaml`; cohort and association
minimum support is 200 because that is the setting in the corresponding
notebooks. The anomaly page stays explicitly unsupported because no detector or
detector schema exists in the repository.

## Start locally

1. Ensure the reviewed prepared Bluetooth input and raw friendship, calls, and
   SMS files are present under `data/`.
2. Run `docker compose up --build` from the repository root.
3. Open `http://localhost:3000`. The web port is bound to loopback.

The API and worker read source data and baseline outputs. Every submitted job
recomputes the contacts, connected gatherings, linked episodes, and communication
features from a single parameter snapshot in its own `outputs/app_runs/<run_id>/work`
directory. Reports and downloads are written beside that work directory. The
pipeline never swaps in global “current” artifacts, and a cache key includes the
configuration, all source input fingerprints, analysis ID, and full parameter set.

## Interactive parameters and dependencies

- **Bluetooth & communication:** RSSI threshold and communication lookback;
  re-filters contacts and refreshes aggregate contact and episode-window event
  summaries.
- **Gatherings:** RSSI threshold and minimum component size; recomputes contacts
  and connected components.
- **Episodes:** RSSI threshold, minimum component size, Jaccard threshold, and
  maximum slot gap; rebuilds gatherings and episode links using those same values.
- **Clustering:** feature representation, k, seed, starts, silhouette sample
  size, plus all upstream settings including communication lookback. Changing k
  reuses the features produced inside the run; changing an upstream setting
  rebuilds all dependent stages.
- **Cohorts and association rules:** upstream episode settings and minimum
  support. Their frequent itemsets and tables are recomputed for each run.
- **Anomalies:** unsupported; no setting is offered because the codebase has no
  implemented detector.

Display search/sort controls apply immediately. Expensive pipeline work requires
the explicit **Run analysis** action. Job states and submitted settings are
recorded in run metadata. Exact successful matches may be reused; every other
run remains isolated from baseline files.

The pipeline uses the existing graph-component, greedy Jaccard continuation,
communication-window, K-Means, association-rule, and closed-itemset definitions.
The one-hour cohort session gap and 95th-percentile episode-size flag are retained
from `notebooks/cohort_analysis.ipynb`, not exposed as new controls. Result tables
omit participant identifiers; run-specific episode IDs are only present in the
clustering assignment download. All timestamps remain relative to the source
time origin.

## Services

- FastAPI: `server.api:app`
- Redis queue: `campusgather`
- RQ worker: `server.worker.execute_analysis`
- Frontend: Next.js App Router in `web/`

The API accepts allowlisted analysis IDs and schema-validated JSON parameters; it
does not accept shell commands, Python code, or client-selected paths. Artifact
access is limited to files declared by a run.
