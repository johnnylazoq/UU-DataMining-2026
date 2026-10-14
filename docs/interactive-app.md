# Interactive analysis app

The web application exposes the implemented CampusGather workflows through the
existing API, Redis/RQ worker, per-run metadata, cache, and artifact routes.
Pipeline parameter defaults come from `config/config.yaml`. The documented
itemset schema uses minimum support 200 by default, with a supported range of
100–400. There is no anomaly page or anomaly control in the app.

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
- **Episodes:** RSSI threshold, minimum component size, Jaccard threshold
  (0.6/0.7/0.8), and maximum slot gap (0/1/2); rebuilds gatherings and episode
  links with the same snapshot. Canonical episode-count, one-slot, and drift
  checks gate the displayed result.
- **Clustering:** feature representation, k, seed, starts, silhouette sample
  size, plus all upstream settings including communication lookback. Changing k
  reuses the features produced inside the run; changing an upstream setting
  rebuilds all dependent stages.
- **Recurring cohorts:** RSSI, minimum gathering size, θ, g, and minimum support
  (100–400). Live runs call `build_transactions`, repository `apriori`,
  `find_closed`, and `find_maximal`; checked runs compare exact supports with
  mlxtend and report both runtimes. Saved cohort measures remain separate.
- **RSSI sensitivity:** precomputed only; reads the saved sensitivity CSV, whose
  canonical -90 dBm counts are checked before display.
- **Cohort drill-down:** reads measures from `recurring_cohorts.csv` and filters
  `episodes_v1.parquet` for the selected member set. It does no new mining.
- **Association rules:** uses repository `apriori` and `make_rules`; confidence,
  lift, and top-N controls filter the returned rule table and scatter plot
  immediately. The toy example and canonical real-data counts are checked.

Display search/sort controls apply immediately. Expensive pipeline work requires
the explicit **Run analysis** action. Job states and submitted settings are
recorded in run metadata. Exact successful matches may be reused; every other
run remains isolated from baseline files.

The pipeline uses the existing graph-component, greedy Jaccard continuation,
communication-window, K-Means, Apriori, and rule-generation functions. The
one-hour cohort session gap remains the notebook implementation's fixed setting.
Itemset and rule tables contain participant IDs as in the source outputs; cohort
drill-down contains aggregate day and hour counts. All timestamps remain relative
to the source time origin.

## Services

- FastAPI: `server.api:app`
- Redis queue: `campusgather`
- RQ worker: `server.worker.execute_analysis`
- Frontend: Next.js App Router in `web/`

The API accepts allowlisted analysis IDs and schema-validated JSON parameters; it
does not accept shell commands, Python code, or client-selected paths. Artifact
access is limited to files declared by a run.
