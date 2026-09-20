# CampusGather
## Discovering and Forecasting Student Gatherings from Temporal Interaction Networks

CampusGather is a temporal-network analytics and forecasting prototype for university
campuses. Using anonymized Bluetooth proximity, phone-call metadata, SMS metadata, and
Facebook friendship links, it discovers recurring student gatherings, characterizes
their structural patterns, detects unusual campus mixing, and forecasts future
participant contacts or gathering occurrence.

The project is evaluated on the Copenhagen Networks Study interaction dataset. It is
designed as a data-mining project, not a student-surveillance system: all analysis uses
anonymized IDs and aggregate structural patterns.

---

## Table of Contents

- [Project Goals](#project-goals)
- [Application Domain](#application-domain)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Bluetooth Data Rules](#bluetooth-data-rules)
- [Methodology](#methodology)
- [Forecasting Targets](#forecasting-targets)
- [Evaluation](#evaluation)
- [Application Features](#application-features)
- [Team and Responsibilities](#team-and-responsibilities)
- [Project Plan](#project-plan)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Ethics and Limitations](#ethics-and-limitations)
- [Definition of Done](#definition-of-done)

---

## Project Goals

The project has four goals:

1. Extract physically plausible student gatherings from Bluetooth proximity data.
2. Find student cohorts that meet repeatedly using association-rule and frequent-itemset
   mining.
3. Discover structural gathering archetypes and unusual patterns using clustering and
   anomaly-detection methods.
4. Forecast future participant contacts or gathering occurrence using only information
   available before the prediction time.

The project does not attempt to assign definitive real-world labels such as "lecture,"
"party," or "study session." Instead, it reports evidence-based structural labels such
as **high-affiliation gathering**, **low-affiliation co-location**, **short transient
encounter**, or **large recurring gathering**.

---

## Application Domain

CampusGather is intended for a **university-campus setting**.

Potential users include:

| User group | Possible use |
|---|---|
| Campus space planners | Identify recurring high-activity time windows and aggregate co-location patterns |
| Student-community researchers | Study recurring anonymous cohorts and social mixing patterns |
| Epidemiological researchers | Analyze aggregate close-contact structure and unusual mixing behavior |
| Data-mining researchers | Explore temporal graphs, association rules, clustering, anomaly detection, and forecasting |

Although the pipeline could later be adapted to other closed communities, such as an
office campus or conference, this project is evaluated only on a university-student
population. Results must therefore be interpreted as campus-specific.

---

## Research Questions

- Which groups of enrolled students repeatedly meet?
- Which gathering structures appear in the campus interaction network?
- Which gatherings are unusual compared with normal time-of-day and day-of-week
  behavior?
- Can future student-to-student contacts be predicted, including first-time observed
  contacts?
- Can the occurrence of a recurring cohort or gathering archetype be forecast for a
  future time window?
- Can the system estimate aggregate presence of devices outside the study during a
  gathering?

---

## Dataset

The project uses the **Copenhagen Networks Study interaction dataset**, containing
multi-layer interaction data collected from university students over several weeks.

| File | Description | Main use |
|---|---|---|
| `bt_symmetric.csv` | Bluetooth proximity scans: timestamp, user A, user B, and RSSI | Gathering extraction and future-contact labels |
| `calls.csv` | Phone-call metadata | Recent communication and social-activity features |
| `sms.csv` | SMS metadata | Recent communication and social-activity features |
| `fb_friends.csv` | Facebook friendship links | Long-term social-affinity features |
| `genders.csv` | Participant gender metadata | Optional descriptive analysis; not used for individual decisions |
| `*.README` | Dataset-specific file documentation | Correct parsing and data cleaning |

Participant identifiers are anonymous categorical identifiers. They are never treated
as numerical measurements, coordinates, rankings, or contiguous array indices.

---

## Bluetooth Data Rules

Each Bluetooth row has this structure:

```text
timestamp, user_a, user_b, rssi
```

`user_a` is the enrolled participant whose phone performed the scan. `user_b` is either
another enrolled participant or a special sentinel value.

| Condition | Meaning | Use in pipeline |
|---|---|---|
| `user_a >= 0` and `user_b >= 0` | Contact between two enrolled participants | Retain as a candidate proximity edge |
| `user_b = -1` and `rssi = 0` | Empty scan; no Bluetooth device was detected | Use only for availability and coverage analysis |
| `user_b = -2` | One or more devices outside the experiment were detected | Do not include in participant graph; optionally use as external-presence outcome |
| Higher RSSI | Stronger signal; e.g., `-73` is stronger than `-88` | Use for filtering candidate close contacts |

All devices outside the experiment share the same `user_b = -2` value. Therefore,
`-2` is **not a person**, is not a persistent external identity, and must never be
included as a graph node.

### Valid participant-contact filter

```python
valid_contacts = bt[
    (bt["user_a"] >= 0)
    & (bt["user_b"] >= 0)
    & (bt["rssi"] < 0)
].copy()
```

A proximity threshold is then applied. The analysis will compare at least:

```text
RSSI >= -80 dBm
RSSI >= -85 dBm
RSSI >= -90 dBm
```

Because RSSI is a logarithmic signal-strength measurement and Bluetooth detection can
be noisy, it is used as a proximity filter rather than an exact measure of distance.

---

## Methodology

### 1. Shared data foundation

All team members use the same preprocessing module and configuration file.

The shared pipeline will:

1. Load raw CSV files with explicit column names.
2. Parse timestamps consistently.
3. Validate participant IDs and special Bluetooth sentinel values.
4. Produce cleaned, versioned intermediate datasets.
5. Record every analysis parameter in `config.yaml`.

Core intermediate outputs:

```text
contacts.parquet
coverage.parquet
gatherings_v1.parquet
gathering_features_v1.parquet
```

### 2. Gathering extraction

Bluetooth scans are sampled on a five-minute grid. For each time slot:

1. Retain valid participant-to-participant Bluetooth contacts.
2. Apply an RSSI threshold.
3. Construct an undirected proximity graph.
4. Extract connected components with a minimum size, initially `k >= 3`.
5. Merge compatible components from consecutive slots into longer gathering episodes.

A gathering episode contains:

```text
gathering_id
start_time
end_time
duration_minutes
participant_ids
n_participants
n_time_slots
mean_rssi
graph_density
rssi_threshold
external_presence
pipeline_version
```

We will compare connected-component and clique-based definitions where feasible, and
we will report sensitivity to the RSSI threshold and minimum group size.

### 3. Recurring cohorts

Each extracted gathering is represented as a transaction containing its participant IDs.

We will mine recurring cohorts using:

- Apriori and/or FP-Growth.
- Frequent itemsets.
- Closed itemsets.
- Maximal itemsets.
- Support, confidence, and lift.

Closed and maximal itemsets will be prioritized to reduce redundant results and control
candidate explosion. Support will be based on distinct gathering episodes, not raw
five-minute Bluetooth observations.

### 4. Gathering archetypes

Each gathering receives a feature vector built from:

| Feature family | Example features |
|---|---|
| Size and duration | Participant count, duration, number of active slots |
| Proximity structure | Edge count, graph density, RSSI summary |
| Temporal context | Hour of day, day of week, recurrence frequency |
| Social affinity | Internal Facebook-friendship density |
| Communication context | Call/SMS volume before the gathering |
| Communication structure | Reciprocity and number of active communicating pairs |
| External presence | Whether one or more non-study devices were detected nearby |

Continuous features will be normalized. Cyclic time features will use appropriate
encoding. We will test dimensionality reduction for visualization and clustering, then
evaluate clustering using silhouette score, Davies-Bouldin index, cluster sizes, and
stability across reasonable parameter settings.

Cluster labels are interpretive summaries of measurable structure, not verified event
names.

### 5. Anomaly detection

We will detect unusual gatherings and unusual campus days with complementary methods:

- Distance-based anomaly scoring.
- Local Outlier Factor (LOF).
- Isolation Forest.

Anomaly scores will be compared against time-of-day and day-of-week baselines. We will
also use temporal shuffle or null-model analyses to assess whether apparent anomalies,
recurring cohorts, or cluster structure are stronger than expected from ordinary daily
activity rhythms.

---

## Forecasting Targets

Forecasting is an application layer built on top of the discovered temporal-network
structure. It does not attempt to identify people outside the study.

### Target A: Future participant contact

For an enrolled pair of students `i` and `j`:

```text
Given all data available up to time t,
will i and j have a valid Bluetooth proximity contact in the next prediction window?
```

This target includes **first-time observed contacts**: two enrolled students may not have
met in the previous data, but the model can estimate the probability that they will meet
later.

Candidate features:

- Mutual Facebook friendships.
- Number of common prior gathering members.
- Shared historical active time slots.
- Recent calls/SMS between the participants.
- Individual activity and proximity degree.
- Previous contact frequency and recency.
- Hour of day and day of week.

### Target B: Future gathering occurrence

```text
Given the interaction history up to time t,
will a known recurring cohort or gathering archetype occur in the next prediction window?
```

The output may include:

- Predicted time window.
- Predicted cohort or likely attendee set.
- Probability/confidence score.
- Predicted structural archetype.
- Estimated probability of external-device presence.

### Target C: External-device presence

For a participant or detected gathering:

```text
Will one or more Bluetooth devices outside the experiment be detected
during the next time window?
```

This target predicts only **aggregate external-device presence**. It cannot identify,
count reliably, or track individual people outside the study because all non-experiment
devices are represented by the same identifier, `-2`.

---

## Evaluation

### Temporal validation

All evaluation uses chronological splits.

```text
Training period -> Validation period -> Future held-out test period
```

No future observation may be used to construct a past feature. Random train/test splits
are avoided because they leak temporal information.

### Metrics

| Task | Main metrics |
|---|---|
| Future contact prediction | Precision, recall, F1, PR-AUC, ROC-AUC |
| Future gathering prediction | Precision/recall at top-k predictions, F1, PR-AUC, calibration |
| Clustering | Silhouette score, Davies-Bouldin index, stability and interpretability |
| Anomaly detection | Cross-method agreement, temporal baseline comparison, null-model comparison |
| Cohort mining | Support, confidence, lift, recurrence counts, stability across thresholds |

### Baselines

At minimum, compare proposed models against:

- Time-of-day and day-of-week activity baseline.
- Recent-contact-frequency baseline.
- Random or popularity-based pair baseline, where appropriate.
- Threshold-sensitivity baseline for Bluetooth RSSI filtering.

---

## Application Features

The final prototype will be an exploratory dashboard or notebook-based app.

### 1. Overview

- Dataset coverage and active participants.
- Current RSSI threshold and minimum gathering size.
- Number of extracted gatherings and recurring cohorts.
- Summary of anomaly and forecast outputs.

### 2. Temporal activity view

- Interaction and gathering activity by date and hour.
- Activity distribution by day of week.
- Optional comparison with external-device-presence frequency.

### 3. Gathering explorer

For a selected detected gathering:

- Start/end time and duration.
- Number of participants.
- Proximity structure and density.
- Communication and friendship features.
- Cluster/archetype assignment.
- Anomaly score.
- External-presence indicator.

### 4. Recurring cohort explorer

- Repeated participant groups.
- Number of occurrences.
- Support, confidence, and lift.
- Time distribution of occurrences.
- Sensitivity across RSSI thresholds.

### 5. Forecast view

- Predicted future participant contacts.
- Predicted recurring cohorts or gathering archetypes.
- Prediction probability/confidence.
- Prediction horizon and time window.
- Clear warning that predictions concern anonymous dataset participants only.

### 6. Limitations view

- No GPS or semantic event labels.
- Bluetooth proximity is not proof of social interaction.
- External devices cannot be individually identified.
- Structural archetypes are not verified real-world event types.
- Results are specific to the observed university-student population.

---

## Team and Responsibilities

| Team member | Core responsibility | Course concepts | Main deliverables |
|---|---|---|---|
| **Emmanouil Vettas** | Shared Bluetooth data foundation, gathering extraction, recurring-cohort analysis | Association-rule mining, Apriori, FP-Growth, closed/maximal itemsets | Clean contact pipeline, versioned gathering table, cohort-mining results, threshold sensitivity analysis |
| **Sotirios Oikonomou** | Gathering feature engineering, dimensionality reduction, clustering and archetype analysis | Feature normalization, distance measures, clustering, cluster validity | Gathering-feature table, clustering pipeline, validity results, archetype descriptions and figures |
| **Johnny Israel Lazo Quinonez** | Anomaly detection, temporal/null validation, forecasting evaluation | Distance-based anomalies, LOF, Isolation Forest, temporal validation | Anomaly pipeline, baseline/null-model comparisons, forecast experiments and evaluation |
| **All members** | Integration, code review, interpretation, app prototype, report, and presentation | Reproducibility, ethical analysis, scientific communication | Shared repository, final outputs, dashboard/demo, report, and presentation |

### Collaboration rules

- Use one shared repository and protected main branch.
- Use one shared `config.yaml` and one loading/preprocessing module.
- Store intermediate tables in documented schemas.
- Version outputs when a definition changes, e.g. `gatherings_v1.parquet`.
- Include parameter values, data version, and random seed in generated outputs.
- Review conclusions collaboratively to prevent over-interpretation.

---

## Project Plan

**Project period:** 20 September 2026 to 14 October 2026.

### Week 1 — Data understanding and shared foundation
**20–26 September**

Goals:

- Set up repository, environment, folder structure, and dependency list.
- Read every dataset README and document the data dictionary.
- Load all files and inspect schemas, missingness, time ranges, and participant coverage.
- Implement correct Bluetooth sentinel handling:
  - `user_b = -1`, `rssi = 0`: empty scan.
  - `user_b = -2`: one or more non-study devices.
- Create a shared `config.yaml`.
- Generate descriptive plots for RSSI, scan coverage, participant activity, and time range.

Deliverables:

```text
01_data_audit.ipynb
src/io.py
src/preprocessing.py
config/config.yaml
docs/data_dictionary.md
Initial exploratory figures
```

Acceptance criteria:

- Every team member can run the data-loading pipeline.
- Bluetooth filtering rules are documented and tested.
- The team agrees on initial values for RSSI threshold, time bin, and minimum gathering size.

### Week 2 — Gathering extraction and feature table
**27 September–3 October**

Goals:

- Construct per-slot participant proximity graphs.
- Extract connected components with the initial definition `k >= 3`.
- Test merging rules for consecutive components.
- Compare RSSI thresholds `-80`, `-85`, and `-90 dBm`.
- Produce `contacts.parquet`, `coverage.parquet`, and `gatherings_v1.parquet`.
- Build the first gathering-level feature table using Bluetooth, friendship, calls, and SMS.
- Define chronological train/validation/test windows and simple prediction baselines.

Deliverables:

```text
02_gathering_extraction.ipynb
src/gatherings.py
src/features.py
data/processed/contacts.parquet
data/processed/coverage.parquet
data/processed/gatherings_v1.parquet
data/processed/gathering_features_v1.parquet
```

Acceptance criteria:

- Gathering extraction is reproducible from raw data.
- Each gathering has a documented schema and unique ID.
- The team has initial visual checks of gathering size, duration, and time distribution.

### Week 3 — Core data-mining analyses
**4–10 October**

Goals:

- Run frequent-itemset and association-rule mining for recurring cohorts.
- Run clustering and select settings with quantitative validity checks.
- Run Isolation Forest, LOF, and a distance-based anomaly method.
- Compare findings with time-of-day/day-of-week baselines.
- Implement temporal shuffle/null-model validation where feasible.
- Draft conservative structural interpretations for discovered patterns.

Deliverables:

```text
03_cohort_mining.ipynb
04_clustering.ipynb
05_anomaly_detection.ipynb
outputs/tables/recurring_cohorts.csv
outputs/tables/gathering_clusters.csv
outputs/tables/anomaly_scores.csv
outputs/figures/
```

Acceptance criteria:

- Each team member has a runnable analysis pipeline and documented results.
- The group has figures/tables for cohorts, clusters, and anomalies.
- Results include sensitivity checks and do not rely on unsupported semantic labels.

### Week 4 — Forecasting, app integration, and finalization
**11–14 October**

Goals:

- Define and train a future-contact or future-gathering prediction model.
- Evaluate predictions on a later held-out period.
- Compare the model to temporal and simple behavioral baselines.
- Integrate recurring cohorts, archetypes, anomalies, and forecasts into the app/demo.
- Finalize documentation, report, figures, and presentation.
- Perform a complete clean-environment reproducibility run.

Deliverables:

```text
06_forecasting.ipynb
src/forecasting.py
outputs/tables/forecast_metrics.csv
outputs/tables/forecast_predictions.csv
app/
docs/methodology.md
docs/presentation_outline.md
Final README.md
```

Acceptance criteria:

- Forecasting uses strictly chronological evaluation.
- The app/demo communicates predictions and uncertainty clearly.
- Final results distinguish known participants, first-time observed contacts, and aggregate external-device presence.
- Repository instructions allow another person to reproduce the principal outputs.

---

## Repository Structure

```text
campusgather/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                       # Usually excluded from Git
│   └── processed/
│       ├── contacts.parquet
│       ├── coverage.parquet
│       ├── gatherings_v1.parquet
│       └── gathering_features_v1.parquet
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_gathering_extraction.ipynb
│   ├── 03_cohort_mining.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_anomaly_detection.ipynb
│   └── 06_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── gatherings.py
│   ├── features.py
│   ├── cohorts.py
│   ├── clustering.py
│   ├── anomalies.py
│   └── forecasting.py
├── app/
│   └── app.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
└── docs/
    ├── data_dictionary.md
    ├── methodology.md
    └── presentation_outline.md
```

---

## Setup

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Suggested packages:

```text
pandas
numpy
pyarrow
networkx
scikit-learn
mlxtend
matplotlib
seaborn
plotly
jupyter
streamlit
pyyaml
```

### 3. Place the raw data

Place the downloaded dataset files in:

```text
data/raw/
```

Expected files include:

```text
bt_symmetric.csv
bt_symmetric.README
calls.csv
calls.README
sms.csv
sms.README
fb_friends.csv
fb_friends.README
genders.csv
```

### 4. Configure analysis parameters

Edit:

```text
config/config.yaml
```

Example:

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed

bluetooth:
  slot_seconds: 300
  rssi_threshold: -85
  min_gathering_size: 3
  merge_overlap_threshold: 0.6

forecasting:
  prediction_horizon_hours: 24
  random_seed: 42
```

### 5. Run the pipeline

```bash
jupyter notebook
```

Run notebooks in numerical order, or provide a command-line workflow later as the project
matures.

---

## Ethics and Limitations

### Privacy-aware use

- IDs are anonymous and must not be re-identified.
- Outputs should be aggregate, research-oriented, and explanatory.
- The tool must not be used for disciplinary action, individual surveillance, or
  real-world behavioral judgments.
- Gender metadata, if used, is limited to aggregate descriptive analysis and must not
  drive individual predictions or decisions.

### Data limitations

- Bluetooth proximity is not proof of conversation, friendship, or intentional social
  interaction.
- RSSI does not provide exact physical distance.
- The data has no GPS locations, calendar labels, message contents, or verified event
  names.
- A gathering may be an incidental co-location rather than a meaningful social event.
- `user_b = -2` signals external devices collectively; it cannot identify or reliably
  count individual non-participants.
- Results are based on a university-student population and may not generalize to other
  populations without separate validation.

---

## Definition of Done

The project is complete when it can:

- Reproduce preprocessing and gathering extraction from raw data.
- Document data quality, coverage, sentinel values, and analysis parameters.
- Identify recurring student cohorts using frequent-itemset/association-rule mining.
- Cluster gatherings and report quantitative cluster-quality measures.
- Detect anomalous gatherings using multiple complementary methods.
- Validate findings against temporal baselines and/or null models.
- Evaluate at least one future-contact or future-gathering forecasting task on a
  held-out future period.
- Provide a dashboard, app, or reproducible notebook-based demonstration.
- Clearly document uncertainty, privacy constraints, and interpretation limits.
