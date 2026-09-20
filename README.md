# CampusGather
## Discovering and Forecasting Student Gatherings from Temporal Interaction Networks

CampusGather is a temporal-network analytics and forecasting prototype for university
campuses. It uses anonymized Bluetooth proximity, phone-call metadata, SMS metadata,
and Facebook friendship links to discover recurring student gatherings, characterize
their structural patterns, identify unusual campus mixing, and forecast future
participant contacts or gathering occurrence.

The project uses the Copenhagen Networks Study interaction dataset. It is a data-mining
and research prototype, not a student-surveillance system: all analysis is conducted
with anonymous participant IDs and aggregate structural patterns.

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

CampusGather has four goals:

1. Extract physically plausible student gatherings from Bluetooth proximity data.
2. Find student cohorts that meet repeatedly using association-rule and frequent-itemset
   mining.
3. Discover gathering archetypes and unusual patterns using clustering and
   anomaly-detection methods.
4. Forecast future participant contacts or gathering occurrence using only information
   available before the prediction time.

The project does not claim that a detected gathering is definitively a lecture, party,
study session, or another named real-world event. Instead, results use conservative
structural labels, such as:

- High-affiliation gathering.
- Low-affiliation co-location.
- Short transient encounter.
- Large recurring gathering.

---

## Application Domain

CampusGather is designed for a **university-campus setting**.

Potential users include:

| User group | Possible use |
|---|---|
| Campus space planners | Identify recurring high-activity time windows and aggregate co-location patterns |
| Student-community researchers | Study recurring anonymous cohorts and campus social mixing |
| Epidemiological researchers | Analyze aggregate close-contact structure and unusual mixing patterns |
| Data-mining researchers | Explore temporal graphs, association rules, clustering, anomaly detection, and forecasting |

Although the pipeline may later be adapted to other closed communities, such as a
workplace or conference, this project is evaluated only on a university-student
population. Results should therefore be interpreted as campus-specific.

---

## Research Questions

- Which groups of enrolled students meet repeatedly?
- Which structural types of gathering occur in the campus interaction network?
- Which gatherings differ from ordinary time-of-day and day-of-week behavior?
- Can future participant-to-participant contacts be predicted, including first-time
  observed contacts?
- Can a recurring cohort or gathering archetype be forecast for a future time window?
- Can the system estimate aggregate presence of devices outside the study during a
  gathering?

---

## Dataset

The project uses the **Copenhagen Networks Study interaction dataset**, which provides
multi-layer interaction data for a university-student population.

| File | Description | Main use |
|---|---|---|
| `bt_symmetric.csv` | Bluetooth proximity scans: timestamp, user A, user B, RSSI | Gathering extraction and future-contact labels |
| `calls.csv` | Phone-call metadata | Recent communication and social-activity features |
| `sms.csv` | SMS metadata | Recent communication and social-activity features |
| `fb_friends.csv` | Facebook friendship links | Long-term social-affinity features |
| `genders.csv` | Participant gender metadata | Optional aggregate descriptive analysis only |
| `*.README` | File-specific documentation | Correct parsing and preprocessing |

Participant IDs are anonymous categorical identifiers. They must not be treated as
numerical measurements, coordinates, ranking variables, or contiguous array indices.

---

## Bluetooth Data Rules

Each Bluetooth record has this form:

```text
timestamp, user_a, user_b, rssi
```

`user_a` is the enrolled participant whose phone scanned. `user_b` is either another
enrolled participant or a special sentinel value.

| Condition | Meaning | Pipeline treatment |
|---|---|---|
| `user_a >= 0` and `user_b >= 0` | Contact between two enrolled participants | Retain as a candidate proximity edge |
| `user_b = -1` and `rssi = 0` | Empty Bluetooth scan | Use only for availability/coverage analysis |
| `user_b = -2` | One or more devices outside the experiment were detected | Exclude from participant graph; optionally use as an aggregate external-presence outcome |
| Higher RSSI | Stronger signal; `-73` is stronger than `-88` | Use as a candidate-close-contact filter |

All devices outside the experiment are assigned the same `user_b = -2` value. Therefore,
`-2` is **not an identifiable person**, cannot be tracked as a persistent individual,
and must never be included as a node in the participant interaction graph.

### Valid participant-contact filter

```python
valid_contacts = bt[
    (bt["user_a"] >= 0)
    & (bt["user_b"] >= 0)
    & (bt["rssi"] < 0)
].copy()
```

A proximity threshold is then applied. We will compare at least:

```text
RSSI >= -80 dBm
RSSI >= -85 dBm
RSSI >= -90 dBm
```

RSSI is a logarithmic signal-strength measurement. It is used as a filtering signal for
likely proximity, not as an exact physical distance or proof of social interaction.

---

## Methodology

### Shared Data Foundation

All team members use one loading/preprocessing module and one configuration file.

The common pipeline will:

1. Load raw CSV files with explicit column names.
2. Parse timestamps consistently.
3. Validate participant IDs and Bluetooth sentinel values.
4. Create cleaned, versioned intermediate datasets.
5. Record parameters, versions, and random seeds in `config.yaml`.

Shared intermediate outputs:

```text
contacts.parquet
coverage.parquet
gatherings_v1.parquet
gathering_features_v1.parquet
```

### Gathering Extraction

Bluetooth scans are sampled on a five-minute grid. For each time slot:

1. Retain valid participant-to-participant Bluetooth contacts.
2. Apply an RSSI threshold.
3. Construct an undirected proximity graph.
4. Extract connected components with a minimum group size, initially `k >= 3`.
5. Merge compatible components across consecutive time slots into gathering episodes.

Each extracted gathering will contain:

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

The analysis will compare connected-component and clique-based gathering definitions
where feasible. It will also report sensitivity to RSSI thresholds and the minimum
gathering size.

### Recurring Cohorts

Each gathering is represented as a transaction containing its participant IDs.

The recurring-cohort pipeline will use:

- Apriori and/or FP-Growth.
- Frequent itemsets.
- Closed itemsets.
- Maximal itemsets.
- Support, confidence, and lift.

Support is calculated from distinct gathering episodes rather than raw Bluetooth scans,
so that one long event is not artificially counted as many repeated events.

### Gathering Archetypes

Each gathering receives a feature vector built from several data layers.

| Feature family | Example features |
|---|---|
| Size and duration | Participant count, duration, active time slots |
| Proximity structure | Edge count, graph density, RSSI summary |
| Temporal context | Hour of day, day of week, recurrence frequency |
| Social affinity | Internal Facebook-friendship density |
| Communication context | Call/SMS volume before a gathering |
| Communication structure | Reciprocity and active communicating pairs |
| External presence | Whether external devices were detected nearby |

Continuous features will be normalized. Temporal features will use appropriate cyclic
encoding. The project may use dimensionality reduction for visualization and clustering.

Cluster quality will be assessed using:

- Silhouette score.
- Davies-Bouldin index.
- Cluster size and balance.
- Stability under reasonable parameter changes.
- Interpretability based on the observed features.

### Anomaly Detection and Validation

The project will use complementary methods to identify unusual gatherings and unusual
campus days:

- Distance-based anomaly scoring.
- Local Outlier Factor (LOF).
- Isolation Forest.

Anomaly scores will be compared with time-of-day and day-of-week baselines. Temporal
shuffle or null-model analyses will be used where feasible to test whether detected
patterns exceed normal diurnal activity rhythms.

---

## Forecasting Targets

Forecasting is an application layer built on top of the temporal-network analysis. It
does not try to identify people outside the study.

### Target A: Future Participant Contact

For an enrolled participant pair `i` and `j`:

```text
Given all information available up to time t,
will i and j have a valid Bluetooth proximity contact
during the next prediction window?
```

This includes **first-time observed contacts**. Two enrolled participants may have no
previous recorded encounter, but the model can estimate the probability of a later
meeting.

Candidate features include:

- Mutual Facebook friendships.
- Number of common prior gathering members.
- Shared historical active time slots.
- Recent calls and SMS between participants.
- Individual activity and proximity degree.
- Previous-contact frequency and recency.
- Hour of day and day of week.

### Target B: Future Gathering Occurrence

```text
Given interaction history up to time t,
will a recurring cohort or gathering archetype occur in the next prediction window?
```

Possible outputs:

- Predicted time window.
- Predicted cohort or likely attendees.
- Probability/confidence score.
- Predicted structural archetype.
- Estimated probability of external-device presence.

### Target C: External-Device Presence

For a participant or detected gathering:

```text
Will one or more Bluetooth devices outside the experiment
be detected during the next prediction window?
```

This predicts only aggregate external-device presence. It cannot identify, count
reliably, or track individual people outside the study because all such devices share
the sentinel identifier `-2`.

---

## Evaluation

### Temporal Validation

Evaluation uses chronological data splits:

```text
Training period -> Validation period -> Later held-out test period
```

No future observation may be used to create a past feature. Random train/test splits are
avoided because they would leak future temporal information.

### Metrics

| Task | Main metrics |
|---|---|
| Future-contact prediction | Precision, recall, F1, PR-AUC, ROC-AUC |
| Future-gathering prediction | Precision/recall at top-k, F1, PR-AUC, calibration |
| Clustering | Silhouette score, Davies-Bouldin index, stability |
| Anomaly detection | Cross-method agreement, temporal baseline comparison, null-model comparison |
| Cohort mining | Support, confidence, lift, recurrence count, threshold stability |

### Baselines

At minimum, the project compares proposed models against:

- Hour-of-day and day-of-week activity baselines.
- Recent-contact-frequency baseline.
- Pair-popularity or random baseline where appropriate.
- Bluetooth RSSI threshold sensitivity comparisons.

---

## Application Features

The final prototype will be an exploratory dashboard or notebook-based application.

### Overview

- Dataset coverage and participant activity.
- Selected RSSI threshold and minimum gathering size.
- Number of gatherings, recurring cohorts, and detected anomalies.
- Summary of forecasts and uncertainties.

### Temporal Activity

- Interaction and gathering volume by date and hour.
- Activity by day of week.
- Optional comparison with external-device-presence frequency.

### Gathering Explorer

For a selected gathering:

- Start/end time and duration.
- Number of participants.
- Proximity-graph density and RSSI summary.
- Friendship and communication context.
- Cluster/archetype assignment.
- Anomaly score.
- External-device-presence indicator.

### Recurring Cohort Explorer

- Repeated participant groups.
- Recurrence count and dates.
- Support, confidence, and lift.
- Threshold and parameter sensitivity.

### Forecast View

- Predicted future participant contacts.
- Predicted recurring cohorts or gathering archetypes.
- Prediction confidence.
- Forecast time horizon.
- Clear warning that predictions concern anonymized study participants only.

### Limitations View

- No GPS locations or semantic event labels.
- Bluetooth proximity is not proof of meaningful interaction.
- External devices are not individually identifiable.
- Archetypes are structural interpretations, not verified real-world event types.
- Results are specific to the observed university-student population.

---

## Team and Responsibilities

### Emmanouil Vettas — Gathering Extraction and Recurring Cohorts

**Primary question:** Which student groups keep meeting?

**Owns:**

- Bluetooth data cleaning and valid participant-contact definition.
- RSSI threshold sensitivity analysis at `-80`, `-85`, and `-90 dBm`.
- Per-time-slot proximity graph construction.
- Connected-component and, if feasible, clique-based gathering extraction.
- Consecutive-slot gathering merging and `gatherings_v*.parquet`.
- Transaction generation from gathering memberships.
- Frequent-itemset and association-rule mining with FP-Growth or Apriori.
- Closed/maximal itemset analysis.
- Support, confidence, lift, and recurrence reporting.

**Main outputs:**

```text
src/preprocessing.py
src/gatherings.py
src/cohorts.py
data/processed/contacts.parquet
data/processed/coverage.parquet
data/processed/gatherings_v1.parquet
outputs/tables/recurring_cohorts.csv
outputs/figures/rssi_sensitivity.*
outputs/figures/cohort_recurrence.*
```

### Sotirios Oikonomou — Gathering Features and Archetype Clustering

**Primary question:** What structural kinds of student gatherings exist?

**Owns:**

- Loading and validating calls, SMS, Facebook-friendship, and gender files.
- Documenting non-Bluetooth dataset schemas.
- Building one feature vector per extracted gathering.
- Feature scaling, transformations, and missing-value handling.
- Mixed-type representation and distance choices.
- Dimensionality reduction for exploration and visualization.
- Clustering selection and parameter tuning.
- Cluster quality assessment with silhouette and Davies-Bouldin scores.
- Conservative structural interpretation of clusters.

**Main outputs:**

```text
src/features.py
src/clustering.py
data/processed/gathering_features_v1.parquet
outputs/tables/gathering_clusters.csv
outputs/tables/cluster_validity.csv
outputs/figures/feature_distributions.*
outputs/figures/cluster_visualization.*
outputs/figures/cluster_profiles.*
```

### Johnny Israel Lazo Quinonez — Anomaly Detection, Validation, and Forecast Evaluation

**Primary question:** Which gatherings are abnormal, and do observed patterns exceed ordinary daily rhythms?

**Owns:**

- Chronological train/validation/test split definition.
- Hour-of-day and day-of-week baseline models.
- External-device-presence outcome definition without treating `-2` as a person.
- Distance-based anomaly scoring.
- Local Outlier Factor (LOF).
- Isolation Forest.
- Comparison and consolidation of anomaly scores.
- Temporal-shuffle or degree-preserving null-model validation.
- Future-contact or future-gathering forecast evaluation on later held-out data.
- Reporting precision, recall, F1, PR-AUC, ROC-AUC, and baseline comparisons.

**Main outputs:**

```text
src/anomalies.py
src/forecasting.py
outputs/tables/anomaly_scores.csv
outputs/tables/anomaly_method_agreement.csv
outputs/tables/null_model_results.csv
outputs/tables/forecast_metrics.csv
outputs/tables/forecast_predictions.csv
outputs/figures/anomaly_distributions.*
outputs/figures/baseline_comparison.*
outputs/figures/forecast_performance.*
```

### Shared Responsibilities

All members jointly own:

- Repository maintenance, code review, reproducibility, and `config.yaml`.
- Agreement on raw-to-processed data schemas.
- Interpretation of results and ethical limitations.
- Dashboard/app integration.
- Preliminary report, final report, slides, and presentation.

No member should change a shared schema or parameter definition without documenting the
change in `config.yaml`, the changelog, and the relevant output version.

---

## Project Plan

**Project period:** 20 September 2026 to 14 October 2026.  
**Preliminary report deadline:** **2 October 2026**.  
**Final project deadline:** **14 October 2026**.

### Milestone Overview

| Date | Milestone | Minimum evidence |
|---|---|---|
| 26 September | Shared data foundation completed | Reproducible loaders, data dictionary, cleaning rules, initial descriptive results |
| 2 October | Preliminary report submitted | Gathering pipeline, preliminary results from all three work streams, remaining-work plan |
| 10 October | Core analyses completed | Cohorts, clusters, anomalies, validation, and initial forecast evaluation |
| 13 October | Integration freeze | Final figures/tables, app/demo, report draft, reproducibility check |
| 14 October | Final submission | Final repository, report, presentation, demo, and outputs |

### Phase 1 — Data Understanding and Shared Foundation
**20–26 September**

#### Emmanouil

- Read `bt_symmetric.README` and implement Bluetooth sentinel handling.
- Load Bluetooth data with explicit column names.
- Inspect timestamp range, participant coverage, RSSI distribution, duplicates, and scan volume.
- Implement the first valid-contact cleaning function.
- Create `contacts.parquet` and `coverage.parquet`.
- Propose initial RSSI threshold, time-slot size, and minimum gathering size.

#### Sotirios

- Read and load `calls.csv`, `sms.csv`, `fb_friends.csv`, and `genders.csv`.
- Document columns, timestamps, participant coverage, duplicates, and missing values.
- Verify participant-ID overlap between network layers.
- Define the gathering-level feature dictionary.
- Write the first feature-schema specification.

#### Johnny

- Define chronological development, validation, and held-out test periods.
- Define time-of-day/day-of-week baseline predictions.
- Define evaluation metrics for forecasting and anomaly detection.
- Define aggregate external-device-presence labels using `user_b = -2`.
- Specify an initial temporal-shuffle/null-model plan.

#### Shared

- Create the repository, environment, folder structure, and `requirements.txt`.
- Create `config/config.yaml`.
- Create `docs/data_dictionary.md`.
- Agree on schemas for `contacts.parquet`, `coverage.parquet`, and `gatherings_v1.parquet`.
- Hold a review meeting by 26 September to approve shared definitions.

#### Required outputs

```text
01_data_audit.ipynb
src/io.py
src/preprocessing.py
config/config.yaml
docs/data_dictionary.md
data/processed/contacts.parquet
data/processed/coverage.parquet
Initial data-audit figures
```

### Phase 2 — Preliminary Report Milestone
**27 September–2 October**

This phase produces the material required for the **preliminary report on 2 October**.
The goal is a reproducible pipeline and preliminary evidence from all three analysis
streams—not a completed final project.

#### Emmanouil

- Build per-five-minute proximity graphs from cleaned Bluetooth contacts.
- Extract connected components using the initial `k >= 3` definition.
- Merge compatible components over consecutive slots into gathering episodes.
- Generate `gatherings_v1.parquet`.
- Compare initial RSSI thresholds: `-80`, `-85`, and `-90 dBm`.
- Convert gathering memberships into transactions.
- Run a first FP-Growth or Apriori analysis.
- Report preliminary recurring cohorts and support values.

#### Sotirios

- Implement the first gathering-level feature table from `gatherings_v1.parquet`.
- Add temporal, Bluetooth, Facebook, and recent communication features.
- Perform descriptive feature analysis and normalization.
- Run an initial clustering workflow.
- Produce a preliminary cluster-quality table.
- Create one cluster visualization and conservative cluster profiles.

#### Johnny

- Create hour-of-day and day-of-week activity baselines.
- Run initial Isolation Forest or LOF on the first gathering feature table.
- Produce a ranked list of preliminary unusual gatherings or days.
- Compare anomaly scores against temporal baselines.
- Finalize chronological train/test split definitions.
- Define and demonstrate one forecasting target, even if only using a baseline model.

#### Shared preliminary-report tasks

- Integrate at least one figure or table from each work stream.
- Document `user_b = -1`, `user_b = -2`, RSSI threshold choice, and gathering definition.
- State research questions, methodology, preliminary results, risks, limitations, and
  remaining work.
- Include each member’s completed contribution.
- Review and submit the report by **2 October**.

#### Mandatory preliminary-report contents

```text
1. Project title, campus domain, and research questions.
2. Dataset overview and Bluetooth cleaning rules.
3. Shared methodology and parameter configuration.
4. Preliminary gathering-extraction results.
5. Preliminary recurring-cohort results.
6. Preliminary gathering-feature and clustering results.
7. Preliminary anomaly and temporal-baseline results.
8. Work completed by each team member.
9. Risks, limitations, and plan from 3–14 October.
```

#### Required outputs

```text
02_gathering_extraction.ipynb
data/processed/gatherings_v1.parquet
data/processed/gathering_features_v1.parquet
outputs/tables/preliminary_cohorts.csv
outputs/tables/preliminary_clusters.csv
outputs/tables/preliminary_anomalies.csv
outputs/figures/preliminary_*.*
docs/preliminary_report.md or preliminary_report.pdf
```

### Phase 3 — Complete the Core Analyses
**3–10 October**

#### Emmanouil

- Refine gathering extraction using preliminary-report feedback.
- Compare connected-component and clique approaches where feasible.
- Run final frequent-itemset analysis across chosen RSSI thresholds.
- Compare Apriori and FP-Growth if practical.
- Focus on closed and maximal itemsets.
- Calculate support, confidence, lift, recurrence times, and threshold stability.
- Produce final cohort tables and figures.

#### Sotirios

- Finalize the feature table and transformations.
- Compare reasonable clustering configurations.
- Select cluster settings based on validity, stability, and interpretability.
- Create cluster profiles and temporal distributions.
- Apply only conservative, feature-supported structural labels.
- Document why each archetype interpretation is justified.

#### Johnny

- Run distance-based, LOF, and Isolation Forest anomaly methods.
- Compare anomaly rankings and method agreement.
- Implement the selected temporal shuffle/null-model validation.
- Test whether findings exceed time-of-day effects.
- Train at least one future-contact or future-gathering model.
- Evaluate it strictly on a later held-out period.
- Compare against time-based and recent-activity baselines.
- Produce final metric tables and performance figures.

#### Shared

- Hold an integration meeting by 8 October.
- Review agreement and conflicts across cohort, clustering, and anomaly results.
- Select final report/app figures and tables.
- Update methodology documentation with final parameter values.

#### Required outputs

```text
03_cohort_mining.ipynb
04_clustering.ipynb
05_anomaly_detection.ipynb
06_forecasting.ipynb
outputs/tables/recurring_cohorts.csv
outputs/tables/gathering_clusters.csv
outputs/tables/anomaly_scores.csv
outputs/tables/null_model_results.csv
outputs/tables/forecast_metrics.csv
Final analytical figures
```

### Phase 4 — App Integration and Final Submission
**11–14 October**

#### Emmanouil

- Prepare reusable outputs for gathering and recurring-cohort app views.
- Add RSSI-threshold and support-filter details to outputs.
- Validate gathering memberships and recurrence summaries.
- Write the final cohort-mining methodology and results sections.

#### Sotirios

- Prepare cluster profiles and feature summaries for the app.
- Create presentation-ready archetype figures.
- Validate feature names, units/scales, and cluster descriptions.
- Write the final clustering methodology and results sections.

#### Johnny

- Prepare anomaly and forecast outputs for the app.
- Add metrics, baseline comparisons, confidence scores, and limitations.
- Verify strict chronological evaluation and absence of time leakage.
- Write the final anomaly, validation, and forecasting sections.

#### Shared

- Integrate the dashboard or notebook demo.
- Add limitations and ethics documentation.
- Complete final README, methodology documentation, and slides.
- Run the pipeline in a clean environment.
- Freeze code and final outputs by 13 October.
- Submit final materials on 14 October.

#### Required outputs

```text
app/
docs/methodology.md
docs/presentation_outline.md
Final report
Final slides/demo
Final reproducible repository
```

### Individual Accountability Matrix

| Deliverable | Owner | Reviewer(s) | Due |
|---|---|---|---|
| Bluetooth cleaning and sentinel handling | Emmanouil | Sotirios, Johnny | 26 Sep |
| Non-Bluetooth schema audit and feature dictionary | Sotirios | Emmanouil, Johnny | 26 Sep |
| Chronological validation/evaluation protocol | Johnny | Emmanouil, Sotirios | 26 Sep |
| `gatherings_v1.parquet` | Emmanouil | Sotirios | 30 Sep |
| `gathering_features_v1.parquet` | Sotirios | Emmanouil | 1 Oct |
| Preliminary anomaly/baseline outputs | Johnny | Sotirios | 1 Oct |
| Preliminary report | All; section owners above | All | 2 Oct |
| Final cohort-mining outputs | Emmanouil | Johnny | 8 Oct |
| Final clustering outputs | Sotirios | Emmanouil | 8 Oct |
| Final anomaly and forecast evaluation | Johnny | Sotirios | 10 Oct |
| App/demo integration | All; view owners above | All | 13 Oct |
| Final report and submission | All | All | 14 Oct |

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
│   ├── raw/                       # Normally excluded from Git
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

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Suggested dependencies:

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

### 3. Add raw data

Place dataset files in:

```text
data/raw/
```

Expected files:

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

### 4. Configure parameters

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

### 5. Run analyses

```bash
jupyter notebook
```

Run notebooks in numerical order. Later, the project may add a command-line workflow for
full pipeline reproduction.

---

## Ethics and Limitations

### Privacy-Aware Use

- Participant IDs are anonymous and must not be re-identified.
- Outputs are aggregate, research-oriented, and explanatory.
- The tool must not support disciplinary action, individual surveillance, or behavioral
  judgments about real students.
- Gender metadata, if used, is limited to aggregate descriptive analysis and must not
  drive individual decisions or predictions.

### Data Limitations

- Bluetooth proximity does not prove conversation, friendship, or intentional social
  interaction.
- RSSI does not provide exact physical distance.
- The dataset has no GPS coordinates, calendar labels, message contents, or verified
  event names.
- A detected gathering may be incidental co-location rather than a meaningful social
  gathering.
- `user_b = -2` represents external devices collectively and cannot identify or reliably
  count individual non-participants.
- Results are based on a university-student population and may not generalize without
  separate validation.

---

## Definition of Done

The project is complete when it can:

- Reproduce preprocessing and gathering extraction from raw data.
- Document data coverage, quality checks, sentinel values, and parameter choices.
- Identify recurring student cohorts using frequent-itemset and association-rule mining.
- Cluster gatherings and report quantitative cluster-quality measures.
- Detect unusual gatherings with multiple complementary anomaly methods.
- Validate findings against temporal baselines and/or null models.
- Evaluate at least one future-contact or future-gathering prediction task on a
  later-held-out period.
- Provide a dashboard, app, or reproducible notebook-based demonstration.
- Document uncertainty, privacy constraints, and interpretation limits.
