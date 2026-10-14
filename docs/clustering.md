# Gathering episode clustering

## Purpose

This analysis groups gathering episodes using structural, Bluetooth,
friendship, and prior-communication features.

The clusters are exploratory descriptions, not verified categories such
as lectures, parties, or study sessions.

## Final representation

The final representation is `social_pair_normalized_without_gap`.
It uses these seven features in the recorded order: [file:196]

- `log_median_size`
- `log_span_minutes`
- `membership_expansion_ratio`
- `median_rssi`
- `internal_fb_density`
- `log_calls_events_per_pair_pre`
- `log_sms_events_per_pair_pre`

`gap_fraction` is not included in this representation, and prior
communication uses a 24-hour lookback window. [file:196]

## Recorded model settings

| Setting | Value |
|---|---|
| Representation | `social_pair_normalized_without_gap` [file:196] |
| Input version | `episodes_v1` [file:196] |
| Feature version | `pair_normalized_v1` [file:196] |
| Fitted episodes | 22,259 [file:196] |
| Random seed | 42 [file:196] |
| Initializations | 20 [file:196] |
| Tested cluster counts | 2–8 [file:192] |
| Coarse partition | 2 clusters [file:196] |
| Exploratory descriptive partition | 5 clusters [file:196] |
| Recorded scikit-learn version | 1.9.1 [file:196] |

The two- and five-cluster solutions should be treated as separate
partitions, not as levels of a demonstrated hierarchy.

## Cluster-count evaluation

| Clusters | Sampled silhouette | Davies–Bouldin |
|---|---:|---:|
| 2 | 0.4554 [file:192] | 1.3841 [file:192] |
| 3 | 0.3295 [file:192] | 1.3626 [file:192] |
| 4 | 0.2425 [file:192] | 1.4193 [file:192] |
| 5 | 0.2645 [file:192] | 1.2906 [file:192] |
| 6 | 0.2719 [file:192] | 1.2641 [file:192] |
| 7 | 0.2614 [file:192] | 1.2339 [file:192] |
| 8 | 0.2647 [file:192] | 1.2354 [file:192] |

Two clusters achieved the highest sampled silhouette among the tested
values. [file:192]

The two-cluster partition is strongly imbalanced: the larger cluster
contains 20,406 episodes, while the smaller contains 1,853. [file:192]

Five clusters were retained for descriptive interpretation, not because
they optimized all validity measures. [file:196]

Six clusters achieved a slightly higher silhouette and lower
Davies–Bouldin score than five; seven achieved the lowest
Davies–Bouldin score among the tested values. [file:192]

## Five-cluster profiles

The following working labels describe the saved profiles.
They are not validated event types.

| Cluster ID | Working label | Episodes | Share |
|---|---|---:|---:|
| 0 | Small, more friendship-connected groups | 5,118 [file:214] | 23.0% [file:214] |
| 1 | Longer episodes with membership expansion | 2,830 [file:214] | 12.7% [file:214] |
| 2 | Larger groups | 3,026 [file:214] | 13.6% [file:214] |
| 3 | Groups with higher prior communication | 1,347 [file:214] | 6.1% [file:214] |
| 4 | Small groups with lower friendship density | 9,938 [file:214] | 44.6% [file:214] |

The table below reports values from the saved profile rows, not
characteristics shared by every episode.

| Cluster ID | Group size | Span (minutes) | RSSI | Friendship density | Membership expansion |
|---|---:|---:|---:|---:|---:|
| 0 | 3 [file:214] | 5 [file:214] | -77 [file:214] | 0.467 [file:214] | 1.00 [file:214] |
| 1 | 4 [file:214] | 55 [file:214] | -81 [file:214] | 0.167 [file:214] | 1.33 [file:214] |
| 2 | 17 [file:214] | 5 [file:214] | -82.5 [file:214] | 0.095 [file:214] | 1.00 [file:214] |
| 3 | 3 [file:214] | 5 [file:214] | -80 [file:214] | 0.400 [file:214] | 1.00 [file:214] |
| 4 | 3 [file:214] | 5 [file:214] | -84 [file:214] | 0.095 [file:214] | 1.00 [file:214] |

Cluster 0 has the highest friendship-density profile value and the
least negative RSSI value among the five profiles. [file:214]

Cluster 1 stands out for its longer span and membership expansion,
while cluster 2 stands out for its larger group size. [file:214]

Cluster 3 has the highest prior-communication profile values:
approximately 0.67 call events and 2.83 SMS events per pair. [file:214]

Cluster 4 is the largest cluster and combines small group size with
relatively low friendship density and the most negative RSSI
profile value. [file:214]

These patterns should not be assigned specific activity labels
without external validation.

## Initialization stability

The five-cluster solution was compared across seeds 1–5, producing
ten pairwise adjusted Rand index measurements. [file:194]

The measurements ranged from 0.9987 to 0.9998, indicating very
strong assignment agreement across these initializations. [file:194]

This test concerns initialization changes only; it should not be
interpreted as evidence of robustness to different features,
episode extraction settings, or input populations.

## Feature sensitivity

| Variant | Reference representation | Clusters | ARI versus reference |
|---|---|---:|---:|
| `structural_without_gap` | `structural` | 3 [file:195] | 0.4447 [file:195] |
| `social_without_gap` | `social_communication` | 5 [file:195] | 0.6882 [file:195] |
| `social_pair_normalized` | `social_communication` | 5 [file:195] | 0.6240 [file:195] |
| `social_pair_normalized_without_gap` | `social_communication` | 5 [file:195] | 0.5598 [file:195] |

The tested representation changes produced materially different
assignments relative to their respective references. [file:195]

The structural comparison uses three clusters and a different
reference from the social comparisons, which use five clusters. [file:195]

Consequently, these rows should not be treated as a directly
comparable ranking of individual feature importance.

## Limitations

- Treat cluster IDs as identifiers, not ordered categories.
- Descriptive labels do not establish real-world event types.
- Do not infer physical distance directly from the RSSI profiles.
- Strong initialization agreement does not establish feature robustness.
- The timestamp origin remains unverified, so calendar-date and
  weekday interpretations are not established. [file:196]
- The metadata does not record the scaler specification, silhouette
  sample size, input fingerprints, or code revision; consult the
  implementation before claiming complete reproducibility. [file:196]

## Supporting artifacts

The documentation uses these exported files:

- `gathering_clustering_v1_metadata.json` — representation and
  recorded model settings. [file:196]
- `cluster_validity_normalized_without_gap.csv` — cluster-count
  evaluation. [file:192]
- `cluster_profiles_normalized_final_candidates-2.csv` — uploaded
  cluster profiles for the candidate partitions. [file:214]
- `cluster_seed_stability_normalized_without_gap-3.csv` — uploaded
  initialization-stability results. [file:194]
- `cluster_feature_sensitivity-4.csv` — uploaded feature-sensitivity
  results. [file:195]

The numeric suffixes above reflect the uploaded filenames.
Check the repository for the corresponding local filenames.

## Deferred work

The anomaly-filtered comparison is deferred and has not been performed.
The current documentation describes the baseline analysis only.