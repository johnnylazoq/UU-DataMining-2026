"""Null models to test statistical significance of recurring cohorts.

Tests whether observed cohort support (frequency) is significantly higher than
expected by chance. Uses permutation-based null models: randomize participant
assignments and count how often each cohort appears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations


def permute_participants(episodes: pd.DataFrame, n_permutations: int = 100) -> list[dict]:
    """Generate null models by randomly shuffling participants across episodes.

    Keeps episode structure (size, time) but randomizes who participates.

    Args:
        episodes: DataFrame with episode_id, participant_set columns
        n_permutations: number of random permutations to generate

    Returns:
        List of dicts, each with permutation_id and null_support for each cohort
    """
    all_participants = set()
    episode_sizes = []

    for participants in episodes["participant_set"]:
        all_participants.update(participants)
        episode_sizes.append(len(participants))

    all_participants = sorted(list(all_participants))

    results = []
    for perm_id in range(n_permutations):
        # Shuffle participant list and assign to episodes in order
        shuffled = np.random.RandomState(42 + perm_id).permutation(all_participants)

        null_episodes = []
        idx = 0
        for size in episode_sizes:
            null_cohort = frozenset(shuffled[idx : idx + size])
            null_episodes.append(null_cohort)
            idx += size

        results.append({
            "permutation_id": perm_id,
            "null_episodes": null_episodes,
        })

    return results


def compute_null_support(
    null_episodes: list[frozenset],
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    """Count how often each cohort appears in null model episodes.

    Args:
        null_episodes: list of frozensets (participant groups)
        cohorts: DataFrame with 'members' column (space-separated IDs)

    Returns:
        DataFrame with null_support for each cohort
    """
    null_support = {}

    for idx, row in cohorts.iterrows():
        members = frozenset(map(int, row["members"].split()))
        count = sum(1 for ep in null_episodes if members == ep)
        null_support[idx] = count

    return pd.Series(null_support, name="null_support")


def evaluate_cohorts(
    episodes: pd.DataFrame,
    cohorts: pd.DataFrame,
    n_permutations: int = 100,
) -> pd.DataFrame:
    """Test each cohort against null model distribution.

    Computes:
    - observed_support: how many times cohort appears in real data
    - null_mean, null_std: mean and std of support in null models
    - z_score: (observed - null_mean) / null_std
    - p_value: fraction of null models with support >= observed (one-tailed)

    Args:
        episodes: DataFrame with participant_set column (arrays or strings)
        cohorts: DataFrame with members column (space-separated strings)
        n_permutations: number of random permutations

    Returns:
        DataFrame with significance metrics for each cohort
    """
    # Extract real participant sets (handle both numpy arrays and strings)
    real_episodes = []
    for p in episodes["participant_set"]:
        if isinstance(p, str):
            real_episodes.append(frozenset(map(int, p.split())))
        else:
            # numpy array or list
            real_episodes.append(frozenset(map(int, p)))

    # Compute observed support
    # Cohorts are "frequent itemsets": check if all members appear *within* an episode
    observed = {}
    for idx, row in cohorts.iterrows():
        members = frozenset(map(int, row["members"].split()))
        count = sum(1 for ep in real_episodes if members.issubset(ep))
        observed[idx] = count

    # Generate null models
    null_supports = {idx: [] for idx in cohorts.index}

    print(f"Generating {n_permutations} null models...")
    for perm_id in range(n_permutations):
        # Shuffle participant list
        all_participants = set()
        for ep in real_episodes:
            all_participants.update(ep)
        all_participants = sorted(list(all_participants))

        shuffled = np.random.RandomState(42 + perm_id).permutation(all_participants)

        # Assign shuffled participants to episode slots
        null_episodes = []
        idx = 0
        for ep in real_episodes:
            size = len(ep)
            null_cohort = frozenset(shuffled[idx : idx + size])
            null_episodes.append(null_cohort)
            idx += size

        # Count support in this null model (check if members appear within episode)
        for cohort_idx, row in cohorts.iterrows():
            members = frozenset(map(int, row["members"].split()))
            count = sum(1 for ep in null_episodes if members.issubset(ep))
            null_supports[cohort_idx].append(count)

        if (perm_id + 1) % 20 == 0:
            print(f"  {perm_id + 1}/{n_permutations}")

    # Compute statistics
    results = []
    for cohort_idx in cohorts.index:
        obs = observed[cohort_idx]
        nulls = np.array(null_supports[cohort_idx])

        null_mean = nulls.mean()
        null_std = nulls.std()

        if null_std > 0:
            z = (obs - null_mean) / null_std
        else:
            z = np.inf if obs > null_mean else 0

        # One-tailed p-value: what fraction of null models >= observed?
        p_value = (nulls >= obs).mean()

        results.append({
            "cohort_idx": cohort_idx,
            "observed_support": obs,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "p_value": p_value,
        })

    return pd.DataFrame(results).set_index("cohort_idx")


def main():
    """Run null model analysis on recurring cohorts."""
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Load clean episodes (anomalies removed)
    episodes = pd.read_parquet(OUTPUTS_DIR / "anomalies" / "episodes_normal_only.parquet")
    print(f"Loaded {len(episodes):,} clean episodes")

    # Load recurring cohorts
    cohorts = pd.read_csv(OUTPUTS_DIR / "tables" / "recurring_cohorts.csv")
    print(f"Loaded {len(cohorts)} recurring cohorts")
    print()

    # Test cohorts against null models
    results = evaluate_cohorts(episodes, cohorts, n_permutations=100)

    # Add cohort info
    results["members"] = cohorts.loc[results.index, "members"].values
    results["observed_support_from_data"] = cohorts.loc[results.index, "support"].values

    print("\n" + "="*80)
    print("NULL MODEL RESULTS")
    print("="*80)
    print(results[["members", "observed_support", "null_mean", "z_score", "p_value"]])

    # Summary statistics
    print(f"\nSignificant cohorts (p < 0.05): {(results['p_value'] < 0.05).sum()}")
    print(f"Highly significant (p < 0.01): {(results['p_value'] < 0.01).sum()}")
    print(f"Mean z-score: {results['z_score'].mean():.2f}")

    # Save results
    null_dir = OUTPUTS_DIR / "null_models"
    null_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(null_dir / "cohort_significance.csv")
    print(f"\n✓ Results saved to {null_dir}/cohort_significance.csv")


if __name__ == "__main__":
    main()
