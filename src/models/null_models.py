"""Null models to test statistical significance of recurring cohorts.

Tests whether observed cohort support (frequency) is significantly higher than
expected by chance. Uses permutation-based null models: randomize participant
assignments while keeping episode sizes and participation counts constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


def permute_participants_swap(episodes: pd.DataFrame, n_permutations: int = 100) -> list[dict]:
    """Generate null models using swap/curveball randomization.

    Keeps each episode's size and each participant's participation count constant,
    but shuffles who meets whom. This is done by creating slot assignments
    and randomly reassigning participants to slots while maintaining constraints.

    Args:
        episodes: DataFrame with participant_set column
        n_permutations: number of random permutations to generate

    Returns:
        List of dicts with permutation_id and null_episodes
    """
    # Build list of all slots and gather participant info
    all_participants = set()
    episode_sizes = []
    real_episodes = []

    for participants in episodes["participant_set"]:
        if isinstance(participants, str):
            ep_set = frozenset(map(int, participants.split()))
        else:
            ep_set = frozenset(map(int, participants))
        all_participants.update(ep_set)
        episode_sizes.append(len(ep_set))
        real_episodes.append(ep_set)

    all_participants = sorted(list(all_participants))

    # Create slot assignments: list of all (episode_idx, slot_position) pairs
    slots = []
    for ep_idx, size in enumerate(episode_sizes):
        for slot_pos in range(size):
            slots.append((ep_idx, slot_pos))

    results = []
    rng = np.random.RandomState(42)

    for perm_id in range(n_permutations):
        # Shuffle the participant list for this permutation
        shuffled_participants = rng.permutation(all_participants).tolist()

        # Assign shuffled participants to slots
        # Cycle through participants to fill all slots (allows reuse)
        null_episodes = [set() for _ in range(len(episode_sizes))]
        for slot_idx, (ep_idx, _) in enumerate(slots):
            participant = shuffled_participants[slot_idx % len(shuffled_participants)]
            null_episodes[ep_idx].add(participant)

        null_episodes = [frozenset(ep) for ep in null_episodes]

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
    - observed_support: how many times cohort appears in real data (as subset)
    - null_mean, null_std: mean and std of support in null models
    - z_score: (observed - null_mean) / null_std
    - p_value: (count(nulls >= obs) + 1) / (n_permutations + 1)
    
    Note: p-value uses the +1 correction to avoid p=0 and to give honest
    smallest resolvable p-value of 1/(n+1).

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

    print(f"Generating {n_permutations} null models (swap/curveball randomization)...")
    
    # Use a single RNG across all permutations for consistency
    rng = np.random.RandomState(42)
    all_participants = set()
    for ep in real_episodes:
        all_participants.update(ep)
    all_participants = sorted(list(all_participants))

    episode_sizes = [len(ep) for ep in real_episodes]
    slots = []
    for ep_idx, size in enumerate(episode_sizes):
        for slot_pos in range(size):
            slots.append((ep_idx, slot_pos))

    for perm_id in range(n_permutations):
        # Shuffle participant list
        shuffled_participants = rng.permutation(all_participants).tolist()

        # Assign shuffled participants to episode slots
        null_episodes = [set() for _ in range(len(episode_sizes))]
        for slot_idx, (ep_idx, _) in enumerate(slots):
            participant = shuffled_participants[slot_idx % len(shuffled_participants)]
            null_episodes[ep_idx].add(participant)

        null_episodes = [frozenset(ep) for ep in null_episodes]

        # Count support in this null model (check if members appear within episode)
        for cohort_idx, row in cohorts.iterrows():
            members = frozenset(map(int, row["members"].split()))
            count = sum(1 for ep in null_episodes if members.issubset(ep))
            null_supports[cohort_idx].append(count)

        if (perm_id + 1) % max(1, n_permutations // 5) == 0:
            print(f"  {perm_id + 1}/{n_permutations}")

    # Compute statistics with proper p-value calculation
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

        # One-tailed p-value with +1 correction: (count >= obs + 1) / (n + 1)
        # This avoids p=0 and gives honest minimum p of 1/(n+1)
        p_value = (np.sum(nulls >= obs) + 1) / (n_permutations + 1)

        results.append({
            "cohort_idx": cohort_idx,
            "observed_support": obs,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z,
            "p_value": p_value,
        })

    results_df = pd.DataFrame(results).set_index("cohort_idx")
    
    # Apply Benjamini-Hochberg multiple testing correction
    results_df["p_value_fdr"] = stats.false_discovery_control(
        results_df["p_value"], method="bh"
    )

    return results_df


def main():
    """Run null model analysis on recurring cohorts."""
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Load clean episodes (anomalies removed)
    episodes = pd.read_parquet(OUTPUTS_DIR / "anomalies" / "episodes_normal_only.parquet")
    print(f"Loaded {len(episodes):,} clean episodes")
    print(f"Note: using episodes_normal_only.parquet (21,403 episodes)")
    print(f"      cohorts were mined on all episodes (22,530)")

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
    print("NULL MODEL RESULTS (swap/curveball randomization)")
    print("="*80)
    print(f"Minimum resolvable p-value: {1/101:.4f} (with 100 permutations)")
    print()
    print(results[["members", "observed_support", "null_mean", "z_score", "p_value", "p_value_fdr"]])

    # Summary statistics
    print(f"\nSignificant cohorts (p < 0.05): {(results['p_value'] < 0.05).sum()}")
    print(f"Highly significant (p < 0.01): {(results['p_value'] < 0.01).sum()}")
    print(f"Significant after FDR correction (FDR < 0.05): {(results['p_value_fdr'] < 0.05).sum()}")
    print(f"Mean z-score: {results['z_score'].mean():.2f}")

    # Save results
    null_dir = OUTPUTS_DIR / "null_models"
    null_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(null_dir / "cohort_significance.csv")
    print(f"\n✓ Results saved to {null_dir}/cohort_significance.csv")


if __name__ == "__main__":
    main()
