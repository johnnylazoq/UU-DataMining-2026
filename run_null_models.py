"""Run null model analysis on recurring cohorts."""

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.null_models import evaluate_cohorts
import pandas as pd

def main():
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Load clean episodes (anomalies removed)
    episodes = pd.read_parquet(OUTPUTS_DIR / "anomalies" / "episodes_normal_only.parquet")
    print(f"Loaded {len(episodes):,} clean episodes")

    # Load recurring cohorts
    cohorts = pd.read_csv(OUTPUTS_DIR / "tables" / "recurring_cohorts.csv")
    print(f"Loaded {len(cohorts)} recurring cohorts\n")

    # Test cohorts against null models
    results = evaluate_cohorts(episodes, cohorts, n_permutations=50)

    # Add cohort info
    results["members"] = cohorts.loc[results.index, "members"].values
    results["support_from_data"] = cohorts.loc[results.index, "support"].values

    print("\n" + "="*80)
    print("NULL MODEL RESULTS - TOP 20 MOST SIGNIFICANT COHORTS")
    print("="*80)

    top_20 = results.sort_values("z_score", ascending=False).head(20)
    display_cols = ["members", "observed_support", "null_mean", "null_std", "z_score", "p_value"]
    for col in display_cols:
        if col in top_20.columns:
            print(f"{col:20s}", end=" ")
    print()
    print("-" * 80)

    for idx, row in top_20.iterrows():
        print(f"{row['members']:20s} {row['observed_support']:8.0f} {row['null_mean']:8.1f} {row['null_std']:8.1f} {row['z_score']:8.2f} {row['p_value']:8.4f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cohorts tested: {len(results)}")
    print(f"Significant (p < 0.05): {(results['p_value'] < 0.05).sum()}")
    print(f"Highly significant (p < 0.01): {(results['p_value'] < 0.01).sum()}")
    print(f"Mean observed support: {results['observed_support'].mean():.1f}")
    print(f"Mean null support: {results['null_mean'].mean():.1f}")
    print(f"Mean z-score: {results['z_score'].mean():.2f}")

    # Save results
    null_dir = OUTPUTS_DIR / "null_models"
    null_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(null_dir / "cohort_significance.csv")
    print(f"\n✓ Results saved to {null_dir}/cohort_significance.csv")

if __name__ == "__main__":
    main()
