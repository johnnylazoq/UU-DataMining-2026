"""Run forecasting analysis on recurring cohorts."""

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forecasting import (
    compute_cohort_recurrence,
    compute_temporal_stability,
    predict_next_encounter,
)
import pandas as pd


def main():
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Load clean episodes
    episodes = pd.read_parquet(OUTPUTS_DIR / "anomalies" / "episodes_normal_only.parquet")
    print(f"Loaded {len(episodes):,} clean episodes")

    # Load recurring cohorts
    cohorts = pd.read_csv(OUTPUTS_DIR / "tables" / "recurring_cohorts.csv")
    print(f"Loaded {len(cohorts)} recurring cohorts\n")

    # Compute recurrence metrics
    print("Computing recurrence rates...")
    recurrence = compute_cohort_recurrence(episodes, cohorts)
    print(f"  Analyzed {len(recurrence)} cohorts\n")

    # Compute temporal stability
    print("Testing temporal stability...")
    stability = compute_temporal_stability(episodes, cohorts)
    print(f"  Analyzed {len(stability)} cohorts\n")

    # Predict next encounters
    print("Predicting next encounters...")
    predictions = predict_next_encounter(episodes, cohorts)
    print(f"  Predicted for {len(predictions)} cohorts\n")

    # Display results
    print("="*80)
    print("RECURRENCE METRICS - TOP 20 MOST FREQUENT COHORTS")
    print("="*80)
    top_recurrence = recurrence.sort_values("recurrence_rate_per_day", ascending=False).head(20)
    print(top_recurrence[["members", "total_occurrences", "time_span_days", "recurrence_rate_per_day", "mean_gap_days"]])

    print("\n" + "="*80)
    print("TEMPORAL STABILITY - TOP 20 MOST UNBALANCED COHORTS")
    print("="*80)
    unstable = stability.sort_values("chi_square", ascending=False).head(20)
    print(unstable[["members", "first_half_count", "second_half_count", "chi_square"]])

    print("\n" + "="*80)
    print("NEXT ENCOUNTER PREDICTIONS - TOP 20 MOST CONFIDENT")
    print("="*80)
    confident = predictions.sort_values("confidence", ascending=False).head(20)
    print(confident[["members", "last_occurrence", "median_gap_days", "days_until_next", "confidence"]])

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nRecurrence rates (per day):")
    print(f"  Mean: {recurrence['recurrence_rate_per_day'].mean():.3f}")
    print(f"  Median: {recurrence['recurrence_rate_per_day'].median():.3f}")
    print(f"  Std: {recurrence['recurrence_rate_per_day'].std():.3f}")

    print(f"\nInter-occurrence gaps (days):")
    print(f"  Mean gap: {recurrence['mean_gap_days'].mean():.2f}")
    print(f"  Median gap: {recurrence['median_gap_days'].median():.2f}")
    print(f"  Min gap: {recurrence['min_gap_days'].min():.2f}")
    print(f"  Max gap: {recurrence['max_gap_days'].max():.2f}")

    print(f"\nTemporal stability:")
    print(f"  Mean chi-square: {stability['chi_square'].mean():.2f}")
    print(f"  Cohorts with stable frequency (chi < 1): {(stability['chi_square'] < 1).sum()}")
    print(f"  Cohorts with drift (chi > 5): {(stability['chi_square'] > 5).sum()}")

    print(f"\nForecastability:")
    print(f"  Mean confidence: {predictions['confidence'].mean():.3f}")
    print(f"  High confidence (> 0.5): {(predictions['confidence'] > 0.5).sum()}")

    # Save results
    forecast_dir = OUTPUTS_DIR / "forecasting"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    recurrence.to_csv(forecast_dir / "cohort_recurrence.csv")
    stability.to_csv(forecast_dir / "cohort_stability.csv")
    predictions.to_csv(forecast_dir / "next_encounter_predictions.csv")

    print(f"\n✓ Results saved to {forecast_dir}/")
    print(f"  - cohort_recurrence.csv")
    print(f"  - cohort_stability.csv")
    print(f"  - next_encounter_predictions.csv")


if __name__ == "__main__":
    main()
