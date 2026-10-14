"""Run forecasting analysis on recurring cohorts."""
from pathlib import Path
import pandas as pd
from src.models.forecasting import (
    compute_cohort_recurrence,
    compute_temporal_stability,
    predict_next_encounter,
)

PROJECT_ROOT = Path.cwd()
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
print(f"  Analyzed {len(recurrence)} cohorts")

# Test temporal stability
print("Testing temporal stability...")
stability = compute_temporal_stability(episodes, cohorts)
print(f"  Analyzed {len(stability)} cohorts")

# Predict next encounters
print("Predicting next encounters...")
predictions = predict_next_encounter(episodes, cohorts)
print(f"  Predicted for {len(predictions)} cohorts")

# Display results
print("\n" + "="*80)
print("RECURRENCE METRICS - TOP 20 MOST FREQUENT COHORTS")
print("="*80)
top_recurrence = recurrence.nlargest(20, "recurrence_rate_per_day")
print(top_recurrence[["members", "total_sessions", "total_episodes", "time_span_days", "recurrence_rate_per_day", "median_gap_days"]])

print("\n" + "="*80)
print("TEMPORAL STABILITY - TOP 20 MOST STABLE COHORTS")
print("="*80)
stable = stability.nlargest(20, "chi_square")
print(f"Stable cohorts (low chi-square, consistent frequency): {(stability['chi_square'] < 1.0).sum()}")
print(f"Drifting cohorts (high chi-square, changing frequency): {(stability['chi_square'] >= 1.0).sum()}")
print()
print(stable[["members", "first_half_count", "second_half_count", "chi_square"]])

print("\n" + "="*80)
print("FORECAST PREDICTIONS - TOP 20 MOST CONFIDENT")
print("="*80)
top_predictions = predictions.nlargest(20, "confidence")
print(f"Highly confident (>0.8): {(predictions['confidence'] > 0.8).sum()}")
print(f"Moderately confident (0.5-0.8): {((predictions['confidence'] >= 0.5) & (predictions['confidence'] <= 0.8)).sum()}")
print(f"Low confidence (<0.5): {(predictions['confidence'] < 0.5).sum()}")
print()
print(top_predictions[["members", "median_gap_days", "predicted_next", "confidence"]])

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Mean recurrence rate (sessions/day): {recurrence['recurrence_rate_per_day'].mean():.2f}")
print(f"Median recurrence rate: {recurrence['recurrence_rate_per_day'].median():.2f}")
print(f"Mean session gap (days): {recurrence['median_gap_days'].mean():.4f}")
print(f"Median session gap (days): {recurrence['median_gap_days'].median():.4f}")
print(f"Median session gap (hours): {recurrence['median_gap_days'].median() * 24:.1f}")
print(f"Median session gap (minutes): {recurrence['median_gap_days'].median() * 24 * 60:.1f}")

# Save results
forecast_dir = OUTPUTS_DIR / "forecasting"
forecast_dir.mkdir(parents=True, exist_ok=True)

recurrence.to_csv(forecast_dir / "cohort_recurrence.csv")
stability.to_csv(forecast_dir / "cohort_stability.csv")
predictions.to_csv(forecast_dir / "next_encounter_predictions.csv")

print(f"\n✓ Results saved to {forecast_dir}/")
