"""Anomaly detection on episodes data."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.features.episodes import build_episodes
from src.models.anomaly_methods import (
    run_isolation_forest,
    run_distance_based_knn,
    run_density_based_lof,
    run_kernel_one_class_svm,
    compare_methods,
)

PROJECT_ROOT = Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def main():
    # Load config and build episodes
    config = load_config("config/config.yaml")
    gatherings = pd.read_parquet(PROCESSED_DIR / "gatherings_v1.parquet")
    episodes = build_episodes(gatherings, config)

    print(f"Loaded {len(episodes):,} episodes")
    print(f"Columns: {episodes.columns.tolist()}\n")

    # Define numeric features for anomaly detection
    FEATURES = ["n_slots", "median_size", "median_rssi", "n_gap_slots", "night_share"]

    # Run Isolation Forest (recommended starting point)
    print("Running Isolation Forest (contamination=0.05)...")
    ranked_iforest = run_isolation_forest(episodes, FEATURES, contamination=0.05)
    print(f"Anomalies detected: {ranked_iforest['is_anomaly'].sum()} of {len(ranked_iforest)}\n")

    print("Top 20 most anomalous episodes:")
    display_cols = FEATURES + ["anomaly_score", "is_anomaly"]
    print(ranked_iforest[display_cols].head(20))

    # Compare multiple methods
    print("\n" + "="*60)
    print("Comparing all methods:")
    print("="*60)
    comparison = compare_methods(episodes, FEATURES, contamination=0.05)
    print(comparison)

    # Save results
    anomalies_dir = OUTPUTS_DIR / "anomalies"
    anomalies_dir.mkdir(parents=True, exist_ok=True)

    normal_only = ranked_iforest[~ranked_iforest["is_anomaly"]]
    anomalies_only = ranked_iforest[ranked_iforest["is_anomaly"]]

    ranked_iforest.to_parquet(anomalies_dir / "episodes_anomalies_iforest.parquet")
    normal_only.to_parquet(anomalies_dir / "episodes_normal_only.parquet")
    anomalies_only.to_parquet(anomalies_dir / "episodes_anomalies_only.parquet")
    comparison.to_csv(anomalies_dir / "method_comparison.csv")

    print(f"\n✓ Results saved to {anomalies_dir}/")
    print(f"  1. episodes_anomalies_iforest.parquet       ({len(ranked_iforest):,} rows, all with flags)")
    print(f"  2. episodes_normal_only.parquet             ({len(normal_only):,} rows, clean)")
    print(f"  3. episodes_anomalies_only.parquet          ({len(anomalies_only):,} rows, outliers only)")
    print(f"  4. method_comparison.csv")

if __name__ == "__main__":
    main()
