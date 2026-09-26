"""Tests for the shared EDA report/transform helpers."""

import pandas as pd

from src.features.eda import (
    apply_missing_strategy,
    drop_duplicate_rows,
    drop_symmetric_duplicates,
    duplicate_report,
    flag_outliers_iqr,
    missingness_report,
    outlier_report_iqr,
    scale_numeric_columns,
    symmetric_duplicate_report,
)


def test_duplicate_report_and_drop():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    report = duplicate_report(df)
    assert report["n_duplicate_rows"] == 1

    deduped = drop_duplicate_rows(df)
    assert len(deduped) == 2


def test_symmetric_duplicate_report_and_drop():
    df = pd.DataFrame({"user_a": [0, 512, 3], "user_b": [512, 0, 3]})
    report = symmetric_duplicate_report(df, "user_a", "user_b")
    assert report["n_self_loops"] == 1
    assert report["n_duplicate_edges"] == 1

    deduped = drop_symmetric_duplicates(df, "user_a", "user_b")
    assert len(deduped) == 1
    assert deduped.iloc[0][["user_a", "user_b"]].tolist() == [0, 512]


def test_missingness_report():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missingness_report(df)
    assert report.loc["a", "n_missing"] == 1
    assert report.loc["b", "n_missing"] == 0


def test_apply_missing_strategy_variants():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})

    assert "a" not in apply_missing_strategy(df, "a", "drop_column").columns

    imputed_median = apply_missing_strategy(df, "a", "impute_median")
    assert imputed_median["a"].isna().sum() == 0
    assert imputed_median.loc[1, "a"] == 2.0

    flagged = apply_missing_strategy(df, "a", "flag_missing")
    assert flagged["a_missing"].tolist() == [False, True, False]


def test_outlier_report_and_flag():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})
    report = outlier_report_iqr(df, "x")
    assert report["n_outliers"] == 1

    flagged = flag_outliers_iqr(df, "x")
    assert flagged["x_outlier"].tolist() == [False, False, False, False, True]


def test_scale_numeric_columns_standard():
    df = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
    scaled, scaler = scale_numeric_columns(df, ["x"], method="standard")
    assert "x_scaled" in scaled.columns
    assert abs(scaled["x_scaled"].mean()) < 1e-9
    assert scaler is not None


def test_scale_numeric_columns_robust_ignores_outlier_magnitude():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    scaled, scaler = scale_numeric_columns(df, ["x"], method="robust")
    assert "x_scaled" in scaled.columns
    assert scaled.loc[2, "x_scaled"] == 0.0
    assert scaler is not None
