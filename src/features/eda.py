"""Reusable EDA report/transform helpers shared by the per-dataset notebooks.

Report functions never mutate their input. Transform functions return a new
dataframe so a preprocessing decision made in a notebook stays explicit and
reusable instead of being re-implemented per notebook.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    """Count exact duplicate rows (or duplicates over `subset` columns)."""
    n_duplicates = int(df.duplicated(subset=subset).sum())
    n_rows = len(df)
    return {
        "n_rows": n_rows,
        "n_duplicate_rows": n_duplicates,
        "pct_duplicate_rows": round(100 * n_duplicates / n_rows, 4) if n_rows else 0.0,
    }


def drop_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def _canonical_pairs(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    lo = df[[col_a, col_b]].min(axis=1)
    hi = df[[col_a, col_b]].max(axis=1)
    return lo.astype(str) + "_" + hi.astype(str)


def symmetric_duplicate_report(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Report duplicate and self-loop edges in an undirected edge list."""
    n_self_loops = int((df[col_a] == df[col_b]).sum())
    n_duplicate_edges = int(_canonical_pairs(df, col_a, col_b).duplicated().sum())
    return {
        "n_edges": len(df),
        "n_self_loops": n_self_loops,
        "n_duplicate_edges": n_duplicate_edges,
    }


def drop_symmetric_duplicates(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    drop_self_loops: bool = True,
) -> pd.DataFrame:
    """Deduplicate an undirected edge list and optionally drop self-loops."""
    result = df.copy()
    if drop_self_loops:
        result = result[result[col_a] != result[col_b]]
    canonical = _canonical_pairs(result, col_a, col_b)
    return result.loc[~canonical.duplicated()].reset_index(drop=True)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values."""
    n_rows = len(df)
    n_missing = df.isna().sum()
    pct_missing = (100 * n_missing / n_rows).round(4) if n_rows else n_missing.astype(float)
    return pd.DataFrame(
        {"n_missing": n_missing, "pct_missing": pct_missing}
    ).sort_values("n_missing", ascending=False)


def apply_missing_strategy(
    df: pd.DataFrame,
    column: str,
    strategy: Literal["drop_column", "impute_median", "impute_mode", "flag_missing"],
) -> pd.DataFrame:
    """Apply one documented missing-data decision to a single column."""
    result = df.copy()
    if strategy == "drop_column":
        return result.drop(columns=[column])
    if strategy == "impute_median":
        result[column] = result[column].fillna(result[column].median())
        return result
    if strategy == "impute_mode":
        mode = result[column].mode(dropna=True)
        if not mode.empty:
            result[column] = result[column].fillna(mode.iloc[0])
        return result
    if strategy == "flag_missing":
        result[f"{column}_missing"] = result[column].isna()
        return result
    raise ValueError(f"Unknown missing-data strategy: {strategy!r}")


def outlier_report_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> dict:
    """IQR-based outlier bounds and count for one numeric column."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    n_outliers = int(((df[column] < lower) | (df[column] > upper)).sum())
    n_rows = len(df)
    return {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": n_outliers,
        "pct_outliers": round(100 * n_outliers / n_rows, 4) if n_rows else 0.0,
    }


def flag_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """Add a boolean `<column>_outlier` flag instead of silently dropping rows."""
    bounds = outlier_report_iqr(df, column, k)
    result = df.copy()
    result[f"{column}_outlier"] = (result[column] < bounds["lower_bound"]) | (
        result[column] > bounds["upper_bound"]
    )
    return result


def scale_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: Literal["standard", "minmax", "robust"] = "standard",
    suffix: str = "_scaled",
):
    """Scale numeric columns into new `<col><suffix>` columns.

    `method="robust"` centers on the median and scales by the IQR instead of
    the mean/std, so it isn't skewed by outliers that were flagged but kept
    (see `flag_outliers_iqr`) rather than dropped.

    Returns `(dataframe_with_scaled_columns, fitted_scaler)`.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    scaler = scalers[method]()
    result = df.copy()
    scaled = scaler.fit_transform(result[columns])
    for i, column in enumerate(columns):
        result[f"{column}{suffix}"] = scaled[:, i]
    return result, scaler
