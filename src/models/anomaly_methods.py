"""Six families of anomaly detection methods behind one common interface.

Every ``run_*`` function returns a copy of the input with two added columns and
is ranked so that Rank 1 is the most anomalous row:

  - anomaly_score: higher = more anomalous (same direction for every method)
  - is_anomaly:    True if the row is above the method's threshold

This makes the methods directly comparable, e.g. with ``evaluate_metrics`` or
``compare_methods``.

Note: ``src/models/anomaly.py::rank_anomalies`` uses scikit-learn's own
Isolation Forest convention (lower score = more anomalous). Here the score is
flipped so that all six methods point the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _check_input(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Fail early with a clear message instead of a deep sklearn traceback."""
    if df.empty:
        raise ValueError("Input data is empty.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if df[list(columns)].isna().any().any():
        raise ValueError(f"Columns {list(columns)} contain missing values.")


def _scaled(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Standardise features so that no single column dominates the distances.

    Needed for kNN, LOF and One-Class SVM: without it, a feature measured in
    thousands (e.g. contact records) swamps one measured in tens (e.g. RSSI).
    """
    return StandardScaler().fit_transform(df[list(feature_cols)])


def _finalize(
    df: pd.DataFrame,
    scores: np.ndarray,
    is_anomaly: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Attach scores, sort (most anomalous first) and add a 1-based rank index."""
    result = df.copy()
    for name, values in (extra or {}).items():
        result[name] = values
    result["anomaly_score"] = scores
    result["is_anomaly"] = np.asarray(is_anomaly, dtype=bool)
    result = result.sort_values("anomaly_score", ascending=False, kind="stable")
    result = result.reset_index(drop=True)
    result.index = pd.RangeIndex(1, len(result) + 1, name="rank")
    return result


def _top_fraction(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Flag the top ``contamination`` share of scores as anomalies."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    return scores > np.quantile(scores, 1 - contamination)


# ---------------------------------------------------------------------------
# 1. Statistical
# ---------------------------------------------------------------------------


def run_statistical_zscore(
    df: pd.DataFrame,
    value_col: str,
    threshold: float = 3.0,
    robust: bool = False,
) -> pd.DataFrame:
    """1. Statistical (univariate Gaussian): flag rows where |z| > threshold.

    With ``robust=True`` the median and MAD are used instead of mean and
    standard deviation. Extreme values inflate the standard deviation and can
    hide themselves ("masking"); the median/MAD version is not affected.
    """
    _check_input(df, [value_col])
    values = df[value_col].astype(float)

    if robust:
        center = values.median()
        # 1.4826 makes the MAD comparable to a standard deviation for normal data
        spread = 1.4826 * (values - center).abs().median()
    else:
        center = values.mean()
        spread = values.std()

    if spread == 0 or np.isnan(spread):
        # All values (nearly) identical: nothing stands out.
        z = np.zeros(len(values))
    else:
        z = ((values - center) / spread).to_numpy()

    scores = np.abs(z)
    return _finalize(df, scores, scores > threshold, extra={"z_score": z})


# ---------------------------------------------------------------------------
# 2-3. Proximity based
# ---------------------------------------------------------------------------


def run_distance_based_knn(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 5,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """2. Distance based (kNN): score = distance to the k-th nearest neighbour.

    Each point is its own nearest neighbour (distance 0) when fitting and
    querying the same data, so k + 1 neighbours are requested and the point
    itself is skipped. The top ``contamination`` share is flagged.
    """
    _check_input(df, feature_cols)
    if k >= len(df):
        raise ValueError(f"k={k} must be smaller than the number of rows ({len(df)}).")

    X = _scaled(df, feature_cols)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    scores = distances[:, -1]
    return _finalize(df, scores, _top_fraction(scores, contamination))


def run_density_based_lof(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """3. Density based (LOF): compares each point's local density to its neighbours'.

    LOF ~ 1 means as dense as the neighbours; clearly above 1 means sparser,
    i.e. an outlier. The top ``contamination`` share is flagged.
    """
    _check_input(df, feature_cols)
    n_neighbors = min(n_neighbors, len(df) - 1)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit(_scaled(df, feature_cols))
    # negative_outlier_factor_ is lower for outliers; negate so higher = anomalous
    scores = -lof.negative_outlier_factor_
    return _finalize(df, scores, _top_fraction(scores, contamination))


# ---------------------------------------------------------------------------
# 4. Isolation based
# ---------------------------------------------------------------------------


def run_isolation_forest(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    contamination: float | str = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """4. Isolation based (iForest): outliers are isolated in fewer random splits.

    Tree based, so the features do not need scaling.
    """
    _check_input(df, feature_cols)
    X = df[list(feature_cols)]

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    ).fit(X)
    scores = -model.decision_function(X)
    return _finalize(df, scores, model.predict(X) == -1)


# ---------------------------------------------------------------------------
# 5. Kernel based
# ---------------------------------------------------------------------------


def run_kernel_one_class_svm(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    nu: float = 0.1,
    gamma: str | float = "scale",
) -> pd.DataFrame:
    """5. Kernel based (One-Class SVM): learns a boundary around the normal data.

    ``nu`` is an upper bound on the share of points treated as outliers.
    Very small ``nu`` (e.g. 0.05) with the default kernel width can make all
    scores collapse to about 0, which ruins the ranking; if that happens,
    raise ``nu`` or use a smaller ``gamma`` (e.g. 0.1) for a wider kernel.
    RBF kernels are distance based, so the features are standardised first.
    Training is roughly quadratic in the number of rows, so sample first on
    large tables (tens of thousands of rows and up).
    """
    _check_input(df, feature_cols)
    X = _scaled(df, feature_cols)

    model = OneClassSVM(nu=nu, gamma=gamma).fit(X)
    scores = -model.decision_function(X)
    return _finalize(df, scores, model.predict(X) == -1)


# ---------------------------------------------------------------------------
# 6. Prediction based
# ---------------------------------------------------------------------------


def run_prediction_based(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    method: str = "rolling",
    window: int = 12,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """6. Prediction based (time series): score = size of the forecast error.

    ``method="rolling"``: predict each value as the median of the previous
    ``window`` values. Only past data is used, and a single spike does not
    drag the prediction along with it.
    ``method="linear"``: fit a straight trend line over time (only suitable
    for series without seasonality).

    The absolute error is divided by a robust spread (MAD) of all errors, so
    ``threshold`` is on the same "number of standard deviations" scale as the
    z-score method.
    """
    _check_input(df, [time_col, value_col])
    data = df.sort_values(time_col, kind="stable")
    values = data[value_col].astype(float)

    if method == "rolling":
        predicted = values.shift(1).rolling(window, min_periods=1).median()
        predicted = predicted.fillna(values.iloc[0])
    elif method == "linear":
        model = LinearRegression().fit(data[[time_col]], values)
        predicted = pd.Series(model.predict(data[[time_col]]), index=data.index)
    else:
        raise ValueError("method must be 'rolling' or 'linear'.")

    residual = values - predicted
    spread = 1.4826 * (residual - residual.median()).abs().median()
    scores = residual.abs() / spread if spread > 0 else residual.abs()

    return _finalize(
        data,
        scores.to_numpy(),
        (scores > threshold).to_numpy(),
        extra={
            "predicted_value": predicted.to_numpy(),
            "residual": residual.to_numpy(),
        },
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_metrics(
    y_true: Sequence[int] | Sequence[bool],
    scores: Sequence[float],
    y_pred: Sequence[int] | Sequence[bool] | None = None,
) -> dict[str, float]:
    """Evaluate anomaly detection against ground-truth labels (1/True = anomaly).

    Score based (no threshold needed; ``scores`` must be higher = more anomalous):
      - AUC_ROC, AUC_PR (average precision; more informative when anomalies are rare)
    Threshold based (only if ``y_pred`` is given):
      - precision, recall, F1, FPR (false positive rate)
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")

    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) == 2:
        metrics["AUC_ROC"] = roc_auc_score(y_true, scores)
        metrics["AUC_PR"] = average_precision_score(y_true, scores)
    else:
        # Both AUCs are undefined when y_true contains only one class.
        metrics["AUC_ROC"] = metrics["AUC_PR"] = float("nan")

    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
        # labels=[0, 1] keeps the matrix 2x2 even if one class is absent
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def compare_methods(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str | None = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Run the four multivariate methods and summarise them in one table.

    With ``label_col`` (ground truth, 1 = anomaly) each method is scored with
    ``evaluate_metrics``. Without labels, the table shows how many rows each
    method flags. The z-score and prediction methods are left out because
    they work on one column and on a time series respectively.
    """
    runs = {
        "kNN": run_distance_based_knn(df, feature_cols, contamination=contamination),
        "LOF": run_density_based_lof(df, feature_cols, contamination=contamination),
        "IsolationForest": run_isolation_forest(
            df, feature_cols, contamination=contamination
        ),
        "OneClassSVM": run_kernel_one_class_svm(df, feature_cols, nu=contamination),
    }
    rows = []
    for name, result in runs.items():
        row: dict[str, float | str] = {
            "method": name,
            "n_flagged": int(result["is_anomaly"].sum()),
        }
        if label_col is not None:
            row.update(
                evaluate_metrics(
                    result[label_col], result["anomaly_score"], result["is_anomaly"]
                )
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")
