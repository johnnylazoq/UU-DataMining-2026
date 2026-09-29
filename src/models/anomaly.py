"""
Rank data points by how anomalous they are, using Isolation Forest.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


def rank_anomalies(
    data: pd.DataFrame,
    contamination: float | str = "auto",
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit an Isolation Forest and return a ranked table where the most
    anomalous point is at Rank 1.

    Columns added to the result:
      - anomaly_score: decision_function (negative = anomaly, lower = more anomalous)
      - is_anomaly:    True if the model classifies the point as an anomaly
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(data)

    result = data.copy()
    result["anomaly_score"] = model.decision_function(data)
    result["is_anomaly"] = model.predict(data) == -1

    result = result.sort_values("anomaly_score", kind="stable").reset_index(drop=True)
    result.index = pd.RangeIndex(1, len(result) + 1, name="rank")
    return result
