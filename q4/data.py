"""Exact synthetic data generation and deterministic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

JOB_IDS = tuple(f"J{i}" for i in range(1, 7))
JOB_FEATURES = np.array(
    [
        [38.0, 1.0, 2.0],
        [49.0, 9.0, 3.0],
        [20.0, 8.0, 3.0],
        [59.0, 8.0, 3.0],
        [50.0, 1.0, 4.0],
        [27.0, 10.0, 4.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class RegressionResult:
    features: np.ndarray
    target: np.ndarray
    predictions: np.ndarray
    mae: float
    r2: float
    coefficients: np.ndarray
    intercept: float


def generate_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate the 100 records with the question's exact RNG call order."""
    rng = np.random.default_rng(72); N = 100
    dist = rng.uniform(5, 60, N); load = rng.uniform(1, 10, N)
    traffic = rng.uniform(1, 5, N)
    time = 0.04*dist + 0.15*load + 0.30*traffic + 0.5 + rng.normal(0, 0.1, N)
    return np.column_stack((dist, load, traffic)), time


def train_and_predict() -> RegressionResult:
    """Evaluate on a fixed holdout, then refit all records for job predictions."""
    features, target = generate_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    holdout_model = LinearRegression().fit(x_train, y_train)
    holdout_prediction = holdout_model.predict(x_test)

    final_model = LinearRegression().fit(features, target)
    predictions = final_model.predict(JOB_FEATURES)
    if not np.all(np.isfinite(predictions)) or np.any(predictions <= 0):
        raise ValueError("All six delivery-time predictions must be finite and positive")

    return RegressionResult(
        features=features,
        target=target,
        predictions=predictions,
        mae=float(mean_absolute_error(y_test, holdout_prediction)),
        r2=float(r2_score(y_test, holdout_prediction)),
        coefficients=final_model.coef_.copy(),
        intercept=float(final_model.intercept_),
    )

