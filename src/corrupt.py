from __future__ import annotations

import numpy as np
import pandas as pd

# Check that a corruption rate is between 0 and 1 (helper function used by all corruption functions)
def _check_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")


# Missing Completely At Random: in each chosen column, blank out a random `rate`
# fraction of the values that are currently present. Which cells go missing has
# nothing to do with their values.
def inject_missing_mcar(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                        columns: list[str] | None = None) -> pd.DataFrame:
    _check_rate(rate)
    X = X.copy()
    columns = list(X.columns) if columns is None else columns

    for col in columns:
        observed = np.flatnonzero(X[col].notna().to_numpy())
        num_remove = int(round(rate * len(observed)))
        if num_remove == 0:
            continue
        rows = rng.choice(observed, size=num_remove, replace=False)
        X.iloc[rows, X.columns.get_loc(col)] = np.nan
    return X


# Missing Not At Random: blank out a `rate` fraction of the values in one numeric
# column, where larger values (or smaller ones, if upper_tail=False) are more
# likely to be removed. Each value's chance of being picked is proportional to
# its rank, so the extremes go missing most often, but not only the extremes.
def inject_missing_mnar(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                        column: str, upper_tail: bool = True) -> pd.DataFrame:
    _check_rate(rate)
    if not pd.api.types.is_numeric_dtype(X[column]):
        raise TypeError(f"MNAR needs a numeric column; {column!r} is {X[column].dtype}")
    X = X.copy()

    values = X[column]
    observed = np.flatnonzero(values.notna().to_numpy())
    num_remove = int(round(rate * len(observed)))
    if num_remove == 0:
        return X

    # Rank 1 = least likely to be removed, rank n = most likely
    ranks = values.iloc[observed].rank(method="average",
                                       ascending=upper_tail).to_numpy()
    weights = ranks / ranks.sum()
    rows = rng.choice(observed, size=num_remove, replace=False, p=weights)
    X.iloc[rows, X.columns.get_loc(column)] = np.nan
    return X


# Create false labels: change a `rate` fraction of labels to a wrong class.
#   symmetric=True  -> any row can be flipped; each picked label is changed to a
#                      different class chosen uniformly (for 0/1 labels: 0<->1).
#   symmetric=False -> only positive labels (1) are flipped to 0, i.e. a `rate`
#                      fraction of the real positives get mislabeled as negatives
#                      (like failures or high earners that were never recorded).
def flip_labels(y: pd.Series, rate: float, rng: np.random.Generator,
                symmetric: bool = True) -> pd.Series:
    _check_rate(rate)
    y = y.copy()
    classes = np.unique(y.to_numpy())

    if symmetric:
        if len(classes) < 2:
            raise ValueError("need at least two classes to flip labels")
        candidates = np.arange(len(y))
    else:
        if set(classes) - {0, 1}:
            raise ValueError("symmetric=False expects 0/1 labels")
        candidates = np.flatnonzero(y.to_numpy() == 1)

    num_flip = int(round(rate * len(candidates)))
    if num_flip == 0:
        return y
    rows = rng.choice(candidates, size=num_flip, replace=False)

    if symmetric:
        old = y.iloc[rows].to_numpy()
        # Pick a random offset 1..k-1 so the new class is never the old one
        old_idx = np.searchsorted(classes, old)
        offsets = rng.integers(1, len(classes), size=num_flip)
        new = classes[(old_idx + offsets) % len(classes)]
    else:
        new = np.zeros(num_flip, dtype=y.dtype)

    y.iloc[rows] = new
    return y


# Introduce inaccurate measurements: in each numeric column, add Gaussian noise
# to a random `rate` fraction of the present values. The noise standard deviation
# is noise_scale times that column's standard deviation, so columns measured in
# different units get comparable amounts of noise. Text columns are left alone.
def perturb_features(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                     noise_scale: float = 1.0) -> pd.DataFrame:
    _check_rate(rate)
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    X = X.copy()
    numeric_cols = X.select_dtypes(include="number").columns

    for col in numeric_cols:
        observed = np.flatnonzero(X[col].notna().to_numpy())
        num_noisy = int(round(rate * len(observed)))
        std = X[col].std()
        if num_noisy == 0 or not np.isfinite(std) or std == 0:
            continue
        rows = rng.choice(observed, size=num_noisy, replace=False)
        # Integer columns become float once noise is added
        X[col] = X[col].astype(float)
        noise = rng.normal(0.0, noise_scale * std, size=num_noisy)
        X.iloc[rows, X.columns.get_loc(col)] += noise
    return X
