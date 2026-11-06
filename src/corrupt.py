from __future__ import annotations

import numpy as np
import pandas as pd

# Intentionally remove data at value from data set.
def inject_missing_mcar(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                        columns: list[str] | None = None) -> pd.DataFrame:
    raise NotImplementedError


# Remove data but not at random, selecting more probable outliers rather than random points.
def inject_missing_mnar(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                        column: str, upper_tail: bool = True) -> pd.DataFrame:
    raise NotImplementedError


# Create false labels, as in, incorrect answers.
def flip_labels(y: pd.Series, rate: float, rng: np.random.Generator,
                symmetric: bool = True) -> pd.Series:
    raise NotImplementedError


# Introduce inaccurate measurements by adding Guassian noise.
def perturb_features(X: pd.DataFrame, rate: float, rng: np.random.Generator,
                     noise_scale: float = 1.0) -> pd.DataFrame:
    raise NotImplementedError
