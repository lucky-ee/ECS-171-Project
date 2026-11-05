import re
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def clean_column_names(df):
    # XGBoost does not allow the characters [ ] < in column names,
    # e.g. AI4I has "Air temperature [K]" -> "Air_temperature_K"
    new_names = []
    for name in df.columns:
        name = re.sub(r"[\[\]<>]", "", name)          # remove [ ] < >
        name = name.strip().replace(" ", "_").replace("-", "_")
        new_names.append(name)
    df.columns = new_names
    return df


# Dataset 1: Adult Income (UCI id 2)
# Task: predict whether a person earns more than $50K per year
def load_adult():
    # Download the dataset from the UCI repository
    df = fetch_ucirepo(id=2).data.original.copy()

    # Remove extra spaces around text values (" Private" -> "Private")
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    # This dataset marks missing values with "?", so replace them with NaN
    df = df.replace("?", np.nan)

    # Some labels have a trailing period (">50K." instead of ">50K"),
    # which would make them look like a different class, so remove it
    df["income"] = df["income"].str.rstrip(".")

    # Target variable: 1 if income > 50K, else 0
    y = (df["income"] == ">50K").astype(int)

    # Input features: drop the target and two unhelpful columns
    #   education -> same information as education-num
    #   fnlwgt    -> census sampling weight, not a property of the person
    X = df.drop(columns=["income", "education", "fnlwgt"])

    # Remove duplicate rows
    is_duplicate = X.duplicated()
    X = X[~is_duplicate].reset_index(drop=True)
    y = y[~is_duplicate].reset_index(drop=True)

    X = clean_column_names(X)
    return X, y

# Dataset 2: AI4I 2020 Predictive Maintenance (UCI id 601)
# Task: predict whether a machine fails
def load_ai4i():
    # Download the dataset from the UCI repository
    df = fetch_ucirepo(id=601).data.original.copy()
    df = clean_column_names(df)

    # Target variable: 1 if the machine failed, else 0
    y = df["Machine_failure"].astype(int)

    # Input features: drop the target and columns the model should not see
    #   UDI, Product_ID         -> ID numbers, they do not predict failure
    #   TWF, HDF, PWF, OSF, RNF -> the TYPE of failure; these give away the
    #                              answer (data leakage), so they must be removed
    X = df.drop(columns=["UDI", "Product_ID", "Machine_failure",
                         "TWF", "HDF", "PWF", "OSF", "RNF"], errors="ignore")
    return X, y


# Train/test split
def split(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test


# Preprocessing
def make_preprocessor(X_train, mode):
    # Split column names into categorical (text) and numeric
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    if mode == "impute":
        numeric_steps = make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler())
        categorical_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                          OneHotEncoder(handle_unknown="ignore"))
    elif mode == "native":
        numeric_steps = "passthrough"   # leave numeric columns unchanged
        categorical_steps = OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=np.nan,
                                           encoded_missing_value=np.nan)
    else:
        raise ValueError("mode must be 'impute' or 'native'")

    # ColumnTransformer applies different steps to different columns
    preprocessor = ColumnTransformer([
        ("numeric", numeric_steps, numeric_cols),
        ("categorical", categorical_steps, categorical_cols),
    ])
    return preprocessor


if __name__ == "__main__":
    for name, load_function in [("Adult", load_adult), ("AI4I", load_ai4i)]:
        X, y = load_function()
        X_train, X_test, y_train, y_test = split(X, y)

        print(f"\n===== {name} =====")
        print("Rows, columns:", X.shape)
        print("Train size:", len(X_train), " Test size:", len(X_test))
        print(f"Fraction of positives: {y.mean():.3f}")

        missing_counts = X.isna().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "  none")