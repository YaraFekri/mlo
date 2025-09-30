
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def process_data(X, categorical_features=[], label=None, training=True, encoder=None, lb=None):
    """
    Processes the data for the machine learning model.

    Args:
        X (pd.DataFrame): The input features.
        categorical_features (list): A list of column names that are categorical.
        label (str): The name of the target column. Defaults to None.
        training (bool): Whether the data is for training (True) or inference (False). Defaults to True.
        encoder (sklearn.preprocessing.OneHotEncoder): Pre-fitted OneHotEncoder. Defaults to None.
        lb (sklearn.preprocessing.LabelBinarizer): Pre-fitted LabelBinarizer. Defaults to None.

    Returns:
        tuple: Processed features (X), optional label (y), fitted encoder, and fitted label binarizer.
    """
    X = X.copy()

    if label:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = pd.Series()

    # Clean column names and string data
    X.columns = X.columns.str.strip()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].str.strip()

    # Replace "?" with NaN and handle missing values (simple imputation for now, can be improved)
    X = X.replace("?", np.nan)
    for col in categorical_features:
        if col in X.columns and X[col].isnull().any():
            # Impute with mode, handle case where mode might be empty (e.g., all NaNs)
            mode_val = X[col].mode()
            if not mode_val.empty:
                X[col] = X[col].fillna(mode_val[0])
            else:
                # If mode is empty, fill with a placeholder or drop, depending on strategy
                # For now, let's fill with a common placeholder like 'Unknown'
                X[col] = X[col].fillna("Unknown")

    if training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_categorical = encoder.fit_transform(X[categorical_features])
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel()
    else:
        X_categorical = encoder.transform(X[categorical_features])
        y = lb.transform(y).ravel() if label else y

    X_numerical = X.drop(columns=categorical_features).values
    X = np.concatenate([X_numerical, X_categorical], axis=1)
    return X, y, encoder, lb

def load_data(path):
    """
    Loads the census dataset from the specified path.
    """
    df = pd.read_csv(path)
    return df


