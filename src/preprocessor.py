import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/content/findex-mlflow-pipeline')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def build_pipeline(model):
    """
    Build a full sklearn Pipeline wrapping preprocessing and a model.

    Args:
        model: Any sklearn-compatible estimator.

    Returns:
        sklearn Pipeline: preprocessor + model ready to fit.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    return pipeline


def get_X_y(df):
    """
    Split dataframe into features and target.

    Args:
        df: Processed Findex DataFrame.

    Returns:
        X: Feature DataFrame
        y: Target Series
    """
    available_features = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f in df.columns]
    missing = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f not in df.columns]
    if missing:
        print(f"Warning: missing features skipped: {missing}")

    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
