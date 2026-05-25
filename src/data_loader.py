import pandas as pd
import os
import sys
sys.path.append('/content/findex-mlflow-pipeline')
from config import (RAW_FILE, FINDEX_FILE, TARGET_COLUMN, ALL_FEATURES,
                    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                    AFRICAN_COUNTRIES, INCLUSION_INDICATORS)


def load_and_process_raw(raw_path=RAW_FILE, save_path=FINDEX_FILE):
    """
    Load raw WB_FINDEX.csv, filter to African countries,
    clean, and save to data/processed/.
    """
    print(f"Loading raw file: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Raw shape: {df.shape}")

    # ── Filter to Africa ──────────────────────────────────────────────────────
    df = df[df['REF_AREA_LABEL'].isin(AFRICAN_COUNTRIES)].copy()
    print(f"After Africa filter: {df.shape}")

    # ── Filter to indicators if specified ────────────────────────────────────
    if INCLUSION_INDICATORS is not None:
        df = df[df['INDICATOR_LABEL'].isin(INCLUSION_INDICATORS)].copy()
        print(f"After indicator filter: {df.shape}")

    # ── Drop rows with missing target ─────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    print(f"Dropped {before - len(df):,} rows with missing OBS_VALUE")

    # ── Flag total-level rows ─────────────────────────────────────────────────
    df['is_total'] = (
        (df['SEX_LABEL'] == 'Total') &
        (df['AGE_LABEL'] == '15 years old and over') &
        (df['URBANISATION_LABEL'] == 'Total')
    ).astype(int)

    # ── Select columns ────────────────────────────────────────────────────────
    keep_cols = ALL_FEATURES + [TARGET_COLUMN, 'is_total']
    df = df[keep_cols].copy()

    # ── Clean TIME_PERIOD ─────────────────────────────────────────────────────
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df = df.dropna(subset=['TIME_PERIOD'])
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)

    df = df.reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\nFinal shape: {df.shape}")
    print(f"Countries:  {df['REF_AREA_LABEL'].nunique()}")
    print(f"Indicators: {df['INDICATOR_LABEL'].nunique()}")
    print(f"Years:      {sorted(df['TIME_PERIOD'].unique())}")
    print(f"\nOBS_VALUE stats:\n{df[TARGET_COLUMN].describe().round(2)}")
    print("\nData processing complete.")
    return df


def load_findex(path=FINDEX_FILE):
    """Load processed file, running raw pipeline if it doesn't exist."""
    if not os.path.exists(path):
        print("Processed file not found — running raw processing pipeline...")
        return load_and_process_raw()
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def get_available_features(df):
    """Returns which configured features are present in the dataframe."""
    return {
        "numeric": [f for f in NUMERIC_FEATURES if f in df.columns],
        "categorical": [f for f in CATEGORICAL_FEATURES if f in df.columns],
    }
