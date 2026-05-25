import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = "/content/findex-mlflow-pipeline"
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports", "figures")
MODEL_DIR      = os.path.join(BASE_DIR, "app")

RAW_FILE       = "/content/WB_FINDEX.csv"
FINDEX_FILE    = os.path.join(DATA_PROCESSED, "findex_ssa_processed.csv")
MODEL_PKL      = os.path.join(MODEL_DIR, "model.pkl")
METADATA_JSON  = os.path.join(MODEL_DIR, "model_metadata.json")

# ── African countries ─────────────────────────────────────────────────────────
AFRICAN_COUNTRIES = [
    'Nigeria', 'Ghana', 'Kenya', 'South Africa', 'Ethiopia', 'Tanzania',
    'Uganda', 'Rwanda', 'Senegal', 'Cameroon', 'Zambia', 'Zimbabwe',
    'Mozambique', 'Mali', 'Burkina Faso', 'Niger', 'Chad', 'Madagascar',
    'Malawi', 'Angola', 'Togo', 'Benin', 'Botswana', 'Namibia', 'Gabon'
]

# ── Indicators — all 108 in African subset (no filter) ───────────────────────
# We use ALL indicators to maximise rows for modelling
# INDICATOR_LABEL becomes a categorical feature
INCLUSION_INDICATORS = None   # None = keep all indicators

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "OBS_VALUE"   # inclusion/behaviour rate 0–100

# ── Features ──────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "TIME_PERIOD",
]

CATEGORICAL_FEATURES = [
    "REF_AREA_LABEL",
    "SEX_LABEL",
    "AGE_LABEL",
    "URBANISATION_LABEL",
    "INDICATOR_LABEL",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "findex-financial-exclusion"
MLFLOW_MODEL_NAME      = "findex-exclusion-regressor"

# ── Model defaults ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
