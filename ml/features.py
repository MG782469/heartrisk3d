"""Feature definitions, encoding and modality groups for the Z-Alizadeh Sani (extension) dataset.

Single source of truth for:
  * which raw columns are model inputs (and which are targets -> never inputs),
  * how each raw column is encoded to a number,
  * which modality (history, symptoms/exam, ECG, labs, echo) a feature belongs to,
  * display metadata (label, unit, normal range, modifiable or not) used by the web app.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Targets. These columns must NEVER be used as model inputs (problem statement, requirement 1d).
TARGETS = {
    "CAD": ("Cath", "CAD"),
    "LAD": ("LAD", "Stenotic"),
    "LCX": ("LCX", "Stenotic"),
    "RCA": ("RCA", "Stenotic"),
}
LEAKAGE_COLUMNS = ["Cath", "LAD", "LCX", "RCA"]

# Raw columns that are constant in the dataset and carry no information.
DROPPED_CONSTANT = ["Exertional CP"]

YN = {"Y": 1, "N": 0}

# name, raw column, type, modality, label, unit, normal (lo, hi) or None, modifiable
# type: "binary" | "numeric" | "ordinal" | "onehot:<value>"
FEATURES = [
    # ---- history / demographics
    ("Age", "Age", "numeric", "history", "Age", "years", None, False),
    ("Sex", "Sex", "binary", "history", "Male sex", "", None, False),
    ("Weight", "Weight", "numeric", "history", "Weight", "kg", None, True),
    ("Length", "Length", "numeric", "history", "Height", "cm", None, False),
    ("BMI", "BMI", "numeric", "history", "BMI", "kg/m2", (18.5, 24.9), True),
    ("DM", "DM", "binary", "history", "Diabetes mellitus", "", None, False),
    ("HTN", "HTN", "binary", "history", "Hypertension", "", None, False),
    ("Current Smoker", "Current Smoker", "binary", "history", "Current smoker", "", None, True),
    ("EX-Smoker", "EX-Smoker", "binary", "history", "Ex-smoker", "", None, False),
    ("FH", "FH", "binary", "history", "Family history of CAD", "", None, False),
    ("Obesity", "Obesity", "binary", "history", "Obesity", "", None, False),
    ("CRF", "CRF", "binary", "history", "Chronic renal failure", "", None, False),
    ("CVA", "CVA", "binary", "history", "Previous stroke (CVA)", "", None, False),
    ("Airway disease", "Airway disease", "binary", "history", "Airway disease", "", None, False),
    ("Thyroid Disease", "Thyroid Disease", "binary", "history", "Thyroid disease", "", None, False),
    ("CHF", "CHF", "binary", "history", "Congestive heart failure", "", None, False),
    ("DLP", "DLP", "binary", "history", "Dyslipidemia", "", None, False),
    # ---- symptoms and physical examination
    ("BP", "BP", "numeric", "symptoms", "Systolic blood pressure", "mmHg", (90, 120), True),
    ("PR", "PR", "numeric", "symptoms", "Pulse rate", "/min", (60, 100), False),
    ("Edema", "Edema", "binary", "symptoms", "Edema", "", None, False),
    ("Weak Peripheral Pulse", "Weak Peripheral Pulse", "binary", "symptoms", "Weak peripheral pulse", "", None, False),
    ("Lung rales", "Lung rales", "binary", "symptoms", "Lung rales", "", None, False),
    ("Systolic Murmur", "Systolic Murmur", "binary", "symptoms", "Systolic murmur", "", None, False),
    ("Diastolic Murmur", "Diastolic Murmur", "binary", "symptoms", "Diastolic murmur", "", None, False),
    ("Typical Chest Pain", "Typical Chest Pain", "binary", "symptoms", "Typical chest pain", "", None, False),
    ("Dyspnea", "Dyspnea", "binary", "symptoms", "Dyspnea", "", None, False),
    ("Function Class", "Function Class", "ordinal", "symptoms", "Functional class (0-3)", "", None, False),
    ("Atypical", "Atypical", "binary", "symptoms", "Atypical chest pain", "", None, False),
    ("Nonanginal", "Nonanginal", "binary", "symptoms", "Non-anginal chest pain", "", None, False),
    ("LowTH Ang", "LowTH Ang", "binary", "symptoms", "Low-threshold angina", "", None, False),
    # ---- ECG
    ("Q Wave", "Q Wave", "binary", "ecg", "Q wave", "", None, False),
    ("St Elevation", "St Elevation", "binary", "ecg", "ST elevation", "", None, False),
    ("St Depression", "St Depression", "binary", "ecg", "ST depression", "", None, False),
    ("Tinversion", "Tinversion", "binary", "ecg", "T-wave inversion", "", None, False),
    ("LVH", "LVH", "binary", "ecg", "Left ventricular hypertrophy", "", None, False),
    ("Poor R Progression", "Poor R Progression", "binary", "ecg", "Poor R-wave progression", "", None, False),
    ("BBB_LBBB", "BBB", "onehot:LBBB", "ecg", "Left bundle branch block", "", None, False),
    ("BBB_RBBB", "BBB", "onehot:RBBB", "ecg", "Right bundle branch block", "", None, False),
    # ---- laboratory
    ("FBS", "FBS", "numeric", "labs", "Fasting blood sugar", "mg/dL", (70, 99), True),
    ("CR", "CR", "numeric", "labs", "Creatinine", "mg/dL", (0.6, 1.3), False),
    ("TG", "TG", "numeric", "labs", "Triglycerides", "mg/dL", (0, 150), True),
    ("LDL", "LDL", "numeric", "labs", "LDL cholesterol", "mg/dL", (0, 100), True),
    ("HDL", "HDL", "numeric", "labs", "HDL cholesterol", "mg/dL", (40, 100), True),
    ("BUN", "BUN", "numeric", "labs", "Blood urea nitrogen", "mg/dL", (7, 20), False),
    ("ESR", "ESR", "numeric", "labs", "ESR", "mm/h", (0, 20), False),
    ("HB", "HB", "numeric", "labs", "Haemoglobin", "g/dL", (12, 17.5), False),
    ("K", "K", "numeric", "labs", "Potassium", "mEq/L", (3.5, 5.0), False),
    ("Na", "Na", "numeric", "labs", "Sodium", "mEq/L", (135, 145), False),
    ("WBC", "WBC", "numeric", "labs", "White blood cells", "/uL", (4000, 11000), False),
    ("Lymph", "Lymph", "numeric", "labs", "Lymphocytes", "%", (20, 40), False),
    ("Neut", "Neut", "numeric", "labs", "Neutrophils", "%", (40, 70), False),
    ("PLT", "PLT", "numeric", "labs", "Platelets", "x1000/uL", (150, 450), False),
    # ---- echocardiography
    ("EF-TTE", "EF-TTE", "numeric", "echo", "Ejection fraction (TTE)", "%", (55, 70), False),
    ("Region RWMA", "Region RWMA", "ordinal", "echo", "Regional wall-motion abnormality score (0-4, not a location)", "", None, False),
    ("VHD", "VHD", "ordinal", "echo", "Valvular heart disease (0 none - 3 severe)", "", None, False),
]

FEATURE_NAMES = [f[0] for f in FEATURES]
MODALITY_OF = {f[0]: f[3] for f in FEATURES}
MODALITIES = ["history", "symptoms", "ecg", "labs", "echo"]
MODALITY_LABEL = {
    "history": "History & demographics",
    "symptoms": "Symptoms & examination",
    "ecg": "ECG",
    "labs": "Laboratory",
    "echo": "Echocardiography",
}
# History and symptoms are always available at the bedside; ECG, labs and echo can be toggled.
OPTIONAL_MODALITIES = ["ecg", "labs", "echo"]

# Healthy reference value used by the what-if "model sensitivity" table (modifiable features only).
WHATIF_TARGETS = {
    "Current Smoker": 0,
    "Weight": None,  # derived from BMI target below
    "BMI": 24.9,
    "BP": 120,
    "FBS": 95,
    "TG": 150,
    "LDL": 100,
    "HDL": 50,
}

VHD_MAP = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}


def modality_sets():
    """All combinations of optional modalities. Key like 'ecg+labs+echo' or 'bedside'."""
    sets = {}
    for r in range(len(OPTIONAL_MODALITIES), -1, -1):
        for combo in itertools.combinations(OPTIONAL_MODALITIES, r):
            key = "+".join(combo) if combo else "bedside"
            sets[key] = ["history", "symptoms", *combo]
    return sets


def features_for(modalities):
    return [n for n in FEATURE_NAMES if MODALITY_OF[n] in modalities]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> numeric feature matrix with columns FEATURE_NAMES (no targets)."""
    out = pd.DataFrame(index=df.index)
    for name, raw, ftype, *_ in FEATURES:
        col = df[raw]
        if ftype == "numeric" or ftype == "ordinal":
            if raw == "VHD":
                out[name] = col.map(VHD_MAP).astype(float)
            else:
                out[name] = pd.to_numeric(col).astype(float)
        elif ftype == "binary":
            if raw == "Sex":
                out[name] = (col == "Male").astype(float)
            elif col.dtype == object:
                out[name] = col.map(YN).astype(float)
            else:
                out[name] = col.astype(float)
        elif ftype.startswith("onehot:"):
            out[name] = (col == ftype.split(":", 1)[1]).astype(float)
        else:
            raise ValueError(ftype)
    assert not set(LEAKAGE_COLUMNS) & set(out.columns), "target leaked into features"
    return out


def labels(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({t: (df[c] == pos).astype(int) for t, (c, pos) in TARGETS.items()}, index=df.index)


def feature_meta(X: pd.DataFrame):
    """Display metadata for the web form."""
    meta = []
    for name, raw, ftype, modality, label, unit, normal, modifiable in FEATURES:
        col = X[name]
        kind = "binary" if ftype in ("binary",) or ftype.startswith("onehot:") else ftype
        meta.append({
            "name": name,
            "label": label,
            "modality": modality,
            "kind": kind,
            "unit": unit,
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "normal": list(normal) if normal else None,
            "modifiable": modifiable,
            "step": 1.0 if kind != "numeric" or float(col.round().eq(col).all()) else 0.1,
            "group": "BBB" if raw == "BBB" else None,
        })
    return meta


def nvessels(y: pd.DataFrame) -> np.ndarray:
    return y[["LAD", "LCX", "RCA"]].sum(axis=1).to_numpy()
