import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# ─────────────────────────────────────────────
# SENSOR COLUMNS USED FOR DETECTION
# Not all 21 sensors are informative in FD001 —
# some have near-zero variance (essentially constant).
# Using only the sensors that actually change.
# ─────────────────────────────────────────────
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]


# ─────────────────────────────────────────────
# 1. LOAD CLEAN DATA
# ─────────────────────────────────────────────
def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[DETECTOR] Loaded {len(df):,} rows from {path}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# Raw sensor readings alone miss degradation trends.
# Adding rolling mean and std per unit captures
# how a sensor is behaving over time — not just
# its current value. This is what separates a
# naive model from a real predictive one.
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[DETECTOR] Engineering features...")
    featured = df.copy()

    for col in INFORMATIVE_SENSORS:
        # Rolling mean over last 5 cycles per engine unit
        featured[f"{col}_rolling_mean"] = (
            featured.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        # Rolling std — captures volatility / instability
        featured[f"{col}_rolling_std"] = (
            featured.groupby("unit_id")[col]
            .transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
        )

    print(f"[DETECTOR] Features expanded: {len(df.columns)} → {len(featured.columns)} columns")
    return featured


# ─────────────────────────────────────────────
# 3. TRAIN ISOLATION FOREST
# Isolation Forest works by randomly partitioning
# data — anomalies are isolated faster (shorter
# path length) than normal points.
# contamination=0.05 means we expect ~5% anomalies.
# ─────────────────────────────────────────────
def train_detector(df: pd.DataFrame) -> tuple:
    print("[DETECTOR] Training Isolation Forest...")

    # Select feature columns: raw sensors + rolling features
    feature_cols = (
        INFORMATIVE_SENSORS
        + [f"{s}_rolling_mean" for s in INFORMATIVE_SENSORS]
        + [f"{s}_rolling_std"  for s in INFORMATIVE_SENSORS]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values

    # Scale features — Isolation Forest is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = IsolationForest(
        n_estimators=100,      # number of trees
        contamination=0.05,    # expected anomaly rate
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )
    model.fit(X_scaled)

    print("[DETECTOR] Model trained ✓")
    return model, scaler, feature_cols


# ─────────────────────────────────────────────
# 4. SCORE AND FLAG
# anomaly_score: raw decision function score
#   more negative = more anomalous
# is_anomaly: 1 = normal, -1 = anomaly (sklearn convention)
#   we remap to True/False for readability
# ─────────────────────────────────────────────
def score_and_flag(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    feature_cols: list
) -> pd.DataFrame:

    print("[DETECTOR] Scoring all readings...")
    flagged = df.copy()

    X = flagged[feature_cols].values
    X_scaled = scaler.transform(X)

    # Raw anomaly score (lower = more anomalous)
    flagged["anomaly_score"] = model.decision_function(X_scaled)

    # Boolean flag: True means anomalous
    flagged["is_anomaly"] = model.predict(X_scaled) == -1

    n_anomalies = flagged["is_anomaly"].sum()
    pct = n_anomalies / len(flagged) * 100
    print(f"[DETECTOR] Flagged {n_anomalies:,} anomalies ({pct:.1f}% of readings)")

    return flagged


# ─────────────────────────────────────────────
# 5. SUMMARY REPORT
# ─────────────────────────────────────────────
def print_anomaly_report(df: pd.DataFrame):
    anomalies = df[df["is_anomaly"] == True]

    print("\n" + "=" * 50)
    print("ANOMALY DETECTION REPORT")
    print("=" * 50)
    print(f"  Total readings     : {len(df):,}")
    print(f"  Anomalies flagged  : {len(anomalies):,}")
    print(f"  Anomaly rate       : {len(anomalies)/len(df)*100:.1f}%")
    print(f"  Units affected     : {anomalies['unit_id'].nunique()} of {df['unit_id'].nunique()}")
    print(f"  Most anomalous unit: {anomalies.groupby('unit_id').size().idxmax()}")

    print("\n  Top 5 most anomalous readings:")
    top5 = (
        df.nsmallest(5, "anomaly_score")
        [["unit_id", "cycle", "anomaly_score", "sensor_2", "sensor_3", "sensor_4", "sensor_7"]]
    )
    print(top5.to_string(index=False))
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# 6. SAVE FLAGGED DATA
# ─────────────────────────────────────────────
def save_flagged(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DETECTOR] Flagged data saved → {path}")


# ─────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    CLEAN_PATH   = "data/clean/sensors_clean.csv"
    FLAGGED_PATH = "data/clean/sensors_flagged.csv"

    print("\n🔍  MAINTENANCE AGENT — ANOMALY DETECTOR\n")

    df_clean    = load_clean(CLEAN_PATH)
    df_featured = engineer_features(df_clean)
    model, scaler, feature_cols = train_detector(df_featured)
    df_flagged  = score_and_flag(df_featured, model, scaler, feature_cols)

    print_anomaly_report(df_flagged)
    save_flagged(df_flagged, FLAGGED_PATH)

    print("✅  Detector complete.\n")