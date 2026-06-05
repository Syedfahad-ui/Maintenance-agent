import pandas as pd
import numpy as np
import os
import random

# ─────────────────────────────────────────────
# 1. COLUMN HEADERS
# NASA CMAPSS has no headers in the raw file.
# Structure: unit id, cycle, 3 op settings, 21 sensors
# ─────────────────────────────────────────────
COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ─────────────────────────────────────────────
# 2. LOAD RAW DATA
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """Load the raw NASA CMAPSS txt file and attach column headers."""
    df = pd.read_csv(
        path,
        sep=r"\s+",       # columns are whitespace-separated, not comma
        header=None,       # no header row in the file
        names=COLUMNS,
    )
    print(f"[LOAD] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# 3. INJECT ARTIFICIAL MESSINESS
# Real IoT feeds have all of these problems.
# ─────────────────────────────────────────────
def inject_mess(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Deliberately corrupt a copy of the dataframe to simulate
    real-world messy IoT sensor data.
    """
    random.seed(seed)
    np.random.seed(seed)
    messy = df.copy()
    n = len(messy)

    # 3a. Random NaN values in sensor columns (~3% of cells)
    sensor_cols = [c for c in messy.columns if c.startswith("sensor_")]
    for col in sensor_cols:
        null_idx = random.sample(range(n), k=int(n * 0.03))
        messy.loc[null_idx, col] = np.nan

    # 3b. Duplicate rows (~2% of rows)
    dup_idx = random.sample(range(n), k=int(n * 0.02))
    duplicates = messy.iloc[dup_idx]
    messy = pd.concat([messy, duplicates], ignore_index=True)

    # 3c. Out-of-range spikes in sensor_2 and sensor_3 (~1% of rows)
    spike_idx = random.sample(range(len(messy)), k=int(len(messy) * 0.01))
    messy.loc[spike_idx, "sensor_2"] = 9999.0
    messy.loc[spike_idx, "sensor_3"] = -9999.0

    # 3d. A handful of fully blank rows
    blank_idx = random.sample(range(len(messy)), k=10)
    messy.loc[blank_idx, sensor_cols] = np.nan

    print(f"[MESS] After injection: {len(messy):,} rows")
    print(f"[MESS] NaNs introduced: {messy.isna().sum().sum():,} cells")
    print(f"[MESS] Duplicate rows added: {len(duplicates)}")
    return messy


# ─────────────────────────────────────────────
# 4. PIPELINE REPORT — "before" snapshot
# ─────────────────────────────────────────────
def print_before_report(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("BEFORE CLEANING")
    print("=" * 50)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Total NaNs    : {df.isna().sum().sum():,}")
    print(f"  Duplicates    : {df.duplicated().sum():,}")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    spikes_s2 = (df["sensor_2"] > 9000).sum()
    spikes_s3 = (df["sensor_3"] < -9000).sum()
    print(f"  Spikes sensor_2 (>9000)  : {spikes_s2}")
    print(f"  Spikes sensor_3 (<-9000) : {spikes_s3}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# 5. CLEAN
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps in order:
      1. Remove duplicate rows
      2. Cap out-of-range spikes
      3. Forward-fill NaN sensor readings
      4. Drop any remaining rows that are entirely NaN
      5. Reset index
    """
    cleaned = df.copy()
    sensor_cols = [c for c in cleaned.columns if c.startswith("sensor_")]

    # Step 1 — drop duplicates
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    print(f"[CLEAN] Dropped {before_dedup - len(cleaned)} duplicate rows")

    # Step 2 — cap spikes to plausible range
    # sensor_2 normal range ~ 550–650, cap at 800
    # sensor_3 normal range ~ 1300–1600, cap at -100 floor
    cleaned["sensor_2"] = cleaned["sensor_2"].clip(upper=800)
    cleaned["sensor_3"] = cleaned["sensor_3"].clip(lower=-0)
    print(f"[CLEAN] Clipped out-of-range spikes on sensor_2 and sensor_3")

    # Step 3 — sort by unit and cycle before forward-fill
    # so fill goes in time order per engine unit
    cleaned = cleaned.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Forward-fill NaNs within each unit group
    cleaned[sensor_cols] = (
        cleaned.groupby("unit_id")[sensor_cols]
        .transform(lambda g: g.ffill().bfill())
    )
    print(f"[CLEAN] Forward/backward filled NaN sensor values")

    # Step 4 — drop rows still fully NaN after fill
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=sensor_cols, how="all")
    print(f"[CLEAN] Dropped {before_drop - len(cleaned)} fully-blank rows")

    # Step 5 — clean reset index
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


# ─────────────────────────────────────────────
# 6. VALIDATE
# Schema assertions — if these fail, the pipeline stops.
# This is production-grade practice.
# ─────────────────────────────────────────────
def validate(df: pd.DataFrame):
    """
    Production-grade validation for IoT sensor data.
    Checks go beyond what was artificially injected —
    catches real-world corruption patterns from any source.
    """
    print("\n[VALIDATE] Running schema checks...")
    errors = []
    warnings = []

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # ── STRUCTURAL CHECKS ─────────────────────────────────────────
    # Are all expected columns present?
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Is the dataframe empty?
    if len(df) == 0:
        errors.append("Dataframe is empty — nothing survived cleaning")

    # ── DUPLICATE CHECKS ──────────────────────────────────────────
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        errors.append(f"{dup_count} duplicate rows still present")

    # ── NULL CHECKS ───────────────────────────────────────────────
    nan_count = df[sensor_cols].isna().sum().sum()
    if nan_count > 0:
        errors.append(f"{nan_count} NaN cells remain in sensor columns")

    # Any NaNs in structural columns (unit_id, cycle)?
    structural_nans = df[["unit_id", "cycle"]].isna().sum().sum()
    if structural_nans > 0:
        errors.append(f"{structural_nans} NaN values in unit_id or cycle columns")

    # ── TYPE CHECKS ───────────────────────────────────────────────
    # unit_id and cycle must be numeric (not strings or objects)
    for col in ["unit_id", "cycle"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric — got {df[col].dtype}")

    # ── RANGE CHECKS — values that are physically impossible ──────
    # Negative cycle numbers make no sense
    if (df["cycle"] <= 0).any():
        errors.append("cycle column contains zero or negative values")

    # unit_id must be positive
    if (df["unit_id"] <= 0).any():
        errors.append("unit_id column contains zero or negative values")

    # Operating settings should be within plausible bounds
    # op_setting_1 is a flight condition — range roughly -0.01 to 0.005 in FD001
    op1_out = df["op_setting_1"].abs() > 1.0
    if op1_out.any():
        warnings.append(
            f"op_setting_1 has {op1_out.sum()} values outside expected range (-1, 1)"
        )

    # ── SENSOR PHYSICS CHECKS — things that break in real hardware ─
    # Sensors that should never be zero (would mean sensor is dead/disconnected)
    never_zero_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7"]
    for col in never_zero_sensors:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                warnings.append(
                    f"{col} has {zero_count} zero readings — possible dead sensor"
                )

    # Sensors that should never go negative (pressure, temperature readings)
    always_positive_sensors = ["sensor_2", "sensor_3", "sensor_4"]
    for col in always_positive_sensors:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(
                    f"{col} has {neg_count} negative values — physically impossible"
                )

    # ── CONSISTENCY CHECKS — cross-column logic ────────────────────
    # Each unit should have a continuous cycle sequence with no huge gaps
    # A gap > 5 cycles suggests missing data that ffill couldn't recover
    max_gap = (
        df.groupby("unit_id")["cycle"]
        .apply(lambda x: x.sort_values().diff().max())
        .max()
    )
    if max_gap > 5:
        warnings.append(
            f"Largest cycle gap across units is {max_gap} — "
            f"possible missing time steps that were not recovered"
        )

    # All units should have at least 10 cycles — fewer suggests truncated data
    min_cycles_per_unit = df.groupby("unit_id")["cycle"].count().min()
    if min_cycles_per_unit < 10:
        warnings.append(
            f"Some units have fewer than 10 cycles — data may be truncated"
        )

    # ── STATISTICAL CHECKS — sudden variance collapse ─────────────
    # A sensor with near-zero standard deviation across all readings
    # is likely stuck / frozen — common real hardware failure
    for col in sensor_cols:
        std = df[col].std()
        if std < 1e-6:
            warnings.append(
                f"{col} has std={std:.2e} — sensor may be frozen/stuck"
            )

    # ── PRINT RESULTS ─────────────────────────────────────────────
    if errors:
        print("\n[VALIDATE] ❌ ERRORS (pipeline should not proceed):")
        for e in errors:
            print(f"           → {e}")
        raise ValueError(
            f"Validation failed with {len(errors)} error(s). "
            f"Fix the issues above before proceeding."
        )

    if warnings:
        print("[VALIDATE] ⚠️  WARNINGS (data passed but review these):")
        for w in warnings:
            print(f"           → {w}")

    print(f"[VALIDATE] ✅ All checks passed — {len(warnings)} warning(s)\n")


# ─────────────────────────────────────────────
# 7. PIPELINE REPORT — "after" snapshot
# ─────────────────────────────────────────────
def print_after_report(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("AFTER CLEANING")
    print("=" * 50)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Total NaNs    : {df.isna().sum().sum():,}")
    print(f"  Duplicates    : {df.duplicated().sum():,}")
    print(f"  Units (engines): {df['unit_id'].nunique()}")
    print(f"  Max cycle     : {df['cycle'].max()}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
def save(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVE] Clean data saved → {path}")


# ─────────────────────────────────────────────
# 9. RUN THE PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    RAW_PATH   = "data/raw/sensors_raw.txt"
    CLEAN_PATH = "data/clean/sensors_clean.csv"

    print("\n🔧  MAINTENANCE AGENT — DATA PIPELINE\n")

    # Load
    df_raw = load_raw(RAW_PATH)

    # Inject mess (simulates real-world dirty IoT data)
    df_messy = inject_mess(df_raw)
    print_before_report(df_messy)

    # Clean
    df_clean = clean(df_messy)

    # Validate
    validate(df_clean)

    # After report
    print_after_report(df_clean)

    # Save
    save(df_clean, CLEAN_PATH)

    print("✅  Pipeline complete.\n")




    