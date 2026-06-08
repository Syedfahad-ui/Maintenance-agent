import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Custom Data Analysis",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RISK_COLORS = {
    "LOW"     : "#1D9E75",
    "MEDIUM"  : "#EF9F27",
    "HIGH"    : "#E05A2B",
    "CRITICAL": "#C0392B"
}

RISK_ICONS = {
    "LOW"     : "✅",
    "MEDIUM"  : "⚠️",
    "HIGH"    : "🔴",
    "CRITICAL": "🚨"
}

MAX_ROWS_WARNING  = 50_000
MAX_SENSOR_COLS   = 20   # cap for LLM context
MIN_SENSOR_COLS   = 3    # minimum required

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Auto-detect column types.
    Returns dict: {col: 'numeric' | 'id' | 'categorical' | 'datetime'}
    Mitigation: shows detected types to user for confirmation.
    """
    types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique <= 20 and nunique / len(df) < 0.05:
                types[col] = "id_or_categorical"
            else:
                types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types


def validate_mapping(mapping: dict, df: pd.DataFrame) -> tuple[bool, list, list]:
    """
    Validate user column mapping.
    Returns (is_valid, errors, warnings)
    Mitigation: strict prerequisites before enabling Run button.
    """
    errors   = []
    warnings = []

    asset_col = mapping.get("asset_id")
    cycle_col = mapping.get("cycle")
    sensor_cols = mapping.get("sensors", [])

    if not asset_col:
        errors.append("Asset ID column is required — identifies which machine each reading belongs to.")
    if not cycle_col:
        errors.append("Time/Cycle column is required — orders readings chronologically.")
    if len(sensor_cols) < MIN_SENSOR_COLS:
        errors.append(f"At least {MIN_SENSOR_COLS} sensor columns required — got {len(sensor_cols)}.")

    if asset_col and asset_col in df.columns:
        if pd.api.types.is_float_dtype(df[asset_col]):
            warnings.append(f"'{asset_col}' looks like a float — asset IDs are usually integers. Check this is correct.")

    if cycle_col and cycle_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[cycle_col]):
            errors.append(f"'{cycle_col}' must be numeric — found {df[cycle_col].dtype}.")
        elif df[cycle_col].min() < 0:
            warnings.append(f"'{cycle_col}' has negative values — cycles should be positive.")

    if len(sensor_cols) > MAX_SENSOR_COLS:
        warnings.append(
            f"You selected {len(sensor_cols)} sensor columns. Only the top {MAX_SENSOR_COLS} "
            f"by variance will be used in the LLM assessment to avoid context limits."
        )

    return len(errors) == 0, errors, warnings


def engineer_features(df: pd.DataFrame, sensor_cols: list, asset_col: str, cycle_col: str) -> tuple:
    """
    Feature engineering: rolling mean + std per asset.
    Returns (featured_df, feature_col_names)
    """
    featured = df.copy()
    feature_cols = []

    for col in sensor_cols:
        featured[f"{col}_rolling_mean"] = (
            featured.groupby(asset_col)[col]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        featured[f"{col}_rolling_std"] = (
            featured.groupby(asset_col)[col]
            .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        )
        feature_cols += [col, f"{col}_rolling_mean", f"{col}_rolling_std"]

    return featured, feature_cols


def run_isolation_forest(df: pd.DataFrame, feature_cols: list, contamination: float = 0.05) -> pd.DataFrame:
    """
    Train and score Isolation Forest on the fly.
    Mitigation: capped feature cols to avoid memory issues.
    """
    X = df[feature_cols].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    result = df.copy()
    result["anomaly_score"] = model.decision_function(X_scaled)
    result["is_anomaly"]    = model.predict(X_scaled) == -1
    return result


def get_top_sensors_by_variance(df: pd.DataFrame, sensor_cols: list, top_n: int = MAX_SENSOR_COLS) -> list:
    """
    Mitigation for LLM context limits:
    Select top N sensors by variance — most informative ones.
    """
    variances = df[sensor_cols].var().sort_values(ascending=False)
    return variances.head(top_n).index.tolist()


def build_dynamic_prompt(
    sensor_descriptions: dict,
    top_anomalies_text: str,
    asset_col: str,
    cycle_col: str
) -> str:
    """
    Build LLM prompt dynamically from user-provided column descriptions.
    Mitigation: only injects top N sensors to avoid token limits.
    """
    sensor_context = "\n".join([
        f"- {col}: {desc}" for col, desc in sensor_descriptions.items()
    ])

    return f"""
You are a senior predictive maintenance engineer analyzing industrial sensor data.

SENSOR CONTEXT (provided by the operator):
{sensor_context}

Asset identifier column: {asset_col}
Time/cycle column: {cycle_col}

ANOMALOUS READINGS TO ASSESS:
{top_anomalies_text}

Based on these anomalous readings and the sensor context above, provide:

1. FAULT ASSESSMENT
   What patterns are visible? What might be causing these anomalies?

2. RISK LEVEL
   State clearly: LOW / MEDIUM / HIGH / CRITICAL

3. AFFECTED SYSTEMS
   Which systems or components appear to be involved?

4. RECOMMENDED ACTION
   What should the maintenance team investigate and with what urgency?

Be specific. Reference the sensor names and values in your assessment.
Keep your response concise and actionable.
"""


def format_top_anomalies(df: pd.DataFrame, sensor_cols: list, asset_col: str, cycle_col: str, top_n: int = 5) -> str:
    anomalies  = df[df["is_anomaly"] == True]
    top        = anomalies.nsmallest(top_n, "anomaly_score")
    display    = [asset_col, cycle_col, "anomaly_score"] + sensor_cols[:8]
    display    = [c for c in display if c in top.columns]

    lines = []
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(f"Reading {i}:")
        lines.append(f"  {asset_col}: {row[asset_col]}  |  {cycle_col}: {row[cycle_col]}")
        lines.append(f"  Anomaly score: {row['anomaly_score']:.4f}")
        for col in sensor_cols[:8]:
            if col in row:
                lines.append(f"  {col}: {row[col]:.4f}")
        lines.append("")
    return "\n".join(lines)


def get_llm_assessment(prompt_text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY not found in .env — LLM assessment unavailable."
    llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)
    chain = PromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
    return chain.invoke({"prompt": prompt_text})


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Custom Data")
    st.markdown("---")
    st.markdown("**Requirements**")
    st.markdown(f"""
    Your CSV must have:
    - One **asset/machine ID** column
    - One **time/cycle** column
    - At least **{MIN_SENSOR_COLS} numeric sensor** columns
    """)
    st.markdown("---")
    st.markdown("**Performance note**")
    st.markdown(f"Files over {MAX_ROWS_WARNING:,} rows will show a warning. Isolation Forest retraining takes 10–30s depending on size.")
    st.markdown("---")
    st.markdown("**Known limitations**")
    st.markdown("""
    - Column mapping relies on user accuracy
    - LLM context capped at top 20 sensors by variance
    - No historical RAG memory (uses fresh model per upload)
    """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.title("🗂️ Custom Data Analysis")
st.markdown(
    "Upload any structured sensor CSV, map your columns, and get AI-powered "
    "anomaly detection and fault assessment on your own equipment data."
)
st.markdown("---")

# ─────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────
st.markdown("### Step 1 — Upload your CSV")

uploaded = st.file_uploader(
    "Upload sensor data CSV",
    type=["csv"],
    help="Must contain an asset ID column, a time/cycle column, and numeric sensor columns."
)

if not uploaded:
    st.info("Upload a CSV file to begin.", icon="📂")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success(f"Loaded: {len(df_raw):,} rows × {len(df_raw.columns)} columns")

if len(df_raw) > MAX_ROWS_WARNING:
    st.warning(
        f"Large file detected ({len(df_raw):,} rows). "
        "Retraining may take 30–60 seconds. Consider sampling to 50k rows for faster results.",
        icon="⚠️"
    )

with st.expander("Preview data (first 5 rows)"):
    st.dataframe(df_raw.head(5), use_container_width=True)

# ─────────────────────────────────────────────
# STEP 2 — COLUMN TYPE DETECTION + CONFIRMATION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 2 — Confirm column types")
st.markdown("We've auto-detected types below. Correct any misclassifications before mapping.")

detected_types = detect_column_types(df_raw)

type_corrections = {}
cols_per_row = 4
all_cols     = list(df_raw.columns)

for i in range(0, len(all_cols), cols_per_row):
    row_cols = all_cols[i:i + cols_per_row]
    grid     = st.columns(len(row_cols))
    for j, col in enumerate(row_cols):
        with grid[j]:
            corrected = st.selectbox(
                col,
                options=["numeric", "id_or_categorical", "categorical", "datetime", "ignore"],
                index=["numeric", "id_or_categorical", "categorical", "datetime", "ignore"]
                      .index(detected_types.get(col, "ignore")),
                key=f"type_{col}"
            )
            type_corrections[col] = corrected

numeric_cols     = [c for c, t in type_corrections.items() if t == "numeric"]
id_cat_cols      = [c for c, t in type_corrections.items() if t == "id_or_categorical"]
all_usable_cols  = list(df_raw.columns) + ["— not mapped —"]

# ─────────────────────────────────────────────
# STEP 3 — COLUMN MAPPING
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 3 — Map your columns")
st.markdown("Tell the system what each column represents.")

map_col1, map_col2 = st.columns(2)

with map_col1:
    asset_col = st.selectbox(
        "Asset / Machine ID column",
        options=["— not mapped —"] + id_cat_cols + numeric_cols,
        help="Identifies which machine or asset each row belongs to (e.g. machine_id, unit_id, asset)"
    )
    asset_col = None if asset_col == "— not mapped —" else asset_col

with map_col2:
    cycle_col = st.selectbox(
        "Time / Cycle column",
        options=["— not mapped —"] + numeric_cols + id_cat_cols,
        help="Orders readings chronologically (e.g. cycle, timestamp, time_step)"
    )
    cycle_col = None if cycle_col == "— not mapped —" else cycle_col

st.markdown("**Select sensor columns** (hold Cmd/Ctrl to select multiple):")
sensor_cols = st.multiselect(
    "Sensor columns",
    options=numeric_cols,
    default=numeric_cols[:min(10, len(numeric_cols))],
    help=f"Select numeric sensor readings. Minimum {MIN_SENSOR_COLS}, maximum {MAX_SENSOR_COLS} recommended."
)

# ─────────────────────────────────────────────
# STEP 4 — SENSOR DESCRIPTIONS
# ─────────────────────────────────────────────
if sensor_cols:
    st.markdown("---")
    st.markdown("### Step 4 — Describe your sensors *(optional but recommended)*")
    st.markdown(
        "These descriptions are injected into the LLM prompt so it can reason about "
        "what your sensors actually measure. Leave blank if unknown."
    )

    sensor_descriptions = {}
    desc_cols = st.columns(2)
    for i, col in enumerate(sensor_cols[:MAX_SENSOR_COLS]):
        with desc_cols[i % 2]:
            desc = st.text_input(
                f"{col}",
                placeholder=f"e.g. 'Temperature at compressor inlet (°C)'",
                key=f"desc_{col}"
            )
            sensor_descriptions[col] = desc if desc else f"Sensor reading: {col}"

# ─────────────────────────────────────────────
# STEP 5 — VALIDATE + RUN
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 5 — Validate and run")

mapping = {
    "asset_id": asset_col,
    "cycle"   : cycle_col,
    "sensors" : sensor_cols
}

is_valid, errors, warnings = validate_mapping(mapping, df_raw)

if errors:
    for e in errors:
        st.error(e, icon="❌")

if warnings:
    for w in warnings:
        st.warning(w, icon="⚠️")

if is_valid:
    st.success("Mapping is valid — ready to run.", icon="✅")

contamination = st.slider(
    "Expected anomaly rate (%)",
    min_value=1, max_value=20, value=5,
    help="Approximate % of readings you expect to be anomalous. Default 5% is a good starting point."
) / 100

run_button = st.button(
    "▶ Run Analysis",
    type="primary",
    disabled=not is_valid
)

# ─────────────────────────────────────────────
# STEP 6 — ANALYSIS
# ─────────────────────────────────────────────
if run_button and is_valid:

    # Clean data
    with st.spinner("Preparing data..."):
        df_work = df_raw[[asset_col, cycle_col] + sensor_cols].copy()
        df_work = df_work.sort_values([asset_col, cycle_col]).reset_index(drop=True)
        df_work[sensor_cols] = df_work[sensor_cols].ffill().bfill()
        df_work = df_work.dropna()

    # Select top sensors by variance (LLM context mitigation)
    top_sensors = get_top_sensors_by_variance(df_work, sensor_cols)
    if len(sensor_cols) > MAX_SENSOR_COLS:
        st.info(
            f"Using top {MAX_SENSOR_COLS} sensors by variance for LLM assessment: {', '.join(top_sensors)}",
            icon="ℹ️"
        )

    # Feature engineering
    with st.spinner("Engineering features..."):
        df_featured, feature_cols = engineer_features(df_work, top_sensors, asset_col, cycle_col)

    # Isolation Forest
    with st.spinner(f"Training Isolation Forest on {len(df_featured):,} rows... (this may take 15–30s)"):
        df_flagged = run_isolation_forest(df_featured, feature_cols, contamination)

    anomaly_count = df_flagged["is_anomaly"].sum()
    anomaly_rate  = anomaly_count / len(df_flagged) * 100

    # ── Results header ───────────────────────
    st.markdown("---")
    st.markdown("### 📊 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Readings",     f"{len(df_flagged):,}")
    m2.metric("Anomalies Detected", f"{anomaly_count:,}")
    m3.metric("Anomaly Rate",       f"{anomaly_rate:.1f}%")
    m4.metric("Assets Affected",    df_flagged[df_flagged["is_anomaly"]][asset_col].nunique())

    # ── Anomaly table ────────────────────────
    st.markdown("**Top 10 most anomalous readings:**")
    display_cols = [asset_col, cycle_col, "anomaly_score"] + top_sensors[:6]
    display_cols = [c for c in display_cols if c in df_flagged.columns]
    st.dataframe(
        df_flagged.nsmallest(10, "anomaly_score")[display_cols].reset_index(drop=True),
        use_container_width=True
    )

    # ── LLM assessment ───────────────────────
    st.markdown("---")
    st.markdown("### 🤖 LLM Fault Assessment")

    with st.spinner("Generating AI assessment..."):
        top_anomalies_text = format_top_anomalies(
            df_flagged, top_sensors, asset_col, cycle_col
        )
        top_descriptions = {k: sensor_descriptions[k] for k in top_sensors if k in sensor_descriptions}
        prompt_text      = build_dynamic_prompt(
            top_descriptions, top_anomalies_text, asset_col, cycle_col
        )
        assessment = get_llm_assessment(prompt_text)

    # Extract risk level
    risk_level = "MEDIUM"
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in assessment.upper():
            risk_level = level
            break

    risk_color = RISK_COLORS.get(risk_level, "#888")
    risk_icon  = RISK_ICONS.get(risk_level, "❓")

    rl_col, assess_col = st.columns([1, 2])
    with rl_col:
        st.markdown("**Risk Level**")
        st.markdown(
            f'<span style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color};'
            f'padding:6px 16px;border-radius:20px;font-weight:600;font-size:1rem">'
            f'{risk_icon} {risk_level}</span>',
            unsafe_allow_html=True
        )
        action_required = risk_level in ["HIGH", "CRITICAL"]
        st.markdown("")
        if action_required:
            st.error("Immediate action required", icon="🔴")
        else:
            st.success("Monitor — no immediate action", icon="✅")

    with assess_col:
        st.markdown("**Assessment**")
        st.markdown(assessment)

    # ── Raw flagged data download ─────────────
    st.markdown("---")
    csv_out = df_flagged[[asset_col, cycle_col, "anomaly_score", "is_anomaly"]].to_csv(index=False)
    st.download_button(
        label="⬇️ Download flagged data as CSV",
        data=csv_out,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )