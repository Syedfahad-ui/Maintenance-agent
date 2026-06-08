import streamlit as st
import pandas as pd
import requests
import json
import time
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Maintenance Intelligence Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
API_BASE = "http://localhost:5001"

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

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2D2D3F;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
    }
    .similar-card {
        background: #1E1E2E;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 3px solid #7F77DD;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #CCC;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2D2D3F;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


def get_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=3)
        if r.status_code == 200:
            return r.json().get("stats", {})
    except:
        pass
    return {}


def run_analysis(use_default=False, file_bytes=None, filename=None):
    try:
        if use_default:
            r = requests.post(
                f"{API_BASE}/analyze",
                json={"use_default": True},
                timeout=120
            )
        else:
            r = requests.post(
                f"{API_BASE}/analyze",
                files={"file": (filename, file_bytes, "text/csv")},
                timeout=120
            )

        if r.status_code == 200:
            return r.json(), None
        else:
            return None, r.json().get("error", "Unknown error")

    except requests.exceptions.Timeout:
        return None, "Request timed out — the pipeline is still running. Try again."
    except Exception as e:
        return None, str(e)


def render_risk_badge(risk_level: str):
    color = RISK_COLORS.get(risk_level, "#888")
    icon  = RISK_ICONS.get(risk_level, "❓")
    st.markdown(
        f'<span class="risk-badge" style="background:{color}22;color:{color};border:1px solid {color}">'
        f'{icon} {risk_level}</span>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Maintenance Agent")
    st.markdown("---")

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.success("API connected", icon="✅")
    else:
        st.error("API offline — run `python app.py`", icon="🔴")

    st.markdown("---")

    # Dataset stats
    st.markdown("**Dataset Overview**")
    stats = get_stats()
    if stats:
        st.metric("Total Readings",  f"{stats.get('total_readings', 0):,}")
        st.metric("Engine Units",    stats.get("total_engines", 0))
        st.metric("Max Cycle",       stats.get("max_cycle", 0))
        st.metric("Anomalies Found", stats.get("total_anomalies", 0))
        st.metric("Anomaly Rate",    f"{stats.get('anomaly_rate_pct', 0)}%")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("""
    - 🔗 LangGraph state machine
    - 🧠 ChromaDB RAG memory
    - 🤖 Groq LLaMA 3.3 70B
    - 🔍 Isolation Forest
    - 🌐 Flask REST API
    """)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("🔧 Predictive Maintenance Intelligence Agent")
st.markdown(
    "AI-powered anomaly detection and fault assessment for turbofan engine sensor data. "
    "Built with LangGraph, RAG memory, and LLM reasoning."
)
st.markdown("---")

if not api_ok:
    st.error(
        "Flask API is not running. Start it with `python app.py` in your terminal, then refresh this page.",
        icon="🔴"
    )
    st.stop()

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown("### 📂 Input Data")

tab1, tab2 = st.tabs(["Upload CSV", "Use Default Dataset"])

uploaded_file = None
use_default   = False

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a sensor CSV file",
        type=["csv"],
        help="File must have columns: unit_id, cycle, op_setting_1-3, sensor_1-21"
    )
    if uploaded_file:
        preview_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        st.success(f"File loaded: {len(preview_df):,} rows × {len(preview_df.columns)} columns")
        st.dataframe(preview_df.head(5), use_container_width=True)

with tab2:
    st.info(
        "Uses the pre-loaded NASA CMAPSS turbofan dataset "
        "(20,643 sensor readings across 100 engine units).",
        icon="ℹ️"
    )
    use_default = st.button("▶ Run Analysis on Default Dataset", type="primary")


# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
run_uploaded = uploaded_file and st.button("▶ Analyse Uploaded File", type="primary")

if use_default or run_uploaded:
    with st.spinner("Running full pipeline — detect → retrieve → assess → recommend..."):
        start = time.time()

        if use_default:
            result, error = run_analysis(use_default=True)
        else:
            result, error = run_analysis(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name
            )

        elapsed = round(time.time() - start, 1)

    if error:
        st.error(f"Pipeline error: {error}", icon="❌")
        st.stop()

    rec = result["recommendation"]
    st.session_state["last_result"] = rec
    st.session_state["elapsed"]     = elapsed


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if "last_result" in st.session_state:
    rec     = st.session_state["last_result"]
    elapsed = st.session_state.get("elapsed", "—")

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    st.caption(f"Pipeline completed in {elapsed}s")

    # ── Top metrics row ─────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", rec["status"].replace("_", " "))
    with col2:
        st.metric("Anomalies Detected", f"{rec['anomaly_count']:,}")
    with col3:
        st.metric("Worst Engine", f"Unit {rec['worst_engine']}")
    with col4:
        st.metric("Worst Cycle", rec["worst_cycle"])

    st.markdown("---")

    # ── Risk level + action ─────────────────
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)
        render_risk_badge(rec["risk_level"])

        st.markdown("")
        action_color = "#E05A2B" if rec["action_required"] else "#1D9E75"
        action_text  = "Immediate action required" if rec["action_required"] else "Monitor — no immediate action"
        st.markdown(
            f'<p style="color:{action_color};font-weight:500;margin-top:12px">'
            f'{"🔴" if rec["action_required"] else "✅"} {action_text}</p>',
            unsafe_allow_html=True
        )

        st.markdown("")
        st.markdown('<p class="section-header">Similar Historical Cases</p>', unsafe_allow_html=True)

        for case in rec["similar_cases"]:
            similarity_pct = int(case["similarity"] * 100)
            st.markdown(
                f'<div class="similar-card">'
                f'<strong>Engine {case["engine"]}</strong> · Cycle {case["cycle"]}<br>'
                f'<span style="color:#7F77DD">{similarity_pct}% similar</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with right:
        st.markdown('<p class="section-header">LLM Fault Assessment</p>', unsafe_allow_html=True)
        st.markdown(rec["assessment"])

    st.markdown("---")

    # ── Raw JSON expander ───────────────────
    with st.expander("View raw API response"):
        st.json(rec)