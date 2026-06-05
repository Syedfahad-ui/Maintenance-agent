from flask import Flask, request, jsonify
import pandas as pd
import io
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env')

sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from graph import build_graph
from memory import build_vector_store

# ─────────────────────────────────────────────
# Flask app setup
# ─────────────────────────────────────────────
app = Flask(__name__)

# Build the LangGraph agent once at startup
print("[APP] Building LangGraph agent...")
agent = build_graph()
print("[APP] Agent ready ✓")

# Ensure vector store is populated at startup
print("[APP] Checking vector memory...")
FLAGGED_PATH = "data/clean/sensors_flagged.csv"
if Path(FLAGGED_PATH).exists():
    df_flagged = pd.read_csv(FLAGGED_PATH)
    build_vector_store(df_flagged)
    print("[APP] Vector memory ready ✓")
else:
    print("[APP] ⚠️  No flagged data found — run detector.py first")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status" : "ok",
        "agent"  : "maintenance-agent-v1",
        "message": "Predictive maintenance agent is running"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint. Accepts a CSV file upload or JSON sensor data.
    Runs the full LangGraph pipeline and returns structured assessment.

    Option 1 — CSV upload:
        POST /analyze
        Content-Type: multipart/form-data
        file: sensors.csv

    Option 2 — JSON body with pre-loaded data path:
        POST /analyze
        Content-Type: application/json
        {"use_default": true}
    """
    try:
        # ── Option 1: CSV file upload ──────────────────
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            content  = file.read().decode("utf-8")
            df_input = pd.read_csv(io.StringIO(content))
            print(f"[API] Received CSV: {len(df_input)} rows")

        # ── Option 2: Use default clean dataset ────────
        elif request.json and request.json.get("use_default"):
            clean_path = "data/clean/sensors_clean.csv"
            if not Path(clean_path).exists():
                return jsonify({"error": "Default dataset not found — run pipeline.py first"}), 404
            df_input = pd.read_csv(clean_path)
            print(f"[API] Using default dataset: {len(df_input)} rows")

        else:
            return jsonify({
                "error"  : "No input provided",
                "options": [
                    "Upload a CSV file with key 'file'",
                    "Send JSON body with {'use_default': true}"
                ]
            }), 400

        # ── Run LangGraph pipeline ─────────────────────
        print("[API] Running LangGraph pipeline...")
        result = agent.invoke({"sensor_df": df_input})

        if result.get("error"):
            return jsonify({"error": result["error"]}), 500

        recommendation = result["recommendation"]
        print(f"[API] Pipeline complete — Risk: {recommendation['risk_level']}")

        return jsonify({
            "success"       : True,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """
    Quick stats endpoint — returns info about the loaded dataset
    without running the full pipeline.
    """
    try:
        flagged_path = "data/clean/sensors_flagged.csv"
        clean_path   = "data/clean/sensors_clean.csv"

        stats_data = {}

        if Path(clean_path).exists():
            df = pd.read_csv(clean_path)
            stats_data["total_readings"] = len(df)
            stats_data["total_engines"]  = int(df["unit_id"].nunique())
            stats_data["max_cycle"]      = int(df["cycle"].max())

        if Path(flagged_path).exists():
            df_f = pd.read_csv(flagged_path)
            stats_data["total_anomalies"]   = int(df_f["is_anomaly"].sum())
            stats_data["anomaly_rate_pct"]  = round(
                df_f["is_anomaly"].sum() / len(df_f) * 100, 1
            )
            stats_data["engines_affected"]  = int(
                df_f[df_f["is_anomaly"]]["unit_id"].nunique()
            )

        return jsonify({"status": "ok", "stats": stats_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌐  MAINTENANCE AGENT — FLASK API\n")
    print("  Endpoints:")
    print("    GET  /health   — health check")
    print("    GET  /stats    — dataset statistics")
    print("    POST /analyze  — run full pipeline\n")
    app.run(debug=True, host="0.0.0.0", port=5001)