import pandas as pd
import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')

sys.path.append(str(Path(__file__).resolve().parent))
from memory import build_vector_store, query_similar_failures, format_similar_for_prompt

# ─────────────────────────────────────────────
# INFORMATIVE SENSORS (same as detector.py)
# ─────────────────────────────────────────────
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]

# ─────────────────────────────────────────────
# 1. AGENT STATE
# TypedDict defines the shape of data flowing
# through the graph. Each node reads from and
# writes to this shared state object.
# This is what makes LangGraph a state machine
# rather than just a chain.
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    # Input
    sensor_df       : Optional[object]    # raw dataframe passed in

    # After node 1: detect
    anomalies_df    : Optional[object]    # flagged anomaly rows
    anomaly_count   : Optional[int]       # total anomalies found
    top_anomaly     : Optional[dict]      # single worst anomaly as dict

    # After node 2: retrieve
    similar_failures: Optional[list]      # retrieved from ChromaDB
    similar_text    : Optional[str]       # formatted for prompt

    # After node 3: assess
    risk_level      : Optional[str]       # LOW / MEDIUM / HIGH / CRITICAL
    assessment      : Optional[str]       # full LLM explanation

    # After node 4: recommend
    recommendation  : Optional[str]       # final structured recommendation

    # Meta
    error           : Optional[str]       # any error message


# ─────────────────────────────────────────────
# 2. NODE FUNCTIONS
# Each node is a plain Python function that
# takes state and returns updated state.
# LangGraph handles the wiring.
# ─────────────────────────────────────────────

def node_detect_anomalies(state: AgentState) -> AgentState:
    """
    Node 1: Run Isolation Forest on input sensor data.
    Produces flagged anomaly dataframe + top anomaly.
    """
    print("[GRAPH] Node 1: Detecting anomalies...")
    try:
        df = state["sensor_df"]

        # Feature engineering
        featured = df.copy()
        for col in INFORMATIVE_SENSORS:
            if col in featured.columns:
                featured[f"{col}_rolling_mean"] = (
                    featured.groupby("unit_id")[col]
                    .transform(lambda x: x.rolling(5, min_periods=1).mean())
                )
                featured[f"{col}_rolling_std"] = (
                    featured.groupby("unit_id")[col]
                    .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
                )

        # Feature columns
        feature_cols = (
            [s for s in INFORMATIVE_SENSORS if s in featured.columns]
            + [f"{s}_rolling_mean" for s in INFORMATIVE_SENSORS if f"{s}_rolling_mean" in featured.columns]
            + [f"{s}_rolling_std"  for s in INFORMATIVE_SENSORS if f"{s}_rolling_std"  in featured.columns]
        )

        X = featured[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(X_scaled)

        featured["anomaly_score"] = model.decision_function(X_scaled)
        featured["is_anomaly"]    = model.predict(X_scaled) == -1

        anomalies   = featured[featured["is_anomaly"] == True]
        top_anomaly = featured.nsmallest(1, "anomaly_score").iloc[0].to_dict()

        print(f"[GRAPH] Node 1 complete: {len(anomalies)} anomalies detected")

        return {
            **state,
            "anomalies_df" : anomalies,
            "anomaly_count": len(anomalies),
            "top_anomaly"  : top_anomaly,
            "error"        : None
        }

    except Exception as e:
        return {**state, "error": f"Detection failed: {str(e)}"}


def node_retrieve_history(state: AgentState) -> AgentState:
    """
    Node 2: Query ChromaDB for similar historical failures.
    Uses the worst anomaly as the retrieval anchor.
    """
    print("[GRAPH] Node 2: Retrieving similar historical failures...")
    try:
        if state.get("error"):
            return state

        top = state["top_anomaly"]
        top_series = pd.Series(top)

        similar      = query_similar_failures(top_series, top_n=3)
        similar_text = format_similar_for_prompt(similar)

        print(f"[GRAPH] Node 2 complete: {len(similar)} similar failures retrieved")

        return {
            **state,
            "similar_failures": similar,
            "similar_text"    : similar_text
        }

    except Exception as e:
        return {**state, "error": f"Retrieval failed: {str(e)}"}


def node_generate_assessment(state: AgentState) -> AgentState:
    """
    Node 3: LLM generates fault assessment using current
    readings + historical context (RAG).
    """
    print("[GRAPH] Node 3: Generating LLM assessment...")
    try:
        if state.get("error"):
            return state

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY")
        )

        template = """
You are a senior predictive maintenance engineer.

{similar_failures}

CURRENT ANOMALOUS SENSOR READINGS:
{sensor_data}

Provide a concise assessment with:
1. FAULT ASSESSMENT — what is likely wrong
2. RISK LEVEL — LOW / MEDIUM / HIGH / CRITICAL
3. AFFECTED COMPONENTS — which parts are involved
4. RECOMMENDED ACTION — what to do and urgency

Be specific and reference historical cases where relevant.
"""
        # Format top 5 anomalies for context
        anomalies_df = state["anomalies_df"]
        display_cols = ["unit_id", "cycle", "anomaly_score", "sensor_2", "sensor_3", "sensor_4"]
        display_cols = [c for c in display_cols if c in anomalies_df.columns]
        top5 = anomalies_df.nsmallest(5, "anomaly_score")[display_cols]

        sensor_data = ""
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            sensor_data += f"Reading {i}: Engine {int(row['unit_id'])} Cycle {int(row['cycle'])} | "
            sensor_data += f"Score: {row['anomaly_score']:.4f} | "
            sensor_data += " | ".join([f"{c}: {row[c]:.2f}" for c in display_cols
                                       if c not in ["unit_id", "cycle", "anomaly_score"]])
            sensor_data += "\n"

        prompt = PromptTemplate(
            template=template,
            input_variables=["similar_failures", "sensor_data"]
        )
        chain      = prompt | llm | StrOutputParser()
        assessment = chain.invoke({
            "similar_failures": state["similar_text"],
            "sensor_data"     : sensor_data
        })

        # Extract risk level from response
        risk_level = "MEDIUM"
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if level in assessment.upper():
                risk_level = level
                break

        print(f"[GRAPH] Node 3 complete: Risk level = {risk_level}")

        return {
            **state,
            "assessment": assessment,
            "risk_level": risk_level
        }

    except Exception as e:
        return {**state, "error": f"Assessment failed: {str(e)}"}


def node_build_recommendation(state: AgentState) -> AgentState:
    """
    Node 4: Build a structured final recommendation object.
    This is what the Flask API returns as JSON.
    """
    print("[GRAPH] Node 4: Building final recommendation...")
    try:
        if state.get("error"):
            return state

        top     = state["top_anomaly"]
        similar = state["similar_failures"] or []

        recommendation = {
            "status"         : "ANOMALY_DETECTED",
            "risk_level"     : state["risk_level"],
            "anomaly_count"  : state["anomaly_count"],
            "worst_engine"   : int(top["unit_id"]),
            "worst_cycle"    : int(top["cycle"]),
            "anomaly_score"  : round(float(top["anomaly_score"]), 4),
            "similar_cases"  : [
                {
                    "engine"    : f["unit_id"],
                    "cycle"     : f["cycle"],
                    "similarity": f["similarity"]
                }
                for f in similar
            ],
            "assessment"     : state["assessment"],
            "action_required": state["risk_level"] in ["HIGH", "CRITICAL"]
        }

        print(f"[GRAPH] Node 4 complete: Recommendation built ✓")

        return {**state, "recommendation": recommendation}

    except Exception as e:
        return {**state, "error": f"Recommendation failed: {str(e)}"}


# ─────────────────────────────────────────────
# 3. CONDITIONAL EDGE
# After detection, check if anomalies exist.
# If none found, skip to END.
# If found, continue to retrieval.
# ─────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("anomaly_count") or state["anomaly_count"] == 0:
        return "end"
    return "retrieve"


# ─────────────────────────────────────────────
# 4. BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect",     node_detect_anomalies)
    graph.add_node("retrieve",   node_retrieve_history)
    graph.add_node("assess",     node_generate_assessment)
    graph.add_node("recommend",  node_build_recommendation)

    # Entry point
    graph.set_entry_point("detect")

    # Conditional edge after detection
    graph.add_conditional_edges(
        "detect",
        should_continue,
        {
            "retrieve": "retrieve",
            "end"     : END
        }
    )

    # Linear edges for remaining nodes
    graph.add_edge("retrieve",  "assess")
    graph.add_edge("assess",    "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 5. RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    CLEAN_PATH   = "data/clean/sensors_clean.csv"
    FLAGGED_PATH = "data/clean/sensors_flagged.csv"

    print("\n🔗  MAINTENANCE AGENT — LANGGRAPH PIPELINE\n")

    # Load data
    df_clean = pd.read_csv(CLEAN_PATH)

    # Ensure vector store is built
    df_flagged = pd.read_csv(FLAGGED_PATH)
    build_vector_store(df_flagged)

    # Build and run graph
    agent = build_graph()
    result = agent.invoke({"sensor_df": df_clean})

    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
    else:
        rec = result["recommendation"]
        print("\n" + "=" * 50)
        print("LANGGRAPH PIPELINE RESULT")
        print("=" * 50)
        print(f"  Status        : {rec['status']}")
        print(f"  Risk Level    : {rec['risk_level']}")
        print(f"  Anomalies     : {rec['anomaly_count']}")
        print(f"  Worst Engine  : Unit {rec['worst_engine']} at cycle {rec['worst_cycle']}")
        print(f"  Action needed : {rec['action_required']}")
        print(f"\n  Similar cases :")
        for c in rec["similar_cases"]:
            print(f"    → Engine {c['engine']} cycle {c['cycle']} | {c['similarity']:.0%} similar")
        print(f"\n  Assessment:\n{rec['assessment']}")
        print("=" * 50)

    print("\n✅  LangGraph pipeline complete.\n")