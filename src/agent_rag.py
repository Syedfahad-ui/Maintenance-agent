import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')

# Import our memory module
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from memory import build_vector_store, query_similar_failures, format_similar_for_prompt


# ─────────────────────────────────────────────
# 1. LOAD LLM
# ─────────────────────────────────────────────
def load_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=api_key
    )
    print("[AGENT] LLM loaded ✓")
    return llm


# ─────────────────────────────────────────────
# 2. UPDATED PROMPT — now includes historical context
# This is the RAG pattern:
# retrieve relevant history → inject into prompt
# → LLM reasons from both current data AND past failures
# ─────────────────────────────────────────────
RAG_TEMPLATE = """
You are a senior predictive maintenance engineer analyzing turbofan engine sensor data.

SENSOR CONTEXT:
- sensor_2  : Total temperature at fan inlet
- sensor_3  : Total temperature at LPC outlet (low-pressure compressor)
- sensor_4  : Total temperature at HPC outlet (high-pressure compressor)
- sensor_7  : Total pressure at HPC outlet
- sensor_9  : Physical core speed
- sensor_11 : Static pressure at HPC outlet
- anomaly_score : How anomalous this reading is (more negative = more abnormal)

{similar_failures}

CURRENT ANOMALOUS READINGS TO ASSESS:
{sensor_data}

Based on BOTH the current readings AND the similar historical failures above, provide:

1. FAULT ASSESSMENT
   What is most likely wrong? How does this compare to the historical cases?

2. RISK LEVEL
   State clearly: LOW / MEDIUM / HIGH / CRITICAL
   Has the risk increased compared to similar past cases?

3. AFFECTED COMPONENTS
   Which engine components are most likely involved?

4. RECOMMENDED ACTION
   What should the maintenance team do and with what urgency?
   Reference the historical cases if relevant.

Keep your response concise and actionable.
"""


# ─────────────────────────────────────────────
# 3. BUILD CHAIN
# ─────────────────────────────────────────────
def build_chain(llm):
    prompt = PromptTemplate(
        template=RAG_TEMPLATE,
        input_variables=["similar_failures", "sensor_data"]
    )
    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# 4. FORMAT CURRENT READINGS
# ─────────────────────────────────────────────
def format_readings(df_anomalies: pd.DataFrame, top_n: int = 5) -> str:
    display_cols = [
        "unit_id", "cycle", "anomaly_score",
        "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_9", "sensor_11"
    ]
    display_cols = [c for c in display_cols if c in df_anomalies.columns]
    top_anomalies = df_anomalies.nsmallest(top_n, "anomaly_score")[display_cols]

    lines = []
    for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        lines.append(f"Reading {i}:")
        lines.append(f"  Engine unit  : {int(row['unit_id'])}")
        lines.append(f"  Cycle        : {int(row['cycle'])}")
        lines.append(f"  Anomaly score: {row['anomaly_score']:.4f}")
        for col in display_cols:
            if col not in ["unit_id", "cycle", "anomaly_score"]:
                lines.append(f"  {col:<12}: {row[col]:.4f}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 5. RAG EXPLAIN — retrieve then generate
# ─────────────────────────────────────────────
def rag_explain(
    df_flagged: pd.DataFrame,
    chain,
    top_n: int = 5
) -> str:

    anomalies = df_flagged[df_flagged["is_anomaly"] == True]
    if len(anomalies) == 0:
        return "No anomalies detected."

    # Get current readings to explain
    top_anomalies = anomalies.nsmallest(top_n, "anomaly_score")

    # For each reading, retrieve similar historical failures
    # Use the single most anomalous reading as the retrieval anchor
    worst = top_anomalies.iloc[0]
    similar = query_similar_failures(worst, top_n=3)
    similar_text = format_similar_for_prompt(similar)

    print(f"[AGENT] Retrieved {len(similar)} similar historical failures from memory")

    # Format current readings
    sensor_data = format_readings(anomalies, top_n=top_n)

    # Invoke chain with both current data AND historical context
    response = chain.invoke({
        "similar_failures": similar_text,
        "sensor_data"     : sensor_data
    })
    return response


# ─────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    FLAGGED_PATH = "data/clean/sensors_flagged.csv"

    print("\n🤖  MAINTENANCE AGENT — RAG EXPLANATION\n")

    df_flagged = pd.read_csv(FLAGGED_PATH)
    print(f"[AGENT] Loaded {len(df_flagged):,} rows, "
          f"{df_flagged['is_anomaly'].sum():,} anomalies flagged")

    # Build vector store if not already built
    print("[AGENT] Initialising vector memory...")
    build_vector_store(df_flagged)

    # Load LLM and chain
    llm   = load_llm()
    chain = build_chain(llm)

    # RAG explanation
    print("[AGENT] Running RAG pipeline...")
    explanation = rag_explain(df_flagged, chain, top_n=5)

    print("\n" + "=" * 50)
    print("LLM MAINTENANCE ASSESSMENT (RAG-enhanced)")
    print("=" * 50)
    print(explanation)
    print("=" * 50 + "\n")

    print("✅  RAG Agent complete.\n")