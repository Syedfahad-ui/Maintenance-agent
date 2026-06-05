import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path

# ─────────────────────────────────────────────
# ChromaDB stores documents as text with
# embeddings for semantic similarity search.
# We convert each anomalous sensor reading
# into a text description, embed it, and store
# it. When a new anomaly comes in, we find the
# most similar historical failures.
# ─────────────────────────────────────────────

CHROMA_PATH = "data/chroma_db"


def get_client():
    """Get or create a persistent ChromaDB client."""
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def get_collection(client):
    """Get or create the failure history collection."""
    # Using ChromaDB's default embedding function
    # (sentence-transformers under the hood)
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="failure_history",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def row_to_text(row: pd.Series) -> str:
    """
    Convert a sensor reading row into a natural language
    description. This is what gets embedded and stored.
    Good text = better similarity matching.
    """
    return (
        f"Engine unit {int(row['unit_id'])} at cycle {int(row['cycle'])}. "
        f"Anomaly score: {row['anomaly_score']:.4f}. "
        f"Sensor readings — "
        f"sensor_2 (fan inlet temp): {row.get('sensor_2', 'N/A'):.2f}, "
        f"sensor_3 (LPC outlet temp): {row.get('sensor_3', 'N/A'):.2f}, "
        f"sensor_4 (HPC outlet temp): {row.get('sensor_4', 'N/A'):.2f}, "
        f"sensor_7 (HPC pressure): {row.get('sensor_7', 'N/A'):.2f}, "
        f"sensor_9 (core speed): {row.get('sensor_9', 'N/A'):.2f}, "
        f"sensor_11 (static pressure): {row.get('sensor_11', 'N/A'):.2f}."
    )


def build_vector_store(df_flagged: pd.DataFrame) -> int:
    """
    Store all anomalous readings in ChromaDB.
    Each document = one anomalous sensor reading.
    Returns number of records stored.
    """
    client = get_client()
    collection = get_collection(client)

    # Only store anomalies — normal readings aren't useful for retrieval
    anomalies = df_flagged[df_flagged["is_anomaly"] == True].copy()

    # Avoid re-adding if collection already has data
    existing = collection.count()
    if existing > 0:
        print(f"[MEMORY] Collection already has {existing} records — skipping rebuild")
        return existing

    print(f"[MEMORY] Building vector store from {len(anomalies)} anomalies...")

    # Build in batches of 100 to avoid memory issues
    batch_size = 100
    total_added = 0

    for i in range(0, len(anomalies), batch_size):
        batch = anomalies.iloc[i:i + batch_size]
        documents = [row_to_text(row) for _, row in batch.iterrows()]
        ids = [f"anomaly_{int(row['unit_id'])}_{int(row['cycle'])}_{i + idx}"
               for idx, (_, row) in enumerate(batch.iterrows())]
        metadatas = [
            {
                "unit_id"      : int(row["unit_id"]),
                "cycle"        : int(row["cycle"]),
                "anomaly_score": float(row["anomaly_score"]),
                "sensor_2"     : float(row.get("sensor_2", 0)),
                "sensor_3"     : float(row.get("sensor_3", 0)),
                "sensor_4"     : float(row.get("sensor_4", 0)),
            }
            for _, row in batch.iterrows()
        ]

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        total_added += len(batch)

    print(f"[MEMORY] Stored {total_added} anomaly records in ChromaDB ✓")
    return total_added


def query_similar_failures(
    row: pd.Series,
    top_n: int = 3
) -> list[dict]:
    """
    Given a new anomalous reading, find the top_n
    most similar historical failures from ChromaDB.
    Returns list of dicts with text + metadata.
    """
    client = get_client()
    collection = get_collection(client)

    if collection.count() == 0:
        return []

    # Convert query row to text, same format as stored docs
    query_text = row_to_text(row)

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_n, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    similar = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similar.append({
            "text"          : doc,
            "unit_id"       : meta["unit_id"],
            "cycle"         : meta["cycle"],
            "anomaly_score" : meta["anomaly_score"],
            "similarity"    : round(1 - dist, 3)   # convert distance to similarity
        })

    return similar


def format_similar_for_prompt(similar_failures: list[dict]) -> str:
    """Format retrieved failures for injection into LLM prompt."""
    if not similar_failures:
        return "No similar historical failures found."

    lines = ["SIMILAR HISTORICAL FAILURES:"]
    for i, f in enumerate(similar_failures, 1):
        lines.append(
            f"{i}. Engine {f['unit_id']} at cycle {f['cycle']} "
            f"(similarity: {f['similarity']:.0%}, "
            f"anomaly score: {f['anomaly_score']:.4f})"
        )
        lines.append(f"   {f['text']}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# RUN STANDALONE — builds the vector store
# ─────────────────────────────────────────────
if __name__ == "__main__":
    FLAGGED_PATH = "data/clean/sensors_flagged.csv"

    print("\n🧠  MAINTENANCE AGENT — VECTOR MEMORY\n")

    df_flagged = pd.read_csv(FLAGGED_PATH)
    count = build_vector_store(df_flagged)

    print(f"\n[MEMORY] Vector store ready — {count} historical failures indexed")

    # Quick test query
    print("\n[MEMORY] Test query — finding similar failures to top anomaly...")
    top_anomaly = df_flagged[df_flagged["is_anomaly"] == True].nsmallest(1, "anomaly_score").iloc[0]
    similar = query_similar_failures(top_anomaly, top_n=3)

    print(f"[MEMORY] Query: Engine {int(top_anomaly['unit_id'])} cycle {int(top_anomaly['cycle'])}")
    print(f"[MEMORY] Found {len(similar)} similar failures:")
    for f in similar:
        print(f"  → Engine {f['unit_id']} cycle {f['cycle']} | similarity {f['similarity']:.0%}")

    print("\n✅  Memory module complete.\n")