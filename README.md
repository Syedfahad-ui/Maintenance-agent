# 🔧 Predictive Maintenance Intelligence Agent

An end-to-end AI agent that detects anomalies in industrial sensor data, retrieves similar historical failures from memory, and generates plain-English maintenance assessments with risk levels and recommended actions.

Built to demonstrate production-grade AI engineering skills relevant to asset performance management in enterprise environments.

---

## 🎯 Problem Statement

Asset-heavy industries — utilities, healthcare facilities, property management — generate continuous streams of sensor telemetry from critical equipment. Maintenance teams struggle to identify which anomalies are genuinely dangerous and which require immediate action, often reacting to failures rather than preventing them.

This agent ingests raw sensor data, detects anomalous patterns, cross-references similar historical failures, and produces actionable maintenance assessments in plain English — combining ML-based detection with LLM-powered reasoning.

---

## 🏗️ Architecture

```
Raw IoT CSV
    ↓
Pandas Data Pipeline (cleaning, validation, feature engineering)
    ↓
Isolation Forest (anomaly detection + scoring)
    ↓
ChromaDB Vector Store (semantic similarity search — RAG memory)
    ↓
LangGraph State Machine (4-node reasoning workflow)
    ↓
Groq LLM — llama-3.3-70b (fault assessment + recommendation)
    ↓
Flask REST API (POST /analyze)
    ↓
Streamlit Dashboard (interactive UI)
```

### LangGraph Pipeline Nodes

| Node | Function |
|------|----------|
| `detect` | Runs Isolation Forest, flags anomalous readings |
| `retrieve` | Queries ChromaDB for 3 most similar historical failures |
| `assess` | LLM generates RAG-enhanced fault assessment |
| `recommend` | Packages structured JSON result with risk level and actions |

---

## ✨ Key Features

- **Production-grade data pipeline** — handles messy real-world IoT data: NaN values, duplicate rows, out-of-range spikes, irregular timestamps, frozen sensors
- **Physics-based validation** — schema checks, type checks, sensor physics constraints (no negative pressure), frozen sensor detection
- **Semantic RAG memory** — ChromaDB with sentence-transformer embeddings retrieves similar historical failures by meaning, not just keyword matching
- **Multi-step LangGraph reasoning** — state machine with conditional edges (skips LLM call if no anomalies detected)
- **Provider-agnostic LLM** — swap Groq, Gemini, or Azure OpenAI with 3 lines
- **REST API** — Flask endpoints for health check, stats, and full pipeline analysis
- **Interactive dashboard** — Streamlit UI with CSV upload, risk badges, similar case cards, and raw JSON viewer

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangChain + LangGraph |
| LLM | Groq (llama-3.3-70b-versatile) |
| Vector memory | ChromaDB + sentence-transformers |
| Anomaly detection | scikit-learn Isolation Forest |
| Data pipeline | Pandas + NumPy |
| API | Flask |
| Dashboard | Streamlit |
| Dataset | NASA CMAPSS Turbofan Engine (FD001) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repo
git clone https://github.com/Syedfahad-ui/Maintenance-agent.git
cd Maintenance-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Setup

1. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

2. Download the NASA CMAPSS dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place `train_FD001.txt` in `data/raw/sensors_raw.txt`

### Running the Pipeline

```bash
# Step 1 — Data pipeline (cleaning + validation)
python src/pipeline.py

# Step 2 — Anomaly detection
python src/detector.py

# Step 3 — Build vector memory
python src/memory.py

# Step 4 — Test LangGraph pipeline standalone
python src/graph.py
```

### Running the Full Application

```bash
# Terminal 1 — Start Flask API
python app.py

# Terminal 2 — Start Streamlit dashboard
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Dataset statistics |
| POST | `/analyze` | Run full pipeline |

**Example request:**
```bash
# Use default dataset
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"use_default": true}'

# Upload custom CSV
curl -X POST http://localhost:5001/analyze \
  -F "file=@your_sensors.csv"
```

**Example response:**
```json
{
  "success": true,
  "recommendation": {
    "status": "ANOMALY_DETECTED",
    "risk_level": "HIGH",
    "anomaly_count": 1033,
    "worst_engine": 14,
    "worst_cycle": 180,
    "action_required": true,
    "similar_cases": [
      {"engine": 14, "cycle": 180, "similarity": 1.0},
      {"engine": 14, "cycle": 123, "similarity": 0.994}
    ],
    "assessment": "FAULT ASSESSMENT: ..."
  }
}
```

---

## 📁 Project Structure

```
Maintenance-agent/
├── data/
│   ├── raw/          ← Place sensors_raw.txt here
│   └── clean/        ← Pipeline outputs (gitignored)
├── src/
│   ├── pipeline.py   ← Data cleaning + validation
│   ├── detector.py   ← Isolation Forest anomaly detection
│   ├── memory.py     ← ChromaDB vector store + RAG
│   ├── graph.py      ← LangGraph state machine
│   └── agent_rag.py  ← Standalone RAG agent
├── app.py            ← Flask REST API
├── streamlit_app.py  ← Streamlit dashboard
├── requirements.txt
└── .env              ← API keys (never committed)
```

---

## 🧠 What I Learned

- Building production-grade data pipelines that handle real-world IoT messiness
- Implementing RAG with semantic embeddings vs keyword matching
- LangGraph state machines with conditional routing and typed state
- Provider-agnostic LLM integration with LangChain LCEL
- Debugging SSL certificate issues with Python 3.13 on macOS
- Separating validation logic from cleaning logic for maintainability

---

## 🗺️ Roadmap (v2)

- [ ] Flexible column mapping — domain-agnostic CSV analysis
- [ ] Fleet monitoring dashboard — all assets ranked by risk
- [ ] Remaining Useful Life (RUL) prediction
- [ ] Conversational agent — ask questions about your data
- [ ] Acoustic anomaly detection (MIMII dataset)
- [ ] Arabic language output toggle (UAE deployment)
- [ ] Docker containerisation

---

## 👤 Author

**Syed Fahad Ehsan**
BSc Artificial Intelligence & Data Science — Middlesex University Dubai

[GitHub](https://github.com/Syedfahad-ui) · [LinkedIn](www.linkedin.com/in/syed-fahad-ehsan-34a824319)

---

*Built as a portfolio project demonstrating production AI engineering — LangGraph, RAG pipelines, and intelligent agent design applied to real-world predictive maintenance.*