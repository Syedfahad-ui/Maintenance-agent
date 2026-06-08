import streamlit as st

st.set_page_config(
    page_title="Maintenance Intelligence Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 Predictive Maintenance Intelligence Agent")
st.markdown(
    "An end-to-end AI agent that detects anomalies in industrial sensor data, "
    "retrieves similar historical failures from memory, and generates plain-English "
    "maintenance assessments with risk levels and recommended actions."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🚀 NASA Turbofan Demo")
    st.markdown(
        "Run the full pipeline on the NASA CMAPSS turbofan engine dataset. "
        "20,643 sensor readings across 100 engine units. Pre-loaded and ready to run."
    )
    st.markdown("""
    - ✅ Pre-trained Isolation Forest model
    - ✅ ChromaDB RAG memory with 1,033 historical failures
    - ✅ LangGraph 4-node reasoning pipeline
    - ✅ Upload your own FD001/FD002/FD003 CSV
    """)
    if st.button("Go to NASA Demo →", type="primary"):
        st.switch_page("pages/1_NASA_Demo.py")

with col2:
    st.markdown("### 🗂️ Custom Data Analysis")
    st.markdown(
        "Upload any structured sensor CSV from your own domain. "
        "Map your columns, validate your data, and get AI-powered anomaly detection "
        "and plain-English assessments on your own equipment data."
    )
    st.markdown("""
    - ✅ Auto column detection + type validation
    - ✅ Interactive column mapping UI
    - ✅ On-the-fly Isolation Forest retraining
    - ✅ Dynamic LLM prompt from your column descriptions
    """)
    if st.button("Go to Custom Analysis →", type="primary"):
        st.switch_page("pages/2_Custom_Data.py")

st.markdown("---")

st.markdown("### 🏗️ Architecture")
st.markdown("""
```
Raw CSV → Data Pipeline → Isolation Forest → ChromaDB RAG → LangGraph → Flask API → Streamlit
```
""")

col3, col4, col5 = st.columns(3)
with col3:
    st.markdown("**Agent Framework**")
    st.markdown("LangChain + LangGraph")
with col4:
    st.markdown("**LLM**")
    st.markdown("Groq llama-3.3-70b")
with col5:
    st.markdown("**Vector Memory**")
    st.markdown("ChromaDB + sentence-transformers")

st.markdown("---")
st.caption("Built by Syed Fahad Ehsan · BSc AI & Data Science · Middlesex University Dubai")