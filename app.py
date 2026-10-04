import os
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go

from pipeline import load_artifacts, predict_severity
from rag_llm import build_vector_store, retrieve_similar_cases, generate_explanation

_ROOT = Path(__file__).resolve().parent

# ── constants ────────────────────────────────────────────────────────────────
THERAPEUTIC_CLASSES = [
    "ANTI INFECTIVES", "GASTRO INTESTINAL", "PAIN ANALGESICS", "NEURO CNS",
    "RESPIRATORY", "CARDIAC", "ANTI DIABETIC", "OPHTHAL", "DERMA", "HORMONES",
    "GYNAECOLOGICAL", "VITAMINS MINERALS NUTRIENTS", "ANTI NEOPLASTICS",
    "BLOOD RELATED", "UROLOGY",
]

SEVERITY_COLORS = {
    "MILD": "#2ecc71",
    "MODERATE": "#f39c12",
    "CRITICAL": "#e74c3c",
}

GROQ_ENV_KEY = os.environ.get("GROQ_API", "")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharmacovigilance System",
    page_icon="💊",
    layout="wide",
)

# ── load ML artifacts once ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model…")
def get_artifacts():
    return load_artifacts()


model, tfidf_name, tfidf_use, label_encoders = get_artifacts()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        value=GROQ_ENV_KEY,
        type="password",
        placeholder="gsk_…",
        help="Required for AI clinical explanations",
    )

    st.divider()
    st.subheader("Vector Store")
    rebuild = st.button("🔄 Build / Rebuild Vector Store", use_container_width=True)
    if rebuild:
        import pandas as pd
        total = len(pd.read_csv(_ROOT / "data" / "drug_reviews" / "medicine_clean.csv"))
        progress = st.progress(0, text="Starting…")
        status_box = st.empty()

        def _progress_callback(done, total=total):
            pct = int(done / total * 100)
            progress.progress(pct, text=f"Embedding {done:,} / {total:,} records…")
            status_box.caption(f"{pct}% complete")

        build_vector_store(force_rebuild=True, progress_callback=_progress_callback)
        progress.progress(100, text="Done!")
        status_box.empty()
        st.success(f"Vector store ready — {total:,} records indexed.")

    st.divider()
    st.markdown(
        """
**About this system**

An MTech-level AI Pharmacovigilance platform that:
- Predicts ADR severity (MILD / MODERATE / CRITICAL) with a trained XGBoost model
- Retrieves similar drug cases via ChromaDB semantic search
- Generates clinical explanations using Groq LLaMA 3

_Always consult a qualified healthcare provider._
"""
    )

# ── main title ────────────────────────────────────────────────────────────────
st.title("💊 AI-Driven Pharmacovigilance System")
st.caption("Adverse Drug Reaction Detection & Clinical Decision Support")
st.divider()

# ── input form ────────────────────────────────────────────────────────────────
with st.form("adr_form"):
    col_l, col_r = st.columns(2)

    with col_l:
        drug_name = st.text_input("Drug Name", placeholder="e.g. augmentin 625 duo tablet")
        drug_uses = st.text_area(
            "Drug Uses", placeholder="e.g. treatment of bacterial infections", height=100
        )
        se_count = st.number_input(
            "Number of Side Effects Reported", min_value=1, max_value=42, value=5, step=1
        )

    with col_r:
        therapeutic_class = st.selectbox("Therapeutic Class", THERAPEUTIC_CLASSES)
        chemical_class = st.text_input("Chemical Class", placeholder="e.g. Penicillin")
        action_class = st.text_input("Action Class", placeholder="e.g. Beta-lactam antibiotic")
        habit_forming_sel = st.selectbox("Habit Forming", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Analyze ADR", use_container_width=True, type="primary")

# ── results ───────────────────────────────────────────────────────────────────
if submitted:
    if not drug_name.strip():
        st.error("Please enter a drug name.")
        st.stop()

    habit_forming = 1 if habit_forming_sel == "Yes" else 0

    # ── ML prediction ──────────────────────────────────────────────────────
    with st.spinner("Running ML prediction…"):
        result = predict_severity(
            drug_name=drug_name,
            drug_uses=drug_uses,
            se_count=int(se_count),
            therapeutic_class=therapeutic_class,
            chemical_class=chemical_class,
            action_class=action_class,
            habit_forming=habit_forming,
            model=model,
            tfidf_name=tfidf_name,
            tfidf_use=tfidf_use,
            label_encoders=label_encoders,
        )

    severity = result["severity"]
    confidence = result["confidence"]
    probs = result["probabilities"]
    color = SEVERITY_COLORS[severity]

    # ── severity badge ──────────────────────────────────────────────────────
    st.markdown(
        f"""
<div style="
    background:{color}22;
    border:2px solid {color};
    border-radius:12px;
    padding:18px 28px;
    margin-bottom:16px;
">
  <span style="font-size:2rem;font-weight:700;color:{color};">⚠ {severity}</span>
  <span style="font-size:1.1rem;color:#555;margin-left:24px;">
    Confidence: <b>{confidence*100:.1f}%</b>
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── two-column results ──────────────────────────────────────────────────
    col_chart, col_cases = st.columns([1, 1])

    with col_chart:
        st.subheader("Probability Breakdown")
        fig = go.Figure(
            go.Bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation="h",
                marker_color=[SEVERITY_COLORS[k] for k in probs],
                text=[f"{v*100:.1f}%" for v in probs.values()],
                textposition="outside",
            )
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            margin=dict(l=10, r=40, t=10, b=10),
            height=220,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cases:
        st.subheader("Similar Cases from Database")
        query = f"{drug_name} {drug_uses} side effects {severity}"
        try:
            similar_cases = retrieve_similar_cases(query, top_k=3)
            for i, c in enumerate(similar_cases, 1):
                with st.expander(f"Case {i}: {c['name']} — {c['severity']}"):
                    st.markdown(f"**Uses:** {c['uses']}")
                    st.markdown(f"**Side effects:** {c['side_effects']}")
        except Exception as e:
            st.warning(
                f"Could not retrieve similar cases ({e}). "
                "Run 'Build / Rebuild Vector Store' in the sidebar first."
            )
            similar_cases = []

    # ── AI clinical explanation ─────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Clinical Explanation")

    if not groq_key:
        st.warning("Enter a Groq API key in the sidebar to generate a clinical explanation.")
    else:
        with st.spinner("Generating clinical explanation via Groq LLaMA 3…"):
            try:
                explanation = generate_explanation(
                    drug_name=drug_name,
                    reported_symptoms=drug_uses,
                    predicted_severity=severity,
                    confidence=confidence,
                    similar_cases=similar_cases if "similar_cases" in dir() else [],
                    groq_api_key=groq_key,
                )
                st.markdown(
                    f"""
<div style="
    background:#f8f9fa;
    border-left:4px solid {color};
    border-radius:8px;
    padding:20px 24px;
    font-size:0.97rem;
    line-height:1.7;
    white-space:pre-wrap;
">
{explanation}
</div>
""",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"LLM call failed: {e}")
