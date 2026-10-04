import os
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = str(_ROOT / "data" / "chroma_db")
CSV_PATH = str(_ROOT / "data" / "drug_reviews" / "medicine_clean.csv")
COLLECTION_NAME = "adr_cases"
EMBED_MODEL = "all-MiniLM-L6-v2"

LABEL_MAP = {0: "MILD", 1: "MODERATE", 2: "CRITICAL"}

SYSTEM_PROMPT = (
    "You are a clinical pharmacovigilance assistant. Your role is to analyze "
    "adverse drug reactions and provide clear, structured clinical explanations "
    "for healthcare professionals. Be concise, accurate, and professional. "
    "Always recommend consulting a healthcare provider for serious reactions."
)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def build_vector_store(csv_path: str = CSV_PATH, force_rebuild: bool = False, progress_callback=None):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if not force_rebuild:
            print("Vector store already exists. Use force_rebuild=True to rebuild.")
            return
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    df = pd.read_csv(csv_path).fillna("")
    embedder = _get_embedder()

    ids, documents, metadatas = [], [], []
    batch_size = 256

    for i, row in df.iterrows():
        severity = LABEL_MAP.get(int(row["label"]), "MILD")
        doc = (
            f"Drug: {row['name']}. "
            f"Uses: {row['all_uses']}. "
            f"Side effects: {row['all_side_effects']}. "
            f"Severity: {severity}."
        )
        ids.append(str(i))
        documents.append(doc)
        metadatas.append(
            {
                "name": str(row["name"]),
                "uses": str(row["all_uses"])[:500],
                "side_effects": str(row["all_side_effects"])[:500],
                "severity": severity,
            }
        )

        if len(ids) == batch_size:
            embeddings = embedder.encode(documents, show_progress_bar=False).tolist()
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            ids, documents, metadatas = [], [], []

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(df)} records...")
            if progress_callback:
                progress_callback(i + 1)

    # flush remainder
    if ids:
        embeddings = embedder.encode(documents, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"Vector store built with {len(df)} records.")


def retrieve_similar_cases(query: str, top_k: int = 3) -> list:
    embedder = _get_embedder()
    query_embedding = embedder.encode([query]).tolist()
    collection = _get_collection()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas"],
    )

    cases = []
    for meta in results["metadatas"][0]:
        cases.append(
            {
                "name": meta.get("name", ""),
                "uses": meta.get("uses", ""),
                "side_effects": meta.get("side_effects", ""),
                "severity": meta.get("severity", ""),
            }
        )
    return cases


def generate_explanation(
    drug_name: str,
    reported_symptoms: str,
    predicted_severity: str,
    confidence: float,
    similar_cases: list,
    groq_api_key: str,
) -> str:
    cases_text = ""
    for i, c in enumerate(similar_cases, 1):
        cases_text += (
            f"\n  Case {i}: {c['name']} — Uses: {c['uses'][:120]} — "
            f"Side effects: {c['side_effects'][:120]} — Severity: {c['severity']}"
        )

    prompt = f"""Patient Information:
- Drug reported: {drug_name}
- Reported symptoms / side effects: {reported_symptoms}
- Predicted ADR severity: {predicted_severity} (confidence: {confidence * 100:.1f}%)

Similar ADR cases retrieved from the pharmacovigilance database:{cases_text}

Based on the above, provide a structured clinical explanation that includes:
1. What the patient is likely experiencing
2. Why this reaction is classified as {predicted_severity}
3. Recommended clinical action
4. Reference to how similar cases were managed"""

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content
