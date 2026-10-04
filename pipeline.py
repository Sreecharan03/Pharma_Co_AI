import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp

BASE = str(Path(__file__).resolve().parent / "data" / "drug_reviews")
FEATURES_DIR = f"{BASE}/features"

LABEL_MAP = {0: "MILD", 1: "MODERATE", 2: "CRITICAL"}


def load_artifacts():
    model = pickle.load(open(f"{BASE}/best_model.pkl", "rb"))
    tfidf_name = pickle.load(open(f"{FEATURES_DIR}/tfidf_name.pkl", "rb"))
    tfidf_use = pickle.load(open(f"{FEATURES_DIR}/tfidf_use.pkl", "rb"))
    label_encoders = pickle.load(open(f"{BASE}/label_encoders.pkl", "rb"))
    return model, tfidf_name, tfidf_use, label_encoders


def predict_severity(
    drug_name: str,
    drug_uses: str,
    se_count: int,
    therapeutic_class: str,
    chemical_class: str,
    action_class: str,
    habit_forming: int,
    model=None,
    tfidf_name=None,
    tfidf_use=None,
    label_encoders=None,
) -> dict:
    if model is None:
        model, tfidf_name, tfidf_use, label_encoders = load_artifacts()

    # TF-IDF features
    name_features = tfidf_name.transform([drug_name])   # (1, 300)
    use_features = tfidf_use.transform([drug_uses])     # (1, 200)

    # Encode categorical features — fallback to 0 for unseen classes
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except Exception:
            return 0

    tc_enc = safe_encode(label_encoders["Therapeutic Class"], therapeutic_class)
    cc_enc = safe_encode(label_encoders["Chemical Class"], chemical_class)
    ac_enc = safe_encode(label_encoders["Action Class"], action_class)

    # Structured features — order must match training
    struct = np.array([[se_count, habit_forming, tc_enc, cc_enc, ac_enc]], dtype=float)
    struct_sparse = sp.csr_matrix(struct)

    # Combine exactly as in training
    X = sp.hstack([name_features, use_features, struct_sparse])

    proba = model.predict_proba(X)[0]
    label = int(np.argmax(proba))

    return {
        "severity": LABEL_MAP[label],
        "label": label,
        "confidence": float(proba[label]),
        "probabilities": {
            "MILD": float(proba[0]),
            "MODERATE": float(proba[1]),
            "CRITICAL": float(proba[2]),
        },
    }
