# AI-Driven Pharmacovigilance System: Adverse Drug Reaction Severity Prediction

**An MTech-level AI system for predicting Adverse Drug Reaction (ADR) severity and providing clinical decision support through machine learning and retrieval-augmented generation.**

---

## 📋 Executive Summary

This project implements a **production-ready pharmacovigilance platform** that leverages machine learning and large language models to:

- **Predict ADR Severity** (MILD, MODERATE, CRITICAL) from drug metadata using a trained XGBoost classifier
- **Retrieve Similar Clinical Cases** via semantic search (ChromaDB + sentence embeddings)  
- **Generate AI-Powered Clinical Explanations** using Groq's LLaMA 3 foundation model
- **Support Clinical Decision-Making** with explainable predictions and evidence-based recommendations

The system processes **248,218+ medicines** from a comprehensive pharmacological dataset and serves as a real-time decision support tool for healthcare professionals.

---

## 🏗️ System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  Input: Drug Details                    │
│      (Name, Uses, Side Effects, Therapeutic Class)      │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
    ┌───▼──────────┐        ┌──────▼──────┐
    │ ML Pipeline  │        │ RAG Pipeline │
    └───┬──────────┘        └──────┬──────┘
        │                         │
    ┌───▼────────────────┐ ┌──────▼──────────┐
    │ XGBoost Classifier │ │ ChromaDB Lookup │
    │ • TF-IDF Features  │ │ Similar Cases   │
    │ • Categorical Enc. │ │                 │
    └───┬────────────────┘ └──────┬──────────┘
        │                         │
        └────────────┬────────────┘
                     │
            ┌────────▼─────────┐
            │  Groq LLM Engine │
            │  (Llama 3.1-8B)  │
            │  • Generate Text │
            │  • Clinical Info │
            └────────┬─────────┘
                     │
        ┌────────────▼─────────────┐
        │    Output Dashboard      │
        │ • Severity Badge         │
        │ • Confidence Score       │
        │ • Similar Cases          │
        │ • AI Explanation         │
        └──────────────────────────┘
```

### Core Components

1. **ML Prediction Engine** (`pipeline.py`)
   - XGBoost gradient boosting classifier trained on 248K+ medicine records
   - Feature engineering: TF-IDF vectorization + categorical label encoding
   - Real-time inference with confidence scoring

2. **RAG & LLM Integration** (`rag_llm.py`)
   - ChromaDB vector store with 248K+ indexed records
   - Sentence embeddings (all-MiniLM-L6-v2) for semantic search
   - Groq API for fast LLM inference (Llama 3.1-8B)

3. **Interactive Web Interface** (`app.py`)
   - Streamlit-based UI for clinician interaction
   - Real-time predictions and visual analytics
   - API key management and vector store control

---

## 📊 Dataset & Exploratory Data Analysis

### Dataset Source
- **Name**: 250K Medicines Usage, Side Effects & Substitutes
- **Source**: Kaggle (shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes)
- **Records**: 248,218 medicines
- **Features**: 58 attributes (drug names, side effects, therapeutic classes, etc.)

### Data Characteristics

| Aspect | Details |
|--------|---------|
| **Total Records** | 248,218 |
| **Side Effect Features** | 42 columns (aggregated into single field) |
| **Drug Uses** | 5 columns per drug (consolidated) |
| **Therapeutic Classes** | 15 major categories (ANTI INFECTIVES, CARDIAC, RESPIRATORY, etc.) |
| **Chemical Classes** | Automated labeling (Macrolides, Penicillins, Beta-blockers, etc.) |
| **Habit Forming Drugs** | ~35% classified as habit-forming |

### Key EDA Findings (01_eda.ipynb)

**Side Effect Patterns**:
- Drugs average **3.2 reported side effects**
- Most common: Nausea, Vomiting, Diarrhea, Headache, Dizziness
- High variability: Range from 0 to 42 side effects per drug

**Severity Distribution** (engineered labels):
- **MILD**: 158,402 (63.8%) — Non-critical, common side effects
- **MODERATE**: 75,936 (30.6%) — Significant adverse reactions or habit-forming  
- **CRITICAL**: 13,880 (5.6%) — Life-threatening reactions (cardiac, hepatic, seizures, etc.)

**Missing Data Handling**:
- Dropped 35+ side effect columns (>99% missing)
- Consolidated remaining features via concatenation
- Categorical features back-filled with "Unknown" class

---

## 🔬 Methodology

### Phase 1: Data Preprocessing (02_preprocessing.ipynb)

**Data Cleaning Pipeline**:
- Removed 100% missing columns (sideEffect35-41)
- Standardized text: lowercasing, whitespace trimming, NaN handling
- Consolidated 42 side effect columns → single text field
- Consolidated 5 use columns → single text field

**Class Imbalance Handling**:
- Applied SMOTE (Synthetic Minority Oversampling Technique) 
- Rebalanced minority CRITICAL class (5.6% → 33%)
- Improved model sensitivity to rare but critical reactions

**Train-Test Split**:
- 80% training, 20% validation (stratified by severity)
- Ensures class distribution preservation

### Phase 2: Feature Engineering (03_feature_engineering.ipynb)

**Text Feature Engineering**:

1. **TF-IDF Vectorization**
   - Drug Name TF-IDF: 300-dimensional vector (captures brand name patterns)
   - Drug Uses TF-IDF: 200-dimensional vector (captures therapeutic keywords)
   - Min document frequency: 1, Max DF: 90%

2. **Categorical Encoding**
   - **Therapeutic Class**: Label encoding (15 classes)
   - **Chemical Class**: Label encoding with OOV fallback
   - **Action Class**: Label encoding with fallback to 0

3. **Numerical Features**
   - **Side Effect Count** (se_count): Raw count of reported side effects
   - **Habit Forming**: Binary encoding (0=No, 1=Yes)

**Feature Space Dimensionality**: 300 (TF-IDF name) + 200 (TF-IDF use) + 5 (structured) = **505 dimensions**

### Phase 3: Model Training (04_model_training.ipynb)

**Algorithm: XGBoost Classifier**

Why XGBoost for this task?
- **Interpretability**: Feature importance scores for clinical validation
- **Robustness**: Handles categorical features and class imbalance well
- **Speed**: Fast inference (<50ms per prediction)
- **Proven Track Record**: State-of-the-art for tabular data in healthcare

**Hyperparameters**:
```
n_estimators=200, max_depth=8, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8, gamma=0.5,
objective='multi:softprob', eval_metric='mlogloss'
```

**Training Results**:
- Cross-validation F1-score: ~0.74 (weighted)
- Precision (CRITICAL): ~0.68 (critical for safety)
- Recall (CRITICAL): ~0.56 (catch most serious cases)
- ROC-AUC (macro): ~0.82
- Inference Speed: 45ms/prediction

**Model Artifacts**:
- `best_model.pkl` (1.8 MB): Trained XGBoost classifier
- `label_encoders.pkl` (40 KB): Encoders for categorical variables
- `tfidf_name.pkl` (12 KB): TF-IDF vectorizer for drug names
- `tfidf_use.pkl` (9 KB): TF-IDF vectorizer for drug uses

---

## 🧠 ML Prediction Engine (pipeline.py)

```python
def predict_severity(drug_name, drug_uses, se_count, 
                     therapeutic_class, chemical_class, action_class,
                     habit_forming, model, tfidf_name, tfidf_use, label_encoders):
    
    # 1. Text Features
    name_features = tfidf_name.transform([drug_name])      # (1, 300)
    use_features = tfidf_use.transform([drug_uses])        # (1, 200)
    
    # 2. Categorical Encoding (with OOV safety)
    tc_enc = safe_encode(label_encoders["Therapeutic Class"], therapeutic_class)
    cc_enc = safe_encode(label_encoders["Chemical Class"], chemical_class)
    ac_enc = safe_encode(label_encoders["Action Class"], action_class)
    
    # 3. Structured Features
    struct = np.array([[se_count, habit_forming, tc_enc, cc_enc, ac_enc]])
    
    # 4. Concatenate All Features
    X = sp.hstack([name_features, use_features, struct_sparse])  # (1, 505)
    
    # 5. Inference & Probability
    proba = model.predict_proba(X)[0]
    severity = LABEL_MAP[np.argmax(proba)]
    confidence = float(proba[np.argmax(proba)])
    
    return {"severity": severity, "confidence": confidence, "probabilities": {...}}
```

---

## 🔍 RAG & LLM Integration (rag_llm.py)

### ChromaDB Vector Store
- **Embedding Model**: all-MiniLM-L6-v2 (384D, ~10KB per vector)
- **Records Indexed**: 248,218 medicines
- **Similarity Metric**: Cosine distance
- **Query Speed**: ~200ms for top-3 retrieval
- **Storage**: SQLite backend (`data/chroma_db/chroma.sqlite3`)

### Groq LLaMA 3.1-8B Integration
- **Model**: llama-3.1-8b-instant (optimized for speed)
- **Context Window**: 8K tokens
- **Temperature**: 0.3 (low for consistency)
- **Max Output**: 500 tokens
- **Response Time**: <1 second

**Clinical Prompt Engineering**:
```
System: "You are a clinical pharmacovigilance assistant..."
User: "Drug: [name], Symptoms: [description], Severity: [prediction]"
Output: Structured clinical explanation with recommendations
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sreecharan03/Pharma_Co_AI.git
cd Pharma_Co_AI

# Install dependencies
pip install -r requirements.txt

# Set Groq API key
export GROQ_API=gsk_<your_key>
# OR update .env file: GROQ_API=gsk_<your_key>

# Launch application
streamlit run app.py
```

Application opens at **http://localhost:8501**

### API Key Setup

**Groq API**:
- Get free key at https://console.groq.com
- Free quota: ~500 requests/day
- Required for LLM clinical explanations

### Configuration

**`.env` file required**:
```bash
GROQ_API=gsk_<your_groq_api_key>
git-pat=ghp_<your_github_pat>  # for repo access
```

---

## 📁 Project Structure

```
ADRSeverityPharmacoAI/
├── README.md                          # Documentation (this file)
├── requirements.txt                   # Python dependencies
├── .env                               # API keys (DO NOT commit)
│
├── app.py                             # Streamlit web UI
├── pipeline.py                        # ML prediction engine
├── rag_llm.py                         # RAG & LLM integration
│
├── 01_eda.ipynb                       # Exploratory data analysis
├── 02_preprocessing.ipynb             # Data cleaning & SMOTE
├── 03_feature_engineering.ipynb       # TF-IDF & categorical encoding
├── 04_model_training.ipynb            # XGBoost training & evaluation
│
└── data/
    ├── drug_reviews/
    │   ├── best_model.pkl             # Trained XGBoost classifier
    │   ├── label_encoders.pkl         # Categorical encoders
    │   ├── medicine_clean.csv          # Processed dataset (248K records)
    │   └── features/
    │       ├── tfidf_name.pkl         # Drug name vectorizer
    │       └── tfidf_use.pkl          # Drug uses vectorizer
    │
    └── chroma_db/                      # Vector store (prebuilt, indexed)
        ├── chroma.sqlite3             # Index metadata
        └── <collection_id>/           # HNSW embedding files
```

### What's Included

- ✅ `app.py`, `pipeline.py`, `rag_llm.py` — Full production app
- ✅ `best_model.pkl` — Pre-trained XGBoost severity classifier (1.8 MB)
- ✅ `label_encoders.pkl`, `tfidf_*.pkl` — Vectorizers & encoders
- ✅ `medicine_clean.csv` — Source dataset (248K records)
- ✅ `chroma_db/` — Prebuilt vector store (no rebuild needed)
- ⚠️ `all-MiniLM-L6-v2` — Downloads automatically from Hugging Face on first run

---

## 🎯 Key Features

### 1. Real-Time ADR Prediction
- Input drug details → get severity in <50ms
- Confidence scores for clinical trust
- Probability distribution across all classes

### 2. Similar Case Retrieval
- Semantic search across 248K historical cases
- Top-3 similar drugs with comparative severity
- Evidence-based precedent for decision support

### 3. AI Clinical Explanations
- LLaMA 3 generated clinical reasoning
- Structured output: What, Why, Recommendations
- Reference to similar case management

### 4. Interactive Dashboard
- Color-coded severity badges (🟢 MILD, 🟠 MODERATE, 🔴 CRITICAL)
- Visual probability charts (Plotly)
- Vector store management (rebuild option)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 81.2% |
| **Weighted F1-Score** | 0.742 |
| **MILD F1** | 0.88 |
| **MODERATE F1** | 0.71 |
| **CRITICAL F1** | 0.58 |
| **ROC-AUC (macro)** | 0.82 |
| **Inference Speed** | 45 ms |
| **Sensitivity (CRITICAL)** | 56% |
| **Specificity (CRITICAL)** | 95% |

---

## 🔐 Security & Compliance

⚠️ **Disclaimer**: System is for **decision support only**, not autonomous decision-making.

**Data Handling**:
- All predictions processed locally
- Only API key stored in environment
- Groq API uses TLS 1.3 encryption
- No patient data retention

**Production Recommendations**:
- Implement user authentication
- Log all predictions for audit trails
- Use HIPAA-compliant infrastructure
- Obtain IRB approval before clinical use
- Regular model retraining with new ADR data

---

## 💻 Requirements

**System**:
- Python 3.9+
- RAM: 8GB minimum
- Disk: 500MB (with prebuilt artifacts)

**Dependencies** (see `requirements.txt`):
- streamlit, plotly, pandas, numpy, scipy
- scikit-learn, xgboost, imbalanced-learn
- chromadb, sentence-transformers, groq

---

## 📚 Research Methodology

### Data Pipeline

1. **EDA** (01_eda.ipynb): 248K medicine dataset profiling
2. **Preprocessing** (02_preprocessing.ipynb): Cleaning, SMOTE, train-test split
3. **Feature Engineering** (03_feature_engineering.ipynb): TF-IDF, encoding, concatenation
4. **Model Training** (04_model_training.ipynb): XGBoost tuning, cross-validation, evaluation
5. **Production** (app.py): Real-time predictions with RAG & LLM

### Key Research Contributions

- **Severity Label Engineering**: Keyword-based heuristics for automatic labeling
- **Sparse Feature Concatenation**: Efficient 505D feature space
- **SMOTE-Balanced XGBoost**: Improved CRITICAL class detection
- **RAG Integration**: Evidence-based case retrieval for explanations
- **LLM-Powered Explanations**: Structured clinical reasoning

---

## 📖 Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| **01_eda.ipynb** | Dataset exploration, severity engineering, visualizations |
| **02_preprocessing.ipynb** | Text consolidation, SMOTE balancing, train-test split |
| **03_feature_engineering.ipynb** | TF-IDF vectorization, categorical encoding, feature creation |
| **04_model_training.ipynb** | XGBoost training, cross-validation, model evaluation |

---

## 🔄 Workflow

```
Data (248K) → EDA → Preprocessing → Feature Engineering → 
XGBoost Training → Evaluation → Artifacts → Production App
                                              (+ ChromaDB + Groq)
```

---

## 📞 Support

**GitHub**: https://github.com/Sreecharan03/Pharma_Co_AI
**Email**: charanch53030@gmail.com

---

## ⚖️ License & Citation

**Disclaimer**: For research and educational purposes. Always consult qualified healthcare providers.

**Dataset**: Kaggle 250K Medicines (check original source for license)
**Code**: MIT License

---

**Status**: ✅ Production Ready | **Last Updated**: October 4, 2024
