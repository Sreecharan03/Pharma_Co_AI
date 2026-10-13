# AI-Driven Pharmacovigilance: ADR Severity Prediction with Retrieval-Augmented Explanation

M.Tech project, Department of Computer Science and Engineering, Vishnu Women's University (2026-27).

The system predicts the severity of adverse drug reactions (ADRs) as **Mild**, **Moderate** or **Critical** from drug metadata. It then retrieves similar historical cases and asks a large language model to write a clinical explanation grounded in those cases.

> This is a research prototype. It is not a medical device and must not replace clinical judgement.

## 1. Motivation and base paper

Most ADR models estimate whether an event is serious, or how often a reaction occurs, and give the clinician no graded severity and no explanation.

The base paper for this work is:

> A. M. Mostafa et al., "Innovative Tailored Semantic Embedding and Machine Learning for Precise Prediction of Drug-Drug Interaction Seriousness," *IEEE Access*, vol. 13, pp. 49249-49270, 2025. DOI: 10.1109/ACCESS.2025.3552239

That paper classifies drug-drug interactions from FAERS reports as serious or non-serious (binary) with BioWordVec embeddings and CatBoost (F1 73.32%, ROC-AUC 0.84). The authors name binary-only classification, a single dataset and limited interpretability as limitations. This project addresses them with three severity levels, a different dataset, and a retrieval-based explanation layer.

Secondary reference: J. Wan et al., "DeepADR: multimodal prediction of adverse drug reaction frequency by integrating early-stage drug discovery information via Kolmogorov-Arnold networks," *Briefings in Bioinformatics* 26(6), bbaf695, 2025.

## 2. System overview

```
Drug details (name, uses, class, habit-forming, side-effect count)
        |
        +--> TF-IDF + structured features (505) --> XGBoost --> severity + confidence
        |
        +--> sentence embedding --> ChromaDB (cosine) --> top-3 similar cases
                                                              |
        severity + confidence + similar cases ---------> LLM --> clinical explanation
```

| File | Role |
|---|---|
| `pipeline.py` | Loads the saved model, TF-IDF vectorizers and encoders; builds the feature vector; returns severity, confidence and class probabilities. Unseen class values fall back to code 0. |
| `rag_llm.py` | Builds and queries the ChromaDB vector store (`all-MiniLM-L6-v2`, cosine) and calls the Groq API (`llama-3.1-8b-instant`, temperature 0.3, 500 tokens). |
| `app.py` | Streamlit interface: input form, severity badge, probability chart, similar cases, explanation, vector-store rebuild button. |

## 3. Dataset

[250K Medicines Usage, Side Effects and Substitutes](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes) (Kaggle): **248,218** medicine records, 58 columns. Side-effect columns 35-41 are empty and were dropped.

The 42 side-effect columns and 5 use columns were merged into two text fields. Missing chemical and action classes (about 44%) were filled with "Unknown", and the 0.03% missing therapeutic classes with the mode.

### Severity labels

The dataset has no severity field, so labels are **derived by rules** (`01_eda.ipynb`):

1. **Critical** if the side-effect text contains any life-threatening keyword (for example cardiac, liver, seizure, stroke, anaphylaxis, bleeding).
2. Otherwise **Moderate** if the drug is habit-forming, has 10 or more side effects, or the text contains a moderate keyword (for example allergic reaction, depression, tremor).
3. Otherwise **Mild**.

| Class | Records | Share |
|---|---|---|
| Mild | 126,918 | 51.1% |
| Moderate | 54,887 | 22.1% |
| Critical | 66,413 | 26.8% |

These are heuristic labels, not clinician annotations. Reported accuracy therefore measures agreement with the rules. See the limitations below.

## 4. Method

**Split.** Stratified 80/20, `random_state=42`: 198,574 training and 49,644 test records.

**Features (505).** TF-IDF of drug name (300 features, 1-2 grams) and drug uses (200 features, 1-2 grams), plus five structured fields: side-effect count, habit-forming flag, and label-encoded therapeutic, chemical and action class. TF-IDF on the side-effect text was removed to avoid leaking the label source.

**Models.** All three use balanced class weights.

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 78.85% | 0.7916 |
| Random Forest (200 trees, depth 20) | 92.25% | 0.9229 |
| **XGBoost** (300 rounds, depth 6, lr 0.1) | **96.48%** | **0.9648** |

XGBoost per-class F1 on the test set: Mild 0.98, Moderate 0.94, Critical 0.96.

**Retrieval.** Each medicine becomes one document (name, uses, side effects) embedded with `all-MiniLM-L6-v2` and stored in ChromaDB. The top three neighbours are shown to the user and given to the LLM together with the prediction and its confidence.

## 5. Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your Groq key
streamlit run app.py        # http://localhost:8501
```

The Groq key is read from `GROQ_API`, or can be pasted into the app sidebar. A free key is available at https://console.groq.com. The sentence-embedding model downloads from Hugging Face on first run.

The repository includes the trained model, vectorizers, encoders, the cleaned CSV and a prebuilt vector store (`data/`), so the app works immediately. The notebooks (`01_eda` to `04_model_training`) reproduce the pipeline but use absolute paths from the original environment and need path edits.

## 6. Known limitations

- **Heuristic labels.** The rules use side-effect count and habit-forming status, which are also model inputs. Part of the accuracy is the model recovering those rules. An ablation without these two features is planned.
- **Single split.** One stratified split, no cross-validation, no hyperparameter search. The test set is also passed as `eval_set` to XGBoost for logging (no early stopping is used).
- **Retrieval query.** The app builds the query from the drug name, uses and the predicted severity, which biases results toward matching labels. The indexed text also contains the severity label.
- **Field naming.** "Drug Uses" is passed to the LLM as the reported symptoms.
- **No evaluation of explanations.** Retrieval precision and explanation faithfulness are not yet measured.
- **Input coverage.** The app dropdown offers 15 therapeutic classes, while the data has 22.

## 7. Planned work

1. Ablation without side-effect count and habit-forming status.
2. Benchmark the base paper's classifier family (CatBoost, LightGBM, XGBoost, Extra Trees, KNN, Naive Bayes) and a stacking ensemble, with Bayesian tuning and repeated cross-validation (mean and standard deviation).
3. Embedding experiment: BioWordVec or BioBERT against TF-IDF.
4. Compare imbalance strategies (downsampling, SMOTE, class weights) and report Critical-class recall.
5. Fix retrieval leakage and add a separate symptoms input.
6. Evaluate retrieval and explanations: precision@k, faithfulness (LLM-as-judge with manual review), with and without retrieval.
7. SHAP explanations for individual predictions.

## 8. Disclaimer

For research and education only. Always consult a qualified healthcare professional about drug safety.
