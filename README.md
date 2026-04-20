# YouTube Video Relevance Labeling & Filtering Pipeline

<div style="display: flex; gap: 20px;">
  <img width="1818" height="941" alt="image" src="https://github.com/user-attachments/assets/b49fa1f0-a993-48ff-bf56-4c0f26e607a2" />

</div>

## Project Overview

This is the second part of a multi-stage personal Youtube learning assistant/librarian. While the [first part](https://github.com/wormzz11/YouTube-Data-Collection-Labeling-System) handles fetching video metadata from YouTube and manual labeling through a UI, this project takes that labeled data and automates the relevance filtering process at scale.

It trains a lightweight classifier on your labeled metadata, runs it against unlabeled video batches, and uses MS MARCO cross-encoder re-ranking as a second filter — producing set of relevant videos ready for downstream, more expensive enrichment.

<img width="727" height="623" alt="image" src="https://github.com/user-attachments/assets/7451dc31-13b5-45eb-922c-ea9bdf00e845" />



## Features

- Trains a relevance classifier on video metadata (title + theme + relevance)
- Supports TF-IDF and MiniLM transformer-based vectorization
- Runs batch predictions with configurable confidence thresholds
- Splits output into certain positives, certain negatives, and manual review
- Accumulates certain positives across runs into a deduplicated master file
- Re-ranks certain positives using MS MARCO cross-encoder for deeper semantic filtering
- Exports filtered results above a user-defined score threshold
- Full Streamlit UI for inspecting data between every stage

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Sentence-Transformers
- Joblib
- Streamlit

## How It Works

**Model Training**
- Loads labeled CSV (exported from the first project or manually prepared)
- Trains a pipeline on `title + theme` as combined text feature and manually set relevancy from possibly the previous tool.
- Evaluates on a held-out test split — accuracy, F1, ROC AUC, confusion matrix

**Batch Prediction & Triage**  

<img width="1818" height="941" alt="image" src="https://github.com/user-attachments/assets/2f3a8d79-c9fc-4fe0-9169-804983b58106" />

- Runs the model on unlabeled data and scores each row with `predict_proba`
- Applies `high_pos` and `high_neg` thresholds to bucket results:
  - **Certain positives** — auto-labeled as relevant(1.0), saved to master file
  - **Certain negatives** — auto-labeled as irrelevant (0.0), saved to master file
  - **Manual review** — uncertain scores saved per batch for human labeling (None)
- Each batch saved individually with a timestamp for auditing (batch_YearMmonthDay_HoursMinutesSeconds)

**MS MARCO Re-ranking**  

<img width="1818" height="941" alt="image" src="https://github.com/user-attachments/assets/7fc36f72-5aa6-495f-8875-0177297e5f33" />


- Loads the certain positives pool and scores each `(theme, title)` pair with a cross-encoder
- User sets a score threshold to keep only the most semantically relevant rows
- Filtered output saved as CSV, ready for enrichment (transcripts, audio, etc.)

## Models

**Models & Evaluation**

The classifier is designed as a high-recall “loose sieve”, where missing relevant videos is more costly than passing some irrelevant ones downstream. Precision is later improved via cross-encoder re-ranking.

**Evaluated Approaches**

Two feature extraction strategies were compared using a logistic regression classifier:  

TF-IDF (bag-of-words)
Transformer embeddings (MiniLM via Sentence-Transformers)

Both models were evaluated on a held-out test set under the same conditions.
| Model               | Features | Recall    | Precision | F1        | ROC AUC   |
| ------------------- | -------- | --------- | --------- | --------- | --------- |
| Logistic Regression | TF-IDF   | 0.937     | 0.353     | 0.513     | 0.761     |
| Logistic Regression | MiniLM   | **0.942** | **0.402** | **0.564** | **0.773** |

**Interpretation**
Both models achieve very high recall (~0.94), satisfying the primary objective of the first-stage filter
Transformer-based features provide a clear improvement in precision and F1 score
Higher ROC AUC indicates better ranking quality, which is important for threshold-based triage
Design Decision

**The pipeline supports both approaches**:

Transformer + Logistic Regression (default)
Used for better semantic understanding and improved filtering quality
TF-IDF + Logistic Regression (fallback)
Available for faster inference on large datasets or limited compute environments

This tradeoff allows the system to adapt between performance and scalability, depending on workload requirements.

Notes
Metrics are computed on a held-out test split
Thresholds are tuned to maintain high recall while controlling the size of the manual review bucket
Final precision is further improved in Stage 2 using MS MARCO cross-encoder re-ranking

## Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd <repo>
```

2. Install dependencies:
```bash
uv sync
```

3. Run the Streamlit UI:
```bash
uv run streamlit run src/Labeling_data_ingestion/app/streamlit_app.py
```

Or run training directly from the CLI without the UI:
```bash
uv run python src/Labeling_data_ingestion/train/run_pipeline.py
```

## Configuration

Thresholds are managed in `src/Labeling_data_ingestion/config.py` via `ThresholdConfig`:

| Parameter | Purpose |
|---|---|
| `high_pos` | Minimum score to auto-label as certain positive |
| `high_neg` | Maximum score to auto-label as certain negative |
| `DATA_PATH` | Default path to training CSV |

All values can be overridden live in the Streamlit sidebar without touching the config file.

## Input / Output

| | Format | Description |
|---|---|---|
| **Input** | CSV | `title`, `theme`, `relevant` — from the [labeling tool](https://github.com/wormzz11/YouTube-Data-Collection-Labeling-System) or manually prepared |
| **Output** | CSV | Ranked and filtered certain positives ready for enrichment |

## Notes & Limitations

- Manual review rows require human labeling before being merged back into training data
- Deduplication on the master file is based on `title`
- MS MARCO scores are raw logits, not probabilities — they can be negative
- First run of MS MARCO downloads the cross-encoder model (~80MB)
- Transformer training is slow on CPU for large datasets — TF-IDF is the faster fallback**
