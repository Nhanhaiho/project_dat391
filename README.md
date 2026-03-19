# Vietnamese Food Review Sentiment Analysis with PhoBERT

A machine learning project for Vietnamese sentiment analysis on food reviews using **PhoBERT** (`vinai/phobert-base`) fine-tuned for **binary classification** (positive/negative).

## 1. Project Overview
This project develops an end-to-end sentiment analysis pipeline for Vietnamese food reviews. The main objective is to classify customer feedback into positive or negative sentiment, helping stakeholders quickly understand user satisfaction trends.

The project combines:
- Data preprocessing and exploratory data analysis (EDA)
- Baseline ML models for benchmarking
- Transformer fine-tuning with PhoBERT
- Evaluation with classification metrics and confusion matrix visualization

Development was performed using **Google Colab** (model training) and **VSCode** (modular code development and project organization).

## 2. Problem Statement
Vietnamese food review data is noisy and linguistically challenging due to:
- Informal writing style
- Local vocabulary and abbreviations
- Inconsistent punctuation and spelling

The goal is to build a robust binary sentiment classifier that can generalize well on unseen Vietnamese food reviews.

## 3. Dataset
- **Primary dataset**: `data/raw/vsa_food_rv_train.csv`
- **Additional files**:
  - `data/raw/vsa_food_rv_test.csv`
  - `data/raw/vietnamese-stopwords-dash.txt`

### Data Splits
Processed splits used for training and evaluation:
- `data/processed/train_split.csv`
- `data/processed/val_split.csv`
- `data/processed/test_split_from_train.csv`

### Target Labels
- `0` -> Negative
- `1` -> Positive

## 4. Methodology
1. **Data Understanding & EDA**
   - Analyze label distribution and review length
   - Inspect potential class imbalance and text characteristics

2. **Preprocessing**
   - Basic text cleaning for baseline models
   - Lightweight preprocessing for PhoBERT (preserve semantic-rich tokens)

3. **Baseline Modeling**
   - Traditional ML baselines (implemented in `src/baseline.py`) for comparison

4. **PhoBERT Fine-tuning**
   - Tokenization with HuggingFace tokenizer
   - Fine-tune `vinai/phobert-base` with PyTorch + Transformers
   - Validate on held-out validation split

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix on test split

## 5. Model Architecture (PhoBERT)
The final model is based on **PhoBERT Base**, a Vietnamese pretrained Transformer model.

- **Backbone**: `vinai/phobert-base`
- **Task head**: Sequence classification head (`num_labels=2`)
- **Framework**: HuggingFace Transformers + PyTorch
- **Training pipeline**: `Trainer` API with early stopping and best-model selection by macro F1

High-level flow:
1. Input review text
2. PhoBERT tokenization
3. Transformer encoding
4. Classification head outputs logits
5. Argmax -> Positive/Negative label

## 6. Project Structure
```text
DAT391/
|
|-- data/
|   |-- processed/
|   |   |-- train_split.csv
|   |   |-- val_split.csv
|   |   `-- test_split_from_train.csv
|   |
|   `-- raw/
|       |-- vsa_food_rv_train.csv
|       |-- vsa_food_rv_test.csv
|       `-- vietnamese-stopwords-dash.txt
|
|-- models/
|   `-- phobert_binary_best/
|
|-- notebooks/
|   |-- 01_data_understanding_preprocessing_and_eda.ipynb
|   |-- 02_baseline_models.ipynb
|   `-- 03_phobert_training.ipynb
|
|-- reports/
|   |-- confusion_matrix_test.png
|   `-- train_phobert.jpeg
|
|-- src/
|   |-- baseline.py
|   |-- data_loader.py
|   |-- eda.py
|   |-- evaluate_phobert.py
|   |-- phobert_pipeline.py
|   |-- predict.py
|   `-- preprocessing.py
|
|-- app.py
|-- .gitignore
`-- README.md
```

## 7. Installation
### Prerequisites
- Python 3.9+
- pip
- (Recommended) virtual environment

### Setup
```bash
# clone repository
git clone <your-repo-url>
cd DAT391

# create virtual environment
python -m venv venv

# activate (Windows)
venv\\Scripts\\activate

# install dependencies
pip install -U pip
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
```

If you maintain a dependency file, you can use:
```bash
pip install -r requirements.txt
```

## 8. Training the Model
Training was primarily run in Google Colab through:
- `notebooks/03_phobert_training.ipynb`

You can also reuse the pipeline functions from `src/phobert_pipeline.py` in a Python script or notebook:
```python
from src.phobert_pipeline import (
    load_splits,
    build_hf_datasets,
    tokenize_datasets,
    build_trainer,
)

train_df, val_df, test_df = load_splits(
    "data/processed/train_split.csv",
    "data/processed/val_split.csv",
    "data/processed/test_split_from_train.csv",
)

train_ds, val_ds, test_ds = build_hf_datasets(train_df, val_df, test_df)
tokenizer, train_tok, val_tok, test_tok = tokenize_datasets(train_ds, val_ds, test_ds)
trainer = build_trainer(train_tok, val_tok, tokenizer, output_dir="models/phobert_binary")
trainer.train()
trainer.save_model("models/phobert_binary_best")
```

## 9. Running Prediction
Use the helper in `src/predict.py`:

```python
from src.predict import predict_sentiment

text = "Quan phuc vu nhanh, mon an ngon va gia hop ly."
print(predict_sentiment(text))  # Positive / Negative
```

To evaluate on the test split and regenerate confusion matrix:
```bash
python src/evaluate_phobert.py
```

## 10. Results
The PhoBERT-based classifier demonstrates strong performance for Vietnamese food review sentiment classification.

### Evaluation Artifacts
- Confusion matrix: `reports/confusion_matrix_test.png`
- Training snapshot: `reports/train_phobert.jpeg`

> Add your final numeric metrics (Accuracy, Precision, Recall, F1) here after your latest run for a fully reproducible portfolio report.

## 11. Future Improvements
- Add hyperparameter tuning (batch size, learning rate, max length)
- Handle class imbalance with weighted loss or resampling
- Expand to multi-class sentiment (negative/neutral/positive)
- Deploy inference API (FastAPI/Flask) for real-time predictions
- Add experiment tracking (Weights & Biases / MLflow)
- Include automated tests and CI for training/evaluation scripts

## 12. Authors
- **Nhanhaiho** - Project implementation, experimentation, and reporting

---
If you use this project in your work, please cite or reference the repository.

