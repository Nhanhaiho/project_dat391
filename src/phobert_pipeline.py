import os
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME = "vinai/phobert-base"


def load_splits(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df[["Comment", "Rating"]].dropna()
    val_df = val_df[["Comment", "Rating"]].dropna()
    test_df = test_df[["Comment", "Rating"]].dropna()

    train_df["Rating"] = train_df["Rating"].astype(int)
    val_df["Rating"] = val_df["Rating"].astype(int)
    test_df["Rating"] = test_df["Rating"].astype(int)

    return train_df, val_df, test_df


def build_hf_datasets(train_df, val_df, test_df):
    train_ds = Dataset.from_pandas(
        train_df.rename(columns={"Comment": "text", "Rating": "label"})[["text", "label"]]
    )
    val_ds = Dataset.from_pandas(
        val_df.rename(columns={"Comment": "text", "Rating": "label"})[["text", "label"]]
    )
    test_ds = Dataset.from_pandas(
        test_df.rename(columns={"Comment": "text", "Rating": "label"})[["text", "label"]]
    )
    return train_ds, val_ds, test_ds


def tokenize_datasets(train_ds, val_ds, test_ds):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)
    test_tok = test_ds.map(tokenize_fn, batched=True)

    return tokenizer, train_tok, val_tok, test_tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
    }


def build_trainer(train_tok, val_tok, tokenizer, output_dir="models/phobert_binary"):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    return trainer


def evaluate_on_test(trainer, test_tok, test_df, figure_path=None):
    preds_output = trainer.predict(test_tok)
    logits = preds_output.predictions
    y_true = test_df["Rating"].to_numpy()
    y_pred = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }

    if figure_path is not None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
        disp.plot()
        plt.title("PhoBERT Confusion Matrix")
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()

    return metrics, y_pred


def save_metrics(metrics, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)