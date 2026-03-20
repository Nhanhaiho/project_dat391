import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

# Config
MODEL_PATH = "models/phobert_binary_best_2"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

SPLITS = {
    "val": "data/processed/val_split.csv",
    "test": "data/processed/test_split_from_train.csv",
    "train": "data/processed/train_split.csv",
}


# Load model
print("Loading model from:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Device: {device}\\n")

if sns is None:
    print("Warning: seaborn is not installed. Falling back to matplotlib heatmaps.")


def draw_cm(ax, cm, xlabels, ylabels, annotate_size=13):
    """Draw confusion matrix using seaborn if available, else matplotlib fallback."""
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=xlabels,
            yticklabels=ylabels,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": annotate_size, "weight": "bold"},
            ax=ax,
        )
        return

    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=annotate_size,
                fontweight="bold",
            )


def predict(texts, batch_size=64):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    return all_preds


def plot_cm(cm, split_name, accuracy, macro_f1, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    draw_cm(
        ax=ax,
        cm=cm,
        xlabels=["Negative", "Positive"],
        ylabels=["Negative", "Positive"],
        annotate_size=14,
    )

    tn, fp, fn, tp = cm.ravel()
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix - {split_name.upper()} set\\n"
        f"Accuracy: {accuracy:.4f}  |  Macro F1: {macro_f1:.4f}  |  "
        f"Neg Recall: {neg_recall:.3f}  Pos Recall: {pos_recall:.3f}",
        fontsize=11,
        pad=12,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# Run evaluation for each split
all_results = {}

for split_name, csv_path in SPLITS.items():
    print(f"\\n{'=' * 50}")
    print(f"Evaluating: {split_name.upper()}  ({csv_path})")
    print(f"{'=' * 50}")

    df = pd.read_csv(csv_path)

    # Auto-detect columns for old/new schema
    if "review" in df.columns:
        texts = df["review"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
    elif "Comment" in df.columns:
        texts = df["Comment"].astype(str).tolist()
        labels = df["Rating"].astype(int).tolist()
    else:
        raise ValueError(f"Cannot find text/label columns in {csv_path}")

    print(f"  Samples: {len(texts)} | Pos: {sum(labels)} | Neg: {len(labels) - sum(labels)}")

    preds = predict(texts)

    acc = accuracy_score(labels, preds)
    prec_mac = precision_score(labels, preds, average="macro", zero_division=0)
    rec_mac = recall_score(labels, preds, average="macro", zero_division=0)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_bin = f1_score(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds)

    all_results[split_name] = {
        "accuracy": acc,
        "macro_f1": f1_mac,
        "macro_prec": prec_mac,
        "macro_rec": rec_mac,
        "binary_f1": f1_bin,
        "cm": cm,
    }

    print("\\n  Classification Report:")
    print(classification_report(labels, preds, target_names=["negative", "positive"], digits=4))

    save_path = os.path.join(REPORT_DIR, f"confusion_matrix_{split_name}_augmented.png")
    plot_cm(cm, split_name, acc, f1_mac, save_path)


# Summary
print(f"\\n{'=' * 50}")
print("SUMMARY - All splits")
print(f"{'=' * 50}")
print(f"{'Split':<8} {'Accuracy':>10} {'Macro F1':>10} {'Macro Prec':>12} {'Macro Rec':>11}")
print("-" * 55)
for split, r in all_results.items():
    print(
        f"{split:<8} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
        f"{r['macro_prec']:>12.4f} {r['macro_rec']:>11.4f}"
    )


# Combined figure (3 confusion matrices)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices - Augmented Multi-domain Dataset", fontsize=14, fontweight="bold", y=1.02)

for ax, (split_name, r) in zip(axes, all_results.items()):
    cm = r["cm"]
    tn, fp, fn, tp = cm.ravel()
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0

    draw_cm(
        ax=ax,
        cm=cm,
        xlabels=["Neg", "Pos"],
        ylabels=["Neg", "Pos"],
        annotate_size=13,
    )

    ax.set_title(
        f"{split_name.upper()}\\nAcc: {r['accuracy']:.4f}  F1: {r['macro_f1']:.4f}\\nNeg Recall: {neg_recall:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
combined_path = os.path.join(REPORT_DIR, "confusion_matrix_all_splits_augmented.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\\nCombined figure saved: {combined_path}")
