from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.data import load_dataset, split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final_eval"
SPLIT_SEED, VALIDATION_SPLIT = 42, 0.25          # same as every notebook

MODELS = [("xgboost", "XGBoost", "#2a78d6"), ("random_forest", "Random forest", "#eb6834"),
          ("neural_network", "Neural network", "#1baf7a"),
          ("logistic_regression", "Logistic regression", "#eda100")]
DATASETS = [("adult", "Adult"), ("ai4i", "AI4I")]
BAR_METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("precision", "Precision"),
               ("recall", "Recall\n(sensitivity)"), ("specificity", "Specificity"), ("mcc", "MCC")]


def test_split(dataset):
    """The same 20% test set every notebook uses."""
    X, y = load_dataset(dataset)
    X_train_full, X_test, y_train_full, y_test = split(X, y, seed=SPLIT_SEED)
    return X_test, y_test


def scores_and_threshold(saved, X):
    """Positive-class probabilities from a saved model, whichever format it was saved in."""
    if isinstance(saved, dict):
        model, threshold = saved["model"], saved.get("threshold", 0.5)
        if "preprocessor" in saved:
            X = saved["preprocessor"].transform(X)
    else:                                   # an old save: the bare pipeline, no tuned threshold
        model, threshold = saved, 0.5
    return model.predict_proba(X)[:, 1], float(threshold)


def evaluate():
    rows, curves, matrices = [], {}, {}
    for ds, _ in DATASETS:
        X_test, y_test = test_split(ds)
        for key, name, _ in MODELS:
            path = ROOT / "models" / key / f"{ds}_clean.joblib"
            if not path.exists():
                print(f"Skipping {name} on {ds}: {path.relative_to(ROOT)} not found")
                continue
            score, threshold = scores_and_threshold(joblib.load(path), X_test)
            pred = (score >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            rows.append({
                "dataset": ds, "model": key, "threshold": threshold,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),      # sensitivity, TPR
                "specificity": tn / (tn + fp),                               # TNR
                "f1": f1_score(y_test, pred, zero_division=0),
                "mcc": matthews_corrcoef(y_test, pred),
                "roc_auc": roc_auc_score(y_test, score),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
            curves[(ds, key)] = roc_curve(y_test, score)
            matrices[(ds, key)] = np.array([[tp, fp], [fn, tn]])           # lecture layout
    return pd.DataFrame(rows), curves, matrices


def plot_bars(metrics, show=False):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(17, 5.5), sharey=True)
    x = np.arange(len(BAR_METRICS))
    present = [m for m in MODELS if m[0] in set(metrics.model)]
    width = 0.8 / len(present)
    for ax, (ds, ds_name) in zip(axes, DATASETS):
        for i, (key, name, color) in enumerate(present):
            row = metrics[(metrics.dataset == ds) & (metrics.model == key)]
            if row.empty:
                continue
            values = [row.iloc[0][m] for m, _ in BAR_METRICS]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, values, width * 0.92, color=color, label=name)
            for p, v in zip(pos, values):
                ax.text(p, max(v, 0) + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in BAR_METRICS])
        ax.set_ylim(0, 1.08)
        ax.set_title(ds_name, fontsize=13, loc="left")
        ax.grid(axis="y", color="#e4e4e0", lw=1)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Test score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=11)
    fig.text(0.5, 0.005, "Final models trained on clean data, scored on the untouched test set. "
             "Predictions use each model's best-F1 threshold from validation.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(OUT / "metrics_bars.png", dpi=200)
    plt.show() if show else plt.close(fig)


def plot_roc(metrics, curves, show=False):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 5.8))
    for ax, (ds, ds_name) in zip(axes, DATASETS):
        for key, name, color in MODELS:
            if (ds, key) not in curves:
                continue
            fpr, tpr, _ = curves[(ds, key)]
            auc = metrics[(metrics.dataset == ds) & (metrics.model == key)].iloc[0]["roc_auc"]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="#555555", lw=1.2, label="Random guess (AUC = 0.5)")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), title=ds_name,
               xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")
        ax.title.set_ha("left"); ax.title.set_x(0)
        ax.grid(color="#e4e4e0", lw=1)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=10)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=200)
    plt.show() if show else plt.close(fig)


def plot_confusion(matrices, show=False):
    import matplotlib.pyplot as plt
    present = [m for m in MODELS if any((ds, m[0]) in matrices for ds, _ in DATASETS)]
    fig, axes = plt.subplots(len(DATASETS), len(present), figsize=(3.6 * len(present), 7.2), squeeze=False)
    positive = {"adult": "> $50k", "ai4i": "Failure"}
    for r, (ds, ds_name) in enumerate(DATASETS):
        for c, (key, name, _) in enumerate(present):
            ax = axes[r, c]
            if (ds, key) not in matrices:
                ax.axis("off")
                continue
            m = matrices[(ds, key)]
            ax.imshow(m / m.sum(axis=0, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    share = m[i, j] / m[:, j].sum()
                    ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=12,
                            color="white" if share > 0.6 else "#14213D")
            ticks = [positive[ds], "Other"]
            ax.set_xticks([0, 1], [f"Actual\n{t}" for t in ticks], fontsize=9)
            ax.set_yticks([0, 1], [f"Predicted\n{t}" for t in ticks], fontsize=9)
            ax.set_title(f"{ds_name}: {name}", fontsize=11, loc="left")
    fig.text(0.5, 0.005, "Rows = predicted, columns = actual (Lecture 10 layout: TP top left, TN bottom right). "
             "Shading = share of each actual class.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "confusion_matrices.png", dpi=200)
    plt.show() if show else plt.close(fig)


def main(show=False):
    """show=True draws the figures in the notebook as well as saving them."""
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, curves, matrices = evaluate()
    if metrics.empty:
        print("No saved models found under models/. Run the model notebooks first.")
        return metrics
    metrics.to_csv(OUT / "metrics.csv", index=False)
    plot_bars(metrics, show)
    plot_roc(metrics, curves, show)
    plot_confusion(matrices, show)
    cols = ["dataset", "model", "accuracy", "macro_f1", "precision", "recall", "specificity", "mcc", "roc_auc"]
    print(metrics[cols].round(3).to_string(index=False))
    print(f"\nSaved metrics.csv, metrics_bars.png, roc_curves.png, confusion_matrices.png to {OUT}")
    return metrics


if __name__ == "__main__":
    main(show=True)