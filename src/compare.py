from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
METRICS = ["accuracy", "precision", "recall", "f1", "average_precision"]


def load_all():
    frames = []
    for path in sorted(RESULTS.glob("*/*/test.csv")):
        model, condition = path.parent.parent.name, path.parent.name
        df = pd.read_csv(path)
        df["model"] = model
        df["corruption_type"] = condition
        if "missing_rate" not in df:
            df["missing_rate"] = 0.0          # clean runs have no corruption rate
        frames.append(df)

    # Logistic regression saved its clean runs as test_metrics.csv, not clean/test.csv
    lr = RESULTS / "logistic_regression" / "test_metrics.csv"
    if lr.exists() and not (RESULTS / "logistic_regression" / "clean" / "test.csv").exists():
        df = pd.read_csv(lr)
        df["model"], df["corruption_type"], df["missing_rate"] = "logistic_regression", "clean", 0.0
        frames.append(df)

    keep = ["model", "dataset", "corruption_type", "missing_rate", "seed", *METRICS]
    return pd.concat(frames, ignore_index=True)[keep]


# One fixed color and marker per model, so a model looks the same in every panel.
# XGBoost with blanks filled in first is the same model family: same blue, dashed.
STYLES = {
    "xgboost":             dict(label="XGBoost",                     color="#2a78d6", marker="o", ls="-"),
    "xgboost_imputed":     dict(label="XGBoost (blanks filled in)",  color="#2a78d6", marker="o", ls="--"),
    "random_forest":       dict(label="Random forest",               color="#eb6834", marker="s", ls="-"),
    "neural_network":      dict(label="Neural network",              color="#1baf7a", marker="^", ls="-"),
    "logistic_regression": dict(label="Logistic regression",         color="#eda100", marker="D", ls="-"),
}
PANELS = [("mcar", "Random missing values"), ("mnar", "Value-dependent missing"),
          ("noise", "Measurement noise"), ("label_flip", "Label flips")]
DATASETS = [("adult", "Adult"), ("ai4i", "AI4I")]


def plot(runs, out=RESULTS / "comparison_ap.png"):
    """Average precision vs. corruption rate: rows = dataset, columns = corruption type."""
    import matplotlib.pyplot as plt

    mean_ap = (runs.groupby(["model", "dataset", "corruption_type", "missing_rate"])
                   ["average_precision"].mean())
    # "mcar_imputed" is XGBoost with blanks filled in; plot it as its own line in the MCAR panel
    clean = mean_ap.xs("clean", level="corruption_type")

    fig, axes = plt.subplots(len(DATASETS), len(PANELS), figsize=(19, 8.5), sharey="row")
    for r, (ds, ds_name) in enumerate(DATASETS):
        for c, (kind, title) in enumerate(PANELS):
            ax = axes[r, c]
            lines = [(m, m, kind) for m in ["xgboost", "random_forest", "neural_network", "logistic_regression"]]
            if kind in ("mcar", "mnar"):
                lines.insert(1, ("xgboost_imputed", "xgboost", f"{kind}_imputed"))
            for style_key, model, cond in lines:
                try:
                    curve = mean_ap.loc[(model if style_key != "xgboost_imputed" else "xgboost", ds, cond)]
                except KeyError:
                    curve = pd.Series(dtype=float)
                if (model, ds) in clean.index:   # start every line at the clean score (0%)
                    curve = pd.concat([pd.Series({0.0: clean.loc[(model, ds)].iloc[0]}), curve])
                if curve.empty:
                    continue
                # A model with no corrupted runs yet shows as a single clean-data point at 0%
                s = STYLES[style_key]
                ax.plot(100 * curve.index, curve.values, color=s["color"], marker=s["marker"],
                        ls=s["ls"], lw=2, ms=8, label=s["label"],
                        markeredgecolor="white", markeredgewidth=1.5)
            ax.set_title(f"{ds_name}: {title}", fontsize=12, loc="left")
            ax.grid(axis="y", color="#e4e4e0", lw=1)
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel("Test average precision")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("Share of training data corrupted")

    # One shared legend: every model that appears in any panel
    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=len(handles),
               frameon=False, fontsize=11)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                "f1": "F1", "average_precision": "Average\nprecision"}
# Share of positives in each dataset: what "always predict the majority class" scores.
# Accuracy of that do-nothing model = 1 - rate; average precision of a random ranking = rate.
POSITIVE_RATE = {"adult": 0.239, "ai4i": 0.034}


def plot_metrics(runs, corruption_type="clean", missing_rate=0.0,
                 out=RESULTS / "metrics_by_model.png"):
    """Grouped bars: every metric for every model, one panel per dataset.
    Defaults to the clean (final) models; pass e.g. corruption_type="mcar", missing_rate=0.4
    to compare the models trained on 40% randomly missing data instead."""
    import numpy as np
    import matplotlib.pyplot as plt

    sel = runs[(runs.corruption_type == corruption_type) & (runs.missing_rate == missing_rate)]
    stats = sel.groupby(["dataset", "model"])[METRICS].agg(["mean", "std"])
    models = [m for m in ["xgboost", "random_forest", "neural_network", "logistic_regression"]
              if m in sel.model.unique()]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5.5), sharey=True)
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)
    for ax, (ds, ds_name) in zip(axes, DATASETS):
        for i, model in enumerate(models):
            if (ds, model) not in stats.index:
                continue
            row = stats.loc[(ds, model)]
            means = [row[(m, "mean")] for m in METRICS]
            stds = [0 if np.isnan(row[(m, "std")]) else row[(m, "std")] for m in METRICS]
            pos = x - 0.4 + width * (i + 0.5)
            s = STYLES[model]
            ax.bar(pos, means, width * 0.92, yerr=stds, color=s["color"], label=s["label"],
                   error_kw=dict(ecolor="#555555", lw=1, capsize=2))
            for p, v, sd in zip(pos, means, stds):   # label sits above the error bar
                ax.text(p, v + sd + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
        # reference lines: what a model that learned nothing would score
        rate = POSITIVE_RATE[ds]
        for metric, value in [("accuracy", 1 - rate), ("average_precision", rate)]:
            j = METRICS.index(metric)
            ax.hlines(value, j - 0.45, j + 0.45, colors="#333333", linestyles="--", lw=1.4,
                      label="Model that learned nothing" if metric == "accuracy" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_NAMES[m] for m in METRICS])
        ax.set_ylim(0, 1.08)
        ax.set_title(ds_name, fontsize=13, loc="left")
        ax.grid(axis="y", color="#e4e4e0", lw=1)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Test score (mean of 3 runs, bars = ±1 std)")
    condition = "clean data" if corruption_type == "clean" else f"{missing_rate:.0%} {corruption_type}"
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=11)
    fig.text(0.5, 0.005, f"Models trained on {condition}; scored on the same untouched test set. "
             "Precision, recall and F1 use each model's best-F1 threshold from validation.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    runs = load_all()
    runs.to_csv(RESULTS / "all_test_results.csv", index=False)

    ap = (runs.groupby(["dataset", "corruption_type", "missing_rate", "model"])["average_precision"]
              .mean().unstack("model").round(3))
    ap.to_csv(RESULTS / "comparison_ap.csv")

    plot(runs)
    plot_metrics(runs)

    pd.set_option("display.width", 200)
    print(ap.to_string())
    missing = sorted(set(runs["model"].unique()) ^ set(p.name for p in RESULTS.iterdir() if p.is_dir()))
    if missing:
        print("\nNo test.csv results found for:", ", ".join(missing))