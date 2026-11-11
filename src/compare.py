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


if __name__ == "__main__":
    runs = load_all()
    runs.to_csv(RESULTS / "all_test_results.csv", index=False)

    ap = (runs.groupby(["dataset", "corruption_type", "missing_rate", "model"])["average_precision"]
              .mean().unstack("model").round(3))
    ap.to_csv(RESULTS / "comparison_ap.csv")

    plot(runs)

    pd.set_option("display.width", 200)
    print(ap.to_string())
    missing = sorted(set(runs["model"].unique()) ^ set(p.name for p in RESULTS.iterdir() if p.is_dir()))
    if missing:
        print("\nNo test.csv results found for:", ", ".join(missing))