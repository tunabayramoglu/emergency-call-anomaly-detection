import matplotlib
matplotlib.use("Agg")  # Headless backend; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common as _common
from common import LABELS, y_of, emotion_features, set_seed
# CLASS_WEIGHTING read live off `_common.CLASS_WEIGHTING` — see
# early.py's comment for why (import-time binding would freeze
# the flag and ignore later toggles from the driver notebook).


def run_majority(splits, emb, seed, device, encoder_name):
    """Majority class baseline. Predicts train-set majority class for all test rows.

    Ignores emb, device, encoder_name; accepted for signature uniformity.

    Args:
        splits: dict with 'train', 'val', 'test' keys containing Row lists
        emb: dict (ignored)
        seed: random seed
        device: device string (ignored)
        encoder_name: encoder name (ignored)

    Returns:
        (y_true_test, y_pred_test) both shape (N_test,) with values in {0,1,2}
    """
    set_seed(seed)

    # Get majority class from train split only
    y_train = y_of(splits["train"])
    majority_class = np.bincount(y_train).argmax()

    # Get true labels for test
    y_test = y_of(splits["test"])

    # Predict majority class for all test rows
    y_pred_test = np.full(len(y_test), majority_class, dtype=np.int64)

    return y_test, y_pred_test


def run_emotion_only(splits, emb, seed, device, encoder_name):
    """Emotion-only baseline. Trains LogisticRegression on 6-dim one-hot emotion features.

    Ignores emb, device, encoder_name; accepted for signature uniformity.
    Fits on TRAIN split, predicts on TEST split.

    Args:
        splits: dict with 'train', 'val', 'test' keys containing Row lists
        emb: dict (ignored)
        seed: random seed for LogisticRegression
        device: device string (ignored)
        encoder_name: encoder name (ignored)

    Returns:
        (y_true_test, y_pred_test) both shape (N_test,) with values in {0,1,2}
    """
    set_seed(seed)

    # Get one-hot emotion features and labels
    X_train = emotion_features(splits["train"], "train", seed)
    y_train = y_of(splits["train"])
    X_test = emotion_features(splits["test"], "test", seed)
    y_test = y_of(splits["test"])

    # Fit LogisticRegression on train, predict on test. class_weight="balanced"
    # honours the same all-or-nothing CLASS_WEIGHTING switch used by the
    # torch-trained methods (common.CLASS_WEIGHTING).
    model = LogisticRegression(
        max_iter=1000, random_state=seed,
        class_weight=("balanced" if _common.CLASS_WEIGHTING else None),
    )
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    return y_test, y_pred_test


def run_emotion_only_logits(splits, emb, seed, device, encoder_name):
    """Same computation as `run_emotion_only`, also returning val/test
    decision-function scores (pseudo-logits) for the post-hoc class-bias
    correction step. No retraining."""
    set_seed(seed)
    X_train = emotion_features(splits["train"], "train", seed)
    y_train = y_of(splits["train"])
    X_val = emotion_features(splits["val"], "val", seed)
    y_val = y_of(splits["val"])
    X_test = emotion_features(splits["test"], "test", seed)
    y_test = y_of(splits["test"])

    model = LogisticRegression(
        max_iter=1000, random_state=seed,
        class_weight=("balanced" if _common.CLASS_WEIGHTING else None),
    )
    model.fit(X_train, y_train)
    val_logits = model.decision_function(X_val)
    test_logits = model.decision_function(X_test)
    return y_val, val_logits, y_test, test_logits


def run_text_only(splits, emb, seed, device, encoder_name):
    """Text-only baseline. Trains LogisticRegression on frozen text embeddings.

    Uses emb['train'] and emb['test']; ignores device, encoder_name.
    Fits on TRAIN split, predicts on TEST split.

    Args:
        splits: dict with 'train', 'val', 'test' keys containing Row lists
        emb: dict with 'train', 'val', 'test' embeddings (N, D) float32
        seed: random seed for LogisticRegression
        device: device string (ignored)
        encoder_name: encoder name (ignored)

    Returns:
        (y_true_test, y_pred_test) both shape (N_test,) with values in {0,1,2}
    """
    set_seed(seed)

    # Get text embeddings and labels
    X_train = emb["train"]
    y_train = y_of(splits["train"])
    X_test = emb["test"]
    y_test = y_of(splits["test"])

    # Fit LogisticRegression on train, predict on test. class_weight="balanced"
    # honours the same all-or-nothing CLASS_WEIGHTING switch used by the
    # torch-trained methods (common.CLASS_WEIGHTING).
    model = LogisticRegression(
        max_iter=1000, random_state=seed,
        class_weight=("balanced" if _common.CLASS_WEIGHTING else None),
    )
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    return y_test, y_pred_test


def run_text_only_logits(splits, emb, seed, device, encoder_name):
    """Same computation as `run_text_only`, also returning val/test
    decision-function scores (pseudo-logits) for the post-hoc class-bias
    correction step. No retraining."""
    set_seed(seed)
    X_train = emb["train"]
    y_train = y_of(splits["train"])
    X_val = emb["val"]
    y_val = y_of(splits["val"])
    X_test = emb["test"]
    y_test = y_of(splits["test"])

    model = LogisticRegression(
        max_iter=1000, random_state=seed,
        class_weight=("balanced" if _common.CLASS_WEIGHTING else None),
    )
    model.fit(X_train, y_train)
    val_logits = model.decision_function(X_val)
    test_logits = model.decision_function(X_test)
    return y_val, val_logits, y_test, test_logits


def tune_class_bias(val_logits: np.ndarray, y_val: np.ndarray,
                    n_passes: int = 3, grid_points: int = 41,
                    lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """
    Post-hoc class-bias correction (deliverable #4): fit a per-class additive
    logit offset (3 scalars, one per LABELS index) that maximises VAL macro-F1
    when added to `val_logits` before argmax. No retraining — this only
    shifts the decision boundary of an already-trained model.

    Method: coordinate search. For each of `n_passes` passes over the 3
    classes, grid-search that class's offset over `grid_points` values in
    [lo, hi] holding the other two fixed at their current best, keep whichever
    value improves val macro-F1 (ties keep the earlier, i.e. smaller-offset,
    value). This is intentionally simple/greedy rather than an exact 3-D
    optimum, in keeping with a POST-HOC correction — the point is to move the
    class-imbalance collapse (f1_borderline==0.0) off dead zero, not to
    squeeze out the last 0.1% of macro-F1.

    `val_logits` may be true logits (torch model output) or decision-function
    scores (sklearn LogisticRegression) — both are pre-argmax, real-valued
    per-class scores and the offset acts identically on either.

    Args:
        val_logits: (N_val, 3) real-valued scores, one column per LABELS index
        y_val: (N_val,) int true labels in 0..2
        n_passes: number of full coordinate-descent sweeps over the 3 classes
        grid_points: number of candidate offsets tried per class per pass
        lo, hi: search range for the additive offset

    Returns:
        (3,) float32 array of per-class additive offsets, indexed like LABELS.
    """
    n_classes = val_logits.shape[1]
    biases = np.zeros(n_classes, dtype=np.float64)
    grid = np.linspace(lo, hi, grid_points)

    def macro_f1_with(b):
        preds = np.argmax(val_logits + b, axis=1)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    best_f1 = macro_f1_with(biases)
    for _ in range(n_passes):
        improved = False
        for c in range(n_classes):
            best_val_for_c = biases[c]
            local_best_f1 = best_f1
            for g in grid:
                trial = biases.copy()
                trial[c] = g
                f1 = macro_f1_with(trial)
                if f1 > local_best_f1:
                    local_best_f1 = f1
                    best_val_for_c = g
            if local_best_f1 > best_f1:
                biases[c] = best_val_for_c
                best_f1 = local_best_f1
                improved = True
        if not improved:
            break

    return biases.astype(np.float32)


def apply_class_bias(logits: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Apply per-class additive offsets from `tune_class_bias` to a logits
    array before argmax. Pure function, no state, no retraining."""
    return logits + biases


def bias_corrected_predictions(val_logits, y_val, test_logits):
    """
    Convenience wrapper: fit the bias offsets on val, apply to test, return
    (uncorrected_test_pred, corrected_test_pred, biases) so callers can
    report both numbers side by side, as the spec requires.
    """
    biases = tune_class_bias(val_logits, y_val)
    uncorrected_pred = np.argmax(test_logits, axis=1)
    corrected_pred = np.argmax(apply_class_bias(test_logits, biases), axis=1)
    return uncorrected_pred, corrected_pred, biases


def results_table(results):
    """Aggregate result dicts by (method, encoder, variant) and compute summary statistics.

    Args:
        results: list of result dicts from run_* methods, each containing
                 'method', 'encoder', 'variant', 'seed', and metric keys
                 (acc, macro_f1, f1_normal, f1_borderline, f1_anomaly, confusion)

    Returns:
        pd.DataFrame with one row per (method, encoder, variant) and columns:
        method, encoder, variant, acc_mean, acc_std, macro_f1_mean, macro_f1_std,
        f1_normal_mean, f1_borderline_mean, f1_anomaly_mean, n_seeds.
        Sorted by variant, encoder, macro_f1_mean descending.
    """
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(results)

    # Group by (method, encoder, variant) and aggregate
    grouped = df.groupby(["method", "encoder", "variant"], as_index=False).agg({
        "acc": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "f1_normal": "mean",
        "f1_borderline": "mean",
        "f1_anomaly": "mean",
        "seed": "count"
    })

    # Flatten multi-level column names
    grouped.columns = ["method", "encoder", "variant",
                       "acc_mean", "acc_std",
                       "macro_f1_mean", "macro_f1_std",
                       "f1_normal_mean", "f1_borderline_mean", "f1_anomaly_mean",
                       "n_seeds"]

    # Sort by variant, encoder, macro_f1_mean descending
    grouped = grouped.sort_values(
        by=["variant", "encoder", "macro_f1_mean"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return grouped


def plot_results(df, out_path):
    """Plot grouped bar chart of macro-F1 scores with error bars.

    Creates one subplot per (encoder, variant) combination, with methods on x-axis,
    grouped bars showing macro_f1_mean with macro_f1_std error bars, and a horizontal
    dashed reference line for the majority baseline in each panel.

    Args:
        df: DataFrame from results_table() with aggregated metrics
        out_path: output path for the PNG file (e.g., 'results_plot.png')
    """
    # Get unique (encoder, variant) combinations, sorted
    encoder_variants = df[["encoder", "variant"]].drop_duplicates().sort_values(
        by=["variant", "encoder"]
    ).reset_index(drop=True)

    n_plots = len(encoder_variants)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    # Get majority baseline macro_f1 (should be same for all encoder/variant)
    majority_rows = df[df["method"] == "majority"]
    if len(majority_rows) > 0 and np.isfinite(majority_rows.iloc[0]["macro_f1_mean"]):
        majority_f1 = float(majority_rows.iloc[0]["macro_f1_mean"])
    else:
        majority_f1 = 0.3  # fallback

    # Get shared y-limits across all subplots
    y_min = 0
    # pandas .std() over a single observation is NaN, so any 1-seed run (a
    # pre-flight, or a deliberately cheap sweep) would otherwise propagate NaN
    # into set_ylim and raise "Axis limits cannot be NaN or Inf". Treat a missing
    # spread as zero spread — with one seed there genuinely is no measured spread.
    means = df["macro_f1_mean"].fillna(0.0)
    stds = df["macro_f1_std"].fillna(0.0)
    y_max = float((means + stds).max()) + 0.05
    if not np.isfinite(y_max) or y_max <= y_min:
        y_max = 1.0

    for plot_idx, (_, enc_var_row) in enumerate(encoder_variants.iterrows()):
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        ax = axes[row_idx, col_idx]

        encoder = enc_var_row["encoder"]
        variant = enc_var_row["variant"]

        # Filter results for this (encoder, variant)
        subset = df[(df["encoder"] == encoder) & (df["variant"] == variant)].copy()
        subset = subset.sort_values("method")  # consistent ordering

        # Prepare bar data
        methods = subset["method"].values
        # Same NaN guard as the y-limit above: a 1-seed run has no measured
        # spread, and matplotlib's yerr also refuses NaN.
        means = subset["macro_f1_mean"].fillna(0.0).values
        stds = subset["macro_f1_std"].fillna(0.0).values

        # Plot grouped bars
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")

        # Add reference line for majority baseline
        ax.axhline(y=majority_f1, color="red", linestyle="--", linewidth=2, label="Majority baseline")

        # Formatting
        ax.set_xlabel("Method", fontsize=10)
        ax.set_ylabel("Macro F1", fontsize=10)
        ax.set_title(f"Encoder: {encoder}, Variant: {variant}", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.legend(fontsize=9)

    # Hide unused subplots
    for plot_idx in range(n_plots, n_rows * n_cols):
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(results, method, encoder, variant, out_path):
    """Plot confusion matrix heatmap for a specific (method, encoder, variant).

    Sums confusion matrices across all run seeds for the requested combination,
    renders as a heatmap with annotated cell counts.

    Args:
        results: list of result dicts (flat, as returned by run_* methods)
        method: method name (e.g., 'majority', 'late', 'intermediate')
        encoder: encoder name (e.g., 'minilm', 'bert')
        variant: variant name (e.g., 'full', 'filtered')
        out_path: output path for the PNG file

    Raises:
        ValueError: if no results match the (method, encoder, variant) filter
    """
    # Filter results
    matching = [r for r in results
                if r["method"] == method and r["encoder"] == encoder and r["variant"] == variant]

    if not matching:
        raise ValueError(f"No results found for method={method}, encoder={encoder}, variant={variant}")

    # Sum confusion matrices across seeds
    confusion_sum = None
    for r in matching:
        conf = np.array(r["confusion"], dtype=np.int64)
        if confusion_sum is None:
            confusion_sum = conf.copy()
        else:
            confusion_sum += conf

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_sum, cmap="Blues", aspect="auto")

    # Add text annotations
    for i in range(confusion_sum.shape[0]):
        for j in range(confusion_sum.shape[1]):
            # Choose text color based on background intensity
            threshold = confusion_sum.max() / 2
            text_color = "white" if confusion_sum[i, j] > threshold else "black"
            ax.text(j, i, str(int(confusion_sum[i, j])), ha="center", va="center",
                   color=text_color, fontsize=14, fontweight="bold")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {method} (encoder={encoder}, variant={variant})",
                fontsize=13, fontweight="bold", pad=20)

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
