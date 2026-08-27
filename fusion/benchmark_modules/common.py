"""the fusion benchmark — shared constants, data loading, splitting and metrics.

LEAKAGE RULE
------------
The only fields allowed as MODEL INPUT are `text` and `gen_emotion`.
`seed_id` is used only for splitting, `uid` only as a cache key, `source_model`
only for the data-variant filter.

Every field in BANNED_FIELDS is forbidden as a feature:

  * `judge_content_risk`, `judge_voice_risk`, `gen_content_risk`,
    `content_risk_seed` — outputs of the same LLM call that produced the
    `anomaly` label. A 6-cell lookup table over
    (judge_voice_risk, judge_content_risk) scores 91.5% on the full dataset.
    That is label leakage, not a result.
  * `event`, `profile`, `target_emotion` — seed metadata. `event` was measured
    by LLM audit (n=1000) to be wrong for 6.7% [5.2-8.3] of rows, because the
    generator drifted off its seeded scenario.
  * `reason`, `judge_*`, `anomaly_votes` — annotation metadata about the label.
  * `source_model` — which LLM wrote the utterance; a style shortcut.

This module has no torch/transformers dependency so it stays importable on a
CPU-only machine.
"""

from __future__ import annotations

import json
import random

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS: list[str] = ["normal", "borderline", "anomaly"]
EMOTIONS: list[str] = ["neutral", "confusion", "fear", "panic", "urgency", "distress"]

ENCODER_IDS: dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim
    "bert": "bert-base-uncased",  # 768-dim, CLS pooling
}

VARIANTS: list[str] = ["full", "filtered"]

# Global, all-or-nothing switch for class-weighted training loss. Every
# trainable method (run_late, run_intermediate, run_early,
# run_text_only_finetuned, run_intermediate_film, run_intermediate_attn) and
# both LogisticRegression baselines (emotion_only, text_only) honour this flag
# via `class_weights()` / `class_weight="balanced"`. The driver must run the
# WHOLE method matrix with this flag at one setting and report both settings
# side by side — never mix weighted and unweighted rows in the same table,
# or a "tuned vs untuned" artefact would masquerade as a fusion-level finding.
CLASS_WEIGHTING: bool = False

# Utterances from these two generators had a measured event-drift rate of
# 18.1% and 11.4%, vs ~1% for deepseek-v4. The "filtered" variant drops them
# as a robustness ablation.
FILTERED_OUT_MODELS: set[str] = {"qwen3.6-flash", "qwen3.5-flash"}

BANNED_FIELDS: set[str] = {
    "judge_content_risk",
    "judge_voice_risk",
    "gen_content_risk",
    "content_risk_seed",
    "event",
    "profile",
    "target_emotion",
    "reason",
    "judge_model",
    "judge_count",
    "judge_agreement",
    "judge_models",
    "anomaly_votes",
    "source_model",
}

_LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
_EMOTION_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(EMOTIONS)}

# --- emotion regime ---------------------------------------------------------
# Which emotion channel the fusion methods see. Read LIVE off this module
# (`_common.EMOTION_REGIME`) — never `from common import
# EMOTION_REGIME`, which would freeze the value at import time and silently
# keep using a stale setting when the driver switches regimes.
#
#   "oracle"   — train on the dataset's `gen_emotion`, test on it too.
#                The upper bound. Not achievable in deployment.
#   "ser_test" — train on the oracle label, test on SER-realistic emotion.
#                Measures how far the current approach falls when it meets the
#                real SER. No retraining, so this is pure robustness measurement.
#   "ser_both" — train AND test on SER-realistic emotion. The fix: the combiner
#                learns to be robust to the SER's measured error pattern.
#
# Compare "ser_test" against "ser_both" to see whether training under noise
# closes the deployment gap. Both share the same test condition, so the
# comparison is fair; "oracle" is the ceiling above them.
EMOTION_REGIMES: list[str] = ["oracle", "ser_test", "ser_both"]
EMOTION_REGIME: str = "oracle"

# --- checkpoint sink --------------------------------------------------------
# When the driver sets this to a dict, each trainable method deposits its
# best-by-val state dict here instead of discarding it, so the driver can decide
# what is worth persisting. Left as None by default: keeping every checkpoint
# would mean ~48 fine-tuned encoders at ~440 MB each (>20 GB), which is useless
# — the app needs ONE deployable model, not forty-eight. Policy applied by the
# driver: keep every small head (they are kilobytes), and of the fine-tuned
# encoders keep only the best val macro-F1 per (encoder, variant).
CHECKPOINT_SINK: dict | None = None

# --- cache locations --------------------------------------------------------
# `run_intermediate_attn` builds its own token-level embedding cache rather than
# receiving one, so the location has to be settable from outside or the cache
# lands on ephemeral Colab local disk and gets rebuilt (~2 GB, several minutes)
# every session. Read LIVE off this module, same discipline as the flags above.
TOKEN_CACHE_DIR: str = "/content/emb_cache_tok"

# --- fine-tuning budget -----------------------------------------------------
# Epochs for the two fine-tuning methods (`early`, `text_only_finetuned`).
# Read LIVE so a smoke test can drop it to 1 and still exercise the real code
# path — the point of a pre-flight check is to prove the expensive branch RUNS,
# not to get a good score from it. 4 is the value the reported results used.
MAX_FINETUNE_EPOCHS: int = 4


def offer_checkpoint(tag: str, state_dict, val_f1: float, meta: dict | None = None) -> None:
    """Hand a trained model to the driver, if it asked for one.

    Deliberately a no-op when CHECKPOINT_SINK is None so that sweeps — which
    train hundreds of throwaway models — cost nothing.
    """
    if CHECKPOINT_SINK is None:
        return
    CHECKPOINT_SINK[tag] = {
        "state_dict": state_dict,
        "val_f1": float(val_f1),
        "meta": dict(meta or {}),
    }


def set_seed(seed: int) -> None:
    """Seed every RNG in play. Torch is optional so this module stays light."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_rows(path: str, variant: str = "full") -> list[dict]:
    """Parse the JSONL dataset. `variant="filtered"` drops the drift-prone models."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if variant == "filtered" and row.get("source_model") in FILTERED_OUT_MODELS:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"no rows loaded from {path!r} for variant {variant!r}")
    return rows


def make_splits(
    rows: list[dict],
    split_seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed_universe: list[int] | None = None,
) -> dict[str, list[dict]]:
    """Split on `seed_id` GROUPS, not rows.

    Utterances sharing a seed_id are near-paraphrases of one another (same
    profile/event/emotion, different generator). A row-level split would put
    paraphrases of the same scenario on both sides and the model would score
    high by memorisation. Fractions apply to the number of seed groups.

    Depends only on `split_seed`, never on a per-run seed, so every method,
    encoder and variant is evaluated on exactly the same held-out scenarios.

    `seed_universe` pins the train/val/test assignment to a fixed set of
    seed_ids. The driver MUST pass the full dataset's seed_ids here for both
    data variants: the "filtered" variant contains fewer unique seeds, so
    letting it shuffle its own universe would give it a different test set and
    confound the ablation — a full-vs-filtered gap would then be split noise
    rather than a data-quality effect. With the universe pinned, the filtered
    splits are strict row-subsets of the full ones.
    """
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be in (0, 1)")

    present = {row["seed_id"] for row in rows}
    if seed_universe is None:
        unique_seeds = sorted(present)
    else:
        unique_seeds = sorted(set(seed_universe))
        missing = present - set(unique_seeds)
        if missing:
            raise ValueError(
                f"{len(missing)} seed_id(s) in rows are absent from seed_universe, "
                f"e.g. {sorted(missing)[:5]}"
            )
    shuffled = list(unique_seeds)
    random.Random(split_seed).shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_seeds = set(shuffled[:n_test])
    val_seeds = set(shuffled[n_test : n_test + n_val])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:  # preserves dataset order within each split
        sid = row["seed_id"]
        if sid in test_seeds:
            splits["test"].append(row)
        elif sid in val_seeds:
            splits["val"].append(row)
        else:
            splits["train"].append(row)
    return splits


def assert_no_seed_overlap(splits: dict[str, list[dict]]) -> None:
    """Raise if any seed_id appears in more than one split."""
    groups = {name: {row["seed_id"] for row in rows} for name, rows in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = groups[a] & groups[b]
        assert not shared, f"seed_id leak between {a} and {b}: {sorted(shared)[:10]}"


def y_of(rows: list[dict]) -> np.ndarray:
    """Target labels as int indices into LABELS."""
    return np.array([_LABEL_TO_IDX[row["anomaly"]] for row in rows], dtype=np.int64)


def emotion_onehot(rows: list[dict]) -> np.ndarray:
    """(N, 6) float32 one-hot of `gen_emotion`."""
    out = np.zeros((len(rows), len(EMOTIONS)), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, _EMOTION_TO_IDX[row["gen_emotion"]]] = 1.0
    return out


def emotion_idx(rows: list[dict]) -> np.ndarray:
    """(N,) int indices of `gen_emotion`.

    Kept for backward compatibility and for the hard-label path. New code should
    prefer `emotion_features`, which also covers the soft/noisy regimes.
    """
    return np.array([_EMOTION_TO_IDX[row["gen_emotion"]] for row in rows], dtype=np.int64)


def emotion_features(
    rows: list[dict],
    split_name: str,
    seed: int,
    regime: str | None = None,
) -> np.ndarray:
    """(N, 6) float32 emotion channel, honouring the active EMOTION_REGIME.

    This is the single place every fusion method gets its emotion input from, so
    a regime switch applies uniformly and cannot be applied to some methods but
    not others.

    Under "oracle" the result is a plain one-hot, which means the models behave
    exactly as they did before this function existed — `nn.Linear(6, k,
    bias=False)` applied to a one-hot is mathematically identical to
    `nn.Embedding(6, k)`, so previously reported hard-label numbers remain valid.

    Args:
        rows: the split's rows.
        split_name: "train", "val" or "test". Decides whether noise applies:
            "ser_test" noises only the test split (the model was trained on the
            oracle label, so val must stay oracle too or model selection would
            not match how it was trained).
        seed: the per-run seed. SER errors are re-drawn per seed, so the spread
            across seeds is a real error bar on the robustness estimate.
        regime: override; defaults to the live module-level EMOTION_REGIME.
    """
    active = EMOTION_REGIME if regime is None else regime
    if active not in EMOTION_REGIMES:
        raise ValueError(f"unknown emotion regime {active!r}, expected {EMOTION_REGIMES}")
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"unknown split name {split_name!r}")

    noisy = active == "ser_both" or (active == "ser_test" and split_name == "test")
    if not noisy:
        return emotion_onehot(rows)

    # Imported lazily: ser_noise imports EMOTIONS from this module,
    # so a top-level import here would be circular.
    from ser_noise import simulate

    # Offset the seed per split so train/val/test do not receive correlated
    # error draws.
    offset = {"train": 0, "val": 1_000, "test": 2_000}[split_name]
    return simulate([row["gen_emotion"] for row in rows], seed + offset)


def class_weights(y_train: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights over LABELS, normalised to mean 1.

    Used to build `nn.CrossEntropyLoss(weight=...)` for every trainable
    method when `CLASS_WEIGHTING` is on, and as the numeric basis for
    `class_weight="balanced"` semantics for the LogisticRegression baselines
    (those call sklearn's own "balanced" option directly; this function is
    for the torch methods, which have no such built-in).

    `borderline` is 12.4% of train in the observed run — this makes the
    minority class(es) worth proportionally more in the loss, without ever
    touching val/test, so it addresses the f1_borderline==0.0 collapse
    without resampling.
    """
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float64)
    if (counts == 0).any():
        # A class absent from train would divide by zero; treat it as
        # maximally rare rather than crash, though a model can't learn a
        # class it never sees regardless of loss weighting.
        counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts
    inv = inv * (len(LABELS) / inv.sum())  # normalise so mean weight == 1
    return inv.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, per-class F1 and the 3x3 confusion matrix (JSON-safe).

    `labels=[0, 1, 2]` is pinned explicitly: a degenerate model that never
    predicts `borderline` would otherwise return a 2-element per-class array
    and silently misalign the f1_* columns.
    """
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_normal": float(per_class[0]),
        "f1_borderline": float(per_class[1]),
        "f1_anomaly": float(per_class[2]),
        "confusion": [[int(v) for v in row] for row in cm],
    }


def make_result(
    method: str,
    encoder: str,
    variant: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """One flat, JSON-serialisable record for the results table."""
    return {
        "method": method,
        "encoder": encoder,
        "variant": variant,
        "seed": int(seed),
        **compute_metrics(y_true, y_pred),
    }


def self_test(path: str) -> None:
    """Smoke test: split integrity and label distribution. Run this first."""
    from collections import Counter

    universe = sorted({row["seed_id"] for row in load_rows(path, "full")})
    test_uids: dict[str, set[int]] = {}

    for variant in VARIANTS:
        rows = load_rows(path, variant=variant)
        splits = make_splits(rows, seed_universe=universe)
        assert_no_seed_overlap(splits)
        test_uids[variant] = {row["uid"] for row in splits["test"]}

        total = sum(len(v) for v in splits.values())
        assert total == len(rows), f"row count changed: {total} != {len(rows)}"
        uids = [row["uid"] for split in splits.values() for row in split]
        assert len(set(uids)) == len(uids), "duplicate uid across splits"

        print(f"[{variant}] {len(rows)} rows, {len({r['seed_id'] for r in rows})} seeds")
        for name in ("train", "val", "test"):
            part = splits[name]
            dist = Counter(row["anomaly"] for row in part)
            share = {k: round(v / len(part), 3) for k, v in sorted(dist.items())}
            print(
                f"  {name:5s} {len(part):5d} rows  "
                f"{len({r['seed_id'] for r in part}):4d} seeds  {share}"
            )

    assert test_uids["filtered"] <= test_uids["full"], (
        "filtered test set is not a subset of the full test set — the two "
        "variants are being evaluated on different scenarios"
    )
    print("filtered test set is a strict row-subset of the full test set")
    print("self_test passed")
