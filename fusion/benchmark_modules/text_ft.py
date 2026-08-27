"""
text_ft.py — text-only, FINE-TUNED control.

Deliverable #1 of the extension: isolates "the encoder was fine-tuned" from
"emotion was visible at the input" (early fusion's actual claim).

`run_text_only` (report.py) freezes the encoder and only trains
a LogisticRegression head on top — so it differs from `run_early` in BOTH
(a) no emotion at input AND (b) no fine-tuning. That confounds the two
questions "does fusion help" and "does fine-tuning help". This module fixes
one variable: same fine-tuning as `run_early`, but the input string is just
`text` — no `[EMOTION]` prefix anywhere.

Implemented as a thin wrapper around `early._run_early_or_textft`
with `use_emotion=False`, so the training loop, hyperparameters, and early
stopping are byte-for-byte the same code path as `run_early` — the only
degree of freedom is the input string. This module intentionally imports
from `early` (not just `common`) to guarantee that
identity; duplicating the loop would risk the two silently drifting apart
after a future edit to one but not the other.
"""

import numpy as np
from common import y_of
from early import _run_early_or_textft


def run_text_only_finetuned(splits, emb, seed, device, encoder_name):
    """
    Text-only baseline with a FINE-TUNED encoder (as opposed to
    `run_text_only`, which uses a frozen embedding + LogisticRegression).

    Identical to `run_early` in every respect (hyperparameters, optimizer,
    batch size, epochs, early stopping, class-weighting switch) except the
    input string has no emotion token prepended — plain `text` only.

    Args:
        splits: dict with "train", "val", "test" lists of Row dicts
        emb: dict with embeddings (ignored — this method builds its own
             representation via fine-tuning, like run_early)
        seed: random seed
        device: torch device string (e.g., "cuda" or "cpu")
        encoder_name: "minilm" or "bert", key to ENCODER_IDS

    Returns:
        tuple[np.ndarray, np.ndarray]: (y_true_test, y_pred_test),
            both int arrays of shape (N_test,) with values in 0..2 indexing LABELS
    """
    y_true, y_pred, _val_logits, _test_logits = _run_early_or_textft(
        splits, emb, seed, device, encoder_name, use_emotion=False
    )
    return y_true, y_pred


def run_text_only_finetuned_logits(splits, emb, seed, device, encoder_name):
    """
    Same computation as `run_text_only_finetuned` but also returns val/test
    logits for the post-hoc class-bias correction step.

    Returns (y_val, val_logits, y_test, test_logits).
    """
    y_test, _y_pred, val_logits, test_logits = _run_early_or_textft(
        splits, emb, seed, device, encoder_name, use_emotion=False
    )
    y_val = y_of(splits["val"])
    return y_val, val_logits, y_test, test_logits
