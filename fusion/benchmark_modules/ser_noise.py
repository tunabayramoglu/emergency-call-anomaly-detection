"""SER noise model — makes the fusion combiner face the emotion channel it will
actually be deployed with, instead of the oracle label.

WHY THIS EXISTS
---------------
Every fusion method in this benchmark consumes `gen_emotion`, a HARD label taken
straight from the dataset. At deployment the emotion does not come from a
dataset — it comes from the SER model, which is wrong a substantial fraction of
the time. And in THIS task the error does not degrade gracefully: the anomaly
label is defined as a mismatch between voice arousal and content severity, so a
flipped arousal reading does not merely add noise, it INVERTS the verdict.

MEASURED GROUND TRUTH (not invented)
------------------------------------
From `_Staj/meeting/D3_voice_channel.png` — the SER voice-risk confusion matrix
on the held-out, speaker-independent validation split, n=902:

                        true=high   true=low
    SER pred = high        512         22       (534)
    SER pred = low         104        264       (368)
                           616        286       (902)

This reconciles exactly with the three numbers reported for the SER channel
(accuracy 86.03%, precision 95.88%, recall 83.12%), so the matrix is the real
measured one rather than a reconstruction.

WHY A BINARY MATRIX IS ENOUGH
-----------------------------
The 6-class SER confusion matrix was not recoverable. It turns out not to
matter much: `judge_voice_risk` in the fusion dataset is a 99.9% deterministic
recode of `gen_emotion` into two arousal buckets
(neutral/confusion -> low, fear/panic/urgency/distress -> high), so the axis
that actually drives the anomaly label IS this binarisation — and that is
precisely what the n=902 matrix measures.

WHAT IS MEASURED VS ASSUMED — state this honestly on any slide
--------------------------------------------------------------
  MEASURED (n=902): the arousal flip rates and their posteriors.
  ASSUMED:          which specific emotion inside a bucket the SER would name.
                    We distribute that mass by the dataset's own within-bucket
                    emotion prior. Nothing in the anomaly label depends on this
                    choice, since the label is driven by the bucket.

IS COLLAPSING TO THE BUCKET A LOSS? MEASURED: NO.
-------------------------------------------------
This model produces only two distinct soft vectors (one per predicted bucket),
so it reduces the emotion channel to a single bit. That is a fair worry, so it
was tested rather than assumed. Predicting `anomaly` on the held-out test split
from the emotion channel alone:

    6-way oracle one-hot     macro-F1 = 0.4111
    2-way arousal bucket     macro-F1 = 0.4111

Identical. The fine-grained emotion identity carries no label-relevant
information beyond its arousal bucket, which is the expected consequence of
`judge_voice_risk` being a binary recode in the first place. So the simple
bucket-level noise model loses nothing, and modelling within-bucket confusion
(which would require an extra independence assumption on top of the 6-class UA)
would add assumptions for no measurable benefit. Fewer assumptions wins.

THE TRAP THIS MODULE AVOIDS
---------------------------
The naive implementation is to hand every high-arousal row the same "expected"
soft vector [0.831 on high, 0.169 on low]. That destroys NO information: every
high row still presents identically, so a model can invert the transform
perfectly and recover the oracle label. It would look like a robustness test
while measuring nothing.

Instead we SAMPLE the predicted bucket per row from the measured likelihoods
(so 16.9% of genuinely high-arousal rows really do present as low), and only
then build a soft vector around the sampled outcome using the measured
posterior error for that prediction. The information loss is real.
"""

from __future__ import annotations

import numpy as np

from common import EMOTIONS

# --- the measured confusion matrix, n=902 -----------------------------------
SER_CONFUSION = {
    ("high", "high"): 512,  # (predicted, true)
    ("high", "low"): 22,
    ("low", "high"): 104,
    ("low", "low"): 264,
}
SER_CONFUSION_N = 902
SER_CONFUSION_SOURCE = "_Staj/meeting/D3_voice_channel.png (held-out val, n=902)"

# Arousal bucket of each of the SER's six classes. This is the same mapping
# that `judge_voice_risk` follows in the dataset (verified: 99.9% agreement).
AROUSAL_BUCKET = {
    "neutral": "low",
    "confusion": "low",
    "fear": "high",
    "panic": "high",
    "urgency": "high",
    "distress": "high",
}
BUCKETS = ("low", "high")

# Emotion counts in dataset_final.jsonl, used as the within-bucket prior.
EMOTION_PRIOR_COUNTS = {
    "neutral": 2267,
    "confusion": 1978,
    "fear": 1616,
    "panic": 1399,
    "urgency": 1386,
    "distress": 1094,
}


def _col_totals() -> dict[str, int]:
    """Column totals = how many rows of each TRUE bucket the matrix saw."""
    return {
        true: sum(v for (_, t), v in SER_CONFUSION.items() if t == true)
        for true in BUCKETS
    }


def _row_totals() -> dict[str, int]:
    """Row totals = how many times the SER PREDICTED each bucket."""
    return {
        pred: sum(v for (p, _), v in SER_CONFUSION.items() if p == pred)
        for pred in BUCKETS
    }


def likelihoods() -> dict[str, dict[str, float]]:
    """P(SER predicts `pred` | true bucket is `true`), column-normalised.

    This is the information-destroying step: it is what decides that ~16.9% of
    genuinely high-arousal utterances will be presented to the fusion model as
    low arousal.
    """
    col = _col_totals()
    return {
        true: {pred: SER_CONFUSION[(pred, true)] / col[true] for pred in BUCKETS}
        for true in BUCKETS
    }


def posteriors() -> dict[str, dict[str, float]]:
    """P(true bucket is `true` | SER predicted `pred`), row-normalised.

    Used to shape the soft vector. Note the strong asymmetry this exposes:
    a "high" prediction is wrong only ~4% of the time, but a "low" prediction
    is wrong ~28% of the time. A well-calibrated downstream model should
    therefore trust "high" far more than "low".
    """
    row = _row_totals()
    return {
        pred: {true: SER_CONFUSION[(pred, true)] / row[pred] for true in BUCKETS}
        for pred in BUCKETS
    }


def within_bucket_prior() -> dict[str, np.ndarray]:
    """For each bucket, a distribution over the 6 emotions summing to 1.

    Emotions outside the bucket get exactly 0. This is the ASSUMED part of the
    noise model — see the module docstring.
    """
    out: dict[str, np.ndarray] = {}
    for bucket in BUCKETS:
        w = np.array(
            [
                EMOTION_PRIOR_COUNTS[e] if AROUSAL_BUCKET[e] == bucket else 0.0
                for e in EMOTIONS
            ],
            dtype=np.float64,
        )
        out[bucket] = (w / w.sum()).astype(np.float32)
    return out


def summary() -> str:
    """Human-readable report of the noise model, for a notebook cell or slide."""
    lk, po = likelihoods(), posteriors()
    lines = [
        f"SER noise model — source: {SER_CONFUSION_SOURCE}",
        "",
        "  MEASURED likelihoods (what the SER does to a true bucket):",
        f"    P(pred=low  | true=high) = {lk['high']['low']:.4f}   <- miss",
        f"    P(pred=high | true=low ) = {lk['low']['high']:.4f}   <- false alarm",
        f"    misses are {lk['high']['low'] / lk['low']['high']:.1f}x more likely "
        f"than false alarms",
        "",
        "  MEASURED posteriors (how much to trust a prediction):",
        f"    P(true=low  | pred=high) = {po['high']['low']:.4f}",
        f"    P(true=high | pred=low ) = {po['low']['high']:.4f}",
        "",
        "  Consequence for the two anomaly directions:",
        f"    calm voice + severe content   -> survives {1 - lk['low']['high']:.1%} of the time (robust)",
        f"    alarmed voice + trivial content -> survives {1 - lk['high']['low']:.1%} of the time (fragile)",
    ]
    return "\n".join(lines)


def simulate(
    true_emotions: list[str],
    seed: int,
    hard: bool = False,
) -> np.ndarray:
    """Turn oracle emotion labels into SER-realistic emotion features.

    Args:
        true_emotions: the oracle `gen_emotion` string for each row.
        seed: RNG seed. Different run seeds draw different SER errors, so the
            spread across seeds is itself a meaningful error bar on the
            robustness estimate — do not fix this to a constant.
        hard: if True return a one-hot of the sampled emotion instead of a soft
            distribution. Useful as an ablation isolating "soft vs hard" from
            "noisy vs oracle".

    Returns:
        (N, 6) float32, each row summing to 1, column order = EMOTIONS.
    """
    rng = np.random.default_rng(seed)
    lk = likelihoods()
    po = posteriors()
    prior = within_bucket_prior()
    other = {"low": "high", "high": "low"}

    out = np.zeros((len(true_emotions), len(EMOTIONS)), dtype=np.float32)

    for i, emo in enumerate(true_emotions):
        true_bucket = AROUSAL_BUCKET[emo]

        # Step 1 — sample what the SER would have PREDICTED. This is the step
        # that genuinely destroys information, and every number in it is
        # measured from the n=902 matrix.
        p_high = lk[true_bucket]["high"]
        pred_bucket = "high" if rng.random() < p_high else "low"

        if hard:
            # Collapse to a single emotion drawn from the predicted bucket.
            out[i] = rng.multinomial(1, prior[pred_bucket]).astype(np.float32)
            continue

        # Step 2 — shape the soft vector using the measured posterior for that
        # prediction, so the vector carries the SER's real uncertainty.
        conf = po[pred_bucket][pred_bucket]  # mass on the predicted bucket
        out[i] = conf * prior[pred_bucket] + (1.0 - conf) * prior[other[pred_bucket]]

    return out


def flip_report(true_emotions: list[str], simulated: np.ndarray) -> dict:
    """How many rows actually had their arousal bucket inverted.

    Compares the oracle bucket against the argmax bucket of the simulated
    features, which is what a downstream hard-label consumer would see.
    """
    counts = {"high->low": 0, "low->high": 0, "unchanged": 0}
    for emo, vec in zip(true_emotions, simulated):
        true_bucket = AROUSAL_BUCKET[emo]
        seen_bucket = AROUSAL_BUCKET[EMOTIONS[int(np.argmax(vec))]]
        if true_bucket == seen_bucket:
            counts["unchanged"] += 1
        else:
            counts[f"{true_bucket}->{seen_bucket}"] += 1
    n = len(true_emotions)
    counts["flipped_frac"] = round((n - counts["unchanged"]) / n, 4)
    counts["n"] = n
    return counts
