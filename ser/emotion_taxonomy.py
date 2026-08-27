"""
Emotion Taxonomy & Arousal Mapper for Emergency Call SER.

Two interfaces:
  1. arousal(label, dataset) -> float 0.0-1.0  (primary — regression target)
  2. map_label(label, source) -> str       (5-class categorical — optional)

Arousal values are 0 (completely calm/asleep) to 1 (maximum panic/agitation).
Values are set consistently across datasets (same emotion ≈ same arousal).

Reference: MSP-Podcast uses activation/valence/dominance;
           our values align with activation (arousal) dimension.
"""

# ── Arousal tables (0.0–1.0) ───────────────────────────────────

AROUSAL = {
    # --- RAVDESS (8 labels) ---
    "ravdess": {
        "neutral":      0.25,
        "calm":         0.10,
        "happy":        0.50,
        "sad":          0.45,
        "angry":        0.85,
        "fearful":      0.80,
        "disgust":      0.70,
        "surprised":    0.75,
    },
    # --- CREMA-D (6 labels) ---
    "cremad": {
        "neutral":      0.30,
        "happiness":    0.55,
        "sadness":      0.45,
        "anger":        0.85,
        "fear":         0.80,
        "disgust":      0.70,
    },
    # --- JL Corpus (10 labels, CC0 Public Domain) ---
    "jl": {
        "neutral":      0.30,
        "happy":        0.50,
        "sad":          0.45,
        "angry":        0.85,
        "excited":      0.85,
        "anxious":      0.75,
        "worried":      0.70,
        "apologetic":   0.40,
        "pensive":      0.35,
        "enthusiastic": 0.75,
    },
    # --- ASVP-ESD (12 labels, CC-BY-4.0) ---
    "asvp_esd": {
        "neutral":        0.25,
        "happiness":      0.55,
        "sadness":        0.50,
        "anger":          0.85,
        "fear":           0.80,
        "surprise":       0.70,
        "disgust":        0.65,
        "excitement":     0.85,
        "pleasure":       0.40,
        "pain":           0.90,
        "disappointment": 0.60,
        "boredom":        0.15,
    },
    # --- ESD (5 emotions, bilingual EN/CN) ---
    "esd": {
        "neutral":      0.30,
        "happy":        0.50,
        "sad":          0.45,
        "angry":        0.85,
        "surprise":     0.75,
    },
    # --- TESS (7 labels) ---
    "tess": {
        "neutral":           0.30,
        "happiness":         0.55,
        "sadness":           0.45,
        "anger":             0.85,
        "fear":              0.80,
        "disgust":           0.70,
        "pleasant_surprise": 0.70,
    },
    # --- EMNS (8 labels, Apache 2.0) ---
    "emns": {
        "neutral":      0.30,
        "happiness":    0.50,
        "sadness":      0.45,
        "anger":        0.85,
        "excitement":   0.85,
        "disgust":      0.70,
        "surprise":     0.75,
        "sarcasm":      0.55,
    },
    # --- Our 911 Qwen2.5-3B labels (11 labels) ---
    "qwen_911": {
        "neutral":      0.30,
        "confused":     0.45,
        "urgent":       0.80,
        "distressed":   0.75,
        "fearful":      0.80,
        "desperate":    0.85,
        "angry":        0.85,
        "panicked":     0.95,
        "sad":          0.50,
        "grieving":     0.55,
        "crying":       0.70,
    },
    # --- Our 911 wav2vec2 (7 labels, unreliable) ---
    "wav2vec2_911": {
        "angry":        0.85,
        "calm":         0.10,
        "disgust":      0.70,
        "fearful":      0.80,
        "happy":        0.50,
        "sad":          0.45,
        "surprised":    0.75,
    },
    # --- IEMOCAP (9 labels, registration-gated, for future) ---
    "iemocap": {
        "neutral":      0.30,
        "happy":        0.50,
        "sad":          0.45,
        "angry":        0.85,
        "frustrated":   0.75,
        "fearful":      0.80,
        "surprised":    0.75,
        "excited":      0.85,
        "disgusted":    0.70,
    },
}

# Valid dataset keys
DATASETS = set(AROUSAL.keys())


def arousal(label: str, dataset: str) -> float | None:
    """Look up arousal value for a label in a dataset.

    Args:
        label: Raw label string (case-insensitive, leading/trailing space OK).
        dataset: Key in AROUSAL table (e.g. 'ravdess', 'cremad', 'jl').

    Returns:
        Float in [0.0, 1.0] or None if label/dataset not found.
    """
    table = AROUSAL.get(dataset)
    if table is None:
        return None
    return table.get(label.strip().lower())


def label_exists(label: str, dataset: str) -> bool:
    """Check if a label exists in a dataset's arousal table."""
    return arousal(label, dataset) is not None


def arousal_or_default(label: str, dataset: str, default: float = 0.5) -> float:
    """Like arousal() but returns default on miss instead of None."""
    val = arousal(label, dataset)
    return val if val is not None else default


# ── Discrete 5-class mapping (secondary, for ablation) ─────────
# Maps from the Roadmap's original taxonomy. Kept for comparison
# against the regression approach.

CLASSES_5 = ["distress", "panic", "urgency", "confusion", "neutral"]

_5CLASS = {
    "qwen_11": {
        "distressed": "distress", "fearful": "distress", "desperate": "distress",
        "angry": "distress", "sad": "distress", "grieving": "distress", "crying": "distress",
        "panicked": "panic", "urgent": "urgency", "confused": "confusion", "neutral": "neutral",
    },
    "omni": {
        "distressed": "distress", "fearful": "distress", "desperate": "distress",
        "angry": "distress", "sad": "distress", "grieving": "distress", "crying": "distress",
        "panicked": "panic", "panic": "panic", "urgent": "urgency",
        "confused": "confusion", "neutral": "neutral",
    },
    "ravdess": {
        "neutral": "neutral", "calm": "neutral", "happy": "neutral",
        "sad": "distress", "angry": "distress", "fearful": "panic",
        "disgust": "distress", "surprised": "urgency",
    },
    "cremad": {
        "neutral": "neutral", "happiness": "neutral",
        "sadness": "distress", "anger": "distress", "fear": "panic", "disgust": "distress",
    },
}


def map_label(label: str, source: str) -> str | None:
    """Map a label to 5-class taxonomy (secondary interface)."""
    table = _5CLASS.get(source)
    if table is None:
        return None
    return table.get(label.strip().lower())


# ── Quick self-test when run directly ──────────────────────────

if __name__ == "__main__":
    print("=== Arousal Mapper Self-Test ===\n")

    # Test arousal lookups
    test_cases = [
        ("ravdess", "fearful",   0.80),
        ("ravdess", "calm",      0.10),
        ("cremad",  "fear",      0.80),
        ("cremad",  "neutral",   0.30),
        ("jl",      "excited",   0.85),
        ("jl",      "anxious",   0.75),
        ("asvp_esd","pain",      0.90),
        ("asvp_esd","boredom",   0.15),
        ("qwen_911","panicked",  0.95),
        ("qwen_911","neutral",   0.30),
        ("qwen_911","confused",  0.45),
    ]

    all_ok = True
    for ds, lbl, expected in test_cases:
        val = arousal(lbl, ds)
        ok = "OK" if val == expected else f"MISMATCH (got {val})"
        if val != expected:
            print(f"  [{ok}] {ds:>12}.arousal('{lbl:>12}') = {val} (expected {expected})")
            all_ok = False

    if all_ok:
        print("  All arousal lookups correct.\n")

    # Coverage per dataset
    print("  Dataset arousal ranges:")
    for ds in sorted(DATASETS):
        vals = list(AROUSAL[ds].values())
        print(f"    {ds:>14}: {len(vals)} labels, range [{min(vals):.2f}, {max(vals):.2f}]")

    print("\nDone.")
