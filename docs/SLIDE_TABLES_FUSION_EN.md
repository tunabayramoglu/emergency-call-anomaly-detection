# B. Fusion — slide tables (EN)

_All numbers measured. Source: `fusion/results_table.csv` (96 grid cells × 3 seeds),
`WINNER_intermediate_attn_bert_full_p2_seed1.pt` → `meta`.
Task: 3-way classification `normal / borderline / anomaly`. Chance = 0.33._

---

## B1 · Fusion level — where the two channels are combined

_Oracle emotion, unweighted, best encoder per method._

| method | encoder | macro-F1 | sd | acc | F1 (borderline) |
|---|---|---|---|---|---|
| majority class | — | 0.2316 | — | 0.5324 | 0.000 |
| text only (frozen BERT) | bert | 0.3440 | — | 0.5324 | 0.000 |
| **late** (two risk bits) | minilm | 0.4058 | 0.0050 | 0.5886 | **0.016** |
| emotion only | — | 0.4111 | — | 0.5932 | 0.000 |
| text only, fine-tuned | bert | 0.4136 | 0.0074 | 0.5072 | 0.173 |
| **intermediate** (concat) | minilm | 0.5498 | 0.0186 | 0.6619 | 0.232 |
| **intermediate + FiLM** | minilm | 0.5571 | 0.0124 | 0.6810 | 0.219 |
| **intermediate + attention** | bert | **0.5684** | 0.0158 | 0.6758 | **0.243** |
| **early** (emotion token) | bert | 0.5917 | 0.0219 | 0.7242 | 0.238 |

**The `F1 (borderline)` column explains the ranking.** Late fusion combines two
scalar risk bits and has no place to represent an intermediate case → 0.016.
Methods that fuse over a shared representation reach 0.22–0.24.

> Fusing earlier, over a joint representation, beats fusing decisions.

---

## B2 · Ablation ladder — what the emotion channel is worth

| step | macro-F1 | gain |
|---|---|---|
| majority class | 0.2316 | — |
| + text (frozen BERT) | 0.3440 | +0.1124 |
| + fine-tune the text encoder | 0.4136 | **+0.0696** |
| + emotion token (early fusion) | 0.5917 | **+0.1782** |

> **Adding the voice-emotion channel is worth 2.56× fine-tuning the text encoder.**
> This single line is why the project exists.

---

## B3 · Robustness across regimes — why `early` was not shipped

| method | oracle, unweighted | oracle, weighted | **real SER emotion** | trained on noise too |
|---|---|---|---|---|
| **intermediate + attention / bert** | 0.5684 | 0.5822 | **0.5259** | 0.5018 |
| intermediate + FiLM / minilm | 0.5571 | 0.5796 | 0.5186 | 0.5141 |
| intermediate / bert | 0.5225 | 0.5810 | 0.5163 | 0.5059 |
| early / bert | **0.5917** | 0.4810 | 0.4610 | 0.4363 |
| late / minilm | 0.4058 | 0.4735 | 0.4265 | 0.4238 |
| emotion only | 0.4111 | 0.4459 | 0.3782 | 0.3782 |

`early` tops the oracle table, then **loses 0.11 macro-F1** the moment class
weighting changes and falls to 0.4610 under real SER output.

> We selected the model that wins **under deployment conditions**, not the one
> that wins with oracle emotion.

---

## B4 · Train/deploy gap — a measured negative result

| | macro-F1 |
|---|---|
| with oracle emotion labels | 0.5799 |
| with real SER output | **0.5224** |
| cost of the gap | **−0.0575** |
| recovered by training under simulated SER noise | −0.009 mean; **2 of 12 configurations improved** |

Noise-aware training does **not** close the gap. The injected noise is
label-independent, so it is irreducible — no amount of exposure to it helps.

> The fusion head is trained on clean emotion labels and deployed on real SER
> output. We measured that cost rather than assuming it away.

---

## B5 · Shipped configuration

| | |
|---|---|
| method | intermediate fusion + emotion-conditioned attention pooling |
| text encoder | `bert-base-uncased`, frozen, per-token hidden states |
| emotion input | 6-way one-hot from the SER branch |
| training regime | oracle emotion, class-weighted |
| seed selection | best of 3 on **validation** macro-F1 (never test) |
| trainable parameters | ~2.5 M (10 MB) |
| macro-F1, real SER | **0.5224** |

---

## Method notes (for the appendix / questions)

**Dataset.** 9,740 utterances, compositionally generated from 1,500
hard-constrained seeds, labelled by a **multi-judge ensemble** with majority
voting — not a single model's opinion.

**Split.** Grouped by `seed_id`: all paraphrases of a scenario go to the same
side. A row-level split would leak near-duplicates. Both data variants share one
pinned seed universe, so the filtered variant is evaluated on a subset of the
same held-out scenarios rather than a freshly reshuffled split.

**Scope, stated plainly.** This is a **fusion-combiner benchmark at the symbolic
level** (emotion label + text), not end-to-end from audio. The pipeline reduces
audio to (text, emotion), and this measures what to do with that pair.

**Excluded features.** `judge_voice_risk` and `judge_content_risk` are outputs of
the same LLM call that produced the label — a 6-cell lookup on them scores 91.5%.
They are not usable as inputs and were never given to any model.
