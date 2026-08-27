# presentation plan

_The numbers live in `SLIDE_TABLES.md`. This file is the running order and what
to say._

---

## FIRST: two corrections to the existing slides (5 min)

**Page 6 — the emotion dataset table.** A real-call corpus is listed, but SER was trained on
**academic data only**.
→ Mark that row "collected, not used in SER training".

**Page 7 — augmentation.** pitch/speed and telephone are listed, but the final
SER used neither.
→ Split it in two: "the pool we tried" vs "**what was applied**: SpecAugment +
pink noise".

You have no answer for either of these if you are asked; once corrected, the
question goes away.

---

## SLIDE ORDER

### 1 · The problem
> "An anomaly in an emergency call is a mismatch between **how** something is
> said and **what** is said. Reporting a fire in a calm voice is an anomaly. So
> is panicking over misplaced keys."

Draw the 2×2 matrix. The whole project fits in that square.

### 2 · Architecture — table C1
> "One **frozen** mHuBERT-147 with two task-specific LoRA adapters on top.
> Memory halves, and trainable parameters drop from 189M to **1.2M**.
> Adding a new task is a 2.4 MB file."

⚠️ **Do not say** "we run the backbone once." The two adapters diverge from layer
1 and the forward pass runs twice. If asked: *"Compute is not shared; restricting
LoRA to the upper layers could save 32%, but we did not measure it — future
work."*

### 3 · ASR: 100h → 300h — table A1
> "No difference on clean read speech: 5.15 → 5.13.
> On accented speech, **20.26 → 15.37 — we erased a quarter of the error.**"

Say this **yourself**, do not wait to be asked:
> "Not more data, more **diverse** data. 748 accents in training, where the
> previous model had none."

### 4 · Accent breakdown — table A2
> "**All six** native languages improve. Not the luck of one accent.
> Vietnamese is both the hardest and the smallest gain — that is our limit."

Naming your own limit beats being asked and having to defend it.

### 5 · Whisper — table A3
> "On L2-ARCTIC we match whisper-base's **accuracy** at 2.28× the speed.
> But the real difference is this: whisper-base is a complete 74-million-parameter
> model; we train **1.20 million**."

Do not hide medium:
> "Medium gets down to 8.10 — better than us. But it is 7.36× slower and produces
> no emotion. Our model gets both the transcript and the emotion out of the same
> frozen backbone."

### 6 · Fusion: why the level matters — table B1
Point at the `f1(borderline)` column.
> "Late fusion combines two risk bits. It has nowhere to represent the middle
> case: **0.016**. The methods that combine through a shared representation reach
> 0.22–0.24. That column is what explains the ordering."

### 7 · How much the emotion channel is worth — table B2
> "Fine-tuning the text buys +0.070. Adding the emotion channel buys **+0.178**.
> **2.56×.** That single line is why this project exists."

### 8 · Model selection — table B3
Your strongest methodology slide.
> "`early` is first in the oracle table. But it **loses 0.11** once class
> weighting is enabled, and falls to 0.4610 with real SER.
> We picked the winner under **deployment conditions**, not under oracle."

### 9 · DEMO
Four clips, in order. **2 and 3 are the actual show.**

| # | what to do | what to say |
|---|---|---|
| 1 | fire report, panicked | "Consistent. Normal." |
| 2 | **same sentence, flat voice** | "Same text. Only the voice changed. **The verdict flipped.**" |
| 3 | looking for keys, panicked | "It catches the reverse too." |
| 4 | asking for directions, calm | "It does not call everything an anomaly." |

About the transcript: *"Beam search is off; we did not want a compiler dependency
on the laptop."*

### 10 · Honesty — table B4
> "The fusion head was trained on **clean emotion labels**, and the demo feeds it
> **real SER output**. We measured the cost: **−0.057 macro-F1.**
> We tried training with noise and it **did not work** — 2 of 12 configurations
> improved. And there is a reason: the injected noise is label-independent, so it
> is irreducible."

A measured negative result is stronger than a hidden weakness.

### 11 · Future work
- Restricting LoRA to the upper layers to share compute (32%, unmeasured)
- End-to-end synthetic data (the current fusion dataset is symbolic level)
- A separate test set for SER

---

## HARD QUESTIONS

**"Why did 300 hours not win on dev-clean?"**
> "We did not expect it to. The baseline's data is 100% LibriSpeech, ours is 33%.
> With 1.20 million trainable parameters this is a capacity trade: we gave up a
> little on clean speech and gained on hard speech. Accented and spontaneous
> speech was the target all along."

**"Isn't macro-F1 0.52 low?"**
> "Three classes, chance is 0.33. The strongest text-only baseline is 0.41. The
> `borderline` class is genuinely hard — even human judges disagree there."

**"Did you tune on the test set?"**
> "We ran both protocols. The strict one tunes on dev-other, which is never
> reported. The second tunes on dev-clean — **which is what the baseline did** —
> so the comparison runs under equal rules. We report both."

**"Is the fusion data real?"**
> "No. It is LLM-generated and symbolic level — no audio, just an emotion label
> and text. It is a benchmark comparing fusion **combiners**, not an end-to-end
> system. The labels come from a multi-judge vote, not from a single model."

**"Why not use Whisper?"**
> "Whisper does not produce emotion. We need two channels and both come out of
> the **same** frozen backbone. We are also 2.28× faster at the base tier."

**"Does SER have no test set?"**
> "It does not. Validation was used for both selection and reporting, which
> creates a mild optimism. That is a gap on our side."

---

## THINGS NOT TO SAY

- "We run the backbone once" — false
- "Better everywhere" — not on dev-clean
- An uncalibrated confidence percentage — show the distribution over the three
  classes, do not invent a single number
- "We tested on real emergency calls" — the model never saw one

---

## FINAL CHECK (before the demo)

- [ ] All four clips produce the expected verdict
- [ ] `demo_prefetch.py` reports READY
- [ ] `app.py:225` is fixed (Pipeline kwarg bug — **the app will not start
      otherwise**)
- [ ] The page 6 and page 7 corrections have been made
