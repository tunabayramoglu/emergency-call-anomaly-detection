# Results

Every number here was measured. Each section names its source.

## 1. ASR

### 1.1 Development ladder (phase 1, dev-clean)

How the architecture was built up. This column is the validation run from
development; it is **not from the same measurement round** as the final table in
1.2, so do not merge the two into one column.

| setup | CER | WER |
|---|---|---|
| baseline — single layer 9, frozen | 8.20% | ~31% |
| weighted-sum [6–12], frozen | 5.17% | 19.1% |
| weighted-sum [8,9,10], frozen | 4.40% | ~17% |
| weighted-sum [9,10], frozen | 4.40% | 16.7% |
| LoRA [9,10] (q_proj/v_proj, r=16) | 3.37% | 12.02% |
| read [9,10,11,12] + adapt [1–12] | 3.19% | 11.35% |
| + KenLM | 1.96% | 5.67% |

Layer 8's weighted-sum weight fell to 0.03 and effectively died — reading more
layers does not add information, it dilutes it.

Two cautions about this table. The read/adapt row is what the shipped phase-1
checkpoint contains: `head.pt` carries a four-element `layer_w` and `adapter.pt`
spans layers 1–12, which is `ws=(9,10,11,12)` and LoRA over 1–12 — an earlier
version of this table said [9,10,11] and [1–11]. And the KenLM row's decoder
settings are not recorded anywhere in the repository. The 300h decoder is
α=0.6 · β=0.0 · beam 50, from `weights/ASR-300.zip → lm_params_clean.json`; the
phase-1 grid in `asr/phase1/kenlm_grid.py` searches α over 0.3/0.5/0.7/0.9 at
beam 100, so the α=0.75 · β=1.5 · beam 256 this row used to name cannot have come
from it.

### 1.2 The final two columns — 100h vs 300h

| system | dev-clean greedy | dev-clean +KenLM | L2-ARCTIC greedy | L2-ARCTIC +KenLM |
|---|---|---|---|---|
| 100h (previous FINAL) | 10.04 | **5.15** | 32.04 | **20.26** |
| 300h | 10.65 | **5.13** | 27.01 | **15.37** |
| difference | +0.61 | −0.02 | −5.03 | **−4.89 (−24%)** |

No difference on clean read speech; on accented speech a quarter of the error
disappears. The story is not "more data" but **more diverse data**: the training
set has 748 accents where the previous one had none.

Source, 300h row: `weights/ASR-300.zip` → `eval_results.json`, which also
confirms L2-ARCTIC's 3,599 utterances. Its `decoder_params` are α=0.6, β=0.0,
beam 50.

Source, 100h row: **nothing in this repository.** `eval_results.json` holds one
system key, `FINAL_300h`, and `asr/phase2_300h/eval_asr.py` scores one run
directory at a time, so it cannot emit a second row. The 100h figures were
carried over from the phase-1 write-up and were not re-measured under this
protocol. Treat the comparison as indicative until someone re-runs the 100h
checkpoint through `eval_asr.py`; §1.3's 100h column has the same status.

One caveat that applies to the 300h row itself: `lm_params_clean.json` records
`tuned_on_a_reported_set: true` — α and β were fitted on dev-clean, the set the
first two columns report. The archive also ships `lm_params_other.json`, the
strict variant fitted on dev-other, which lands on the same α=0.6 and β=0.0 and
differs only in beam width. The choice does not appear to move the number, but it
should be stated rather than inferred.

### 1.3 By accent (L2-ARCTIC, +KenLM, WER %)

| native language | n | 100h | 300h | difference |
|---|---|---|---|---|
| Arabic | 599 | 19.7 | 13.3 | −32.8% |
| Hindi | 600 | 11.5 | 7.7 | −32.8% |
| Korean | 600 | 16.3 | 11.8 | −27.6% |
| Spanish | 600 | 18.9 | 14.2 | −24.8% |
| Chinese | 600 | 24.1 | 18.2 | −24.4% |
| Vietnamese | 600 | 31.0 | 27.0 | −13.0% |

All six native languages improve — not the luck of a single accent. Vietnamese is
both the hardest and the least improved, which is an honest limit.

### 1.4 Whisper comparison (L2-ARCTIC)

| system | WER | CER | RTF | trainable params |
|---|---|---|---|---|
| whisper-base | 15.38 | 7.97 | 0.012 | 74M (full model) |
| **ours 300h + KenLM** | **15.37** | **7.40** | **0.005** | **1.20M** |
| whisper-small | 10.59 | 5.45 | 0.020 | 244M |
| whisper-medium | 8.10 | 4.22 | 0.038 | 769M |

The Whisper rows count whole models, because the whole model is what gets
trained. The row for this system counts what was actually trained on the ASR side, which
is the LoRA adapter plus the CTC head — 589,824 + 613,666, see §4. The backbone
under it stays frozen.

Same accuracy as whisper-base at 2.28× the speed with 61× fewer trainable
parameters. Medium is more accurate but 7.36× slower and a full model.

On CPU (phase-1 measurement): RTF 0.055 and ~505 MB RAM. whisper-medium under the
same conditions: RTF 1.33 and 3 GB, i.e. slower than real time. These four CPU
figures come from `bench/cpu_bench.ipynb`, whose outputs were not kept, so
unlike the rest of this table they cannot be re-checked from the repo.

### 1.5 Telephone-band robustness (phase 1)

| condition | greedy WER | +KenLM WER |
|---|---|---|
| clean | 11.35% | 5.67% |
| tel (bandpass 300–3400 Hz) | 14.92% | 7.60% |
| tel8k (bandpass + 8k resample) | 15.89% | 8.37% |

The degradation multiplier stays around 1.35 with and without the LM — the LM does
not absorb the domain shift, it just scales with it. The optimal α stayed at 0.75
in all three conditions.

### 1.6 Training data (300 hours)

| corpus | hours | why |
|---|---|---|
| LibriSpeech train.100 | 100 | comparability with the 100h baseline |
| Common Voice | 106 | accent diversity — 748 accents |
| AMI | 50 | spontaneous / meeting speech, ihm+sdm |
| VCTK | 44 | clean accent diversity |
| **total** | **300.00** | ~196.6k rows |

L2-ARCTIC never entered training; it was used for evaluation only.

## 2. SER

Config: weighted-sum [7,8,9,11,12] + LoRA [1–12] (r=16, α=32), meanstd pooling,
256-dimensional head, dropout 0.3, class-weighted CE with 0.05 label smoothing,
SpecAugment and pink noise (p=0.3). Telephone and 8k augmentation are off.

| metric | value |
|---|---|
| best epoch | 12 / 22 |
| UA (6 classes) | **59.2%** |
| WA (6 classes) | **61.3%** |
| chance | 16.7% |
| binary risk detector | **86.0% acc** · 95.9% precision · 83.1% recall (n=902) |
| validation set | 1,148 clips |

The learned layer weights — a direct measure of which layer the model actually
uses:

| L7 | L8 | L9 | L11 | L12 |
|---|---|---|---|---|
| 0.098 | 0.284 | 0.069 | 0.229 | **0.320** |

Emotion information concentrates in the top layer but not exclusively; L8 takes
the second largest share and L9 in between is nearly dead. This is the evidence
that picking a single layer is not enough.

Source: `weights/SER.zip` → `config.json`, `summary.json`, for everything except
the binary-detector row. Those three figures come from a 902-clip confusion
matrix that is not stored in the checkpoint; it is reproduced in
`fusion/benchmark_modules/ser_noise.py:82-88`, where 776/902, 512/534 and 512/616
give the three percentages exactly. Note that 902 is not the 1,148-clip
validation set — the two rows describe different populations.

**Three limits.** The validation set was used for both model selection and
reporting, so the numbers carry some selection optimism and there is no separate
test set.

The split is not speaker-independent in the way that phrase implies. Speaker ids
are parsed from filenames, but four of the seven corpora emit a constant: CREMA-D
takes the parent directory name, RAVDESS tests for eight hyphen-separated fields
where the filenames have seven and so always falls through to `"unknown"`, and JL
and ASVP-ESD are hardcoded. The grouping is therefore corpus-level, not
speaker-level, so 59.2% UA measures generalisation to a held-out *corpus*.
`ser/download_datasets.py:73,120,185,225` is where the ids are assigned.

Nothing records which branch of the split ran, so it has to be recovered
arithmetically: the random fallback would have produced a validation set of
int(28,698 × 0.15) = 4,304 clips, and the recorded size is 1,148, so the
speaker branch is the one that ran.

### 2.1 Collected emotion data

| source | rows | hours |
|---|---|---|
| CREMA-D | 7,442 | 5.3 |
| RAVDESS | 1,440 | 1.7 |
| JL-Corpus | 2,396 | 1.4 |
| ASVP-ESD | 13,802 | 17.8 |
| TESS | 2,800 | 1.6 |
| SAVEE | 480 | 0.5 |
| Kaggle Emergency | 338 | 0.3 |
| **total** | **28,698** | **28.6** |

The six-class taxonomy was built around arousal and valence rather than emotion
names: `panic` ← pain · `fear` ← fearful · `urgency` ← surprised, excited ·
`distress` ← angry, disgust, sad, anxious, worried · `confusion` ← pensive ·
`neutral` ← neutral, calm, happy, apologetic, enthusiastic.

All seven corpora are acted studio recordings. That is the main data risk in
this half of the project: the model may be learning performed distress rather
than the real thing, and nothing here measures the gap.

A corpus of real emergency-call audio was collected and pseudo-labelled during
the internship as a possible bridge. It was excluded from training — no speaker
diarization had been applied, so a segment could carry both the caller and the
dispatcher, and no systematic label audit was done. Neither that audio nor
anything derived from it is part of this repository.

Source: `weights/SER.zip` → `config.json`, and `ser/download_datasets.py` for the
corpus list.

## 3. Fusion

### 3.1 Dataset

| | |
|---|---|
| rows | **9,740** |
| unique seeds | 1,499 (1,500 generated) |
| profiles × events | 20 × 47 |
| generator models | 10 |
| judge models | 23 |
| mean length | 11.2 words |

Class distribution: normal 4,926 (50.6%) · anomaly 3,569 (36.6%) · borderline
1,245 (12.8%).

Emotion distribution: neutral 2,267 · confusion 1,978 · fear 1,616 · panic 1,399 ·
urgency 1,386 · distress 1,094.

Mean judge agreement is 0.941 and the judges are unanimous on 87.2% of rows. Only
one judge saw 46% of the rows — the ensemble did not run at full strength on every
row.

Source: `fusion/dataset_final.jsonl`.

Two LLM spot-check audits were run over samples of this dataset and are kept in
`fusion/dataset_audit/`. They measure different things and are easy to
misread together. The judge-accuracy audit reviewed 72 judged rows and found 58
of them correctly labelled, about 81%. The utterance-quality audit reviewed the
generated text itself and flagged roughly half of what it sampled — 48 OK
against 44 ISSUE — almost always for register rather than for labelling: a
panicking teenager who says *"He has collapsed, I am beginning chest
compressions now"* is labelled correctly but does not sound like a real caller.

The benchmark in §3.2 onward depends on the labels, not on the prose, so this
does not undercut the numbers below. It does bound reuse. Anyone treating
`dataset_final.jsonl` as a corpus of realistic emergency-call language, rather
than as a labelled mismatch benchmark, should read
`fusion/dataset_audit/README.md` first. Both audits are partial — seven of ten
batches produced a written verdict in each — and neither was used to filter the
dataset.

### 3.2 Fusion level (oracle emotion, unweighted, best encoder)

| method | encoder | macro-F1 | sd | acc | f1(borderline) |
|---|---|---|---|---|---|
| majority (class prior) | — | 0.2316 | — | 0.5324 | 0.000 |
| text_only (frozen BERT) | bert | 0.3440 | — | 0.5324 | 0.000 |
| late (two risk bits) | minilm | 0.4058 | 0.0050 | 0.5886 | 0.016 |
| emotion_only | — | 0.4111 | — | 0.5932 | 0.000 |
| text_only_finetuned | bert | 0.4136 | 0.0074 | 0.5072 | 0.173 |
| intermediate | minilm | 0.5498 | 0.0186 | 0.6619 | 0.232 |
| intermediate + FiLM | minilm | 0.5571 | 0.0124 | 0.6810 | 0.219 |
| **intermediate + attention** | bert | **0.5684** | 0.0158 | 0.6758 | **0.243** |
| early (emotion token) | bert | 0.5917 | 0.0219 | 0.7242 | 0.238 |

The `f1(borderline)` column explains the ordering. Late fusion combines two risk
bits and has nowhere to represent the middle case, so it sits at 0.016; the
methods that combine through a shared representation reach 0.22–0.24.

### 3.3 Ablation ladder — how much the emotion channel is worth

| step | macro-F1 | gain |
|---|---|---|
| majority | 0.2316 | — |
| + text (frozen BERT) | 0.3440 | +0.1124 |
| + text fine-tuning | 0.4136 | **+0.0696** |
| + emotion token (early) | 0.5917 | **+0.1782** |

The emotion channel is worth **2.56×** what fine-tuning the text is worth. This
row is the reason the project exists.

### 3.4 Regime robustness — why `early` was not chosen

| method | oracle, unweighted | oracle, weighted | **real SER** | trained with noise |
|---|---|---|---|---|
| **intermediate_attn / bert** | 0.5684 | 0.5822 | **0.5259** | 0.5018 |
| intermediate_film / minilm | 0.5571 | 0.5796 | 0.5186 | 0.5141 |
| intermediate / bert | 0.5225 | 0.5810 | 0.5163 | 0.5059 |
| early / bert | **0.5917** | 0.4810 | 0.4610 | 0.4363 |
| late / minilm | 0.4058 | 0.4735 | 0.4265 | 0.4238 |
| emotion_only | 0.4111 | 0.4459 | 0.3782 | 0.3782 |

`early` leads the oracle table but loses 0.11 once class weighting is enabled and
drops to 0.4610 with real SER output. The selection rule was not "who won under
oracle" but **"who won under deployment conditions"**.

### 3.5 Train/deploy gap — a negative result

| | macro-F1 |
|---|---|
| with oracle emotion | 0.5799 |
| with real SER output | **0.5224** |
| cost | **−0.0575** |
| recovered by training with noise | −0.009 on average, **2 of 12 cells** |

Noise-robust training does not help, because the injected noise is
label-independent and therefore irreducible. A negative result with a mechanism.

Source: the benchmark run output (96 cells, 3 seeds) and
`weights/WINNER_intermediate_attn_bert_full_p2_seed1.pt` → `meta`. The raw
`results_table.csv` was written by the notebook to its Colab output directory
(`fusion_benchmark.ipynb`, cell 33) but was not committed to this repo.

## 4. Architecture and capacity

Counted directly from the checkpoints. The BERT row counts `BertModel`, which is
what `app/pipeline.py` loads through `AutoModel.from_pretrained`. An earlier
version of this table gave 110,106,428, the size of `BertForPreTraining` — that
figure includes the masked-language-model and next-sentence-prediction heads,
which are in the published checkpoint file but are never loaded here.

| component | parameters | status |
|---|---|---|
| mHuBERT-147 backbone | 94,371,712 | frozen, shared by both tasks |
| — conv feature extractor + projection | 4,595,456 | |
| — transformer encoder (12 layers) | 89,775,488 | |
| — `masked_spec_embed` | 768 | belongs to neither row above |
| ASR LoRA adapter | 589,824 | trainable |
| ASR head (weighted-sum + CTC) | 613,666 | trainable |
| SER LoRA adapter | 589,824 | trainable |
| SER head (weighted-sum + classifier) | 398,091 | trainable |
| **audio side total** | **96,563,117** | 2,191,405 trainable |
| bert-base-uncased (text encoder) | 109,482,240 | frozen, not fine-tuned |
| fusion head (intermediate_attn) | 2,618,051 | trainable |
| **system total** | **208,663,408** | **4,809,456 trainable** (2.30%) |

For comparison: a separate full model per task would need 189.8M parameters and
twice the memory. The shared-backbone setup does the same work with 96.6M, ships
in about 10 MB, and adding a new task costs a 2.4 MB adapter.

**Limits.** Compute is not shared — the two adapters diverge from layer 1 onward,
so the forward pass runs twice. The saving is in memory and file size, not speed.
Restricting LoRA to the upper layers could save around 32%, but that was never
measured. Also, the 110.1M BERT is the single largest component in the system; the
"small model" claim holds for the audio side, not for the whole pipeline.

Source: `weights/*.pt` and
`app/models/backbone/mHuBERT-147/model.safetensors`.

## 5. Honesty notes

| topic | note |
|---|---|
| no difference on dev-clean | the 300h model does not win on clean speech; the gain is on accented and difficult speech |
| train/deploy gap | −0.058 macro-F1, measured, and training with noise did not close it |
| fusion dataset | symbolic level (emotion label + text), not end-to-end |
| SER evaluation | validation was used for both selection and reporting; no separate test set |
| SER training data | seven acted studio corpora, no real emergency-call audio; performed distress may not transfer |
| recipe difference | 100h used `bb_dropout=0.05` and synthetic noise, 300h used real RIR/MUSAN — the recipe changed, not just the data |
| the 100h column | carried over from the phase-1 write-up, not re-measured under the 300h protocol and not backed by any file here (§1.2) |
| KenLM tuning | α and β were fitted on dev-clean, which §1.2 also reports; the strict dev-other variant ships alongside and agrees (§1.2) |
| SER split | grouped by corpus rather than by speaker, because four of seven corpora emit a constant speaker id (§2) |
| fusion results | §3.2 and §3.4 were transcribed by hand from a run whose output table was not committed; only §3.5's two figures can be re-checked, against the winning checkpoint's `meta` |
| fusion selection | the sweep picks a configuration on one seed's validation score and reports the 3-seed spread afterwards, so that spread is not independent of the selection |
| fusion winner | `intermediate_attn` swaps between bert and minilm; it should be reported as "no significant difference" |
