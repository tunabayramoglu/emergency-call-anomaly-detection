# slide tables

_Every number was measured. Sources: `fusion/results_table.csv` (96 cells, 3
seeds), `benchmark_20260730_224322.json` (all of dev-clean's 2,703 and
L2-ARCTIC's 3,599 utterances), and the `meta` block of `WINNER_...pt`._

---

## A. ASR — the headline

**Table A1 · 100h → 300h, two test sets**

| system | dev-clean greedy | dev-clean +KenLM | L2-ARCTIC greedy | L2-ARCTIC +KenLM |
|---|---|---|---|---|
| 100h (previous FINAL) | 10.04 | **5.15** | 32.04 | **20.26** |
| 300h (new) | 10.65 | **5.13** | 27.01 | **15.37** |
| difference | +0.61 | −0.02 | −5.03 | **−4.89 (−24%)** |

> No difference on clean read speech; on accented speech a quarter of the error
> is erased. The story is not "more data" but **more diverse data**: 748 accents
> in training, where the previous model had none.

**Table A2 · By accent (L2-ARCTIC, +KenLM, WER %)**

| native language | n | 100h | 300h | difference |
|---|---|---|---|---|
| Arabic | 599 | 19.7 | 13.3 | −32.8% |
| Hindi | 600 | 11.5 | 7.7 | −32.8% |
| Korean | 600 | 16.3 | 11.8 | −27.6% |
| Spanish | 600 | 18.9 | 14.2 | −24.8% |
| Chinese | 600 | 24.1 | 18.2 | −24.4% |
| Vietnamese | 600 | 31.0 | 27.0 | −13.0% |

> All six native languages improve — not the luck of one accent. Vietnamese is
> both the hardest and the smallest gain: an honest limit.

**Table A3 · Whisper comparison (L2-ARCTIC)**

| system | WER | CER | RTF | trainable params |
|---|---|---|---|---|
| whisper-base | 15.38 | 7.97 | 0.012 | 74M (full model) |
| **ours 300h + KenLM** | **15.37** | **7.40** | **0.005** | **1.20M** |
| whisper-small | 10.59 | 5.45 | 0.020 | 244M |
| whisper-medium | 8.10 | 4.22 | 0.038 | 769M |

> Same accuracy as whisper-base, **2.28× faster**, 61× fewer trainable
> parameters. Medium is more accurate but 7.36× slower and a complete model — we
> are an adapter plus a head bolted onto a frozen backbone. The Whisper rows
> count whole models; ours counts the ASR LoRA adapter plus the CTC head
> (589,824 + 613,666), which is what was actually trained.

**Table A4 · Data mix (300 hours)**

| corpus | hours | why |
|---|---|---|
| LibriSpeech train.100 | 100 | comparability with the 100h baseline |
| Common Voice | 106 | accent diversity — 748 accents |
| AMI | 50 | spontaneous / meeting speech, ihm+sdm |
| VCTK | 44 | clean accent diversity |
| **total** | **300.00** | 196,620 rows |

---

## B. Fusion

**Table B1 · Fusion level (oracle emotion, unweighted, best encoder)**

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

> **The `f1(borderline)` column explains the ordering.** Late fusion combines two
> risk bits and has nowhere to put the middle case → 0.016. The methods that
> combine through a shared representation reach 0.22–0.24.

**Table B2 · Ablation ladder — how much the emotion channel is worth**

| step | macro-F1 | gain |
|---|---|---|
| majority | 0.2316 | — |
| + text (frozen BERT) | 0.3440 | +0.1124 |
| + text fine-tuning | 0.4136 | **+0.0696** |
| + emotion token (early) | 0.5917 | **+0.1782** |

> **The emotion channel is worth 2.56× what fine-tuning the text is worth.** This
> is why the project exists.

**Table B3 · Regime robustness — why `early` was not chosen**

| method | oracle, unweighted | oracle, weighted | **real SER** | trained with noise |
|---|---|---|---|---|
| **intermediate_attn / bert** | 0.5684 | 0.5822 | **0.5259** | 0.5018 |
| intermediate_film / minilm | 0.5571 | 0.5796 | 0.5186 | 0.5141 |
| intermediate / bert | 0.5225 | 0.5810 | 0.5163 | 0.5059 |
| early / bert | **0.5917** | 0.4810 | 0.4610 | 0.4363 |
| late / minilm | 0.4058 | 0.4735 | 0.4265 | 0.4238 |
| emotion_only | 0.4111 | 0.4459 | 0.3782 | 0.3782 |

> `early` tops the oracle table but **loses 0.11** once class weighting is on, and
> falls to 0.4610 with real SER. The selection rule is not "who won under oracle"
> but **"who won under deployment conditions"**.

**Table B4 · Train/deploy gap — a negative result**

| | macro-F1 |
|---|---|
| with oracle emotion | 0.5799 |
| with real SER output | **0.5224** |
| cost | **−0.0575** |
| recovered by training with noise | −0.009 on average, **2 of 12 configurations** |

> Noise-robust training **does not work**, because the injected noise is
> label-independent and therefore irreducible. A negative result with a
> mechanism.

---

## C. Architecture

**Table C1 · The shared frozen backbone**

| | two separate full models | ours |
|---|---|---|
| total parameters | 189.4M | **95.9M** |
| memory (fp32) | 0.76 GB | **0.38 GB** |
| trainable parameters | 189.4M | **1.18M** (2 × 590k) |
| shipped files | ~760 MB | **~10 MB** |
| cost of a new task | +95M parameters | **+2.4 MB adapter** |

> Compute is not shared: the two adapters diverge from layer 1 onward and the
> forward pass runs twice. Restricting LoRA to the upper layers could save around
> 32% — **future work, not measured.**

> Note: `RESULTS.md` §4 carries a parameter count taken directly from the
> checkpoints (96.56M total on the audio side, 2.19M trainable). This table
> counts only the LoRA adapters as trainable and leaves out the heads. Use one of
> the two consistently.

---

## D. Honesty notes to say out loud

| topic | what to say |
|---|---|
| no difference on dev-clean | the 300h model does not win on clean speech; the gain is on accented and difficult speech |
| train/deploy gap | −0.057 macro-F1, measured, and training with noise did not close it |
| fusion dataset | symbolic level (emotion label + text), not end-to-end |
| SER evaluation | validation was used for both selection and reporting; there is no separate test set |
| recipe difference | 100h used `bb_dropout=0.05` and synthetic noise, 300h used real RIR/MUSAN — the recipe changed too, not just the data |

---

### Appendix: which number came from where

| table | source file |
|---|---|
| A1, A2, A3 | `benchmark_20260730_224322.json` |
| A4 | `manifest_combined.stats.json` |
| B1, B2, B3 | `fusion/results_table.csv` |
| B4 | `WINNER_intermediate_attn_bert_full_p2_seed1.pt` → `meta` |
| C1 | parameter count, from the `[LORA] 589,824 trainable params` log line |
