# Demo — build brief

_For the agent building the interactive app. Everything here is measured, not assumed. Do not retrain anything; all weights exist._

## 1. What the demo does

This system detects an **anomaly in an emergency call**: a mismatch between **how** something is said (voice emotion) and **what** is said (text content). A calm voice reporting a fire is an anomaly; a panicked voice reporting a fire is not.

Audio in → three models → one verdict (`normal` / `borderline` / `anomaly`).

```
                    ┌── LoRA(ASR) + WS head + CTC ──► transcript ──► BERT tokens ──┐
mic audio ──► mHuBERT-147 (FROZEN, shared)                                          ├──► fusion head ──► verdict
                    └── LoRA(SER) + head ───────────► emotion (6 classes) ─────────┘
```

**The frozen backbone is loaded ONCE and shared by both adapters.** That sharing is the project's architectural claim — do not load two copies.

## 2. Files

All present locally; nothing needs downloading except the two HuggingFace models.

| what | file(s) | notes |
|---|---|---|
| ASR | `asr/config.json`, `adapter.pt`, `head.pt` | ~5 MB |
| ASR decoder settings | `asr/lm_params_clean.json` | **use these**, see §4 |
| SER | `ser/config.json`, `adapter.pt`, `head.pt` | ~5 MB, 6 emotion classes |
| Fusion | `fusion/WINNER_intermediate_attn_bert_full_p2_seed1.pt` | ~10 MB, `state_dict` + `meta` |
| backbone | `utter-project/mHuBERT-147` (HF) | ~380 MB, downloaded once |
| text encoder | `bert-base-uncased` (HF) | ~440 MB |
| KenLM (optional) | `3-gram.pruned.1e-7.arpa` | 98 MB, see §4 |

`demo_prefetch.py` verifies all of the above and pre-downloads the HF models so the demo works offline. **Its glob expects `fusion/BEST_*.pt`; the file is named `WINNER_*.pt` — fix one or the other before demo day.**

## 3. How to load each model — reuse existing code, do not rewrite

**ASR.** `asr/eval_asr.py` has `load_our_model(run_dir, device)` which builds the backbone, injects the LoRA adapter and the weighted-sum CTC head, and returns `(bb, head, ws, vocab)`. Import it. It reads `config.json` for `ws` / `lora_layers` / `lora_r` / `lora_alpha`, so it stays correct if those ever change.

Greedy decoding: `eval_asr.greedy_decode(logprobs, vocab)`.

**SER.** `train_ser.py` loads its adapter/head the same way. There are **two
different orderings of the same six labels**, and crossing between them by index
silently mislabels four of the six, three of them high-risk.

```python
# ser/train_ser.py CLASSES — what the SER head's logit index means
["neutral", "distress", "fear", "urgency", "panic", "confusion"]

# fusion/benchmark_modules/common.py EMOTIONS — the fusion head's 6-dim one-hot
["neutral", "confusion", "fear", "panic", "urgency", "distress"]
```

Cross the boundary **by name**, never by index. `app/pipeline.py:699` builds the
one-hot with `FUSION_EMOTIONS.index(emotion)` for exactly this reason. Neither
ordering is recorded in any checkpoint — `ser/config.json` says only `n_cls: 6` —
so these two constants are the only record.

**Fusion.** The checkpoint is a `state_dict`, so you need the class. Copy `_AttnFusionModel` out of `fusion/benchmark_modules/attn.py` (or import the module). Construct it with the values the checkpoint was trained with:

```python
model = _AttnFusionModel(text_dim=768, hidden=256, num_heads=4)   # dropout=0.3
model.load_state_dict(payload["state_dict"])
model.eval()
```

Its `forward(emo_onehot, token_states, attn_mask)` takes:

- `emo_onehot` — `(N, 6)` float, one-hot over the emotion list above
- `token_states` — `(N, T, 768)` float, **per-token** `bert-base-uncased` hidden states, **not** pooled CLS
- `attn_mask` — `(N, T)` int, `1` = real token, `0` = padding

Tokenise exactly as the benchmark did or the features will not match:

```python
tokenizer(text, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
```

Output is `(N, 3)` logits over `["normal", "borderline", "anomaly"]`.

`payload["meta"]` carries the full provenance and the measured scores — read it rather than hardcoding numbers.

## 4. Gotchas that will otherwise cost you an hour each

**Use the tuned decoder settings.** `lm_params_clean.json` holds `alpha`, `beta`, `beam_width` fitted for this exact checkpoint. pyctcdecode's defaults are not these, and the reported WER was measured with the tuned values.

**Cast to float32 before numpy.** The forward runs under `torch.autocast(bfloat16)`. `logits.log_softmax(-1).cpu().numpy()` raises `TypeError: Got unsupported ScalarType BFloat16`. Do `.float()` first. This bit us in `eval_asr.py`.

**KenLM is optional and is a build hazard.** The python bindings ship a pre-generated Cython file that does not compile on Python ≥3.12 (`_PyLong_AsByteArray` signature changed); it needs regenerating with Cython 3. **Greedy decoding is fine for the demo** — it drops the transcript from 5.13% to 10.65% WER on clean read speech, which does not change the anomaly verdict for demo-length utterances, and it removes a dependency that can fail on stage. If you do want KenLM, use Python ≤3.12.

**The backbone must be in `eval()`.** It is frozen; `train()` would enable its internal dropout.

**mHuBERT expects 16 kHz mono float32** in `[-1, 1]`. Resample microphone input.

## 5. Numbers you can state, and what they mean

Measured on the full test sets, not samples.

| | dev-clean | L2-ARCTIC (accented) |
|---|---|---|
| ASR 300h, greedy | 10.65% WER | 27.01% |
| ASR 300h, +KenLM | **5.13%** | **15.37%** |
| previous 100h model, +KenLM | 5.15% | 20.26% |
| whisper-base | — | 15.38% |
| whisper-medium | — | 8.10% |

On accented speech the model improved **24% relative** over the previous one and matches whisper-base at **2.28× the speed** and less memory. On clean speech it is unchanged — the gain came from training on 748 accents, not from more data in general.

Fusion, 3-class macro-F1: **0.5224** with real SER emotion (0.5799 with oracle emotion). Chance is 0.33; the strongest text-only baseline is 0.41.

## 6. Say this out loud in the demo

The fusion head was trained on **clean emotion labels** and the demo feeds it **real SER output**. The measured cost is **−0.057 macro-F1** for this checkpoint (0.5799 → 0.5224). Training under simulated SER noise did **not** recover it (2 of 12 configurations improved) because the injected noise is label-independent and therefore irreducible.

State it as a measured limitation. It is more convincing than pretending the gap is not there, and the checkpoint's `payload["meta"]["deployment_note"]` states **−0.052** instead — that is the 3-seed mean (0.5822 → 0.5259), while the −0.057 above is this single checkpoint's own drop. Both are correct; say which one you are quoting.

## 7. Non-goals

- Do not retrain anything. Every weight is final and reported.
- Do not swap the language model. The 100h-vs-300h comparison rests on both rows using the same LM.
- Do not use the `early` fusion method even though it tops the oracle table. It scores 0.5917 with oracle emotion and 0.4610 under real SER, and it collapses by 0.11 when class weighting changes. `intermediate_attn` is the one that holds up.
- Do not report a per-utterance confidence you have not calibrated. Show the three-way logits or a label, not a fabricated percentage.
