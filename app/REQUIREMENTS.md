# requirements (as delivered)

_2026-07-30. Written **after** implementation, describing the system that exists._

This supersedes the target table in `docs/Roadmap.md` §7 for the purposes of
acceptance. The roadmap described a different architecture (cross-attention
fusion over mel spectrograms, Silero VAD, 5 emotion classes, MSP-Podcast). What
was built is a weighted-sum + LoRA backbone with a symbolic-level fusion
combiner over 6 classes. Deviations are listed in §4 rather than hidden.

**User** = Tuna, in two roles: presenter of the demo, and researcher reading its
output. There is no third-party operator in scope for this PoC.

---

## 1. Functional requirements

| ID | Requirement | Verified by |
|---|---|---|
| FR-01 | The ASR channel converts 16 kHz mono audio to a transcript over the vocabulary `A–Z`, apostrophe, space. | TC-01, TC-02, TC-20 |
| FR-02 | The ASR channel decodes with KenLM beam search when a language model is available, and falls back to greedy CTC otherwise. Each result reports which decoder actually ran, and — when it was KenLM — the α/β/beam it used, so a silent fallback cannot pass for a tuned run. | TC-03, TC-03b, TC-03c, TC-21 |
| FR-03 | The SER channel converts the same audio to one of six emotions (`neutral, distress, fear, urgency, panic, confusion`) with a probability distribution. | TC-22 |
| FR-04 | A binary `voice_risk` is derived from the emotion: `high` for `distress, fear, urgency, panic`; `low` otherwise. | TC-04 |
| FR-05 | Both channels run on **one** frozen mHuBERT-147. LoRA adapters stay unmerged and are injected under separate names; exactly one is active per forward pass. | TC-23, TC-24 |
| FR-06 | The fusion head (`intermediate_attn`) maps (6-dim emotion one-hot, **768-dim per-token BERT `last_hidden_state` of shape (N, T, 768) — no CLS pooling, no normalisation**, plus its `(N, T)` attention mask) to one of `normal / borderline / anomaly` with a probability distribution. Text is tokenised with `max_length=64, truncation=True, padding="max_length"`, matching `encoders.embed_texts_tokenwise`. | TC-25, TC-25h |
| FR-06a | Padded token positions must not influence the output: the key-padding mask is applied inside the attention. | TC-25h |
| FR-06b | A checkpoint that is not an `intermediate_attn` head is rejected with an error naming the file to use. | TC-25c |
| FR-07 | The emotion label crosses the SER→fusion boundary **by name**. The two modules order the same six labels differently. | TC-05 |
| FR-08 | The app analyses a prepared clip selected from `app/clips/`. | TC-30 (manual) |
| FR-09 | The app analyses audio recorded from the microphone. | TC-31 (manual) |
| FR-10 | The app offers an audio-free mode driving the fusion head from typed text and a chosen emotion. | TC-26, TC-32 (manual) |
| FR-11 | Each analysis displays: transcript, emotion distribution and verdict distribution. Every text block sets its own colour, so no element can render invisibly under a theme it did not anticipate. | TC-18b, TC-18d, TC-33 (manual) |
| FR-12 | Checkpoint provenance (name, regime, class weighting, recorded macro-F1) is available on the `Pipeline` object and is logged at startup. It is **not** shown in the UI. | TC-27 |
| FR-13 | The pipeline is usable from the command line, with `--text` and `--emotion` overrides that bypass either channel. | TC-06, TC-28 |
| FR-14 | Loading an adapter whose keys do not fully match the injected adapter is an error, not a warning. | TC-07, TC-24 |
| FR-15 | Audio at any sample rate, channel count or integer dtype is coerced to 16 kHz mono float32 before inference, with no amplitude normalisation. | TC-08, TC-09, TC-10 |
| FR-16 | Both programs expose a command-line entry point that parses arguments and reports usage without loading any model. | TC-13, TC-14, TC-16 |
| FR-17 | Decoder parameters (`alpha`, `beta`, `beam_width`) come from `asr/lm_params_clean.json` and are **not** overridable from the command line, so nothing can be decoded at values the reported WER was never measured at. | TC-15, TC-15b |
| FR-18 | The app carries no metrics panel. No figure the demo does not measure live appears on screen. | TC-17 |
| FR-19 | Text that reaches the renderer is HTML-escaped. | TC-18c |
| FR-20 | Clips are discovered by extension, not by filename. libsndfile-native formats (`.wav`, `.flac`, `.ogg`, `.mp3`, `.aiff`, `.au`) and fallback formats (`.m4a`, `.mp4`, `.aac`, `.wma`) are both listed; non-audio files are ignored. | TC-18e, TC-18f |
| FR-21 | A file no available decoder can read produces a named error carrying the fix, rendered in the UI — not a traceback in the console. `load_wav` tries libsndfile, then PyAV if installed. | TC-29, TC-29b, TC-29c |

## 2. Non-functional requirements

| ID | Requirement | Value | Verified by |
|---|---|---|---|
| NFR-01 | Runs CPU-only. No GPU, no CUDA install required. | — | TC-40 |
| NFR-02 | End-to-end latency for a 5 s clip on the target laptop CPU. | ≤ 5 s (RTF ≤ 1.0) | TC-41 |
| NFR-03 | Resident memory during inference. | ≤ 2 GB | TC-42 |
| NFR-04 | Cold model load. | ≤ 120 s | TC-43 |
| NFR-05 | Determinism: identical input yields identical output within a process. | exact | TC-44 |
| NFR-06 | Inference requires no network once model files are local. | `HF_HUB_OFFLINE=1` succeeds | TC-45 |
| NFR-07 | Degradation is graceful and named: a missing LM falls back to greedy with a log line; missing model files raise an actionable error naming each file. | — | TC-11, TC-21 |
| NFR-08 | The UI states no figure it does not compute live, so nothing on screen can go stale or need a caveat beside it. Provenance and the reported metrics belong to the presenter and the write-up, not the demo. | — | TC-17 |

### Model quality — reported, deliberately not tested here

The test suite asserts **functional correctness only**. No test asserts a WER,
an F1 or an accuracy: those figures belong to the ASR/SER/fusion benchmarks that
produced them, and re-asserting them in an acceptance suite would either
duplicate that work or turn a research result into a brittle CI threshold.

| | Figure | Source |
|---|---|---|
| ASR (300 h) | WER 10.65% greedy / 5.13% KenLM (dev-clean); 27.01% / 15.37% (L2-ARCTIC) | `models/asr/eval_results.json` |
| SER | UA 59.2 / WA 61.3 six-class; 86.0% as a binary risk detector | `_Staj/meeting/D3_voice_channel.png` |
| Fusion | macro-F1 0.5799 oracle / 0.5224 with real SER | the checkpoint's own `meta` |

These belong to the presenter and the write-up. The UI shows none of them
(NFR-08); TC-27 checks only that the pipeline exposes the loaded checkpoint's
provenance, not what it equals.

## 3. User requirements

| ID | As the user I need to… | Verified by |
|---|---|---|
| UR-01 | run the entire demo on my own laptop, with no GPU and no internet connection. | TC-40, TC-45 |
| UR-02 | show all four quadrants of the content×voice matrix on demand, deterministically. | TC-46, TC-30 (manual) |
| UR-03 | demonstrate live that the emotion channel changes the decision — hold the text fixed, change the emotion, watch the verdict move. | TC-26 |
| UR-04 | still demonstrate the fusion layer if the audio path fails mid-presentation. | TC-26, TC-32 (manual) |
| UR-05 | not have the demo state a figure it did not just compute, so nothing on screen can be stale or need a caveat I must remember. | TC-17 |
| UR-06 | start the app from a cold laptop in under two minutes. | TC-43 |
| UR-07 | know which checkpoint and decoder are loaded without reading source code — from the startup log, and from the decoder label on every result. | TC-03b, TC-27, TC-34 (manual) |

## 4. Deviations from `docs/Roadmap.md` §7

| Roadmap item | Status | Note |
|---|---|---|
| WER < 10% clean | **met with KenLM (5.1%), missed by 0.1 pt greedy (10.1%)** | State the decoder alongside the number. |
| Inference < 2 s per 5 s chunk on T4/A100 | **exceeded, and on weaker hardware** | Measured CPU RTF 0.055 for ASR; the requirement assumed a GPU. |
| Weighted F1 > 0.65 | **not comparable** | SER was selected and reported on UA (macro-recall) and WA, not weighted F1. UA 59.2 / WA 61.3. |
| Per-class recall on Distress/Panic > 0.70 | **not measured** | The per-class breakdown was never reported. Honest gap. |
| Cross-attention fusion | **replaced** | Delivered as intermediate feature-level fusion (MLP). `intermediate_attn` was benchmarked but its winner flips between encoders, so no winner is claimed. |
| Silero VAD | **out of scope** | Not implemented; clips are pre-trimmed. |
| MSP-Podcast / cross-corpus evaluation | **out of scope** | SER trained and evaluated on academic corpora only. |
| 80/10/10 split | **changed** | Speaker-independent split; validation used for both selection and reporting, with no separate test set. |
