# test cases

Traces `REQUIREMENTS.md`. **Functional correctness only** — no test asserts a
WER, an F1 or an accuracy. Model quality lives in the ASR/SER/fusion benchmarks.

```
pytest tests/ -v                        # functional suite
pytest tests/ -v -m "not needs_models"  # before setup_weights.py
pytest tests/ -v -m slow                # resource budgets (opt-in)
```

Status on a machine with the weights laid out and gradio installed, but
**without** torch: **49 passed · 26 skipped · 0 failed.** Skips are named, never
silent.

Writing these found four real defects, all fixed: the missing-weights check sat
behind `import torch` (so a half-installed environment got the wrong error);
`Paths` did not coerce a string root; `_lora_layers` handled only one of the two
config dialects actually on disk; and the fusion checkpoint glob did not match
the file's real name.

---

## Tier 1 — automated, no model files

| TC | Requirement | What it proves | Status |
|---|---|---|---|
| TC-01 | FR-01 | CTC vocabulary is exactly 30 symbols: `A–Z`, apostrophe, `\|`, `[UNK]`, `[PAD]`; `[PAD]`=29 is the blank. | pass |
| TC-01b | FR-01 | No digit or punctuation crept into the charset — KenLM was built on LibriSpeech-normalised text. | pass |
| TC-02 | FR-01 | Greedy decode collapses repeats, drops blanks, maps `\|`→space. | pass |
| TC-02b | FR-01 | A blank *between* two identical characters preserves both (`A [PAD] A` → `AA`). The classic CTC off-by-one. | pass |
| TC-02c | FR-01 | `[UNK]` is dropped, not rendered. | pass |
| TC-03 | FR-02 | The result object names the decoder that actually ran. | pass |
| TC-03b | FR-02 | When KenLM ran, the label carries its tuned α/β/beam, while the bare `decoder` name stays comparable. | pass |
| TC-03c | FR-02 | Greedy and manual labels claim no parameters — a greedy fallback must not be mistakable for a tuned KenLM run, they differ by roughly half the WER. | pass |
| TC-04 | FR-04 | `voice_risk` partitions the six emotions; only `neutral` and `confusion` are low. | pass |
| **TC-05** | **FR-07** | The two modules order the same six emotions **differently**, and the pipeline maps by name. | pass |
| **TC-05b** | **FR-07** | Quantifies the trap: an index-based mapping would corrupt 4 of 6 emotions, 3 of them high-risk. | pass |
| TC-06 | FR-13 | CLI rejects an empty invocation. | pass |
| TC-06b | FR-13 | `--self-test` returns 0. | pass |
| TC-07 | FR-05 | Adapter key remap renames only LoRA tensors; a non-LoRA key containing `default` is left alone. | pass |
| TC-07b | FR-05 | Remap preserves tensor count. | pass |
| TC-07c | FR-05 | Both config dialects work: `asr/config.json` writes an explicit `lora_layers` list, `ser/config.json` writes `lora_lo`/`lora_hi`. Supporting one silently adapts the wrong layers on the other. | pass |
| TC-07d | FR-05 | An explicit list wins over lo/hi when both are present. | pass |
| TC-12 | NFR-07 | Fusion glob matches `WINNER_*.pt` and `BEST_*.pt`, with WINNER first — the brief flags this exact naming mismatch. | pass |
| TC-19 | NFR-06 | A truncated `model.safetensors` counts as absent, not as a local backbone. | pass |
| TC-19b | NFR-06 | A complete one is used. | pass |
| TC-19c | NFR-06 | A config with no weights beside it is absent. | pass |
| TC-19d | NFR-06 | A nested `backbone/mHuBERT-147/` layout is found — the zip wraps its contents, and requiring a manual move is a step that gets skipped under time pressure. | pass |
| **TC-19e** | **NFR-06** | A truncated copy in the preferred location does **not** shadow a complete one further down the list. Otherwise the demo re-downloads 380 MB it already has. | pass |
| TC-19f | NFR-06 | Layouts beside the app, not just under `models/`, are searched. | pass |
| TC-13 | FR-16 | `pipeline --help` exits 0 and advertises `--self-test`. | pass |
| TC-14 | FR-16 | `app` imports and its parser produces the documented defaults, with no model loaded. | pass |
| **TC-15** | **FR-17** | `app` exposes no `--alpha` / `--beta` / `--beam`. Decoder parameters come from `lm_params_clean.json` only. | pass |
| TC-15b | FR-17 | `pipeline` likewise rejects `--alpha`. | pass |
| TC-16 | FR-16 | An unknown flag is rejected rather than ignored. | pass |
| TC-17 | FR-18, NFR-08 | The metrics panel removal is **complete**: no `caveats`, `load_reported_metrics` or `METRICS_PATH` survives, and `reported_metrics.json` is gone. A helper left behind with no caller is how dead code returns. | pass |
| TC-18b | FR-11 | Render helper emits the verdict and transcript. | pass |
| TC-18c | FR-19 | Transcript is HTML-escaped before rendering. | pass |
| TC-18d | FR-11 | The transcript block sets its own text colour — it rendered white-on-white when it inherited one. | pass |
| TC-18e | FR-20 | Clips are found by extension across wav/mp3/flac/m4a, with arbitrary filenames; `.txt` and `.png` in `clips/` are ignored. | pass |
| TC-18f | FR-20 | `.m4a` is listed despite libsndfile not reading it — it has a fallback and, failing that, explains itself. A file silently missing from the dropdown is worse than one that says why. | pass |
| **TC-29** | **FR-21** | An undecodable file raises `AudioDecodeError` naming the file **and a way out**, not just a cause. | pass |
| TC-29b | FR-21 | A missing file reports through the same path. | pass |
| TC-29c | FR-21 | The message is what the UI renders — Gradio's default is a red toast plus a console traceback, neither readable from the back of a room. | pass |
| TC-08 | FR-15 | Resamples 8 k / 22.05 k / 44.1 k / 48 k to 16 kHz. | pass |
| TC-08b | FR-15 | Very short clips are padded rather than crashing the backbone. | pass |
| TC-08c | FR-15 | 16 kHz input passes through bit-identical. | pass |
| TC-09 | FR-15 | Stereo collapses to mono float32. | pass |
| TC-10 | FR-15 | int16 scales by 1/32768 exactly as training did. | pass |
| **TC-10b** | **FR-15** | Amplitude is **not** normalised — training did not, so inference must not. | pass |
| TC-11 | NFR-07 | Missing model files raise `FileNotFoundError` naming each one — and do so *before* importing torch, so a half-installed environment still gets the real message. | pass |
| TC-11b | NFR-07 | All gaps are listed, not just the first. | pass |
| TC-11c | NFR-07 | Absent LM directory reports `None` rather than raising. | pass |
| TC-25 | FR-06 | `intermediate_attn` head reconstructs `text_dim` and `hidden` from checkpoint tensor shapes across three configurations. | skip (torch) |
| TC-25b | FR-06 | Emotion projection is bias-free — a bias breaks equivalence with `nn.Embedding` on a one-hot. | skip (torch) |
| TC-25c | FR-06 | A pooled `intermediate` checkpoint is rejected with an error naming the file to use instead. | skip (torch) |
| TC-25d | FR-06 | Dropout is inert at inference (two identical calls agree). | skip (torch) |
| TC-25g | FR-06 | A `num_heads` that does not divide `text_dim` is rejected. | skip (torch) |
| **TC-25h** | **FR-06** | Changing the **padded** token positions does not change the output — the key-padding mask is genuinely applied. Without it a short utterance's verdict would depend on 64 tokens of padding. | skip (torch) |

## Tier 2 — automated, model files required

| TC | Requirement | What it proves |
|---|---|---|
| TC-20 | FR-01 | Transcript contains only `A–Z`, apostrophe and space. |
| TC-21 | FR-02, NFR-07 | Active decoder is `greedy` or `kenlm` and matches what the pipeline advertises. |
| TC-22 | FR-03 | SER returns all six classes, probabilities sum to 1, argmax matches the reported label. |
| **TC-23** | **FR-05** | Every LoRA layer carries **both** `asr` and `ser` adapters — one backbone, two adapters, verified structurally rather than claimed. |
| TC-23b | FR-05 | Backbone is fully frozen. |
| TC-23c | FR-05, NFR-05 | Backbone and all three heads are in eval mode — SpecAugment and dropout are off, so the same clip cannot give two answers. |
| TC-24 | FR-14 | An adapter whose keys do not match raises, rather than half-loading. |
| TC-25e | FR-06 | Fusion returns a valid label and a normalised distribution. |
| TC-25f | FR-06 | An unknown emotion string is rejected. |
| TC-26 | UR-03, UR-04, FR-10 | Holding text fixed and sweeping all six emotions changes the verdict on at least one probe sentence — the emotion channel is genuinely wired through. |
| TC-26b | UR-03 | Even where the label holds, the distribution moves measurably with emotion. |
| TC-27 | FR-12, UR-07 | Checkpoint name, regime, class-weighting flag and recorded val-F1 are all exposed. Presence only; the value is not asserted. |
| TC-28 | FR-13 | `--text` / `--emotion` overrides bypass their channels and are labelled `manual`. |
| TC-40 | NFR-01, UR-01 | Everything is on CPU. |
| TC-44 | NFR-05 | Same input, same output — transcript, emotion, verdict and probabilities. |
| TC-35 | FR-08..FR-12 | The Gradio Blocks graph builds against a real pipeline — broken component wiring is caught at test time, not demo time. The server is not started. |
| TC-46 | UR-02 | All four quadrants of the content×voice matrix are constructible and repeatable. |

## Tier 3 — resource budgets (`-m slow`, opt-in)

| TC | Requirement | Budget |
|---|---|---|
| TC-41 | NFR-02 | 5 s clip end-to-end ≤ 5 s wall clock (RTF ≤ 1.0). |
| TC-42 | NFR-03 | Resident memory ≤ 2 GB. |
| TC-43 | NFR-04, UR-06 | Cold pipeline load ≤ 120 s. |
| TC-45 | NFR-06, UR-01 | Starts with `HF_HUB_OFFLINE=1` in a fresh subprocess — proves no hidden network call. |

## Tier 4 — manual (UI, hardware)

| TC | Requirement | Steps | Expected |
|---|---|---|---|
| TC-30 | FR-08, UR-02 | Prepared clips tab → select each of the four clips → Analyse. | Each renders a verdict; `high_flat` and `low_panic` land in a mismatch quadrant, the other two in a congruent one. |
| TC-31 | FR-09 | Microphone tab → record ~4 s → Analyse. | Transcript and emotion appear; the header note about live audio being noisier is visible. |
| TC-32 | FR-10, UR-04 | Fusion-only tab → keep the text, change the emotion through all six. | Verdict and/or distribution move. Works with the audio path untouched. |
| TC-33 | FR-11 | Any analysis. | Transcript, emotion bars and verdict bars are on screen and legible — dark text, not inheriting the theme. |
| TC-34 | FR-12, UR-07 | Read the startup log in the terminal. | It names the backbone and where it loaded from, the ASR and SER runs, the fusion checkpoint with its regime and encoder, and the active decoder. The UI shows none of this by design — it is the presenter's to state. |

---

## Known functional gaps

- **TC-31 cannot be automated.** Microphone capture is browser-side; it is verified by hand before the presentation.
- **Tier 2 and 3 have never been executed here** — this environment has neither the model files nor torch. They are written and will run on the target laptop; a green Tier 1 is not evidence about them.
- **No end-to-end accuracy gate.** By design, per the scope decision above. If a regression silently degrades the acoustic path, this suite will not catch it — the transcript charset check (TC-20) is the only guard, and it only catches corruption, not drift.
