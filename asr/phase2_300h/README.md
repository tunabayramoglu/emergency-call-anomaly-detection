# ASR — 300h retrain

Extends the existing ASR pipeline (`ablation_engine.py`, `kenlm_grid.py`)
from a 100h LibriSpeech-only model to a 300h model that adds accent,
spontaneous-speech and channel/noise robustness for the Friday live demo
(laptop mic, in a room, non-native speaker).

## Files, and which of them is authoritative

`asr_300h_marimo.py` (46 cells) is the one to edit. It is the live
notebook, and it holds each of the ten modules below as a string literal that it
writes to disk when it runs. The `.py` files next to it are **generated output**:
an edit made directly in one of them is silently overwritten on the next run.
Each generated file now says so in its header.

`asr_300h.ipynb` is the earlier Jupyter version, superseded by the marimo
notebook and kept for reference.

The ten generated modules:

- `prepare_data.py` — download, filter, normalise, manifest-build for all
  four training sources. Never touches L2-ARCTIC (hard `assert` gate).
- `augment.py` — GPU-side augmentation module (runs on the batched waveform
  tensor inside the training loop, not per-sample on CPU).
- `fetch_noise_banks.py` — downloads the noise and RIR banks augmentation reads.
- `verify_data.py` — manifest and cache sanity checks before a long run.
- `fetch_kenlm.py` — downloads and verifies the OpenSLR 3-gram ARPA.
- `gdrive_sync.py` — checkpoint mirroring, returns `(ok, reason)` and never
  fails silently.
- `build_cache.py` — builds the int16 audio cache from the manifests.
- `train_asr.py` — the training loop with per-epoch checkpointing.
- `tune_lm.py` — the alpha/beta grid, fitted on dev-other.
- `eval_asr.py` — the two-column evaluation (dev-clean / L2-ARCTIC).

Standalone helpers that are **not** generated: `check_notebook.py` (marimo cell
validation), `CLOSEOUT_CELL.py` and `FINAL_BENCHMARK_CELL.py`.

## Architecture (unchanged from ablation_engine.py — this is a data/aug retrain,
not an architecture change)

- Frozen `utter-project/mHuBERT-147` backbone.
- LoRA adapters (`peft.LoraConfig`, r=16, alpha=32, `target_modules=["q_proj","v_proj"]`)
  injected on transformer layers 1–12.
- Weighted-sum head reads hidden_states layers `[9, 10, 11, 12]` (FINAL config),
  learnable softmax weights (`Head.layer_w`), + 2-layer MLP + CTC output.
- Character vocabulary: `A-Z, ', |, [UNK], [PAD]` (`[PAD]`=CTC blank) — identical
  to `ablation_engine.py`/`kenlm_grid.py`. This is REQUIRED for the existing
  KenLM 3-gram (LibriSpeech-normalised) to keep working; no new characters are
  ever introduced by the 300h data.

## Environment

Target: **molab** (marimo cloud, RTX PRO 6000), which ships Python 3.13 + uv.
The notebook creates a **Python 3.11 uv venv** instead of using the host
3.13, because:
  - 3.13 causes dependency conflicts with the pinned stack here, and
  - KenLM does not build on 3.13 at all. Building on 3.11 means KenLM builds
    on the SAME machine as training, removing the previous dependency on a
    separate Colab environment for decoding.

`torch`, `torchaudio` and `torchvision` are pinned to the **same version**
from the `cu128` index. A version mismatch between them previously produced
a `torchvision::nms` crash that took down `transformers` entirely — this is
a known trap, not a hypothetical one.

Per-epoch checkpointing (`last.pt`, same convention as `ablation_engine.py`) is
mandatory: this is an ~8h unattended run on a cloud notebook, and a
disconnect without checkpoints means starting over.

## Data composition (300h exactly)

| Source | Hours | Notes |
|---|---|---|
| LibriSpeech train-clean-100 | 100 | comparability anchor, unchanged |
| Common Voice 22 EN | 106 | accent-stratified (per-accent cap), fetched via `snapshot_download` (English shards only — `load_dataset` cannot be used, see below) |
| AMI | 50 | 25h `ihm` + 25h `sdm`, disjoint meetings, 4 filters |
| VCTK | 44 | accent, studio-clean |

### Why Common Voice can't use `load_dataset`
`fsicoli/common_voice_22_0` ships a `.py` loading script, and `datasets` v4.0
removed script support entirely (`trust_remote_code=True` no longer exists).
`prepare_data.py` uses `huggingface_hub.snapshot_download(allow_patterns=...)`
to pull only the English shards (the full repo is 578 GB across 100+
languages) and parses `validated.tsv` + the clip files directly.

### AMI filters (the delicate part)
1. **CTC feasibility** (mandatory — prevents inf/nan CTC loss): require
   `duration_s * 50 >= 2 * len(text)` (≈40 ms/char at the backbone's 50
   frames/sec). AMI genuinely contains 0.02s clips that would otherwise
   silently poison training.
2. **Pure filler stoplist**: drop rows whose entire text is one of
   `MM, MMM, HMM, HM, UH, UM, ERM, AH, OH, EH, MM-HMM, UH-HUH`.
3. **Keep short real words** (`YEAH`, `NO`, `OKAY`, ...) — no blanket
   "fewer than 3 words" cut. Subsampled to `AMI_SHORT_WORD_KEEP_RATE = 0.30`
   (named constant in `prepare_data.py`), never zeroed — the demo will
   contain them, and they teach the model to emit little when little was
   said (anti-hallucination signal).
4. **Character-rate sanity**: drop rows outside `2–25` chars/sec, which
   catches AMI's truncated automatic-alignment segments.

Disjoint meetings are enforced by partitioning `meeting_id` values into two
sets up front (`ihm` draws from one set, `sdm` from the other) and asserting
the two sets never intersect, both before and after row collection.

## Augmentation (GPU, batched, `augment.py`)

Reused from `aug_night_v2.py` / `aug_sweep_v1.py` / `ablation_engine.py`: the
config-of-probabilities shape (`p_clean` escape hatch + one probability per
effect), the physical chain ordering (source → room → noise → channel →
spec masking), and the tuned ranges (5–20 dB SNR, 0.15–0.50s T60, 300–3400 Hz
telephone band, moderate not always-on probabilities). NOT reused verbatim:
the numpy/FFT *implementations*, because those ran per-sample on CPU and the
task requires batched GPU ops (`torchaudio.functional.add_noise`, batched
FFT convolution, `torchaudio.transforms.Frequency/TimeMasking`).

Differences from the SER work this borrows conventions from:
- **Speed perturbation (0.9/1.0/1.1) is ENABLED.** SER avoided speed/pitch
  because it corrupts emotion labels; that concern is ASR-irrelevant, and
  speed perturbation is a standard, cheap WER win here.
- **Reverb is not down-weighted** — the demo is a laptop mic in a room, not
  a phone line, so room acoustics matter more than channel/codec artifacts.
- **No codec APIs** (`io.AudioEffector` / `apply_codec`): torchaudio has been
  in maintenance mode since 2.8 and encode/decode moved to TorchCodec, so
  those APIs may not exist on molab's stack. The telephone-band effect is
  obtained with `functional.resample` 16k→8k→16k instead.

Datasets to fetch for augmentation: MUSAN noise subset (~6h), DEMAND (~6h),
OpenSLR-28 RIRs — loaded into RAM once (~1.4 GB fp32) by `AudioBank` in
`augment.py`, then sampled per-batch on GPU.

## 50h probe (before the full 300h run)

A fast, high-LR probe to test whether more diverse data shifts the optimal
weighted-sum layer selection, run **with augmentation enabled** (tuning it
off would select a config for a training condition never actually used).

| arm | WS layers | LoRA layers |
|---|---|---|
| control (current FINAL) | [9,10,11,12] | 1–12 |
| lower-A | [5,6,7,8] | 1–12 |
| lower-B | [7,8,9,10] | 1–12 |

Hypothesis under test: lower layers are more phonetic, upper layers more
lexical; accent/noise robustness is fundamentally a phonetic problem, so a
lower-layer arm winning on the probe would be a real, actionable finding —
not just noise.

## Evaluation design (the headline table)

Two TEST COLUMNS, not two separate tables — the same set of systems is
scored on both:
- **dev-clean** — comparability anchor. Both the 100h and 300h models are
  measured here so the new number is directly comparable to the published
  10.1% / 5.1% WER.
- **L2-ARCTIC** — held-out OOD accent test. NEVER used in training
  (enforced by `assert_no_l2arctic` in `prepare_data.py`).

Each column reports **greedy** and **+KenLM** separately (KenLM gain is
expected to shrink out-of-domain, since the LM was built on LibriSpeech
text — that shrinkage is reported as a finding, not hidden). Whisper
(base/small/medium, same protocol as `whisper_bench.ipynb`) is
re-run on both columns as the external reference point.

Efficiency stats follow the existing project's convention
(`cpu_bench.ipynb`, `whisper_bench.ipynb`): CPU RTF (`psutil`
process, wall time / audio duration), GPU RTF (`torch.cuda` wall time /
audio duration), and peak RAM (`psutil` on CPU, `torch.cuda.max_memory_allocated`
on GPU).

## What was NOT verified in this environment

This was written without GPU/torch/dataset access (no deep-learning stack
installation and no multi-GB downloads are possible here). Verified: syntax
(`py_compile`) of both `.py` modules, and the JSON structure of the
notebook. NOT verified: actual dataset schemas at runtime (Common Voice 22
repo layout, VCTK's exact HF repo id — a short fallback list is tried in
`build_vctk`), the real accent-field distribution, torchaudio's exact
`fftconvolve` availability on the target torchaudio version (a manual
`torch.fft` fallback is included for that reason), and end-to-end training
convergence. These should be smoke-tested on molab with a tiny subset before
committing to the full ~8h run.
