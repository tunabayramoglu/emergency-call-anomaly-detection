# Emergency-call anomaly detection

Anomaly detection for emergency calls. The anomaly is a mismatch between **what
is said** and **how it is said**: reporting a fire in a calm voice is an anomaly,
and so is panicking over misplaced keys.

One frozen mHuBERT-147 backbone is shared by two LoRA adapters — one for ASR, one
for SER — and a fusion head combines the two channels into a
`normal` / `borderline` / `anomaly` verdict.

```
                 ┌── LoRA(ASR) + WS head + CTC ──► transcript ──► BERT tokens ──┐
mic audio ──► mHuBERT-147 (FROZEN, shared)                                      ├──► fusion head ──► verdict
                 └── LoRA(SER) + WS head ────────► emotion (6 classes) ─────────┘
```

The backbone is loaded **once** and shared by both adapters. That sharing is the
architectural claim of the project; do not load two copies.

## Results

| what | result |
|---|---|
| ASR, dev-clean, +KenLM | **5.13% WER** / 1.82% CER |
| ASR, L2-ARCTIC (accented), +KenLM | **15.37% WER** — was 20.26% with the 100h model |
| ASR, CPU | RTF 0.055 · ~505 MB RAM (whisper-medium: RTF 1.33 · 3 GB) |
| SER, 6 classes, held-out | UA 59.2% / WA 61.3% (chance 16.7%) |
| SER, binary risk detector | 86.0% acc · 95.9% precision · 83.1% recall |
| Fusion, selected method | intermediate + attention, macro-F1 0.5684 (oracle) / 0.5259 (real SER) |
| Trainable parameters | 4.81M of 208.7M — 2.3% of the system |

Every number and where it came from: [RESULTS.md](RESULTS.md).

## Repository map

| directory | contents |
|---|---|
| `asr/phase1/` | 100-hour ASR: ablation engine (`ablation_engine.py`, marimo), KenLM decoding, phase-1 notebooks |
| `asr/phase2_300h/` | 300-hour retrain: `asr_300h_marimo.py` (46 cells) + the 10 modules it generates. Details in `asr/phase2_300h/README.md` |
| `ser/` | SER training (`train_ser.py`, marimo), dataset download, the 6-class taxonomy |
| `fusion/` | Seed generator, multi-model generation, multi-judge labelling, benchmark notebook and `dataset_final.jsonl` (9,740 rows). Layout in `fusion/README.md` |
| `app/` | CPU-only demo. Audio in, one verdict out |
| `bench/` | CPU and Whisper comparison notebooks |
| `weights/` | ASR-300, SER and the winning fusion checkpoint (~18 MB) |
| `figures/` | Architecture diagram and all result charts |
| `docs/` | Roadmap, augmentation literature survey, presentation, handoff chain |
| `archive/` | Superseded versions. Kept for reference — **do not use them** |

Everything is in English, documentation and code comments alike.

One name survives from the internship. Several notebooks read and write
`MyDrive/CLEAR/...` on Google Drive — `CLEAR` was the project's working name and
is the actual folder name in the author's Drive account. Those strings are left
as they are because changing them here would break the code against a folder that
still exists under that name. They are external references, not part of this
repository's own naming.

Third-party models and corpora, and their licence terms, are listed in
[THIRD_PARTY.md](THIRD_PARTY.md). Read it before reusing any of this: the
mHuBERT-147 backbone the whole system is built on is **CC-BY-NC 4.0**, which
rules out commercial use of the released weights. This repository carries no
licence of its own for the code yet.

## Installation

### Prerequisites

| what | required | why |
|---|---|---|
| Python 3.9+ | yes | Only to run `setup.py`. uv downloads Python 3.11 for the virtualenv itself |
| Internet | yes | uv, pip packages, HuggingFace models, language model |
| ~3 GB disk | yes | torch and dependencies ~2 GB, models ~820 MB, language model 94 MB |
| **cmake + a C++ compiler** | **no** | Only for KenLM beam search. Without it the demo decodes greedily, see below |

**You need both.** kenlm ships no wheels on PyPI; the sdist builds through cmake,
and even with cmake installed the configure step fails when there is no compiler.
This repo was tested on Windows 11 with Python 3.11.14: cmake 4.2.1 was present,
Visual Studio was not installed at all, and the build failed at
`cmake ... -A x64` with exit 1. `setup.py` checks the two separately and tells you
which one is missing.

| platform | what to install |
|---|---|
| Windows | [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) → "Desktop development with C++" workload **and** cmake (`winget install Kitware.CMake`) |
| Debian / Ubuntu | `sudo apt install build-essential cmake` |
| macOS | `xcode-select --install && brew install cmake` |

### Running it

One command. `setup.py` needs nothing beyond the standard library.

```
python setup.py
```

In order: installs `uv` if missing (asks first), creates a Python 3.11
environment under `app/.venv`, installs dependencies, unpacks the archives in
`weights/` into the `app/models/` layout, downloads the KenLM language model,
pre-fetches mHuBERT-147 (~380 MB) and bert-base-uncased (~440 MB) from
HuggingFace, and verifies everything. Every step is idempotent; running it twice
skips what is already done.

| flag | effect |
|---|---|
| `--check` | Changes nothing, only reports what is missing |
| `--yes` | Never prompts, including for the uv install |
| `--skip-kenlm` | Does not download the 94 MB language model |
| `--skip-prefetch` | Leaves the ~820 MB HuggingFace download to first run |
| `--skip-deps` | Skips the pip step |

When it finishes:

```powershell
app\.venv\Scripts\activate
python app\app.py
```

### A note on KenLM

The environment is created on Python 3.11 because the kenlm bindings do not build
on 3.12+. But 3.11 alone is no guarantee: if the cmake and compiler requirement
above is not met, the source build fails. Linux and macOS usually have a system
compiler already; Windows usually does not.

If it cannot be installed, `setup.py` says so and carries on, and it does not
download the language model — the `.arpa` file is useless without the package.
The demo then decodes greedily: the transcript degrades from 5.13% to 10.65% WER,
which does not change the verdict for demo-length utterances
(`docs/DEMO_BRIEF.md` §4). Details in `app/README.md`.

The KenLM file is OpenSLR-11 `3-gram.pruned.1e-7`. **Do not swap it**: both rows
of the results table were measured with this model, and a different LM moves both
at once. `asr/phase2_300h/fetch_kenlm.py` already refuses to "upgrade" it and
explains why.

## What is not in the repository

| what | why | how to get it |
|---|---|---|
| Training audio (~29 hours raw) | Size and licensing | `ser/download_datasets.py` |
| KenLM ARPA file (94 MB) | Downloadable | `asr/phase2_300h/fetch_kenlm.py` |
| mHuBERT-147 and bert-base-uncased | On HuggingFace | `app/setup_weights.py` |
| `fusion/results_table.csv` | Written by the benchmark notebook to its Colab output directory, but not committed; the numbers are recorded in `RESULTS.md` §3 | Re-run `fusion/fusion_benchmark.ipynb` |

## If you are taking this over

Start with [HANDOFF.md](HANDOFF.md). It covers where the work stopped, which
decisions are closed, and which traps have already been found and fixed. The
chronological handover documents are under `docs/handoff/`.

Questions about the project, or about the data that is not in the repository,
go to the author: **tunabayram35@gmail.com**.
