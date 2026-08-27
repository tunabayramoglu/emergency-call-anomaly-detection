# reference for the phase-1 marimo engines

This is the "which command does what" reference for three marimo notebooks:
`asr/phase1/ablation_engine.py`, `asr/phase1/kenlm_grid.py` and `ser/train_ser.py`.
They are the engines behind the phase-1 ASR ablations, the KenLM grid and the
SER model, and all three are still current for what they cover.

**It does not cover the rest of the project.** The 300-hour ASR retrain lives in
`asr/phase2_300h/` and is documented in that directory's own README; the fusion
benchmark is in `fusion/` with `fusion/README.md`; the demo application is in
`app/`. Start from the root `README.md` for the whole picture, and from
`RESULTS.md` for the numbers.

**There are no imports between these three scripts**, because every notebook is
uploaded to Colab/molab on its own — shared logic such as the Drive push is
embedded separately in each one.

Shared path convention: if `/marimo` exists it is the root, otherwise the working
directory. Outputs go under `runs/` (ASR) and `runs_ser/` (SER), one subdirectory
per run.

---

## ablation_engine.py — the ASR ablation engine (marimo)

A single-file marimo notebook. Frozen mHuBERT-147 + LoRA (589,824 parameters) +
weighted-sum head, CTC. LibriSpeech train-clean-100. It sweeps augmentation and
regularization axes through controlled ablation, merges the winners into a COMBO,
and trains the FINAL model by hand.

Flow: **BASE** (from scratch) → **X_\<axis\>** (warm start from BASE, each axis
independent) → **COMBO** (the winners) → **FINAL** (the winning recipe, from
scratch, long).

To run: open the cells in order in marimo, set the HF token and the budget, then
"start the night run". For FINAL:
`replace(Cfg(run="FINAL"), epochs=50, bb_dropout=0.05, aug=Aug(...))`.

Output, per run, under `runs/<run>/`: `config.json`, `head.pt`, `adapter.pt`,
`summary.json`, `history.jsonl`, and `last.pt` (for resuming; not pushed to
Drive).

**Automatic Drive push:** after each run writes `summary.json`, it calls
`push_run(cfg.dir, phase=1)` (see below). No manual push is needed.

---

## train_ser.py — the SER baseline (phase 2)

Frozen **base** mHuBERT-147 (*not* the ASR adapter — combo3 suppresses emotional
cues) → mean+std pooling over 12 layers written to disk **once** → a small head
trained on top of that cache (seconds per epoch). Eight academic sources mapped
onto six emotion classes — see `RESULTS.md` §2.1 for the full list and the row
counts. Model selection is on UA (macro recall).

```bash
python train_ser.py --selftest                 # logic test, no GPU needed
python train_ser.py --precompute               # feature cache (once)
python train_ser.py --train --feature-src base --split speaker
```

**Split (§7.4):** `--split speaker` (the default, and the **honest** one) keeps
all clips of one speaker on the same side, so a voice heard in training cannot
leak into test. `--split stratified` reproduces the old, inflated baseline.
Speaker identity is parsed from the filename (CREMA-D field 0, RAVDESS field 6)
and stored in the audio cache; if `speaker` is requested and the identity is
missing, it falls back to stratified with a loud warning (refresh with
`--precompute`). `summary.json` logs which split was used and whether there was
any leakage.

Output under `runs_ser/<run>/`: `config.json`, `head.pt`, `summary.json` (test
WA/UA plus the learned `layer_weights`, i.e. which layer carries the emotion).
Pushed to Drive Phase 2 when it finishes.

---

## kenlm_grid.py — KenLM beam decoding and the α/β grid

Two stages. **Stage 1 (marimo, with ablation_engine loaded):**
`dump_dev_logits("FINAL")` → `dev_logits.npz` (log-softmax logits per utterance
plus the reference). **Stage 2 (standalone, no torch needed):** read the npz,
install pyctcdecode and kenlm, sweep α and β.

```bash
python kenlm_grid.py --selftest
python kenlm_grid.py --grid --npz runs/FINAL/dev_logits.npz \
    --lm lm_work/3-gram.pruned.1e-7.arpa \
    --alphas 0.3,0.5,0.7,0.9 --betas 0.5,1.0,1.5,2.0
```

If the LM is absent it is downloaded from OpenSLR automatically. Vocabulary
mapping into pyctcdecode: `|` → space, `[PAD]`/`[UNK]` → blank.

**Normaliser (§7.8):** the reference *and* every hypothesis go through the
**same** `normalize()` (uppercase → drop punctuation and digits → collapse
whitespace). That keeps WER from being inflated artificially; the old
`.upper().strip()` inconsistency is closed. The grid is written to
`kenlm_grid.json` and the run directory is pushed to Drive Phase 1.

Note: the FINAL backbone differs from D, so sweep α and β **from scratch** — the
old 0.7/1.5 may no longer be optimal.

---

## anomaly_flag.py — anomaly-flag test (option 2b, a **marimo notebook**)

Like `ablation_engine`, this is a **marimo notebook**: upload it to molab and the cells
run on their own. It tests the **anomaly flag** at the very end of the
architecture end to end, *without* training a SER head. Voice emotion comes from
the ground-truth label ("how it is said"); text emotion comes from the FINAL ASR
transcript fed to a HuggingFace text-emotion classifier ("what is said"); the
anomaly fires when the two disagree. The ASR inference code (vocabulary,
backbone, head, decoding) is identical to `ablation_engine` and copied in here, since
the notebook has to stand alone.

Cells: (1) selftest — runs automatically without GPU or network and prints
green/red; (2) settings — edit `final_dir` and `limit`, then press **▶ run the
anomaly test** (the heavy work runs only on the button press, not on upload);
(3) manual Drive push. Output goes to `runs_anomaly/<run>/` (`records.jsonl` and
`summary.json`) and is pushed to Drive Phase 2 when the run finishes.

**Honesty warning:** CREMA-D and RAVDESS use fixed, neutral sentences, so the
text emotion is almost always neutral and the anomaly flag degenerates into a
"the voice is not neutral" detector. That validates the plumbing but does not
measure how discriminative the mismatch signal really is. The report prints the
number of unique transcripts and the text-emotion distribution so this effect is
visible.

This script now lives in `archive/`; the fusion benchmark supersedes it.

---

## Automatic Google Drive push — embedded in every notebook

This removes the manual push. There is **no shared module**, because notebooks
are uploaded independently; each script carries its own copy of `push_gdrive(...)`
and calls it when a run finishes:

```python
push_gdrive(cfg.dir, phase=1)   # ASR → CLEAR/Phase 1/runs/<run>/...
push_gdrive(cfg.dir, phase=2)   # SER → CLEAR/Phase 2/runs/<run>/...
```

Rules (handoff §5): fsspec headless OAuth (authenticate once, then the token is
cached); `put` overwrites, so copies do not accumulate; one subdirectory per run;
only a **finished** run — one that has a `summary.json` — is uploaded (the KenLM
grid output is the exception, via `require_summary=False`); `last.pt` is skipped.

**Safety:** with no credentials or no connection, or with `ECAD_GDRIVE=0`, it
skips the upload and never brings training down. It says so when it skips — the
phase-2 version in `asr/phase2_300h/gdrive_sync.py` returns `(ok, reason)` and
the phase-1 notebooks print the reason once, because a silent skip is exactly
the state in which someone leaves an eight-hour run believing the checkpoints
are safe. In a subprocess without stdin
the OAuth prompt hits EOFError and is skipped, which is why you should
authenticate interactively in a marimo cell **the first time** — after that the
token is cached and the overnight flow uploads without asking. Dependency:
`gdrivefs` (or `gdrive_fsspec`) must be installed in the environment.

---

## Selftests

`kenlm_grid.py` and `train_ser.py` each carry a `--selftest` that runs without
network or GPU (label mapping, vocabulary, split leakage, the normaliser, and so
on). After changing anything:

The two files are in different directories, so run them from the repo root:

```bash
python asr/phase1/kenlm_grid.py --selftest
python ser/train_ser.py --selftest
```
