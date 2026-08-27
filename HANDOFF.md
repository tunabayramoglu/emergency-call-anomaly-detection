# Handover

Last updated 31 July 2026. The internship ended on that date; what follows is the
state of the project as of then.

## Status in one line

ASR-100h ✔ · ASR-300h ✔ (trained and benchmarked) · SER ✔ · Fusion dataset ✔ ·
Fusion benchmark ✔ · Demo app ✔ · real emergency-call audio explored but **not used in
training**.

## First half hour

Set the environment up first — you can start reading while it downloads:

```
python setup.py
```

Then read, in order:

1. `README.md` — what the system does and how to run it.
2. `RESULTS.md` — every measured number and which file it came from.
3. `docs/handoff/2026-07-30-v2.md` — the most detailed technical handover. §3
   fusion, §4 ASR-300h, §7 the classes of bug that were closed.
4. `docs/handoff/2026-07-29-v1.md` — §2b on SER and §3 on how the fusion dataset
   was built are still accurate. The parts calling the fusion benchmark "pending"
   are out of date.
5. `docs/Roadmap.md` — the reasoning behind the architectural decisions.

## Do not re-litigate these

All of them were measured and settled. Reopening them is wasted time.

- **Label leakage.** `judge_voice_risk` and `judge_content_risk` are outputs of
  the same LLM call that produced the `anomaly` label. A six-cell lookup table on
  those two columns scores 91.5%. They cannot be used as input features; use
  `gen_emotion` and `text`.
- **The split must be by seed.** Utterances from the same seed are near
  paraphrases of each other, so a row-level split leaks. `seed_universe` has to be
  pinned as well, otherwise the "filtered" variant builds its own test set.
- **0.4141 is a fingerprint, not a score.** It is exactly the emotion-only
  validation macro-F1. A config reporting it has collapsed to ignoring the text.
- **The `ws=(5,6,7,8)` ablation arm is confounded.** If the head only consumes
  layers 5–8, layers 9–12 receive no gradient at all; that arm answers "8-layer
  model vs 12-layer model", not "which layers are best".
- **The KenLM file does not change.** OpenSLR-11 `3-gram.pruned.1e-7`. Swapping it
  shifts both rows of the results table at once. `fetch_kenlm.py` refuses to
  "upgrade" and says why.
- **The CTC vocabulary does not get extended.** `A-Z` plus apostrophe, `|`,
  `[UNK]` and `[PAD]`. KenLM was built on LibriSpeech normalisation; emitting a
  character the LM has never seen collapses the beam search.
- **L2-ARCTIC never enters training.** `prepare_data.assert_no_l2arctic` and
  `verify_data.py` check this from two separate places.

## Open work

In order of how much it matters:

1. **`fusion/results_table.csv` is not on disk.** The raw 96-cell output of the
   benchmark was never written out; the numbers in `RESULTS.md` were recorded by
   hand from the run output. If you need the raw table, re-run
   `fusion/fusion_benchmark.ipynb`.
2. **The noise/RIR banks report 0 clips.** The files were downloaded and verified
   and the trainer still saw nothing. The *diagnosis* is fixed (`AudioBank` now
   tries soundfile first and prints the real error, and §7 has a "Check
   decodability" button), the *cause* is not known. Prime suspect: torchaudio 2.8+
   moving decoding to TorchCodec. **Run that check before retraining on 300h** —
   reverb is the demo-critical augmentation, and a run with an empty bank looks
   perfectly healthy in the log.
3. **SER has no separate test set.** Validation was used for both model selection
   and reporting, so the numbers carry some selection optimism. Carving out a
   fresh held-out set is the first thing to do.
4. **SER never saw real emergency-call audio.** All seven training corpora are
   acted studio recordings, so the model may have learned performed distress
   rather than the real thing. Nothing in this repository measures that gap, and
   closing it needs real call audio with speaker diarization and an audited
   labelling pass.
5. **The fusion winner flips with the encoder.** `intermediate_attn` swaps places
   between bert and minilm; report it as "no significant difference", not as a
   win. `early/bert` collapses from 0.5917 to 0.4810 once class weighting is on —
   state that rather than hiding the config.
6. **Unmeasured opportunity:** restricting LoRA to the upper layers could save
   roughly 32% of the forward cost. Calculated, never tried.
7. **Dev WER is a single mixed number.** It does not show which corpus carries the
   error. A per-corpus breakdown was proposed and never done.

## Environment

- **molab** (marimo cloud), Python 3.13, RTX PRO 6000. No CLI — you upload a `.py`
  and it runs as a notebook. There is no `/content/drive`, so the OAuth path is
  the one that executes.
- **marimo semantics:** a cell renders its last expression; `return` is an export
  declaration and displays nothing. Top-level names that do not start with an
  underscore must be globally unique across cells.
- **KenLM does not build on molab.** If you need to *build* a new LM, use Colab
  (Python ≤3.12). Decoding with `pyctcdecode` and a prebuilt ARPA is fine there.
- `datasets` v4/v5 removed loading scripts and routes audio decoding through
  torchcodec; `Audio(decode=False)` sidesteps it.
- `torchvision` must match torch's CUDA build or transformers crashes with
  `torchvision::nms`.
- Notebooks are self-contained and do not import from each other.

## Known traps

These happened once each. They are written down so they do not happen again.

- **A module written to disk and never invoked.** Hit three times.
  `asr/phase2_300h/check_notebook.py` walks the marimo cells with AST and catches
  the class of mistake underneath it — a name defined in two cells, a cell
  argument nothing produces, sections numbered out of order. It does **not**
  check that each of the ten modules is actually invoked; nothing does. If you
  add a module, confirm by hand that something calls it.
- **Silent swallowing.** `except Exception: pass` plus reporting only the outcome.
  It killed the noise banks and `sync_checkpoint`. Every `gdrive_sync` function
  now returns `(ok, reason)`.
- **One knob doing two jobs.** `batch` scaled both the memory footprint and the
  optimisation batch. Now `micro_secs` is memory only and `effective_secs` is
  optimisation only.
- **A grid search without error bars picks noise.** Every tuned number now comes
  with a standard error and a significance verdict.
- **Stale GPU processes.** Training starts through `subprocess.Popen`;
  interrupting the marimo cell stops the stream, not the child. If you see an OOM:
  `pkill -f train_asr.py`.

## Contact

For questions about taking this over: **tunabayram35@gmail.com**
