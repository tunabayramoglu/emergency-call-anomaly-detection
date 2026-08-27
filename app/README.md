# interactive demo

Local, CPU-only. One frozen mHuBERT-147 carrying two LoRA adapters (ASR + SER)
feeding an `intermediate_attn` fusion head over per-token BERT states.

Built against `DEMO_BRIEF.md`; §5 below lists where the brief and the artefacts
on disk disagree.

## 1. Environment

The supported path is the repository's own installer, which does all of this
and verifies the result:

```powershell
python setup.py
```

By hand, from the repository root:

```powershell
uv venv --python 3.11 app\.venv
app\.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers "peft>=0.11" gradio soundfile numpy av pytest
```

`av` is not optional: all three shipped clips are `.m4a` and libsndfile cannot
read AAC. `pytest` is needed for the test suite in section 4. Leaving either
out is how the delivered `app/.venv` ended up unable to open a single clip.

Python 3.11 is not optional if you want KenLM — the bindings ship a
pre-generated Cython file that does not compile on ≥3.12.

## 2. Weights

Everything is already local. This unpacks it into the layout the pipeline expects:

```powershell
python setup_weights.py          # builds app/models/
python setup_weights.py --check  # verify only
```

| built from | into |
|---|---|
| `_Staj/ASR-300.zip` | `models/asr/` — config, adapter, head, `lm_params_clean.json` |
| `_Staj/SER.zip` | `models/ser/` |
| `_Staj/WINNER_intermediate_attn_bert_full_p2_seed1.pt` | `models/fusion/` |
| `_Staj/mHuBERT-147.zip` | `models/backbone/` — **no HF download** |
| any `*.arpa` found | `models/lm/` (optional) |

The backbone comes out of the zip, not off the hub. Two of the zip's five
members are skipped: `checkpoint_best.pt` (1.14 GB, fairseq-era, never read by
transformers) and `pytorch_model.bin` (redundant once `model.safetensors` is
present). 360 MB lands on disk instead of 1.75 GB, and the 380 MB download does
not happen — which also sidesteps the Windows symlink warning from the HF cache.

`bert-base-uncased` is not in any zip, so it is still fetched:

```powershell
python ..\demo_prefetch.py --weights models --encoder bert
```

**The backbone directory does not have to be exactly `models/backbone/`.** The
zip wraps its contents in a `mHuBERT-147/` folder, so unzipping it in place is
fine — each candidate is probed one level down too. Searched, in order:

```
models/backbone/            models/backbone/mHuBERT-147/
app/mHuBERT-147/            app/mHuBERT-147/mHuBERT-147/
_Staj/mHuBERT-147/          _Staj/mHuBERT-147/mHuBERT-147/
```

`setup_weights.py --check` prints which one was found, and asks the pipeline
rather than deciding for itself — otherwise `--check` could pass while the demo
still downloads.

Once `--check` reports `ok backbone at …`, the zip is redundant and can be
deleted. If you extracted the whole archive, `checkpoint_best.pt` (1.14 GB) and
`pytorch_model.bin` (378 MB) can go too; transformers reads `model.safetensors`,
`config.json` and `preprocessor_config.json` and nothing else.

Extraction is the slowest step and can be interrupted. A truncated
`model.safetensors` is detected by size and skipped — by `--check`, and by the
pipeline, which treats it as absent rather than loading it and failing
cryptically. A half-finished copy in the first location does not shadow a
complete one further down the list.

## 3. Demo clips

Drop audio into `app/clips/`. `.wav`, `.mp3`, `.flac`, `.ogg` and `.m4a` are
accepted, filenames carry no meaning, and non-audio files are ignored — the
dropdown lists whatever is there.

To show the mismatch rather than just the pipeline, vary content and tone
independently: the same sentence said in panic and said flatly, and a harmless
sentence said both ways. The academic emotion corpora cannot substitute here —
their sentences are scripted and content-neutral, so the text channel is flat
across every clip and everything reads as congruent.

### Formats

`soundfile` (libsndfile) reads `.wav`, `.flac`, `.ogg`, `.aiff`, `.au`, and
`.mp3` since libsndfile 1.1 which recent wheels bundle.

It does **not** read AAC in an MP4 container — `.m4a`, which is what phones and
Windows Voice Recorder produce by default. Those files are still listed in the
dropdown (a file silently missing is worse than one that explains itself) and
`load_wav` falls back to PyAV:

```powershell
pip install av
```

Without PyAV, clicking an `.m4a` shows a readable message naming the file and
both ways out, rather than a console traceback. Converting to `.wav` is the
other way out and needs nothing installed.

## 4. Run

```powershell
python pipeline.py --self-test          # offline invariants, no weights needed
python pipeline.py clips\high_flat.wav  # CLI, one clip
python app.py                           # Gradio at http://127.0.0.1:7860
pytest tests\ -v                              # functional suite
```

## 5. Where this differs from `DEMO_BRIEF.md`

**The brief's SER emotion order is wrong, and it is overridden here.** Brief §3
gives the SER classes as `[neutral, confusion, fear, panic, urgency, distress]`.
That is `common.EMOTIONS`, the *fusion* one-hot order. The SER head
emits `train_ser.CLASSES` = `[neutral, distress, fear, urgency, panic, confusion]`.
`ser/config.json` records only `n_cls: 6`, so nothing on disk contradicts this.
Using the brief's list as SER's output order mislabels **4 of 6 emotions, three
of them high-risk**, and raises no error. The pipeline maps by name; TC-05
asserts it.

**`asr/eval_asr.py::load_our_model` is deliberately not reused.** Brief §3 says
to import it, but it builds its own backbone — importing it for both channels
gives two copies and falsifies the shared-backbone claim of brief §1. The LoRA
config, head shape, vocabulary and decode semantics here are equivalent to it;
only the backbone ownership differs. TC-23 verifies both adapters live on one model.

**Two deployment-gap numbers are in circulation, both correct.** The
checkpoint's `meta["deployment_note"]` quotes **−0.052** (0.5822 → 0.5259,
seed-averaged benchmark reference). The brief §6 quotes **−0.057**, which is this
specific checkpoint's own drop (0.5799 → 0.5224, seed 1). Say which one you mean.
The "2 of 12 configurations" figure matches the checkpoint; the older handoff's
"2 of 18" was wrong and has since been corrected there.

**The fusion checkpoint is `regime: oracle` / `p2_oracle_weighted`.** An earlier
concern that phase tags overwrote each other does not apply to this file — it was
exported with an explicit name and full provenance in `meta`.

**`num_heads` is not recoverable from the checkpoint.** `MultiheadAttention`'s
parameter shapes are identical for any head count dividing `embed_dim`, so 4 is
taken from the brief and the module default and is exposed as `--num-heads`. A
wrong value changes the attention pattern silently.

**`demo_prefetch.py` globs `fusion/BEST_*.pt`; the file is `WINNER_*.pt`.** The
pipeline accepts either, so this cannot break demo day. Fixing `demo_prefetch.py`
itself is still worth doing.

## 6. Other things that are load-bearing

**Downloading the `.arpa` is not enough to get KenLM decoding.** The file is
data; the decoder is code. Two packages are also required:

```powershell
pip install pyctcdecode pypi-kenlm
```

`python setup_weights.py --check` ends with which decoder will actually run and
why. If it says `-> decoder: GREEDY` while `lm/` holds an `.arpa`, one of those
two packages is missing — `kenlm` is the usual culprit on Windows.

**Decoder parameters come from `lm_params_clean.json`** (alpha 0.6, beta 0.0,
beam 50). pyctcdecode's defaults are 0.5 / 1.0 / 100 and the reported WER was not
measured with those. If an `.arpa` is present but the params file is not, the
pipeline refuses to guess and decodes greedily.

**Greedy is an acceptable demo decoder.** 10.65% vs 5.13% WER on clean read
speech; it does not change the anomaly verdict at demo utterance lengths, and it
removes a dependency that can fail on stage.

**`.float()` before numpy.** Under `autocast(bfloat16)` a bf16 tensor raises
`TypeError: Got unsupported ScalarType BFloat16`. Applied at every boundary.

**Everything is in `eval()`.** SpecAugment and dropout are training-time
regularisers; left on, the same clip would give different answers on each click.
TC-23c asserts it.

**Adapter loading rejects partial matches.** 40 of 48 tensors landing is an
error, not a warning — a half-loaded adapter produces fluent garbage.

**Do not extend the CTC vocabulary** past `A–Z` + apostrophe + `|` + `[UNK]` +
`[PAD]`. KenLM was built on LibriSpeech-normalised text.

**bert-base-uncased lowercases its input**, so the ASR head's uppercase output
costs nothing. Punctuation still differs from the fusion head's training text and
is not corrected.
