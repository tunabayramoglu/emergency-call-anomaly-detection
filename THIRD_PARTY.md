# Third-party models, corpora and services

Everything this project depends on that someone else owns. Compiled by tracing the
model and dataset identifiers that actually appear in the tracked code, not from
the planning documents — `docs/Roadmap.md` lists candidates that were evaluated
and dropped, and should not be read as a record of what shipped.

**This file is a factual inventory, not legal advice, and it is not complete
enough to clear a commercial release on its own.** Several entries below are
recorded as "research use" or "non-commercial" and at least one of them —
the backbone — constrains the whole system. Anyone planning to ship this should
have Legal read the primary licence text for each row.

## The constraint that matters most

`utter-project/mHuBERT-147` is released under **CC-BY-NC 4.0**. It is the
backbone of the entire system: both LoRA adapters sit on it, `app/` downloads it
at setup, and every ASR and SER number in `RESULTS.md` was produced on top of it.
The NC clause rules out commercial use of those released weights. Replacing it
means retraining both adapters and re-measuring everything.

## Models

| model | used for | where | licence |
|---|---|---|---|
| `utter-project/mHuBERT-147` | the frozen shared backbone | everywhere | **CC-BY-NC 4.0** |
| `google-bert/bert-base-uncased` | frozen text encoder in the fusion head | `fusion/benchmark_modules/encoders.py`, `app/` | Apache-2.0 |
| `sentence-transformers/all-MiniLM-L6-v2` | second text encoder in the benchmark matrix | `fusion/benchmark_modules/` | Apache-2.0 |
| OpenAI Whisper (base / small / medium) | baseline comparison in `RESULTS.md` §1.4 | `bench/` | MIT |
| `facebook/bart-large-mnli` | zero-shot text risk scoring in the SER demo cells | `ser/train_ser.py` | MIT |
| OpenSLR-11 `3-gram.pruned.1e-7.arpa` | KenLM language model for beam decoding | `asr/phase2_300h/fetch_kenlm.py` | derived from LibriSpeech texts, see below |

## Hosted LLM APIs

The fusion dataset was generated and labelled through a DashScope-compatible
endpoint. Ten generator models and 23 judge models were used; the full list of
identifiers is in `fusion/run_multi_model.py` and `fusion/run_multi_judge.py`,
and every row in `fusion/dataset_final.jsonl` records which model produced it in
`source_model` and `judge_model`.

The families involved are Qwen (`qwen3.5-*`, `qwen3-max`, `qwen-plus`), GLM
(`glm-5.1`, `glm-5.2`) and DeepSeek (`deepseek-v4-pro`, `deepseek-v4-flash`,
`deepseek-v3.2`). **Provider terms of service govern whether the generated
dataset can be redistributed or used commercially, and those terms were not
reviewed during the internship.** This needs checking before the dataset leaves
the company.

## Speech corpora — ASR (300 hours, `RESULTS.md` §1.6)

| corpus | hours | identifier | licence |
|---|---|---|---|
| LibriSpeech train-clean-100 | 100 | `openslr/librispeech_asr` | CC-BY 4.0 |
| Common Voice 22 | 106 | `fsicoli/common_voice_22_0` | CC0 1.0 |
| AMI (ihm + sdm) | 50 | `edinburghcstr/ami` | CC-BY 4.0 |
| VCTK | 44 | `CSTR-Edinburgh/vctk` | CC-BY 4.0 |
| L2-ARCTIC | held out for evaluation only | `KoelLabs/L2Arctic` | CC-BY 4.0, research use |

Augmentation during the 300-hour run also drew on two noise corpora, downloaded
by `asr/phase2_300h/fetch_noise_banks.py:69-77`. They are training inputs of the
shipped ASR checkpoint and belong in any licence review.

| corpus | used for | identifier | licence |
|---|---|---|---|
| MUSAN | additive noise | OpenSLR-17 | CC-BY 4.0 |
| RIRS_NOISES | room impulse responses | OpenSLR-28 | Apache-2.0 |
| ESC-50, UrbanSound8K | phase-1 noise bank only | `ashraq/esc50`, `danavery/urbansound8K` | CC-BY-NC 3.0 / CC-BY-NC 3.0, verify upstream |

L2-ARCTIC is never trained on. Two asserts in `asr/phase2_300h/prepare_data.py`
enforce that.

## Speech corpora — SER (`RESULTS.md` §2.1)

| corpus | rows | licence |
|---|---|---|
| CREMA-D | 7,442 | ODbL |
| RAVDESS | 1,440 | **CC-BY-NC-SA 4.0** |
| ASVP-ESD | 13,802 | non-commercial research, verify upstream |
| TESS | 2,800 | CC-BY-NC 4.0 |
| JL-Corpus | 2,396 | CC-BY 4.0 |
| SAVEE | 480 | research use, registration required |
| Kaggle "Speech emotion recognition for emergency calls" | 338 | see the Kaggle dataset page |

Three of these are explicitly non-commercial. The shipped SER checkpoint was
trained on all of them together.

## Tooling

`torch`, `transformers`, `peft`, `datasets`, `gradio`, `soundfile`, `jiwer`,
`pyctcdecode`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `marimo`, `uv`,
`kenlm`, `av`. Most are BSD / MIT / Apache-2.0, but **two are not**: `kenlm` is
LGPL-2.1 and `av` (PyAV) is LGPL-2.1+ and links FFmpeg. LGPL carries
redistribution conditions that "permissively licensed" does not cover, and both
ship in `setup.py`'s dependency list. `setup.py` pins no versions — see
`HANDOFF.md` for the version mismatches that have already bitten.

## This repository's own code

Orion Innovation has cleared the author to share this project independently, so
the company side of the question is settled. What is left is the author's own
choice of licence, and it is still bounded by the third-party terms above rather
than by that clearance. The released mHuBERT-147 weights are CC-BY-NC 4.0 and
three of the SER training corpora are non-commercial, so no licence chosen here
can grant commercial reuse of the shipped checkpoints. Until a licence is added,
default copyright applies to this repository's own code.
