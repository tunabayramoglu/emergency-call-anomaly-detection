# Fusion

Generates the labelled mismatch dataset and benchmarks fusion methods on it.
Numbers are reported in `../RESULTS.md` §3.

## Layout

```
seeds.jsonl              1,499 scenario seeds — 20 caller profiles x 47 events
dataset_final.jsonl      9,740 labelled utterances, the deliverable
seeds.py                 builds seeds.jsonl
prompts.py               generation and judge prompt templates
generate.py              one model, one pass: seeds -> utterances -> judged rows
run_multi_model.py       fans generation across 10 models, merges and re-uids
run_multi_judge.py       fans judging across 23 models, merges, dedupes -> dataset_final.jsonl
check_quality.py         distribution and red-flag summary over dataset_final.jsonl
fusion_benchmark.ipynb   the driver that runs the benchmark matrix
WINNER_RETRAIN_CELL.py   retrains the winning configuration and writes the checkpoint
inspect_checkpoints.py   prints the metadata stored inside a checkpoint
benchmark_modules/       one module per fusion method, see its CONTRACT.md
generation/              intermediate per-model outputs, see below
dataset_audit/           two LLM spot-check audits of the final dataset
```

## generation/

Everything the two fan-out runners write while they work: per-model utterance
files, the merged utterance file, the seed and utterance chunk files, and one
`dataset_judge_*.jsonl` per judge model. `run_multi_model.py` and
`run_multi_judge.py` write there by construction (`RUNS` at the top of each), so
re-running the pipeline does not scatter files back into this directory.

Only `seeds.jsonl` at the start and `dataset_final.jsonl` at the end live here.
Everything in `generation/` is reproducible from those two plus the scripts, and
is kept only so the run can be inspected without re-spending the API budget.

## Running it

Run from this directory. Both runners need an API key, either `--api-key` or
`OPENAI_API_KEY`, and default to the DashScope-compatible endpoint.

```
python seeds.py --n 1500 --seed 42
python run_multi_model.py --workers-per-model 2
python run_multi_judge.py --workers-per-model 2 --judge-batch 5
python check_quality.py
```

`dataset_final.jsonl` is committed, so the benchmark can be reproduced without
re-running any of the above.

## Before using the dataset

Two constraints are not optional and are explained in
`benchmark_modules/CONTRACT.md`.

Splitting must be done on `seed_id` groups, never on rows. Utterances sharing a
seed are near-variants of each other, so a row-level split puts almost the same
sentence on both sides.

The judge risk columns must never be used as features. They are outputs of the
same LLM call that produced the label, and a six-cell lookup over them scores
91.5%. `CONTRACT.md` lists the full set of banned fields.

`dataset_audit/README.md` records what the quality spot-checks found and what
the dataset should and should not be reused for.
