# Dataset audit

Two independent spot-check audits of `fusion/dataset_final.jsonl` (9,740 rows).
Both were run by prompting an LLM to review sampled rows; neither is a
statistical evaluation, and neither was used to filter the dataset. They are
kept because they are the only qualitative evidence about the generated data
that exists.

Nothing in the pipeline reads these files. They are evidence, not inputs.

## What the two audits measured

They answer different questions, and this distinction matters — read together
they look contradictory, and they are not.

| | `utterance_quality/` | `judge_accuracy/` |
|---|---|---|
| Question | Does this utterance read like something a real caller would say? | Given the utterance, did the judge assign the right labels? |
| Input fields | seed, profile, event, target emotion, text, generated emotion / content risk | the above **plus** `judge_voice_risk`, `judge_content_risk`, `anomaly`, `reason`, `judge_model` |
| Sampled | 10 batches × 10 rows = 100 | 10 batches × 10 rows = 100 |
| Batches with a written verdict | 7 of 10 (0, 2, 3, 4, 6, 8, 9) | 7 of 10 (1, 2, 3, 4, 5, 6, 9) |

## Results

**Utterance quality — roughly half the sampled rows were flagged.** Across the
five markdown verdict files: 48 OK, 44 ISSUE. Batch 8 scores itself hardest at
3 OK / 7 ISSUE. The recurring complaints are consistent across batches:

- Register too formal for spontaneous speech. *"He has collapsed, I am beginning
  chest compressions now"* attributed to a panicking teenager;
  *"My brother is experiencing tonic-clonic activity right now"* likewise.
- Emotion label not matching the text's tone — a composed, declarative sentence
  carrying `gen_emotion=panic`.
- Content risk inflated or deflated relative to the text — `mid` for
  domestic violence in progress, `mid` for a harmless locked-out scenario.
- Occasional profile/text tension, e.g. an "uninvolved bystander" whose text
  describes being personally coerced.

**Judge accuracy — 58 of 72 reviewed rows were judged correctly, about 81%.**
Per batch: 9/11, 10/10, 10/10, 8/10, 9/10, 6/11, 6/10.

These two results are compatible. The first says the *text* is often not
convincingly spontaneous; the second says that, whatever the text is, the labels
attached to it are usually right. `RESULTS.md` §3.1 reports inter-judge
agreement of 0.941 with 87.2% unanimity — that is a measure of judges agreeing
with each other, which is again a different quantity from either audit here.

## What this means for the benchmark

The fusion benchmark measures whether a model can detect a mismatch between
voice channel and text channel. That task depends on the *labels* being right,
not on the text being stylistically perfect, so the low realism score does not
by itself invalidate the fusion results in `RESULTS.md` §3.

It does bound what the dataset can be used for. Anyone reusing
`dataset_final.jsonl` as a corpus of realistic emergency-call language — rather
than as a labelled mismatch benchmark — should read `utterance_quality/` first.

## Known defects in these files

Kept as-is rather than cleaned, because rewriting an audit after the fact would
misrepresent what was actually produced.

- Naming is inconsistent: `_review.md`, `_result.md`, `_results.json`,
  `_summary.md`, `_summary.json`, `_audit.md`, `_audit.json`, `_audit.jsonl`,
  `_verdicts.jsonl`, `_result.jsonl` all appear.
- Six different JSON/JSONL schemas across the seven judge-audit files; the
  verdict key is `verdict` in five of them, `judge_verdict` in one, and
  `judge_accuracy` in another.
- Three batches in each audit produced no written verdict at all.
- `utterance_quality/review_batch_0_review.md` line 11 contains an untranslated
  Chinese fragment mid-sentence ("stiff and书面").
- `utterance_quality/review_batch_4_summary.json` has a `details` field that
  narrates its own file writes rather than describing the data.
- Batches 1 and 6 of the judge audit contain 11 verdict rows against a 10-row
  input batch.

## Files

```
utterance_quality/
  review_batch_0.jsonl … review_batch_9.jsonl    the 100 sampled rows, 10 per batch
  review_batch_{0,2,4,6,8}_*.md                  written verdicts
  review_batch_{3,4,8}_*.json                    machine-readable verdicts
  review_batch_{2,9}_result*.jsonl               per-row verdicts

judge_accuracy/
  judge_review_0.jsonl … judge_review_9.jsonl    the 100 sampled rows, 10 per batch
  judge_review_3_audit.md                        written verdict
  judge_review_{3,4}_audit.json                  machine-readable verdicts
  judge_review_{1,2,5,6,9}_*.jsonl               per-row verdicts
```
