# Judge Review 3 — Audit Report

**File:** `judge_review_3.jsonl` | **Judge Models:** qwen3.7-max-2026-06-08, glm-5.1, qwen3.6-flash-2026-04-16, deepseek-v4-pro, qwen3.7-max-preview, deepseek-v4-flash
**Total Utterances:** 10 | **Correct:** 10 | **Wrong:** 0

## Per-Utterance Verdicts

| # | uid | Profile | Event | Voice Risk | Content Risk | Label | Verdict |
|---|-----|---------|-------|-----------|-------------|-------|---------|
| 1 | 1624 | teenager | minor fender-bender | high (distress) | low | anomaly | CORRECT |
| 2 | 671 | mental-health crisis | lost pet | high (distress) | mid* | borderline | CORRECT |
| 3 | 5805 | bystander | armed robbery | low (neutral) | high | anomaly | CORRECT |
| 4 | 6770 | caregiver | anaphylaxis | high (fear) | high | normal | CORRECT |
| 5 | 3709 | elderly resident | structure fire | low (neutral) | high | anomaly | CORRECT |
| 6 | 4848 | intoxicated adult | seizure | high (urgency) | high | normal | CORRECT |
| 7 | 578 | kidnapping victim | abduction | high (urgency) | high | normal | CORRECT |
| 8 | 3719 | security guard | childbirth | low (neutral) | mid | borderline | CORRECT |
| 9 | 6281 | teenager | crash with injuries | low (neutral) | low | normal | CORRECT |
| 10 | 5723 | bystander | kitchen fire out | high (fear) | low | anomaly | CORRECT |

*\*content_risk inflated from low→mid; borderline label still appropriate*

## Key Findings

- **Voice-risk assessment:** Perfect alignment with gen_emotion across all 10 utterances. Judge correctly maps distress/fear/urgency → "high" and neutral → "low".
- **Content-risk assessment:** 9/10 perfectly aligned with text meaning. Utterance 2 slightly inflated (low→mid for lost pet text), but not enough to break the label.
- **Anomaly logic:** All divergence-based labels are correctly derived:
  - High voice + low content → anomaly (utts 1, 3, 5, 10)
  - High voice + mid content → borderline (utt 2)
  - Low voice + mid content → borderline (utt 8)
  - Same-level risks → normal (utts 4, 6, 7, 9)
- **Contextual awareness:** Judge correctly upgraded content_risk from gen values when seed context warranted it (utts 5, 7) and correctly ignored hidden event context when assessing text alone (utt 9).

## Accuracy Score: 9/10

**Rationale:** All 10 anomaly labels are correct and well-reasoned. One point deducted for utt 2 where content_risk was inflated from low→mid based on contextual inference rather than text content alone — a minor tendency to overweight implicit danger beyond what the words convey. This didn't affect the final label but represents a slight calibration imprecision.
