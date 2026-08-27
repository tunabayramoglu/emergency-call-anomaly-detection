# Batch 2 Review — Detailed Verdicts

## Entry-by-Entry Analysis

### 1. UID 8928 — Teenager / Choking / Panic / High
**Text:** "He has collapsed, I am beginning chest compressions now."
- **Verdict: ISSUE**
- Text is too clinical ("beginning chest compressions") for a teenager in panic. Natural speech would be fragmented and colloquial.
- Emotion label (panic) doesn't match the composed, declarative tone.
- Content risk (high) is correct.

### 2. UID 6873 — Non-native speaker / Seizure / Confusion / High
**Text:** "Mommy? Mommy not moving? I don't know."
- **Verdict: OK**
- Excellent realism for the profile. Simple grammar and trailing uncertainty are natural.
- Minor emotion drift: gen_emotion "fear" vs target "confusion" — acceptable overlap.
- Content risk (high) is correct.

### 3. UID 3611 — Teenager / Lost pet / Urgency / Low
**Text:** "I shall maintain vigilance near the property entrance until assistance arrives."
- **Verdict: ISSUE**
- Completely unrealistic register. No teenager uses "I shall maintain vigilance near the property entrance."
- Zero urgency conveyed despite the label.
- Content risk (low) is correct.

### 4. UID 7359 — Teenager / Seizure / Distress / High
**Text:** "he's been like this for a minute, maybe? I dunno, it feels forever"
- **Verdict: ISSUE**
- Excellent text realism and emotion match.
- content_risk mislabeled: should be "high" (seizure of unknown duration), not "mid".

### 5. UID 9654 — Under influence / Non-emergency info / Confusion / Low
**Text:** "Nah wait, hold up... my boy said y'all already got calls about this, so like... what's the deal with that?"
- **Verdict: OK**
- Colloquial register fits the profile. All labels align correctly.

### 6. UID 4557 — Teenager / Seizure / Fear / High
**Text:** "Wait, another individual is interrupting my response to you."
- **Verdict: ISSUE**
- Formal/dispatcher language ("another individual is interrupting my response to you") is unrealistic for a teenager.
- No fear or panic conveyed in text.
- content_risk mislabeled: should be "high" (seizure), not "mid".

### 7. UID 106 — Elderly resident / Armed robbery / Confusion / High
**Text:** "He appears to be holding a metallic object, though I cannot be entirely certain."
- **Verdict: OK**
- Slightly formal but conveys confusion through uncertainty. Acceptable for the profile.
- Content risk (high) is correct.

### 8. UID 2615 — Stroke victim / Stroke / Urgency / High
**Text:** "Can't lift my left arm... fingers are... stuck. Need help."
- **Verdict: OK**
- Excellent realism. Fragmented speech matches stroke symptoms. Best entry in the batch.
- All labels align perfectly.

### 9. UID 6924 — Security guard / Physical assault / Fear / High
**Text:** "He's wearing a dark hoodie, jeans, maybe six feet tall, I can't see his face, he's... he's just focused on the victim."
- **Verdict: ISSUE**
- Good realism and emotion match.
- content_risk mislabeled: should be "high" (active assault in progress), not "mid".

### 10. UID 5574 — Mental health crisis / Broken traffic light / Distress / Low
**Text:** "How long till someone comes? Do you know where they're at?"
- **Verdict: OK**
- Acceptable realism. All labels align.
- Generic text but consistent with the profile.

## Scorecard

| # | UID | Verdict | Realism | Emotion Match | Risk Match | Notes |
|---|------|---------|---------|---------------|------------|-------|
| 1 | 8928 | ISSUE | Low | No | ✓ | Clinical language, no panic |
| 2 | 6873 | OK | High | ~✓ | ✓ | Minor emotion drift |
| 3 | 3611 | ISSUE | Very Low | No | ✓ | Absurdly formal for teen |
| 4 | 7359 | ISSUE | High | ✓ | ✗ | Risk downgraded to mid |
| 5 | 9654 | OK | Good | ✓ | ✓ | All consistent |
| 6 | 4557 | ISSUE | Very Low | No | ✗ | Triple failure |
| 7 | 106 | OK | Moderate | ✓ | ✓ | Slightly formal |
| 8 | 2615 | OK | Excellent | ✓ | ✓ | Best entry |
| 9 | 6924 | ISSUE | Good | ✓ | ✗ | Risk downgraded to mid |
| 10 | 5574 | OK | Acceptable | ✓ | ✓ | Generic but OK |

## Overall Quality Score: 5/10

5 OK / 5 ISSUE

Key findings:
1. Three entries (4, 6, 9) have content_risk downgraded from "high" to "mid" — systematic drift in risk labeling
2. Three entries (1, 3, 6) have unrealistically formal language that doesn't match the caller profile
3. Two entries (3, 6) are severe quality failures
4. Best entries (8, 2, 5) show the pipeline can produce excellent results
