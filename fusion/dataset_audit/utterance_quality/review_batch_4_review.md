# Batch 4 Review — ReviewBatch4

## Per-Utterance Verdicts

### 1. uid 4333 — OK
- **Text:** "Dispatch, major boom just went down on my position."
- **Profile:** On-duty first responder, building collapse/explosion
- **gen_emotion=urgency ✓** | **gen_content_risk=high ✓**
- Colloquial radio-traffic shorthand is realistic. All fields align.

### 2. uid 711 — OK
- **Text:** "why he point it at the shed door like that."
- **Profile:** Non-native/L2 speaker, active shooter/gunfire
- **gen_emotion=confusion ✓** | **gen_content_risk=high ✓**
- Broken grammar matches L2 profile. Threat-level content is high-risk.

### 3. uid 7527 — ISSUE
- **Text:** "Yeah, uh, somebody took my kid, man. They just drove off."
- **Profile:** Non-native/L2 speaker, kidnapping/abduction
- **gen_emotion=neutral ✗** | **gen_content_risk=high ✓**
- A parent whose child was just abducted would typically show fear or panic. "Neutral" is defensible only if modeling dissociative shock, which should be explicitly noted. More likely a mislabel.

### 4. uid 8785 — ISSUE
- **Text:** "Can't say where."
- **Profile:** Deceptive perpetrator / swatting caller, domestic violence in progress
- **gen_emotion=neutral ✓** | **gen_content_risk=mid ✗**
- The evasive text alone is mid-risk, but the event is domestic violence in progress (seed = high). gen_content_risk should match the event severity.

### 5. uid 2045 — ISSUE
- **Text:** "I am systematically tracking the exact interval between each passing vehicle."
- **Profile:** Person in mental-health crisis, broken traffic light (minor civic issue)
- **gen_emotion=neutral ✓** | **gen_content_risk=mid ✗**
- The prose is grammatically flawless and uses formal vocabulary ("systematically," "interval"). This is unrealistic for a mental-health crisis caller — the register is too polished. Also gen_content_risk=mid doesn't match content_risk_seed=low.

### 6. uid 6201 — OK (borderline)
- **Text:** "He's walking around the house."
- **Profile:** Parent/family member, trespasser on property
- **gen_emotion=fear ✓** | **gen_content_risk=high ≈**
- Fear fits the parent reporting a stranger on their property. content_risk upgrade from mid→high is slightly aggressive but defensible given the family context.

### 7. uid 1291 — OK
- **Text:** "He keeps saying his head is spinning and he can't see straight, but he's still talking to me."
- **Profile:** Bystander/witness, person feeling faint/dizzy
- **gen_emotion=urgency ✓** | **gen_content_risk=mid ✓**
- Natural bystander description with hope. All fields align cleanly.

### 8. uid 9044 — OK
- **Text:** "He's still goin', he's still... oh god, is he... is he gonna be okay? Please..."
- **Profile:** Intoxicated adult, seizure
- **gen_emotion=panic ✓** | **gen_content_risk=high ✓**
- Fragmented speech, repetition, trailing off match panic. Ongoing seizure is high-risk.

### 9. uid 4803 — ISSUE
- **Text:** "She's screaming, I can't calm her down, listen."
- **Profile:** Off-duty medical professional, active childbirth/labor
- **gen_emotion=panic ✗** (target=urgency) | **gen_content_risk=mid ✗** (seed=high)
- Two deviations: (1) A medical professional should show urgency/composure, not panic. (2) Active childbirth without professional support is high-risk, not mid.

### 10. uid 5925 — ISSUE
- **Text:** "I will hang up and attempt to recall the specific number..."
- **Profile:** Intoxicated adult, wrong number/misdial
- **gen_emotion=confusion ✓** | **gen_content_risk=low ✓**
- The text is unrealistically formal for an intoxicated caller. No drunk person says "attempt to recall the specific number." Register mismatch with profile.

---

## Summary Table

| # | uid | Verdict | Emotion OK? | Content Risk OK? | Profile Match? |
|---|---|---|---|---|---|
| 1 | 4333 | OK | ✓ | ✓ | ✓ |
| 2 | 711 | OK | ✓ | ✓ | ✓ |
| 3 | 7527 | ISSUE | ✗ (neutral for abduction) | ✓ | ✓ |
| 4 | 8785 | ISSUE | ✓ | ✗ (mid vs high) | ✓ |
| 5 | 2045 | ISSUE | ✓ | ✗ (mid vs low) | ✗ (too formal) |
| 6 | 6201 | OK | ✓ | ~ (borderline) | ✓ |
| 7 | 1291 | OK | ✓ | ✓ | ✓ |
| 8 | 9044 | OK | ✓ | ✓ | ✓ |
| 9 | 4803 | ISSUE | ✗ (panic vs urgency) | ✗ (mid vs high) | ✗ (doctor should be composed) |
| 10 | 5925 | ISSUE | ✓ | ✓ | ✗ (too formal) |

**4 OK, 5 ISSUE (1 borderline OK)**

## Overall Quality Score: 6 / 10

### Recurring Issues
1. **Register/formality mismatches** (uid 2045, uid 5925): Generation models produce textbook prose instead of realistic casual/impaired speech for the given profile.
2. **Content risk drift** (uid 8785, uid 2045): gen_content_risk doesn't align with the seed's event severity.
3. **Emotion-profile conflict** (uid 4803, uid 7527): Emotion labels don't account for profile-specific behavior (medical professional should be composed; parent of abducted child should be distressed).
