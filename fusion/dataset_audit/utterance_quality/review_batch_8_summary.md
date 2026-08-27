# Review Batch 8 — Emergency Call Utterance QA

## Overall Quality Score: 4 / 10

**Pass rate: 30%** (3 OK, 7 ISSUE out of 10 utterances)

---

## Per-Utterance Verdicts

### #1 — uid 3752 · "lock clicked... oh wow... floor tilts sideways"
**Profile:** drug-influenced person · **Event:** locked out  
**Verdict: ISSUE**
- Content risk inflated: gen=mid but text is harmless (no violence/threats); should be low
- Emotion off-target: gen=distress vs target=confusion; reads more disoriented than distressed
- Spatial inconsistency: "locked out" event but text describes an indoor floor tilting

### #2 — uid 525 · "The... um... the person on the ground is... not moving at all."
**Profile:** bystander/witness · **Event:** physical assault  
**Verdict: OK ✓**
- Natural bystander speech with realistic hesitation
- All labels match targets (neutral/high)
- Fully consistent

### #3 — uid 5168 · "There is dust everywhere. The ceiling looks broken above me."
**Profile:** security guard · **Event:** building collapse  
**Verdict: ISSUE**
- Content risk downgraded: gen=mid but broken ceiling in a collapse = life-threatening; should be high
- Emotion and realism are fine

### #4 — uid 6572 · "I need to report a medical emergency."
**Profile:** bystander · **Event:** cardiac arrest  
**Verdict: OK ✓**
- Valid opening line, all labels match (neutral/high)
- Generic but realistic for a call opener

### #5 — uid 4386 · "Wait, I need you to send fire anyway, I smell something burning."
**Profile:** dispatcher/operator · **Event:** fender-bender  
**Verdict: ISSUE (FATAL)**
- **Role inversion:** profile says "dispatcher" but text is clearly a caller requesting fire services
- Dispatchers send units — they don't ask callers to "send fire"
- Fundamental labeling/generation error

### #6 — uid 1084 · "My psychological state is rapidly deteriorating, and I require immediate professional intervention..."
**Profile:** acute mental-health crisis · **Event:** prank/test call  
**Verdict: ISSUE**
- Text unrealistically clinical for someone in acute crisis; reads like a medical report
- Content risk inflated: gen=high but text has no violent/threatening content; should be low
- Emotion mismatch: gen=panic but text is completely composed

### #7 — uid 3456 · "nope no tracks just heavy gas pooling low, cover your mouth and shuffle backwards slow."
**Profile:** off-duty medical professional · **Event:** chemical spill  
**Verdict: ISSUE**
- Excellent text quality, but content risk downgraded: gen=mid but heavy gas pooling is life-threatening; should be high
- Emotion and realism are perfect

### #8 — uid 9292 · "I require an estimated restoration time with some degree of urgency."
**Profile:** acute mental-health crisis · **Event:** power outage inquiry  
**Verdict: ISSUE**
- Severe profile/text mismatch: crisis profile with mundane, composed utility inquiry text
- Content risk inflated: gen=mid but routine power outage has no dangerous content; should be low
- Emotion label matches target on paper but neither matches the actual flat text tone

### #9 — uid 5155 · "Wait, I see movement comin' up behind the truck over yonder."
**Profile:** security guard · **Event:** suspicious activity  
**Verdict: ISSUE**
- Content risk inflated: gen=high but suspicious movement is unconfirmed; should be mid per seed
- Text quality and emotion are good

### #10 — uid 3483 · "guy next to me ain't responding, and his shirt is soaked through with dark blood"
**Profile:** retired professional · **Event:** crash with injuries  
**Verdict: OK ✓**
- Very realistic, vivid crash scene description
- All labels aligned (distress≈urgency, high/high)
- Fully consistent

---

## Recurring Issue Categories

| Category | Count | Affected UIDs |
|---|---|---|
| Content risk mislabeling | 6/10 | 3752, 5168, 1084, 3456, 9292, 5155 |
| Profile/text mismatch | 2/10 | 4386 (fatal), 9292 |
| Unrealistic register for profile | 2/10 | 1084, 9292 |
| Emotion label ≠ text tone | 3/10 | 3752, 1084, 9292 |

**Primary failure mode:** Content risk mislabeling — systematic and bidirectional (inflating low risks AND deflating high risks).

**Secondary failure mode:** Profile-text incongruence — two utterances where the generated speech fundamentally contradicts the assigned speaker role.
