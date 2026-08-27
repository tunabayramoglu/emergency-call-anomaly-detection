#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the fusion — generation prompt (seed -> labeled utterances).

The DRIVER injects one seed into GEN_USER_TEMPLATE and sends {GEN_SYSTEM, user}
to the LLM. The LLM never sees or runs the seed generator; it is a pure function
prompt(seed) -> JSON. Strict MUST framing, JSON-only output.
"""

GEN_SYSTEM = (
    "You are a deterministic data-generation engine that produces realistic "
    "emergency-call transcripts for a research dataset on voice-vs-content anomaly "
    "detection. You output ONLY a single valid JSON object and nothing else — no "
    "prose, no markdown, no code fences. You MUST follow every constraint exactly."
)

GEN_USER_TEMPLATE = """You MUST generate one emergency-call transcript from this EXACT specification. \
You MUST NOT change, ignore, or add to the specification.

SPECIFICATION (fixed — you MUST obey all fields):
- caller_profile : {profile}
- event          : {event}
- channel        : {channel}
- context        : {context}
- twist          : {twist}
- dispatcher_style : {dispatcher_style}
- caller_register  : {register}
- target_delivery  : {target_emotion}   (this is HOW the caller SOUNDS, not what they say)
- n_utterances     : {n_utterances}

HARD RULES (MUST):
1. You MUST write a realistic multi-turn phone dialogue between the CALLER (exactly the \
caller_profile) and a DISPATCHER, about the event, over the given channel.
2. The CALLER's vocal delivery MUST predominantly convey target_delivery. The delivery is \
tone/prosody — it MAY diverge from what the words mean (that divergence is the point).
3. The DISPATCHER MUST stay neutral and professional in the given dispatcher_style. Dispatcher \
lines are NOT part of the output.
4. The dialogue MUST reflect context, twist, and caller_register.
5. LOGICAL CONSISTENCY: The caller's words, punctuation, and sentence structure MUST be \
physically plausible given the scenario. A person hiding from an attacker MUST NOT yell or \
use exclamation marks — they whisper or speak in short, hushed fragments. A person who is \
injured or weak MUST NOT produce long, forceful sentences. A drunk caller MUST slur or \
trail off. Match the physical reality of the situation.
6. You MUST then select EXACTLY {n_utterances} distinct CALLER lines (never dispatcher lines).
7. For EACH selected caller line you MUST assign, independently:
   - "emotion": EXACTLY one of ["neutral","distress","fear","urgency","panic","confusion"] \
= how the caller SOUNDS saying it (delivery/tone).
   - "content_risk": EXACTLY one of ["low","mid","high"] = how dangerous/urgent the MEANING \
of the words is, judged from text alone, ignoring tone.
8. "emotion" and "content_risk" are INDEPENDENT axes and MAY diverge within the same line.
9. The caller's dominant emotion across lines MUST be target_delivery, but you MUST allow \
natural drift consistent with twist (e.g. escalation if the situation worsens).
10. You MUST NOT output any anomaly/normal/risk-match judgement. A separate judge does that.
11. Each caller line MUST be natural spoken English, 3–25 words, matching caller_register. \
You MUST NOT invent real names/addresses; use generic placeholders (e.g. "my address", "him").
12. You MUST NOT generate utterances about phone signal issues, losing connection, \
giving/reciting addresses, or call quality problems. Stay focused on the event itself.
13. You MUST output ONLY this JSON object, with EXACTLY {n_utterances} items:

{{"utterances": [{{"text": "<caller line>", "emotion": "<one of the six>", "content_risk": "<low|mid|high>"}}]}}
"""


def build_gen_prompt(seed: dict):
    """Return (system, user) messages for one seed."""
    import json
    user = GEN_USER_TEMPLATE.format(
        profile=seed["profile"], event=seed["event"], channel=seed["channel"],
        context=seed["context"], twist=seed["twist"],
        dispatcher_style=seed["dispatcher_style"], register=seed["register"],
        target_emotion=seed["target_emotion"], n_utterances=seed["n_utterances"])
    return GEN_SYSTEM, user


# ============================ JUDGE (separate lens) ============================
# A DIFFERENT persona from the generator (bias guardrail). It has NO access to
# the seed, the intended label, or the generator's reasoning — it judges each
# utterance on its own merits, in batches.

JUDGE_SYSTEM = (
    "You are an independent blind judge for a voice-vs-content anomaly detection "
    "research dataset. You see ONLY the caller utterances and their generated "
    "emotion labels — you have NO access to the seed, the intended label, or "
    "the generator's reasoning. You output ONLY a JSON array, nothing else."
)

JUDGE_USER_TEMPLATE = """You MUST label every caller utterance in the INPUT array.

For EACH item you MUST assign:
- "content_risk": EXACTLY one of ["low","mid","high"] = how dangerous/urgent the \
MEANING of the words is, judged from text alone.
- "voice_risk": EXACTLY one of ["low","high"] = how alarming/distressed the caller \
SOUNDS based on the provided emotion label.
- "anomaly": EXACTLY one of ["normal","anomaly","borderline"] — \
"anomaly" if the voice_risk and content_risk DIVERGE (one high, one low), \
"normal" if they MATCH (both high or both low), \
"borderline" if uncertain or mixed.
- "reason": one short sentence explaining your judgement.

RULES:
1. You MUST use the "emotion" field as the ONLY signal for how the caller sounds.
2. You MUST NOT be influenced by the event or profile — judge each utterance independently.
3. You MUST output a JSON array with EXACTLY the same number of items as the INPUT, \
in the same order, each with "id", "content_risk", "voice_risk", "anomaly", "reason".

INPUT:
{items_json}

OUTPUT (JSON array only):
"""


def build_judge_prompt(items):
    """items: list of {"id": int, "text": str, "emotion": str}. Returns (system, user)."""
    import json
    return JUDGE_SYSTEM, JUDGE_USER_TEMPLATE.format(items_json=json.dumps(items, ensure_ascii=False))


if __name__ == "__main__":
    import json, sys
    seed = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "profile": "elderly person", "event": "house fire",
        "channel": "911 mobile", "context": "night",
        "twist": "no twist", "dispatcher_style": "calming/slow",
        "register": "hesitant/halting", "target_emotion": "panic",
        "n_utterances": 5}
    sysmsg, usr = build_gen_prompt(seed)
    print("=== SYSTEM ===\n" + sysmsg + "\n\n=== USER ===\n" + usr)
