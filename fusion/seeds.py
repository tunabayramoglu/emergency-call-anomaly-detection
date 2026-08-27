#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the fusion — compositional seed generator with HARD compatibility constraints.

The SCRIPT randomizes the combination but only ever emits COHERENT seeds, so the
LLM never receives an impossible combo (no 'skip' escape hatch, full yield).

    WHO (profile)  x  WHAT (event)  x  HOW (emotion)  x  channel/context/twist/...

Constraints:
  * EVENT   -> content_risk + category + allowed channels
  * PROFILE -> allowed emotions + allowed event categories
The (profile, emotion) pairing bakes the REASON into the sample: e.g. an off-duty
nurse (neutral) on a high-risk event is a calm+alarm anomaly *because of* the role.

content_risk comes from the event; emotion is drawn from the profile's allowed set;
a preliminary label balances the sample; the separate judge assigns the FINAL label.

Output: seeds.jsonl   ·   Usage: python seeds.py --n 1500 --seed 42
"""
import argparse, json, random
from collections import Counter, defaultdict

# ============ EVENTS: (desc, content_risk, category) ============
EVENTS = [
    ("someone choking", "high", "medical"),
    ("cardiac arrest / heart attack", "high", "medical"),
    ("stroke symptoms", "high", "medical"),
    ("drug overdose", "high", "medical"),
    ("severe allergic reaction (anaphylaxis)", "high", "medical"),
    ("seizure", "high", "medical"),
    ("active childbirth / labor", "high", "medical"),
    ("self-harm or suicide risk", "high", "mental_health"),
    ("structure fire", "high", "fire_hazmat"),
    ("gas leak with symptoms", "high", "fire_hazmat"),
    ("building collapse / explosion", "high", "fire_hazmat"),
    ("chemical spill / hazmat", "high", "fire_hazmat"),
    ("single-vehicle crash with injuries", "high", "accident"),
    ("multi-vehicle pileup / mass casualties", "high", "accident"),
    ("minor fender-bender, no injuries", "low", "accident"),
    ("active shooter / gunfire", "high", "crime"),
    ("armed robbery in progress", "high", "crime"),
    ("physical assault", "high", "crime"),
    ("domestic violence in progress", "high", "crime"),
    ("kidnapping / abduction", "high", "crime"),
    ("suspicious activity or package", "mid", "crime"),
    ("possible gunshots heard, unconfirmed", "mid", "crime"),
    ("verbal altercation escalating", "mid", "crime"),
    ("trespasser on the property", "mid", "crime"),
    ("drowning", "high", "disaster"),
    ("flood or natural disaster", "high", "disaster"),
    ("severe hypothermia / exposure", "high", "disaster"),
    ("rising water nearby, not yet trapped", "mid", "disaster"),
    ("person feeling faint or dizzy, still conscious", "mid", "medical"),
    ("high fever, stable for now", "mid", "medical"),
    ("laceration needing stitches, bleeding controlled", "mid", "medical"),
    ("smell of smoke, source unknown", "mid", "fire_hazmat"),
    ("small kitchen fire, now extinguished", "mid", "fire_hazmat"),
    ("car stuck in a ditch, no injuries", "mid", "accident"),
    ("road hazard / debris on the highway", "mid", "accident"),
    ("person distressed but not in immediate danger", "mid", "mental_health"),
    ("missing person", "mid", "mundane"),
    ("welfare check request", "mid", "mundane"),
    ("noise complaint", "low", "mundane"),
    ("lost pet", "low", "mundane"),
    ("lost wallet or keys", "low", "mundane"),
    ("locked out of home/car", "low", "mundane"),
    ("power outage inquiry", "low", "mundane"),
    ("broken traffic light / minor civic issue", "low", "mundane"),
    ("request for non-emergency information", "low", "mundane"),
    ("wrong number / misdial", "low", "mundane"),
    ("prank or test call", "low", "mundane"),
]

CAT_CHANNELS = {
    "medical":       ["911 voice call", "police/fire dispatch radio", "hospital triage line"],
    "mental_health": ["911 voice call", "crisis/helpline"],
    "fire_hazmat":   ["911 voice call", "police/fire dispatch radio"],
    "accident":      ["911 voice call", "police/fire dispatch radio"],
    "crime":         ["911 voice call", "police/fire dispatch radio"],
    "disaster":      ["911 voice call", "coast guard radio", "police/fire dispatch radio"],
    "mundane":       ["non-emergency line", "911 voice call"],
}

# ============ PROFILES: (desc, allowed_emotions, allowed_categories) ============
E = ["neutral", "distress", "fear", "urgency", "panic", "confusion"]
PROFILES = [
    ("off-duty medical professional (nurse/paramedic)", ["neutral", "urgency"],
     ["medical", "accident", "disaster", "fire_hazmat"]),
    ("on-duty first responder relaying from the scene", ["neutral", "urgency"],
     ["medical", "fire_hazmat", "accident", "crime", "disaster"]),
    ("retired professional (former nurse/officer)", ["neutral", "urgency", "distress"],
     ["medical", "accident", "crime"]),
    ("trained emergency dispatcher/operator", ["neutral", "urgency"],
     ["medical", "fire_hazmat", "accident", "crime", "disaster"]),
    ("security guard or venue staff", ["neutral", "urgency", "fear"],
     ["crime", "fire_hazmat", "medical"]),
    ("parent or close family member of the affected person", ["panic", "fear", "distress", "urgency"],
     ["medical", "accident", "crime", "fire_hazmat"]),
    ("uninvolved bystander / witness", ["neutral", "urgency", "confusion", "fear"],
     ["accident", "crime", "fire_hazmat", "medical", "disaster"]),
    ("the direct victim, still able to speak", ["distress", "fear", "panic", "urgency", "neutral"],
     ["medical", "crime", "accident", "fire_hazmat", "disaster"]),
    ("young child (under ~10)", ["fear", "panic", "confusion"],
     ["medical", "crime", "fire_hazmat", "mundane"]),
    ("teenager", ["fear", "panic", "distress", "neutral", "urgency"],
     ["medical", "crime", "accident", "mundane"]),
    ("elderly resident", ["confusion", "distress", "fear", "neutral"],
     ["medical", "crime", "mundane", "fire_hazmat"]),
    ("person with dementia / cognitive impairment", ["confusion"],
     ["medical", "mundane", "crime"]),
    ("intoxicated adult (alcohol)", ["confusion", "panic"],
     ["accident", "crime", "mundane", "medical"]),
    ("person under the influence of drugs", ["confusion", "fear", "panic"],
     ["crime", "medical", "mundane"]),
    ("non-native / limited-English speaker", ["confusion", "fear", "neutral", "distress"],
     ["medical", "crime", "fire_hazmat", "accident"]),
    ("coerced person or hostage speaking under duress", ["neutral", "fear"],
     ["crime"]),
    ("deceptive perpetrator or false/swatting caller", ["neutral"],
     ["crime", "fire_hazmat"]),
    ("person in an acute mental-health crisis", ["neutral", "distress", "panic", "confusion", "fear"],
     ["mental_health", "mundane"]),
    ("caregiver of the affected person", ["neutral", "distress", "urgency", "fear"],
     ["medical"]),
    ("uninvolved civilian with a non-emergency matter", ["neutral", "confusion"],
     ["mundane"]),
]

HIGH_EMO = {"distress", "fear", "urgency", "panic"}
CONTEXT_MODIFIERS = ["late at night", "rush hour", "poor line/static", "background noise",
                     "rural area", "crowded place", "rainy/stormy", "weekend",
                     "weak phone signal", "multiple callers, same incident"]
SITUATION_TWISTS = ["situation worsens mid-call", "call drops and reconnects", "a second person cuts in",
                    "caller cannot give the address", "dispatcher misunderstands", "caller floods with questions",
                    "another voice in the background", "caller suddenly goes silent", "a child takes the phone",
                    "wrong info given, then corrected", "no twist (plain)"]
DISPATCHER_STYLES = ["standard professional", "calming/slow", "fast/directive", "procedural/checklist"]
REGISTERS = ["everyday speech", "short/fragmented sentences", "formal/precise",
             "hesitant/halting", "slang/informal", "repetitive/scattered"]


def prelim_label(emotion, content):
    """Preliminary label (for sample balancing only; judge gives the final label)."""
    if content == "mid":
        return "borderline"
    if content == "high":
        return "anomaly" if emotion in ("neutral", "confusion") else "normal"
    return "anomaly" if emotion in HIGH_EMO else "normal"  # content == low


def valid_combos():
    """All coherent (profile, event, emotion) triples under the constraints."""
    combos = []
    for pi, (pdesc, pemos, pcats) in enumerate(PROFILES):
        for ev, content, cat in EVENTS:
            if cat not in pcats:
                continue
            for emo in pemos:
                combos.append((pi, pdesc, ev, content, cat, emo))
    return combos


def make_seeds(n, seed):
    rng = random.Random(seed)
    combos = valid_combos()
    targets = {"anomaly": 0.55, "normal": 0.32, "borderline": 0.13}
    seeds, seen = [], set()
    tries = 0
    while len(seeds) < n and tries < n * 80:
        tries += 1
        pi, pdesc, ev, content, cat, emo = rng.choice(combos)
        lab = prelim_label(emo, content)
        if rng.random() > targets.get(lab, 0.0) / max(targets.values()):
            continue
        s = {
            "profile": pdesc,
            "event": ev,
            "target_emotion": emo,
            "content_risk": content,
            "category": cat,
            "prelim_label": lab,
            "channel": rng.choice(CAT_CHANNELS[cat]),
            "context": rng.choice(CONTEXT_MODIFIERS),
            "twist": rng.choice(SITUATION_TWISTS),
            "dispatcher_style": rng.choice(DISPATCHER_STYLES),
            "register": rng.choice(REGISTERS),
            "n_utterances": rng.randint(4, 9),
            "gen_seed": rng.randint(1, 10**9),
        }
        key = (pi, ev, emo, s["channel"], s["context"], s["twist"], s["dispatcher_style"], s["register"])
        if key in seen:
            continue
        seen.add(key)
        s["seed_id"] = len(seeds)
        seeds.append(s)
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="seeds.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seeds = make_seeds(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    lab = Counter(s["prelim_label"] for s in seeds)
    emo = Counter(s["target_emotion"] for s in seeds)
    con = Counter(s["content_risk"] for s in seeds)
    nvalid = len(valid_combos())
    surf = len(CONTEXT_MODIFIERS) * len(SITUATION_TWISTS) * len(DISPATCHER_STYLES) * len(REGISTERS)
    print(f"[seeds] {len(seeds)} seeds -> {a.out}")
    print(f"[prelim label] {dict(lab)}")
    print(f"[emotion] {dict(emo)}")
    print(f"[content] {dict(con)}")
    print(f"[space] {nvalid:,} COHERENT (who,what,how) triples x ~{surf*3:,} surface  (all valid, no skip)")
    print(f"[est. utterances] ~{sum(s['n_utterances'] for s in seeds)}")


if __name__ == "__main__":
    main()
