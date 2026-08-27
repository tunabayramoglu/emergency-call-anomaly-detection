#!/usr/bin/env python3
"""Dataset quality check script."""
import json, random
from pathlib import Path
from collections import Counter

DATASET = Path(__file__).resolve().parent / "dataset_final.jsonl"
rows = [json.loads(l) for l in open(DATASET, encoding="utf-8")]
total = len(rows)
seeds = len(set(r["seed_id"] for r in rows))

print(f"Total: {total} utterances, {seeds} seeds\n")

# --- 1. Class distribution ---
anom = Counter(r["anomaly"] for r in rows)
emo = Counter(r["gen_emotion"] for r in rows)
vrisk = Counter(r["judge_voice_risk"] for r in rows)
crisk = Counter(r["judge_content_risk"] for r in rows)

print("=== ANOMALY ===")
for k, v in anom.most_common():
    print(f"  {k:12s} {v:4d} ({v/total*100:.0f}%)")

print("\n=== EMOTION ===")
for k, v in emo.most_common():
    print(f"  {k:12s} {v:4d} ({v/total*100:.0f}%)")

print("\n=== VOICE RISK ===")
for k, v in vrisk.most_common():
    print(f"  {k:12s} {v:4d} ({v/total*100:.0f}%)")

print("\n=== CONTENT RISK ===")
for k, v in crisk.most_common():
    print(f"  {k:12s} {v:4d} ({v/total*100:.0f}%)")

# --- 2. Anomaly examples ---
print("\n=== ANOMALY EXAMPLES (voice != content) ===")
anomalies = [r for r in rows if r["anomaly"] == "anomaly"]
for r in anomalies[:10]:
    print(f"  emo={r['gen_emotion']:<10} vrisk={r['judge_voice_risk']:<4} crisk={r['judge_content_risk']:<4} | {r['text']}")

print("\n=== NORMAL EXAMPLES (voice ~= content) ===")
normals = [r for r in rows if r["anomaly"] == "normal"]
for r in normals[:10]:
    print(f"  emo={r['gen_emotion']:<10} vrisk={r['judge_voice_risk']:<4} crisk={r['judge_content_risk']:<4} | {r['text']}")

# --- 3. Fifteen random samples ---
print("\n=== RANDOM 15 ===")
for r in random.sample(rows, min(15, total)):
    tag = "*" if r["anomaly"] == "anomaly" else " "
    print(f"  {tag} [{r['anomaly']:>9}] emo={r['gen_emotion']:<10} | {r['text']}")

# --- 4. Red flags ---
print("\n=== RED FLAGS ===")
short = [r for r in rows if len(r["text"].split()) < 3]
long = [r for r in rows if len(r["text"].split()) > 25]
dupes = [t for t, c in Counter(r["text"] for r in rows).items() if c > 1]
print(f"  Too short (<3 words): {len(short)}")
print(f"  Too long (>25 words): {len(long)}")
print(f"  Duplicated: {len(dupes)}")
if dupes:
    for t in dupes[:5]:
        print(f"    \"{t}\"")

# --- 5. Divergence ---
diverge = sum(1 for r in rows if r["gen_emotion"] != r["target_emotion"])
print(f"\n=== DIVERGENCE ===")
print(f"  gen_emotion != target_emotion: {diverge}/{total} ({diverge/total*100:.0f}%)")
