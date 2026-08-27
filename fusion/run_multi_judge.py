#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the fusion — multi-model judge orchestrator.
Splits utterances across N models, runs judge in parallel, merges results.

Usage:
    python run_multi_judge.py --utts utterances_merged.jsonl --workers-per-model 2 --judge-batch 5
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Intermediate per-model outputs and chunk files live in one place so that
# re-running the pipeline does not scatter dozens of files across fusion/.
RUNS = HERE / "generation"
RUNS.mkdir(exist_ok=True)

# 18 judge models — fast + diverse + date-stamped fallbacks
JUDGE_MODELS = [
    # qwen3.7
    "qwen3.7-max",
    "qwen3.7-max-2026-06-08",
    "qwen3.7-max-2026-05-20",
    "qwen3.7-plus",
    # qwen3.6
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.6-flash-2026-04-16",
    # qwen3.5
    "qwen3.5-plus",
    "qwen3.5-plus-2026-04-20",
    "qwen3.5-flash",
    "qwen3.5-flash-2026-02-23",
    # deepseek
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    # glm
    "glm-5.2",
    "glm-5.1",
    # other
    "qwen3-max",
    "qwen-plus-latest",
]

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def split_utts(utts_path, n_chunks):
    """Split utterances into n_chunks files."""
    lines = open(utts_path, encoding="utf-8").readlines()
    chunk_size = len(lines) // n_chunks + 1
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        chunk_path = str(RUNS / f"utts_chunk_{len(chunks)}.jsonl")
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.writelines(chunk)
        chunks.append((chunk_path, len(chunk)))
    return chunks


def run_judge(model, utts_file, batch, api_key):
    """Run judge for one model in a subprocess."""
    safe = model.replace(".", "_")
    out_path = str(RUNS / f"dataset_judge_{safe}.jsonl")
    cmd = [
        sys.executable, "generate.py",
        "--utts", utts_file,
        "--out", out_path,
        "--model", model,
        "--provider", "openai",
        "--stage", "judge",
        "--judge-batch", str(batch),
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = BASE_URL

    print(f"[launch-judge] {model} -> {out_path}", flush=True)
    return subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))


def merge_judge_results(models):
    """Merge all judge output files, keeping only labeled utterances."""
    merged = str(RUNS / "dataset_merged.jsonl")
    uid = 0
    with open(merged, "w", encoding="utf-8") as fout:
        for model in models:
            safe = model.replace(".", "_")
            jfile = str(RUNS / f"dataset_judge_{safe}.jsonl")
            if not os.path.exists(jfile):
                print(f"  [warn] {jfile} not found, skipping", flush=True)
                continue
            count = 0
            for line in open(jfile, encoding="utf-8"):
                rec = json.loads(line)
                rec["uid"] = uid
                rec["judge_model"] = model
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                uid += 1
                count += 1
            print(f"  [merge] {model}: {count} labeled", flush=True)

    # dedup by text (same utterance judged by multiple models -> keep first)
    seen = set()
    deduped = str(HERE / "dataset_final.jsonl")
    kept = 0
    with open(deduped, "w", encoding="utf-8") as fout:
        for line in open(merged, encoding="utf-8"):
            rec = json.loads(line)
            key = rec["text"]
            if key not in seen:
                seen.add(key)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"\n[judge-merged] {merged} — {uid} total labels", flush=True)
    print(f"[judge-deduped] {deduped} — {kept} unique utterances", flush=True)
    return deduped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", default=str(RUNS / "utterances_merged.jsonl"))
    ap.add_argument("--workers-per-model", type=int, default=2)
    ap.add_argument("--judge-batch", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()

    api_key = a.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY or pass --api-key", flush=True)
        sys.exit(1)

    models = a.models or JUDGE_MODELS
    n_models = len(models)

    print(f"{'='*60}", flush=True)
    print(f"Fusion — Multi-Model Judge", flush=True)
    print(f"  Models: {n_models}", flush=True)
    print(f"  Batch size: {a.judge_batch}", flush=True)
    print(f"  Workers/model: {a.workers_per_model}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # split utterances
    chunks = split_utts(a.utts, n_models)
    total = sum(c for _, c in chunks)
    print(f"Split {total} utterances into {len(chunks)} chunks:", flush=True)
    for cp, cnt in chunks:
        print(f"  {cp}: {cnt} utterances", flush=True)
    print()

    # launch all judges in parallel
    procs = []
    for i, model in enumerate(models):
        chunk_path = chunks[i][0] if i < len(chunks) else chunks[-1][0]
        p = run_judge(model, chunk_path, a.judge_batch, api_key)
        procs.append((model, p))

    # wait
    print(f"\n[waiting] {len(procs)} judges running...", flush=True)
    failed = []
    for model, p in procs:
        rc = p.wait()
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{status}] {model}", flush=True)
        if rc != 0:
            failed.append(model)

    # cleanup chunks
    for cp, _ in chunks:
        if os.path.exists(cp):
            os.remove(cp)

    # merge
    print(f"\n{'='*60}", flush=True)
    print(f"MERGING JUDGE RESULTS", flush=True)
    deduped = merge_judge_results([m for m, _ in procs if m not in failed])

    total_kept = sum(1 for _ in open(deduped, encoding="utf-8"))
    print(f"\n{'='*60}", flush=True)
    print(f"DONE — {total_kept} unique labeled utterances", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)
    print(f"Output: {deduped}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
