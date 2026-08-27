#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the fusion — multi-model orchestrator.
Splits seeds across N models, runs each in a subprocess, merges results.
Each model gets its own utterances file; merged at the end.

Usage:
    python run_multi_model.py --seeds seeds.jsonl --workers-per-model 2
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Intermediate per-model outputs and chunk files live in one place so that
# re-running the pipeline does not scatter dozens of files across fusion/.
RUNS = HERE / "generation"
RUNS.mkdir(exist_ok=True)

# 10 models from Qwen Cloud free tier — diverse architectures
MODELS = [
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
    "qwen3.5-plus",
    "qwen3.5-flash",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "glm-5.2",
]

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def split_seeds(seeds_path, n_chunks):
    """Split seeds.jsonl into n_chunks files."""
    seeds = [json.loads(l) for l in open(seeds_path, encoding="utf-8")]
    chunk_size = len(seeds) // n_chunks + 1
    chunks = []
    for i in range(0, len(seeds), chunk_size):
        chunk = seeds[i:i + chunk_size]
        chunk_path = str(RUNS / f"seeds_chunk_{len(chunks)}.jsonl")
        with open(chunk_path, "w", encoding="utf-8") as f:
            for s in chunk:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        chunks.append((chunk_path, len(chunk)))
    return chunks


def run_model(model, seeds_file, workers, api_key):
    """Run generation for one model in a subprocess."""
    out_utts = str(RUNS / f"utterances_{model.replace('.', '_')}.jsonl")
    cmd = [
        sys.executable, "generate.py",
        "--seeds", seeds_file,
        "--utts", out_utts,
        "--out", str(RUNS / f"dataset_{model.replace('.', '_')}.jsonl"),
        "--model", model,
        "--provider", "openai",
        "--workers", str(workers),
        "--stage", "gen",  # only generation, judge later
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = BASE_URL

    print(f"[launch] {model} -> {out_utts} (workers={workers})", flush=True)
    return subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))


def merge_results(model_names):
    """Merge all utterance files into one, re-uid."""
    merged = str(RUNS / "utterances_merged.jsonl")
    uid = 0
    with open(merged, "w", encoding="utf-8") as fout:
        for model in model_names:
            utts_file = str(RUNS / f"utterances_{model.replace('.', '_')}.jsonl")
            if not os.path.exists(utts_file):
                print(f"  [warn] {utts_file} not found, skipping", flush=True)
                continue
            count = 0
            for line in open(utts_file, encoding="utf-8"):
                rec = json.loads(line)
                rec["uid"] = uid
                rec["source_model"] = model
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                uid += 1
                count += 1
            print(f"  [merge] {model}: {count} utterances", flush=True)
    print(f"\n[merged] {merged} — {uid} total utterances from {len(model_names)} models", flush=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(HERE / "seeds.jsonl"))
    ap.add_argument("--workers-per-model", type=int, default=2)
    ap.add_argument("--models", nargs="+", default=None, help="override model list")
    ap.add_argument("--api-key", default=None, help="or set OPENAI_API_KEY env var")
    a = ap.parse_args()

    api_key = a.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY or pass --api-key", flush=True)
        sys.exit(1)

    models = a.models or MODELS
    n_models = len(models)

    print(f"{'='*60}", flush=True)
    print(f"Fusion — Multi-Model Run", flush=True)
    print(f"  Models: {n_models}", flush=True)
    print(f"  Workers/model: {a.workers_per_model}", flush=True)
    print(f"  Total workers: {n_models * a.workers_per_model}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # split seeds
    chunks = split_seeds(a.seeds, n_models)
    print(f"Split {sum(c for _, c in chunks)} seeds into {len(chunks)} chunks:", flush=True)
    for cp, cnt in chunks:
        print(f"  {cp}: {cnt} seeds", flush=True)
    print()

    # launch all models in parallel
    procs = []
    for i, model in enumerate(models):
        chunk_path = chunks[i][0] if i < len(chunks) else chunks[-1][0]
        p = run_model(model, chunk_path, a.workers_per_model, api_key)
        procs.append((model, p))

    # wait for all
    print(f"\n[waiting] {len(procs)} models running...", flush=True)
    failed = []
    for model, p in procs:
        rc = p.wait()
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{status}] {model}", flush=True)
        if rc != 0:
            failed.append(model)

    # cleanup chunk files
    for cp, _ in chunks:
        if os.path.exists(cp):
            os.remove(cp)

    # merge
    print(f"\n{'='*60}", flush=True)
    print(f"MERGING RESULTS", flush=True)
    merged = merge_results([m for m, _ in procs if m not in failed])

    # summary
    total = sum(1 for _ in open(merged, encoding="utf-8"))
    print(f"\n{'='*60}", flush=True)
    print(f"DONE — {total} utterances from {len(procs) - len(failed)}/{len(procs)} models", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)
    print(f"Output: {merged}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
