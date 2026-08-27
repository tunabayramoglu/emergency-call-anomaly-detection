#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the fusion — dataset driver.
Reads seeds.jsonl, injects each seed into the generation prompt (the LLM never
runs the seed generator), collects labeled caller utterances, then runs a SECOND
pass with the separate judge to assign the final holistic anomaly label.

Output: dataset.jsonl  (one caller utterance per line, generation + judge labels)

Usage:
    python generate.py --seeds seeds.jsonl --out dataset.jsonl \
        --provider openai --model mimo-v2.5 --limit 0 --workers 3
Resumable: re-running skips seed_ids already present in --out.
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import build_gen_prompt, build_judge_prompt

VALID_EMO = {"neutral", "distress", "fear", "urgency", "panic", "confusion"}
VALID_RISK = {"low", "mid", "high"}
VALID_VRISK = {"low", "high"}
VALID_ANOM = {"normal", "anomaly", "borderline"}


# ------------------------- token counter (thread-safe) -------------------------
class TokenCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt_tok, completion_tok):
        with self._lock:
            self.prompt_tokens += prompt_tok
            self.completion_tokens += completion_tok
            self.calls += 1

    def summary(self):
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}


# ------------------------- LLM adapter (fill your provider) -------------------------
# Each thread gets its own client instance to avoid sharing state.
_openai_clients = {}
_anthropic_clients = {}
_client_lock = threading.Lock()


def _get_openai_client():
    tid = threading.current_thread().ident
    with _client_lock:
        if tid not in _openai_clients:
            from openai import OpenAI
            _openai_clients[tid] = OpenAI()
    return _openai_clients[tid]


def _get_anthropic_client():
    tid = threading.current_thread().ident
    with _client_lock:
        if tid not in _anthropic_clients:
            import anthropic
            _anthropic_clients[tid] = anthropic.Anthropic()
    return _anthropic_clients[tid]


def call_llm(system, user, model, provider, temperature=0.9, retries=6):
    """Returns (text, prompt_tokens, completion_tokens)."""
    last = None
    for k in range(retries):
        try:
            if provider == "openai":
                cli = _get_openai_client()
                r = cli.chat.completions.create(
                    model=model, temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                usage = r.usage
                return r.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens
            elif provider == "anthropic":
                cli = _get_anthropic_client()
                r = cli.messages.create(
                    model=model, max_tokens=2000, temperature=temperature,
                    system=system, messages=[{"role": "user", "content": user}])
                return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
            else:
                raise ValueError(f"unknown provider {provider}")
        except Exception as e:
            last = e
            # exponential backoff: 2, 4, 8, 16, 32, 64 seconds
            wait = min(2 ** (k + 1), 64)
            if k < retries - 1:
                print(f"  [retry {k+1}/{retries}] {type(e).__name__}: {e} — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} tries: {type(last).__name__}: {last}")


def parse_json(text):
    """Extract the first JSON object/array from a model reply (tolerant of stray text)."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return json.loads(m.group(1) if m else text)


# ------------------------------ generation pass ------------------------------
def _process_seed(seed, model, provider, counter):
    """Process a single seed: build prompt, call LLM, return list of records."""
    sysm, usr = build_gen_prompt(seed)
    reply, in_tok, out_tok = call_llm(sysm, usr, model, provider)
    counter.add(in_tok, out_tok)
    utts = parse_json(reply).get("utterances", [])
    records = []
    for u in utts:
        emo = str(u.get("emotion", "")).lower().strip()
        cr = str(u.get("content_risk", "")).lower().strip()
        txt = str(u.get("text", "")).strip()
        if not txt or emo not in VALID_EMO or cr not in VALID_RISK:
            continue
        records.append({
            "seed_id": seed["seed_id"], "profile": seed["profile"],
            "event": seed["event"], "target_emotion": seed["target_emotion"],
            "content_risk_seed": seed["content_risk"], "text": txt,
            "gen_emotion": emo, "gen_content_risk": cr})
    return records


def generate(seeds, out_utts, model, provider, workers=1):
    done = set()
    if os.path.exists(out_utts):
        with open(out_utts, encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line)["seed_id"])
                except Exception: pass
    pending = [s for s in seeds if s["seed_id"] not in done]
    if not pending:
        print("[gen] all seeds already done, skipping")
        return sum(1 for _ in open(out_utts, encoding="utf-8")), TokenCounter()

    uid = sum(1 for _ in open(out_utts, encoding="utf-8")) if os.path.exists(out_utts) else 0
    fout = open(out_utts, "a", encoding="utf-8")
    write_lock = threading.Lock()
    ok = bad = 0
    counter = TokenCounter()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(pending), desc="[gen]", unit="seed", dynamic_ncols=True)
    except ImportError:
        pbar = None

    def _handle(seed):
        nonlocal uid, ok, bad
        try:
            records = _process_seed(seed, model, provider, counter)
            with write_lock:
                for rec in records:
                    rec["uid"] = uid
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    uid += 1
                fout.flush()
                if records:
                    ok += 1
        except Exception as e:
            with write_lock:
                bad += 1
            msg = f"  ! seed {seed['seed_id']} failed: {type(e).__name__}: {e}"
            tqdm.write(msg) if pbar else print(msg, flush=True)
        finally:
            if pbar:
                pbar.update(1)

    if workers <= 1:
        for seed in pending:
            _handle(seed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_handle, s): s for s in pending}
            for f in as_completed(futures):
                pass  # _handle writes results via side effect

    if pbar:
        pbar.close()
    fout.close()
    tok = counter.summary()
    print(f"[gen] done · seeds ok {ok} · utterances {uid} · fail {bad}", flush=True)
    print(f"[gen] tokens · in {tok['prompt_tokens']:,} · out {tok['completion_tokens']:,} · total {tok['total_tokens']:,} ({tok['calls']} calls)", flush=True)
    return uid, counter


# -------------------------------- judge pass --------------------------------
def judge(utts_path, out_path, model, provider, batch=40):
    rows = [json.loads(l) for l in open(utts_path, encoding="utf-8")]
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try: done.add(json.loads(l)["uid"])
            except Exception: pass
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        print("[judge] all utterances already labeled, skipping")
        return TokenCounter()

    fout = open(out_path, "a", encoding="utf-8")
    n_batches = (len(todo) + batch - 1) // batch
    counter = TokenCounter()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_batches, desc="[judge]", unit="batch", dynamic_ncols=True)
    except ImportError:
        pbar = None

    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        items = [{"id": r["uid"], "text": r["text"], "emotion": r["gen_emotion"]} for r in chunk]
        try:
            sysm, usr = build_judge_prompt(items)
            reply, in_tok, out_tok = call_llm(sysm, usr, model, provider, temperature=0.0)
            counter.add(in_tok, out_tok)
            labels = {int(x["id"]): x for x in parse_json(reply)}
        except Exception as e:
            msg = f"  ! judge batch {i} failed: {type(e).__name__}: {e}"
            tqdm.write(msg) if pbar else print(msg, flush=True)
            if pbar: pbar.update(1)
            continue
        for r in chunk:
            lb = labels.get(r["uid"], {})
            an = str(lb.get("anomaly", "")).lower().strip()
            if an not in VALID_ANOM:
                continue
            r.update({
                "judge_content_risk": str(lb.get("content_risk", "")).lower().strip(),
                "judge_voice_risk": str(lb.get("voice_risk", "")).lower().strip(),
                "anomaly": an, "reason": str(lb.get("reason", ""))[:60]})
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        if pbar:
            pbar.update(1)
        else:
            print(f"[judge] {min(i+batch,len(todo))}/{len(todo)}", flush=True)

    if pbar:
        pbar.close()
    fout.close()
    tok = counter.summary()
    print(f"[judge] tokens · in {tok['prompt_tokens']:,} · out {tok['completion_tokens']:,} · total {tok['total_tokens']:,} ({tok['calls']} calls)", flush=True)
    return counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="seeds.jsonl")
    ap.add_argument("--utts", default="utterances.jsonl", help="intermediate (generation only)")
    ap.add_argument("--out", default="dataset.jsonl", help="final (generation + judge)")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default="mimo-v2.5")
    ap.add_argument("--limit", type=int, default=0, help="0 = all seeds")
    ap.add_argument("--workers", type=int, default=3, help="concurrent LLM calls for generation")
    ap.add_argument("--judge-batch", type=int, default=40)
    ap.add_argument("--stage", default="all", choices=["gen", "judge", "all"])
    a = ap.parse_args()
    seeds = [json.loads(l) for l in open(a.seeds, encoding="utf-8")]
    if a.limit:
        seeds = seeds[:a.limit]

    gen_counter = TokenCounter()
    judge_counter = TokenCounter()

    if a.stage in ("gen", "all"):
        _, gen_counter = generate(seeds, a.utts, a.model, a.provider, workers=a.workers)
    if a.stage in ("judge", "all"):
        judge_counter = judge(a.utts, a.out, a.model, a.provider, a.judge_batch)
        n = sum(1 for _ in open(a.out, encoding="utf-8")) if os.path.exists(a.out) else 0
        print(f"[done] dataset -> {a.out} ({n} labeled utterances)")

    # combined summary
    gt = gen_counter.summary()
    jt = judge_counter.summary()
    print(f"\n{'='*50}")
    print(f"TOTAL TOKEN USAGE")
    print(f"  gen   · {gt['calls']:>4d} calls · {gt['prompt_tokens']:>10,} in · {gt['completion_tokens']:>10,} out · {gt['total_tokens']:>10,} total")
    print(f"  judge · {jt['calls']:>4d} calls · {jt['prompt_tokens']:>10,} in · {jt['completion_tokens']:>10,} out · {jt['total_tokens']:>10,} total")
    print(f"  ALL   · {gt['calls']+jt['calls']:>4d} calls · {gt['prompt_tokens']+jt['prompt_tokens']:>10,} in · {gt['completion_tokens']+jt['completion_tokens']:>10,} out · {gt['total_tokens']+jt['total_tokens']:>10,} total")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
