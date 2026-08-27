# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile==0.14.0", "datasets==5.0.0"]
# ///

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""
ASR -- 300h retrain, packed-cache builder.

Converts manifest_combined.jsonl (produced by prepare_data.py) into the SAME
packed int16 binary + meta.json cache format ablation_engine.py's `prepare()`
uses (audio.i16 + offsets + texts), so the training script's Dataset class
can stay close to SpeechDS in ablation_engine.py.

Design decision (not specified verbatim in the task, made here for
consistency with the existing codebase): LibriSpeech and AMI rows carry an
`hf_index` into their HF dataset split rather than a standalone audio file
(that's how ablation_engine.py and the AMI loader consume them -- parquet-backed,
audio decoded from in-memory arrow arrays, no extracted wav files on disk).
Common Voice and VCTK rows carry a real `audio_path` on disk. This script
handles both: it re-opens the relevant HF dataset split once per corpus and
indexes into it for hf_index rows, and reads directly from disk for
audio_path rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def log(*a):
    print(*a, flush=True)


def _decode_soundfile(path: str, sr_target: int) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32")
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if int(sr) != sr_target:
        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr_target / sr)),
                      np.arange(len(w)), w).astype(np.float32)
    return w


def _decode_hf_cell(cell, sr_target: int) -> np.ndarray:
    import soundfile as sf

    if isinstance(cell, dict) and cell.get("array") is not None:
        w, sr = np.asarray(cell["array"], np.float32), cell.get("sampling_rate", sr_target)
    elif isinstance(cell, dict) and cell.get("bytes"):
        w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    elif isinstance(cell, dict) and cell.get("path"):
        w, sr = sf.read(cell["path"], dtype="float32")
    else:
        raise ValueError(f"unrecognised audio cell: {type(cell)}")
    w = np.asarray(w, np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if int(sr) != sr_target:
        w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr_target / sr)),
                      np.arange(len(w)), w).astype(np.float32)
    return w


_HF_SPLIT_CACHE = {}


def _hf_split(corpus: str, source: str):
    key = (corpus, source)
    if key in _HF_SPLIT_CACHE:
        return _HF_SPLIT_CACHE[key]
    from datasets import load_dataset

    if corpus == "librispeech":
        # Should be unreachable: prepare_data.py streams LibriSpeech and stages
        # the encoded bytes to disk, so its rows carry `audio_path`. Naming the
        # split does NOT limit the download -- a run was observed generating
        # train.360 (104,014 examples) despite split="train.100" -- so silently
        # falling back to load_dataset here would re-download hundreds of hours.
        raise ValueError(
            "LibriSpeech rows must carry `audio_path` (streamed bytes), not "
            "`hf_index`. Rebuild the manifest with `--corpus librispeech`."
        )
    elif corpus == "ami":
        # AMI no longer reaches this path. prepare_data.py streams AMI and
        # writes FLAC to disk, so its rows carry a real `audio_path` and are
        # read directly. Reaching here means a stale manifest built by the old
        # random-access code is being fed to a new cache builder -- that
        # mismatch would silently re-download ~160h of AMI, so fail instead.
        raise ValueError(
            "AMI rows must carry `audio_path` (streamed FLAC), not `hf_index`. "
            "This manifest was built by an older prepare_data.py -- rebuild it "
            "with `--corpus ami` before running build_cache."
        )
    else:
        raise ValueError(f"no hf_index loader for corpus={corpus}")
    _HF_SPLIT_CACHE[key] = ds
    return ds



def _stratified_subset(rows: list[dict], target_hours: float, seed: int = 1337) -> list[dict]:
    """Take `target_hours` while PRESERVING the corpus mix.

    Why not just slice the first N rows: the combined manifest is written corpus
    by corpus, so `rows[:N]` is pure LibriSpeech. A WS-layer ablation run on
    100% clean read speech would answer the wrong question entirely -- the whole
    reason for the probe is that accent, spontaneous speech and noise might move
    the optimal layers, and none of those appear in the first slice.

    Each corpus contributes the same PROPORTION it has in the full 300h mix, so
    a 50h probe cache is a scale model of the real training set.
    """
    import random as _random

    by_corpus: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_corpus[r["corpus"]].append(r)
    total_s = sum(r["duration_s"] for r in rows)
    rng = _random.Random(seed)

    out: list[dict] = []
    for corpus, crows in sorted(by_corpus.items()):
        corpus_s = sum(r["duration_s"] for r in crows)
        share = corpus_s / total_s if total_s else 0.0
        want_s = target_hours * 3600.0 * share
        rng.shuffle(crows)
        got = 0.0
        for r in crows:
            if got >= want_s:
                break
            out.append(r)
            got += r["duration_s"]
        log(f"  [subset] {corpus:14s} {got / 3600.0:6.2f}h "
            f"({share:.1%} of the mix, {len([x for x in out if x['corpus'] == corpus]):,} rows)")
    rng.shuffle(out)
    log(f"  [subset] TOTAL {sum(r['duration_s'] for r in out) / 3600.0:.2f}h "
        f"/ {len(out):,} rows (target {target_hours:.0f}h)")
    return out


def build_cache(manifest_path: Path, cache_dir: Path, sr: int = 16000,
                 max_rows: int | None = None, hours: float | None = None) -> Path:
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if max_rows:
        rows = rows[:max_rows]
    if hours:
        rows = _stratified_subset(rows, hours, seed=1337)

    key = hashlib.md5(f"{manifest_path}|{len(rows)}|{sr}|{hours}".encode()).hexdigest()[:12]
    tag = f"{hours:.0f}h" if hours else "full"
    d = cache_dir / f"combined_{tag}_{key}"
    bin_p, meta_p = d / "audio.i16", d / "meta.json"
    if bin_p.exists() and meta_p.exists():
        log(f"[cache] found existing cache at {d}")
        return d

    d.mkdir(parents=True, exist_ok=True)
    t0, offs, texts, corpora, pos = time.perf_counter(), [0], [], [], 0
    with open(bin_p, "wb") as f:
        for i, row in enumerate(rows):
            try:
                if row.get("audio_path") and Path(row["audio_path"]).exists():
                    w = _decode_soundfile(row["audio_path"], sr)
                else:
                    ds = _hf_split(row["corpus"], row["source"])
                    cell = ds[row["hf_index"]]["audio"]
                    w = _decode_hf_cell(cell, sr)
            except Exception as e:
                log(f"[cache] WARN skip row {i} ({row.get('corpus')}): {type(e).__name__}: {e}")
                continue
            q = np.clip(np.rint(w * 32768.0), -32768, 32767).astype(np.int16)
            f.write(q.tobytes())
            pos += q.size
            offs.append(pos)
            texts.append(row["text"])
            corpora.append(row["corpus"])
            if (i + 1) % 2000 == 0:
                log(f"  [cache] {i + 1}/{len(rows)} ({pos * 2 / 1e9:.1f} GB, "
                    f"{time.perf_counter() - t0:.0f}s)")

    meta_p.write_text(json.dumps({"offsets": offs, "texts": texts, "corpora": corpora}))
    total_h = (offs[-1]) / sr / 3600.0
    log(f"[cache] {len(texts)} utterances, {total_h:.2f}h -> {d} "
        f"({(time.perf_counter() - t0) / 60:.1f} min)")
    return d


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/marimo/data/manifest_combined.jsonl")
    ap.add_argument("--cache", default="/marimo/cache")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None,
                    help="build a corpus-stratified subset of this many hours "
                         "(e.g. 50 for the WS ablation probe) instead of the full mix")
    args = ap.parse_args()
    build_cache(Path(args.manifest), Path(args.cache),
                max_rows=args.max_rows, hours=args.hours)
