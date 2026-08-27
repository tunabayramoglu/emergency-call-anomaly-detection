# /// script
# requires-python = ">=3.11"
# dependencies = ["soundfile==0.14.0"]
# ///

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""Pre-flight check: is the 300h training data actually complete and usable?

Run this AFTER prepare_data.py / build_cache.py and BEFORE starting an
8-hour training run. Every failure mode it looks for is one that has already
bitten this pipeline at least once:

  * a corpus that silently produced 0 rows (Common Voice kept 0.00h because the
    audio lived inside tar shards nobody opened)
  * a manifest whose `audio_path` entries point at files that are not there
  * an hours total that quietly falls short of the 300h the slides claim
  * rows that violate the CTC length constraint -- `duration_s * 50 < 2 * len(text)`
    produces inf/nan loss and poisons training, and AMI genuinely contains
    0.02-second clips
  * out-of-vocabulary characters surviving normalisation, which would break
    KenLM decoding
  * L2-ARCTIC leaking into a training manifest (it is the held-out OOD test set)
  * a packed cache whose hour count does not match its manifest
  * empty noise/RIR banks, which make augmentation a silent no-op

Exit codes: 0 = ready to train, 2 = at least one FAIL.

Usage:
    python verify_data.py --data /marimo/data --cache /marimo/cache \\
        --noise-dir /marimo/noise --rir-dir /marimo/rir
    python verify_data.py --data ... --check-all-audio     # stat every file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
ALLOWED = CHARS | {" "}

# Backbone frame rate: 16 kHz audio -> 50 frames/sec. CTC needs roughly two
# frames per target character (blanks between repeats), so a row is infeasible
# when duration_s * 50 < 2 * len(text).
FRAMES_PER_SEC = 50.0
EXPECTED = {"librispeech": 100.0, "common_voice": 106.0, "ami": 50.0, "vctk": 44.0}
TOTAL_TARGET = 300.0
L2_MARKERS = ("l2-arctic", "l2_arctic", "l2arctic")

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    return ok


def load_manifest(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def verify_manifests(data_dir: Path, check_all_audio: bool, sample_n: int) -> dict:
    print("\n1. Per-corpus manifests")
    per_corpus_rows: dict[str, list[dict]] = {}
    for corpus, want_h in EXPECTED.items():
        p = data_dir / f"manifest_{corpus}.jsonl"
        if not p.exists():
            check(f"{corpus}: manifest exists", False, f"missing {p.name}")
            continue
        rows = load_manifest(p)
        per_corpus_rows[corpus] = rows
        h = sum(r["duration_s"] for r in rows) / 3600.0
        # 10% short is a real shortfall, not rounding
        check(f"{corpus}: {h:.2f}h / {want_h:.0f}h target, {len(rows):,} rows",
              len(rows) > 0 and h >= want_h * 0.9,
              "" if h >= want_h * 0.9 else "more than 10% short")
    return per_corpus_rows


def verify_combined(data_dir: Path, per_corpus: dict) -> list[dict]:
    print("\n2. Combined manifest")
    p = data_dir / "manifest_combined.jsonl"
    if not p.exists():
        check("combined manifest exists", False, f"missing {p.name} — run --combine")
        return []
    rows = load_manifest(p)
    total_h = sum(r["duration_s"] for r in rows) / 3600.0
    check(f"combined total {total_h:.2f}h / {TOTAL_TARGET:.0f}h, {len(rows):,} rows",
          total_h >= TOTAL_TARGET * 0.9,
          "" if total_h >= TOTAL_TARGET * 0.9 else "more than 10% short")

    by = Counter(r["corpus"] for r in rows)
    missing = [c for c in EXPECTED if by.get(c, 0) == 0]
    check("all four corpora present in combined", not missing,
          f"absent: {missing}" if missing else f"{dict(by)}")

    # combine() applies a trainability gate (duration in [0.2, 30] s and CTC
    # feasibility), so the combined manifest is deliberately SMALLER than the sum
    # of its parts. Asserting equality here was my own bug: it contradicted a
    # filter I had just added. What actually matters is that combined never
    # EXCEEDS the parts, and that the gate removed a small fraction rather than
    # silently eating the dataset.
    sum_parts = sum(len(v) for v in per_corpus.values())
    dropped = sum_parts - len(rows)
    frac = dropped / sum_parts if sum_parts else 0.0
    check("combined row count <= sum of per-corpus", len(rows) <= sum_parts,
          f"combined {len(rows):,} EXCEEDS parts {sum_parts:,} — duplicated rows?"
          if len(rows) > sum_parts else "")
    check(f"trainability gate dropped {dropped:,} rows ({frac:.3%})", frac < 0.01,
          f"parts {sum_parts:,} -> combined {len(rows):,}"
          + ("  — over 1% is too much, check the combine log for the reason"
             if frac >= 0.01 else ""))
    return rows


def verify_rows(rows: list[dict], check_all_audio: bool, sample_n: int) -> None:
    if not rows:
        return

    print("\n3. L2-ARCTIC leakage gate")
    leaks = [r for r in rows
             if any(m in " ".join(str(r.get(k, "")) for k in
                                  ("corpus", "source", "audio_path", "speaker")).lower()
                    for m in L2_MARKERS)]
    check("no L2-ARCTIC rows in training data", not leaks,
          f"{len(leaks)} leaked rows!" if leaks else "held-out test set is clean")

    print("\n4. Text / vocabulary")
    oov, empty = [], 0
    for r in rows:
        t = r.get("text") or ""
        if not t:
            empty += 1
            continue
        bad = set(t) - ALLOWED
        if bad:
            oov.append((r["corpus"], sorted(bad)[:5], t[:40]))
    check("no empty transcripts", empty == 0, f"{empty} empty" if empty else "")
    check("no out-of-vocabulary characters", not oov,
          f"{len(oov)} rows, e.g. {oov[:3]}" if oov else "all rows fit A-Z + apostrophe")

    print("\n5. CTC feasibility (duration_s * 50 >= 2 * len(text))")
    infeasible = [r for r in rows
                  if r["duration_s"] * FRAMES_PER_SEC < 2 * len(r.get("text", ""))]
    by_corpus = Counter(r["corpus"] for r in infeasible)
    check("no rows would produce inf/nan CTC loss", not infeasible,
          f"{len(infeasible)} infeasible rows {dict(by_corpus)}" if infeasible
          else "every row has enough frames for its transcript")

    print("\n6. Duration sanity")
    durs = [r["duration_s"] for r in rows]
    tiny = sum(1 for d in durs if d < 0.2)
    huge = sum(1 for d in durs if d > 30.0)
    check("no sub-0.2s clips", tiny == 0, f"{tiny} clips shorter than 0.2s" if tiny else "")
    check("no clips over 30s", huge == 0, f"{huge} clips longer than 30s" if huge else "")
    print(f"        min {min(durs):.2f}s  median {sorted(durs)[len(durs)//2]:.2f}s  "
          f"max {max(durs):.2f}s")

    print("\n7. Audio files on disk")
    paths = [r.get("audio_path") for r in rows]
    dupes = len(paths) - len(set(paths))
    check("no duplicate audio_path", dupes == 0, f"{dupes} duplicates" if dupes else "")
    to_check = rows if check_all_audio else random.Random(0).sample(
        rows, min(sample_n, len(rows)))
    missing = [r["audio_path"] for r in to_check
               if not r.get("audio_path") or not Path(r["audio_path"]).exists()]
    label = f"all {len(to_check):,}" if check_all_audio else f"{len(to_check):,} sampled"
    check(f"audio present on disk ({label})", not missing,
          f"{len(missing)} missing, e.g. {missing[:2]}" if missing
          else "every checked file exists")

    print("\n8. Per-corpus breakdown")
    agg = defaultdict(lambda: [0, 0.0])
    for r in rows:
        agg[r["corpus"]][0] += 1
        agg[r["corpus"]][1] += r["duration_s"]
    for c, (n, s) in sorted(agg.items()):
        print(f"        {c:14s} {n:>8,} rows  {s / 3600.0:7.2f}h  "
              f"mean {s / max(n, 1):.2f}s")

    accents = Counter()
    for r in rows:
        for a in r.get("accents", []) or []:
            accents[a] += 1
    if accents:
        print(f"\n        Common Voice accents kept: {len(accents)} distinct")
        for a, n in accents.most_common(8):
            print(f"          {a[:52]:54s} {n:>7,}")


def verify_cache(cache_dir: Path, rows: list[dict]) -> None:
    print("\n9. Packed cache")
    caches = sorted(cache_dir.glob("combined_*"))
    if not caches:
        check("packed cache exists", False, "no combined_* dir — run build_cache.py")
        return
    d = caches[-1]
    bin_p, meta_p = d / "audio.i16", d / "meta.json"
    if not (bin_p.exists() and meta_p.exists()):
        check("cache files present", False, f"{d.name} missing audio.i16 or meta.json")
        return
    meta = json.loads(meta_p.read_text())
    n_utt = len(meta.get("texts", []))
    cache_h = meta["offsets"][-1] / 16000 / 3600.0
    size_gb = bin_p.stat().st_size / 1e9
    check(f"cache: {n_utt:,} utterances, {cache_h:.2f}h, {size_gb:.1f} GB",
          n_utt > 0 and cache_h > 0)
    if rows:
        # build_cache skips rows it cannot decode; a large gap means many were lost
        keep = n_utt / len(rows)
        check("cache kept >=95% of manifest rows", keep >= 0.95,
              f"only {keep:.1%} of {len(rows):,} manifest rows made it into the cache")
        expect_bytes = meta["offsets"][-1] * 2
        check("cache byte size matches offsets", abs(expect_bytes - bin_p.stat().st_size) < 1024,
              f"offsets imply {expect_bytes:,} B, file is {bin_p.stat().st_size:,} B")


def verify_banks(noise_dir: Path | None, rir_dir: Path | None) -> None:
    print("\n10. Augmentation banks")
    for label, d in (("noise", noise_dir), ("rir", rir_dir)):
        if d is None:
            continue
        n = sum(1 for _ in d.rglob("*.wav")) if d.exists() else 0
        check(f"{label} bank populated ({n:,} wavs)", n > 0,
              "empty -> this augmentation is a SILENT no-op" if n == 0 else "")


def verify_kenlm(lm_dir: Path | None) -> None:
    """The LM is not needed to TRAIN, but without it the +KenLM column of the
    results table is empty -- and that column carries the headline 10.1 -> 5.1
    WER improvement. Flagged here so it is noticed before the run, not after."""
    if lm_dir is None:
        return
    print("\n11. KenLM language model")
    arpa = lm_dir / "3-gram.pruned.1e-7.arpa"
    if not arpa.exists():
        check("KenLM ARPA present", False,
              f"missing {arpa} -- evaluation will be greedy-only")
        return
    # Structural check, NOT a size guess. An earlier version demanded >500 MB and
    # was wrong -- 3-gram.pruned.1e-7 is ~98 MB decompressed, so the threshold
    # rejected a perfectly good model. Every ARPA ends with `\\end\\`; a
    # truncated download still gunzips and still has a valid header, but cannot
    # have the terminator.
    size = arpa.stat().st_size
    head_ok = tail_ok = False
    try:
        with open(arpa, "rb") as fh:
            head = fh.read(4096).decode("utf-8", errors="replace")
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
        head_ok = "\\data\\" in head or "ngram 1=" in head
        tail_ok = "\\end\\" in tail
    except Exception:
        pass
    detail = ("" if (head_ok and tail_ok) else
              ("no ARPA header" if not head_ok else
               "no \\end\\ terminator -- truncated download"))
    check(f"KenLM ARPA present and complete ({size / 1e6:.0f} MB)",
          size >= 10 * 1024 * 1024 and head_ok and tail_ok, detail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--rir-dir", default=None)
    ap.add_argument("--lm-dir", default=None)
    ap.add_argument("--check-all-audio", action="store_true",
                    help="stat every audio file instead of a random sample "
                         "(slower, but proves nothing is missing)")
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    print("=" * 66)
    print("ASR — training data verification")
    print("=" * 66)

    data_dir = Path(args.data)
    per_corpus = verify_manifests(data_dir, args.check_all_audio, args.sample)
    rows = verify_combined(data_dir, per_corpus)
    verify_rows(rows, args.check_all_audio, args.sample)
    if args.cache:
        verify_cache(Path(args.cache), rows)
    verify_banks(Path(args.noise_dir) if args.noise_dir else None,
                 Path(args.rir_dir) if args.rir_dir else None)
    verify_kenlm(Path(args.lm_dir) if args.lm_dir else None)

    fails = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 66)
    if fails:
        print(f"NOT READY — {len(fails)} check(s) failed:")
        for f in fails:
            print(f"  - {f}")
        print("Fix these before starting an 8-hour run.")
    else:
        print(f"READY — all {len(_results)} checks passed. Safe to start training.")
    print("=" * 66)
    sys.exit(2 if fails else 0)


if __name__ == "__main__":
    main()
