# /// script
# requires-python = ">=3.11"
# dependencies = []   # stdlib only
# ///

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""Fetch the KenLM language model used for CTC beam-search decoding.

This is the SAME model the 100h baseline used, so the 300h numbers stay
comparable: LibriSpeech **3-gram.pruned.1e-7** from OpenSLR-11. `kenlm_grid.py`
hardcodes that exact URL, and this script reuses it rather than picking a
"better" LM -- swapping the LM would change the decoder underneath both rows of
the results table and make the 100h vs 300h comparison meaningless.

The 4-gram from the same resource is deliberately NOT fetched. It is several
times larger, and the project already decided against it.

WHY THE LM MUST MATCH THE ACOUSTIC VOCABULARY
---------------------------------------------
This LM was built on LibriSpeech-normalised text: A-Z, apostrophe, no digits, no
punctuation. That is exactly why `prepare_data.normalize_text` expands digits to
words and drops rows with out-of-vocabulary characters instead of extending the
CTC output vocabulary. Emitting a character the LM has never seen collapses the
beam search and throws away the 10.1 -> 5.1 WER gain that KenLM provides.

Usage:
    python fetch_kenlm.py --lm-dir /marimo/lm
    python fetch_kenlm.py --lm-dir /marimo/lm --verify-only
    python fetch_kenlm.py --lm-dir /marimo/lm --jobs 8
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Same file kenlm_grid.py uses. OpenSLR mirrors: the primary host is often slow
# and its us. subdomain has had a certificate hostname mismatch, so try in order.
LM_NAME = "3-gram.pruned.1e-7.arpa"
LM_URLS = [
    f"https://us.openslr.org/resources/11/{LM_NAME}.gz",
    f"https://www.openslr.org/resources/11/{LM_NAME}.gz",
    f"https://openslr.elda.org/resources/11/{LM_NAME}.gz",
]
# Sanity FLOOR only -- deliberately low. An earlier version guessed 500 MB and
# was simply wrong: 3-gram.pruned.1e-7 decompresses to about 98 MB (it is a
# PRUNED model, ~200k unigrams / ~2.45M bigrams), so the guess deleted a
# perfectly good file and re-downloaded it. Size is a bad completeness test.
MIN_ARPA_BYTES = 10 * 1024 * 1024


def arpa_is_complete(path: Path) -> tuple[bool, str]:
    """Structural check instead of a size guess.

    Every ARPA file ends with the literal line `\end\`. A download truncated
    mid-stream still gunzips and still has a valid-looking header, so the header
    alone proves nothing -- but it cannot have the terminator. Reading the last
    kilobyte is O(1) and is the actual definition of "complete".
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat: {exc}"
    if size < MIN_ARPA_BYTES:
        return False, f"only {size / 1e6:.0f} MB -- far too small even for a pruned model"

    with open(path, "rb") as fh:
        head = fh.read(4096).decode("utf-8", errors="replace")
        fh.seek(max(0, size - 4096))
        tail = fh.read().decode("utf-8", errors="replace")

    if "\\data\\" not in head and "ngram 1=" not in head:
        return False, "no ARPA header (\\data\\ / ngram 1=) in the first 4 KB"
    if "\\end\\" not in tail:
        return False, ("no \\end\\ terminator in the last 4 KB -- the download was "
                       "truncated mid-file")
    ngrams = [l.strip() for l in head.splitlines() if l.strip().startswith("ngram ")]
    return True, f"{size / 1e6:.0f} MB, {', '.join(ngrams[:3])}, \\end\\ present"


def log(*a):
    print(*a, flush=True)


def _supports_ranges(url: str, timeout: int = 30) -> tuple[bool, int]:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "clear-asr/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            return (r.headers.get("Accept-Ranges", "").lower() == "bytes" and size > 0), size
    except Exception:
        return False, 0


def _download_ranges(url: str, dest: Path, size: int, jobs: int) -> None:
    chunk = size // jobs
    spans = [(i * chunk, (size - 1) if i == jobs - 1 else (i + 1) * chunk - 1)
             for i in range(jobs)]
    with open(dest, "wb") as fh:
        fh.truncate(size)
    done, lock, t0 = 0, threading.Lock(), time.time()

    def worker(span):
        nonlocal done
        start, end = span
        req = urllib.request.Request(
            url, headers={"User-Agent": "clear-asr/1.0", "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "r+b") as fh:
            fh.seek(start)
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                with lock:
                    done += len(buf)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(worker, sp) for sp in spans]
        last = 0.0
        while any(not f.done() for f in futures):
            time.sleep(0.25)
            now = time.time()
            if now - last < 5.0:
                continue
            last = now
            with lock:
                d = done
            el = max(now - t0, 1e-6)
            log(f"  [lm] {d / 1e6:.0f}/{size / 1e6:.0f} MB ({100 * d / size:.0f}%) "
                f"@ {d / 1e6 / el:.0f} MB/s, {jobs} streams")
        for f in futures:
            f.result()
    if dest.stat().st_size != size:
        raise RuntimeError(f"size mismatch: {dest.stat().st_size:,} != {size:,}")


def download_lm_archive(dest_gz: Path, jobs: int) -> None:
    if dest_gz.exists() and dest_gz.stat().st_size > 0:
        log(f"  [lm] reusing existing {dest_gz.name} ({dest_gz.stat().st_size / 1e6:.0f} MB)")
        return
    if shutil.which("aria2c"):
        log(f"  [lm] aria2c found -- {jobs} connections")
        for url in LM_URLS:
            r = subprocess.run(["aria2c", "-x", str(jobs), "-s", str(jobs), "-k", "10M",
                                "--console-log-level=warn", "--auto-file-renaming=false",
                                "--allow-overwrite=true", "-d", str(dest_gz.parent),
                                "-o", dest_gz.name, url])
            if r.returncode == 0 and dest_gz.exists():
                return
    last = None
    for url in LM_URLS:
        log(f"  trying {url}")
        try:
            ok, size = _supports_ranges(url)
            if ok and jobs > 1:
                log(f"  [lm] range support, {size / 1e6:.0f} MB -> {jobs} parallel streams")
                _download_ranges(url, dest_gz, size, jobs)
            else:
                log("  [lm] no range support -- single stream")
                req = urllib.request.Request(url, headers={"User-Agent": "clear-asr/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dest_gz, "wb") as fh:
                    shutil.copyfileobj(r, fh, length=1 << 20)
            return
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
            dest_gz.unlink(missing_ok=True)
    raise RuntimeError(f"all mirrors failed; last error: {last}")


def fetch_lm(lm_dir: Path, jobs: int = 8) -> Path:
    lm_dir.mkdir(parents=True, exist_ok=True)
    arpa = lm_dir / LM_NAME
    if arpa.exists():
        ok, why = arpa_is_complete(arpa)
        if ok:
            log(f"[lm] already present and complete: {arpa}")
            log(f"[lm]   {why}")
            return arpa
        log(f"[lm] {arpa.name} is unusable ({why}) -- re-downloading")
        arpa.unlink()

    gz = lm_dir / f"{LM_NAME}.gz"
    download_lm_archive(gz, jobs)

    log(f"[lm] decompressing {gz.name} -> {arpa.name}")
    t0 = time.time()
    with gzip.open(gz, "rb") as f_in, open(arpa, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1 << 22)
    gz.unlink(missing_ok=True)
    log(f"[lm] ready: {arpa} ({arpa.stat().st_size / 1e9:.2f} GB, "
        f"{time.time() - t0:.0f}s to decompress)")
    return arpa


def verify(lm_dir: Path) -> bool:
    arpa = lm_dir / LM_NAME
    log("")
    log("KenLM status")
    log("=" * 30)
    if not arpa.exists():
        log(f"  {arpa} -> MISSING")
        log("  NOT READY — evaluation will fall back to greedy decoding only, "
            "and the +KenLM column of the results table will be empty.")
        return False
    log(f"  {arpa}")
    ok, why = arpa_is_complete(arpa)
    log(f"  {why}")
    log("  READY — beam-search decoding with KenLM is available." if ok else
        "  NOT READY — " + why)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    lm_dir = Path(args.lm_dir)
    if not args.verify_only:
        try:
            fetch_lm(lm_dir, jobs=args.jobs)
        except Exception as exc:
            log(f"[lm] FAILED: {type(exc).__name__}: {exc}")
    sys.exit(0 if verify(lm_dir) else 2)


if __name__ == "__main__":
    main()
