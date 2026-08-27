# /// script
# requires-python = ">=3.11"
# dependencies = []   # stdlib only: urllib + tarfile + zipfile
# ///

# GENERATED FILE - do not edit here.
# The authoritative copy is the string literal in asr_300h_marimo.py,
# which writes this file to disk when the notebook runs. An edit made here is
# silently overwritten on the next run; change it in the notebook instead.
"""Download the background-noise and room-impulse-response banks.

WHY THIS FILE EXISTS
--------------------
The notebook created `noise/` and `rir/` directories, wired `--noise-dir` and
`--rir-dir` through to the trainer, and `augment.py` read them — but nothing
ever populated them, and `AudioBank` treats an empty folder as "no bank":

    if not self.rir_bank.empty()   and rng.random() < cfg.p_rir:   ...
    if not self.noise_bank.empty() and rng.random() < cfg.p_noise: ...

So with empty folders the reverb and additive-noise effects are skipped in
silence. Training completes, the log looks healthy, and the two augmentation
axes that matter MOST for the Friday demo (a laptop microphone in a room, not a
studio) were never applied at all.

WHAT IT FETCHES
---------------
  OpenSLR-28  rirs_noises.zip   ~4 GB   simulated + real RIRs, plus pointsource
                                        and isotropic noises. Highest value per
                                        byte: it covers BOTH axes on its own.
  OpenSLR-17  musan.tar.gz      ~11 GB  we keep only `musan/noise/**` (~6 h,
                                        930 files). The archive is monolithic so
                                        the whole thing crosses the wire, but
                                        members are filtered while streaming so
                                        only the noise subset ever hits disk.

MUSAN is therefore OPTIONAL: 11 GB of transfer for ~6 h of extra noise variety,
on top of what OpenSLR-28 already provides. Start with RIR only if bandwidth or
time is tight.

DEMAND is deliberately NOT automated here. It is distributed as per-scene Zenodo
archives and I could not verify the current URLs from this environment; guessing
at them would produce a downloader that fails at 3 a.m. Fetch it by hand into
the noise directory if you want those scenes.

Usage:
    python fetch_noise_banks.py --noise-dir /marimo/noise --rir-dir /marimo/rir
    python fetch_noise_banks.py --noise-dir ... --rir-dir ... --skip-musan
    python fetch_noise_banks.py --noise-dir ... --rir-dir ... --verify-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import shutil as _shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# OpenSLR is mirrored; the primary host is frequently slow, so try in order.
MUSAN_URLS = [
    "https://us.openslr.org/resources/17/musan.tar.gz",
    "https://www.openslr.org/resources/17/musan.tar.gz",
    "https://openslr.elda.org/resources/17/musan.tar.gz",
]
RIR_URLS = [
    "https://us.openslr.org/resources/28/rirs_noises.zip",
    "https://www.openslr.org/resources/28/rirs_noises.zip",
    "https://openslr.elda.org/resources/28/rirs_noises.zip",
]


def log(*a):
    print(*a, flush=True)


def count_wavs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav")) if root.exists() else 0


def _open_stream(urls: list[str], timeout: int = 60):
    """First URL that responds. Returns (response, url)."""
    last = None
    for url in urls:
        try:
            log(f"  trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "clear-asr/1.0"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp, url
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"all mirrors failed; last error: {last}")



# ============================================================================
# Parallel download
# ============================================================================
#
# OpenSLR serves a single stream slowly, and MUSAN is ~11 GB, so a plain
# sequential read is a patience test. Splitting the file into byte ranges and
# fetching them concurrently is usually several times faster, because the
# bottleneck is per-connection throughput rather than total bandwidth.
#
# THE TRADE-OFF, stated plainly: the streaming path never puts the 11 GB tar on
# disk (it filters members as they arrive and keeps only ~6 h of noise wavs).
# Range-parallel downloading REQUIRES the whole archive on disk first, because
# ranges arrive out of order. So this buys wall-clock time at the cost of ~11 GB
# of temporary disk. The tar is deleted straight after extraction.
#
# Not all servers honour Range. We probe first and fall back to the streaming
# path rather than silently downloading the file 8 times or getting garbage.


def _supports_ranges(url: str, timeout: int = 30) -> tuple[bool, int]:
    """(accepts_ranges, content_length). Both are needed to split the work."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "clear-asr/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("Content-Length") or 0)
            accepts = (r.headers.get("Accept-Ranges", "").lower() == "bytes")
            return (accepts and size > 0), size
    except Exception:
        return False, 0


def _download_ranges(url: str, dest: Path, size: int, jobs: int = 8) -> None:
    """Fetch `url` into `dest` using `jobs` concurrent byte-range requests."""
    chunk = size // jobs
    spans = [(i * chunk, (size - 1) if i == jobs - 1 else (i + 1) * chunk - 1)
             for i in range(jobs)]

    with open(dest, "wb") as fh:      # preallocate so each worker can seek
        fh.truncate(size)

    done = 0
    lock = threading.Lock()
    t0 = time.time()

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
        # Poll often, LOG rarely: a coarse sleep would put a hard floor under
        # every download (a 40 MB fetch took 5s purely because of a sleep(5)).
        last_log = 0.0
        while any(not f.done() for f in futures):
            time.sleep(0.25)
            now = time.time()
            if now - last_log < 5.0:
                continue
            last_log = now
            with lock:
                d = done
            el = max(now - t0, 1e-6)
            eta = (size - d) / max(d / el, 1.0)
            log(f"  [dl] {d / 1e9:.2f}/{size / 1e9:.2f} GB ({100 * d / size:.0f}%) "
                f"@ {d / 1e6 / el:.0f} MB/s, {jobs} streams, ETA {eta / 60:.1f} min")
        for f in futures:
            f.result()        # surface any worker exception

    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"size mismatch: got {got:,} B, expected {size:,} B")
    log(f"  [dl] done: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")


def _download_aria2(url: str, dest: Path, jobs: int = 8) -> bool:
    """Use aria2c when it happens to be installed. Returns True on success.

    aria2c is strictly better than our thread pool when present -- it resumes,
    retries per-connection, and handles redirects -- but it is not installed by
    default on molab, so it is opportunistic rather than required.
    """
    if not _shutil.which("aria2c"):
        return False
    log(f"  [dl] aria2c found -- {jobs} connections")
    cmd = ["aria2c", "-x", str(jobs), "-s", str(jobs), "-k", "10M",
           "--console-log-level=warn", "--summary-interval=10",
           "--auto-file-renaming=false", "--allow-overwrite=true",
           "-d", str(dest.parent), "-o", dest.name, url]
    try:
        return subprocess.run(cmd).returncode == 0 and dest.exists()
    except Exception as exc:
        log(f"  [dl] aria2c failed ({exc}) -- falling back")
        return False


def download_file(urls: list[str], dest: Path, jobs: int = 8) -> Path:
    """Download the first working URL to `dest`, in parallel when possible."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  [dl] reusing existing {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    last = None
    for url in urls:
        log(f"  trying {url}")
        try:
            if _download_aria2(url, dest, jobs):
                return dest
            ok, size = _supports_ranges(url)
            if ok and jobs > 1:
                log(f"  [dl] server accepts ranges, {size / 1e9:.2f} GB "
                    f"-> {jobs} parallel streams")
                _download_ranges(url, dest, size, jobs)
                return dest
            log("  [dl] no range support -- single stream")
            resp, _ = _open_stream([url])
            with open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            return dest
        except Exception as exc:
            last = exc
            log(f"    failed: {type(exc).__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"all mirrors failed; last error: {last}")


def fetch_musan_noise(noise_dir: Path, jobs: int = 8) -> int:
    """Stream musan.tar.gz and extract ONLY `musan/noise/**`.

    Streaming ('r|gz') cannot seek, which is exactly what we want over HTTP: we
    walk members in order and write out just the noise subset, so the ~11 GB
    archive never lands on disk in full.
    """
    noise_dir.mkdir(parents=True, exist_ok=True)
    target = noise_dir / "musan_noise"
    if count_wavs(target) > 100:
        log(f"[musan] already populated ({count_wavs(target)} wavs) — skipping")
        return count_wavs(target)

    target.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()

    if jobs > 1:
        # Parallel path: the whole 11 GB archive lands on disk first (ranges
        # arrive out of order, so they cannot be piped through the tar reader),
        # then we extract only musan/noise/** and delete it again.
        tmp = noise_dir / "_musan.tar.gz"
        free_gb = shutil.disk_usage(noise_dir).free / 1e9
        if free_gb < 14:
            log(f"[musan] only {free_gb:.1f} GB free -- need ~12 GB for the parallel "
                "path, falling back to streaming (slower, no temp file)")
        else:
            download_file(MUSAN_URLS, tmp, jobs=jobs)
            log(f"[musan] extracting musan/noise/** from {tmp.name}")
            with tarfile.open(tmp, mode="r:gz") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    if "/noise/" not in name or not name.endswith(".wav"):
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    with open(target / Path(name).name, "wb") as fh:
                        shutil.copyfileobj(src, fh)
                    n += 1
                    if n % 100 == 0:
                        log(f"  [musan] {n} noise wavs kept ({time.time() - t0:.0f}s)")
            tmp.unlink(missing_ok=True)
            log(f"[musan] extracted {n} noise wavs into {target} "
                f"(temp archive deleted)")
            return n

    resp, url = _open_stream(MUSAN_URLS)
    log(f"[musan] streaming from {url} (extracting musan/noise/** only)")
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if "/noise/" not in name or not name.endswith(".wav"):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            dest = target / Path(name).name
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
            if n % 50 == 0:
                log(f"  [musan] {n} noise wavs kept ({time.time() - t0:.0f}s)")
    log(f"[musan] extracted {n} noise wavs into {target}")
    return n


def _classify_slr28(name: str) -> str | None:
    """Is this OpenSLR-28 member a room impulse response, a noise, or neither?

    Matching on DIRECTORY COMPONENTS, not a substring of the whole path. The
    naive `"rir" in path` test is wrong here because the archive's top-level
    directory is literally `RIRS_NOISES`, so every pointsource NOISE file also
    contains "rir" and lands in the RIR bank — which is exactly the bug a
    synthetic-archive test caught: 7 files classified as RIRs, 0 as noise.

    Layout:
        RIRS_NOISES/simulated_rirs/{small,medium,large}room/RoomXXX/*.wav  -> rir
        RIRS_NOISES/real_rirs_isotropic_noises/*.wav                      -> mixed,
            decided by filename ("*rir*" vs "*noise*")
        RIRS_NOISES/pointsource_noises/*.wav                              -> noise
    """
    parts = [p.lower() for p in Path(name).parts]
    stem = Path(name).name.lower()
    if "pointsource_noises" in parts:
        return "noise"
    if "simulated_rirs" in parts:
        return "rir"
    if "real_rirs_isotropic_noises" in parts:
        return "rir" if "rir" in stem else "noise"
    return None


def fetch_rirs(rir_dir: Path, with_pointsource_noise: Path | None = None,
               jobs: int = 8) -> int:
    """Download OpenSLR-28 and extract the RIR wavs.

    If `with_pointsource_noise` is given, the archive's pointsource/isotropic
    noise wavs are extracted there too — free extra noise variety, since the
    bytes have already been paid for.
    """
    rir_dir.mkdir(parents=True, exist_ok=True)
    target = rir_dir / "rirs"
    if count_wavs(target) > 100:
        log(f"[rir] already populated ({count_wavs(target)} wavs) — skipping")
        return count_wavs(target)
    target.mkdir(parents=True, exist_ok=True)

    # zipfile needs random access, so this one has to land on disk first.
    # A zip needs random access, so this one always lands on disk -- which means
    # the parallel downloader costs nothing extra here and is pure win.
    tmp_zip = rir_dir / "_rirs_noises.zip"
    download_file(RIR_URLS, tmp_zip, jobs=jobs)

    n_rir, n_noise = 0, 0
    with zipfile.ZipFile(tmp_zip) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".wav"):
                continue
            kind = _classify_slr28(info.filename)
            if kind == "rir":
                dest, is_rir = target, True
            elif kind == "noise" and with_pointsource_noise is not None:
                dest, is_rir = with_pointsource_noise / "slr28_noise", False
            else:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            # Flatten: AudioBank globs recursively but flat names avoid
            # collisions between the simulated_rirs sub-trees.
            out = dest / (info.filename.replace("/", "_"))
            with z.open(info) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
            if is_rir:
                n_rir += 1
            else:
                n_noise += 1
    log(f"[rir] extracted {n_rir} RIR wavs into {target}"
        + (f", {n_noise} noise wavs alongside" if n_noise else ""))
    tmp_zip.unlink(missing_ok=True)
    return n_rir


def verify(noise_dir: Path, rir_dir: Path) -> bool:
    """Report bank sizes and say plainly whether augmentation will actually run."""
    n_noise, n_rir = count_wavs(noise_dir), count_wavs(rir_dir)
    log("")
    log("Augmentation bank status")
    log("=" * 30)
    log(f"  noise dir : {noise_dir}  ->  {n_noise} wav files")
    log(f"  rir dir   : {rir_dir}  ->  {n_rir} wav files")
    log("")
    if n_noise == 0:
        log("  ⚠ noise bank EMPTY — additive-noise augmentation will be SILENTLY SKIPPED")
    if n_rir == 0:
        log("  ⚠ rir bank EMPTY — reverb augmentation will be SILENTLY SKIPPED")
        log("    (this is the demo-critical one: a laptop mic in a room is reverberant)")
    ok = n_noise > 0 and n_rir > 0
    log("  READY — noise and reverb will both be applied." if ok else
        "  NOT READY — train_asr.py will run, but those effects will do nothing.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--skip-musan", action="store_true",
                    help="skip the 11 GB MUSAN transfer; OpenSLR-28 still gives "
                         "RIRs plus its own pointsource noises")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel download connections (1 = old single-stream "
                         "path, which avoids the ~11 GB temp file for MUSAN)")
    args = ap.parse_args()

    noise_dir, rir_dir = Path(args.noise_dir), Path(args.rir_dir)

    if not args.verify_only:
        try:
            fetch_rirs(rir_dir, with_pointsource_noise=noise_dir, jobs=args.jobs)
        except Exception as exc:
            log(f"[rir] FAILED: {type(exc).__name__}: {exc}")
        if not args.skip_musan:
            try:
                fetch_musan_noise(noise_dir, jobs=args.jobs)
            except Exception as exc:
                log(f"[musan] FAILED: {type(exc).__name__}: {exc}")
        else:
            log("[musan] skipped (--skip-musan)")

    ok = verify(noise_dir, rir_dir)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
