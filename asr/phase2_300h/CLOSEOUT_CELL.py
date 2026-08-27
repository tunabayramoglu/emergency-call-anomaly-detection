# =============================================================================
# PASTE AS A NEW marimo CELL AND RUN IT BEFORE CLOSING THE SESSION.
# (needs: mo, subprocess, Path, json, py_bin, asr_dir, base_dir, runs_dir, data_dir)
#
# WHAT IS AND IS NOT ALREADY SAFE
# -------------------------------
# Safe: training checkpoints (train_asr.py mirrored them every epoch),
# lm_params*.json (tune_lm.py mirrors its own), eval/benchmark json
# (eval_asr.py and bench_all.py mirror theirs).
#
# NOT safe, and this is the one that is easy to miss: the MANIFESTS. They are
# the distilled result of hours of downloading -- LibriSpeech streamed to dodge
# train.360, Common Voice pulled out of tar shards, AMI streamed per meeting
# with a deterministic ihm/sdm assignment, then a global trainability gate.
# Rebuilding them means re-running all of that. They are tens of megabytes and
# they record exactly which 196,620 utterances the reported model was trained
# on, which is also the only thing that makes the run reproducible.
#
# The 35 GB packed cache is deliberately NOT saved: it is a mechanical
# transformation of the manifests plus the audio, so it is rebuildable in
# 10-30 minutes once the audio is back. The manifests are not.
#
# Caveat recorded here so nobody trips on it later: manifest rows carry
# `audio_path` pointing at THIS machine's disk. On a new machine those paths are
# stale -- the row identity (corpus, text, duration, hf_index/speaker) survives,
# the path does not, and prepare_data.py must be re-pointed at fresh audio.
# =============================================================================

_CLOSEOUT = r'''
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdrive_sync

def size_mb(p):
    return p.stat().st_size / 1e6

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    jobs = []   # (path, drive_folder)

    data = Path(args.data)
    for p in sorted(data.glob("manifest_*.jsonl")) + sorted(data.glob("*.json")):
        jobs.append((p, "manifests"))

    runs = Path(args.runs)
    if runs.is_dir():
        keep = ("config.json", "summary.json", "history.jsonl", "lm_params.json",
                "lm_params_clean.json", "lm_params_other.json", "eval_results.json",
                "head.pt", "adapter.pt")
        for d in sorted(x for x in runs.iterdir() if x.is_dir()):
            for n in keep:
                f = d / n
                if f.is_file():
                    jobs.append((f, d.name))

    bench = runs.parent / "benchmark"
    if bench.is_dir():
        for p in sorted(bench.glob("*.json")):
            jobs.append((p, "benchmark"))

    for e in args.extra:
        p = Path(e)
        if p.is_file():
            jobs.append((p, "misc"))

    total = sum(size_mb(p) for p, _ in jobs)
    print(f"{len(jobs)} file(s), {total:.1f} MB total\n")

    ok = bad = 0
    for p, folder in jobs:
        try:
            good, why = gdrive_sync.sync_checkpoint(p, folder)
        except Exception as exc:
            good, why = False, f"{type(exc).__name__}: {exc}"
        print(f"  {'OK  ' if good else 'FAIL'} {size_mb(p):8.2f} MB  {folder}/{p.name}"
              + ("" if good else f"   -- {why}"), flush=True)
        ok += good; bad += (not good)

    print(f"\nmirrored {ok}, failed {bad}")
    if bad:
        print("Do NOT close the session while anything above says FAIL. Run "
              "`python gdrive_sync.py` for the credential diagnosis.")
        raise SystemExit(2)
    print("Everything small is on Drive. What stays behind on purpose:")
    print("  * the packed cache (~35 GB)  -- rebuildable from the manifests in 10-30 min")
    print("  * the downloaded audio       -- re-fetchable, hours")
    print("  * the noise/RIR banks, the KenLM ARPA, the venv -- all re-fetchable")
    print("  * manifest `audio_path` values point at THIS disk and will be stale")

if __name__ == "__main__":
    main()
'''

_p = asr_dir / "closeout.py"
_p.write_text(_CLOSEOUT, encoding="utf-8")

_cmd = [str(py_bin), str(_p), "--data", str(data_dir), "--runs", str(runs_dir)]
print("=== CLOSEOUT: mirroring everything small to Drive ===", flush=True)
_c = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, bufsize=1)
for _line in _c.stdout:
    print(_line, end="", flush=True)
_c.wait()

closeout_msg = mo.md(
    "✓ **Safe to close the session.** Everything small is mirrored; the log above "
    "lists each file. The 35 GB cache and the downloaded audio stay behind on "
    "purpose — the cache rebuilds from the manifests in 10-30 minutes, and the "
    "manifests are now on Drive."
    if _c.returncode == 0 else
    f"❌ **Do not close yet** (exit {_c.returncode}). At least one file failed to "
    "mirror; the log names it and why. If it is a credential problem, run "
    "`python gdrive_sync.py` for the full diagnosis.")
closeout_msg
