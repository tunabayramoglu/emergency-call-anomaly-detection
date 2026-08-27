# =============================================================================
# PASTE THIS AS A SINGLE NEW marimo CELL (it needs: mo, subprocess, Path,
# json, py_bin, asr_dir, runs_dir, lm_dir)
#
# WHAT WAS WRONG WITH RUNNING §15 TWICE
# -------------------------------------
# §15 evaluates ONE run and stores it under the hardcoded key "FINAL_300h",
# whatever run it actually scored. Running it for baseline_100h and then for the
# 300h model therefore produced two files that both claim to be FINAL_300h, and
# merging them overwrites one with the other. It also reloaded both datasets and
# re-ran Whisper for each system -- Whisper being a fixed external reference that
# cannot change between them.
#
# This cell evaluates all three systems in ONE pass:
#   * each dataset is loaded once,
#   * Whisper runs once per test set and is shared,
#   * each ASR system decodes with ITS OWN tuned alpha/beta/beam (that is the
#     entire point of §14 -- a shared guess would hand one system an advantage),
#   * results are keyed by the real run name,
#   * the output file is timestamped, so a second run never overwrites the first.
# =============================================================================

BENCH_SRC = r'''
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_asr import (load_devclean, load_l2arctic, run_our_model, run_whisper,
                      sync_to_drive, log, _selfstamp)


def params_for(run_dir: Path) -> tuple[float, float, int, str]:
    """Tuned decoder settings for THIS run, or the old guess with a warning.

    alpha trades the acoustic posterior against the LM, so it belongs to one
    checkpoint. Reusing another system's value is how a comparison quietly stops
    being fair -- hence the run_dir check rather than a shared default.
    """
    for name in ("lm_params_clean.json", "lm_params.json"):
        p = run_dir / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        owner = d.get("run_dir", "")
        if owner and Path(owner).name != run_dir.name:
            raise SystemExit(f"{p} was tuned on {owner}, not {run_dir} -- refusing "
                             "to decode one system with another's LM weight")
        return d["alpha"], d["beta"], d["beam_width"], name
    log(f"[bench] !! {run_dir.name}: no lm_params -- falling back to the UNTUNED "
        "alpha=0.5 beta=1.0 beam=100 guess. This system is being handicapped "
        "relative to any system that has tuned params.")
    return 0.5, 1.0, 100, "untuned-default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--whisper", default="openai/whisper-base,openai/whisper-small,openai/whisper-medium")
    ap.add_argument("--whisper-on", default="l2-arctic",
                    choices=["both", "dev-clean", "l2-arctic", "none"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    log(f"[src] {_selfstamp()}")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs_dir = Path(args.runs_dir)

    # Fail before any GPU work if a run directory is unusable. An hour into a
    # benchmark is the wrong moment to discover a missing adapter.
    for r in args.runs:
        d = runs_dir / r
        missing = [f for f in ("config.json", "adapter.pt", "head.pt")
                   if not (d / f).is_file()]
        if missing:
            raise SystemExit(f"{d}: missing {missing}")

    # Loaded ONCE, shared by every system. Re-loading per system was pure waste
    # and, worse, meant two systems could be scored on different random subsets
    # when --limit is set.
    sets = {"dev-clean": load_devclean(args.limit),
            "l2-arctic": load_l2arctic(args.limit)}
    for k, v in sets.items():
        log(f"[bench] {k}: {len(v)} utterances")

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "device": device, "limit": args.limit, "lm": args.lm,
           "systems": {}, "whisper": {}, "sets": {k: len(v) for k, v in sets.items()}}

    for r in args.runs:
        d = runs_dir / r
        a, b, beam, src = params_for(d)
        log(f"\n===== {r} | alpha={a} beta={b} beam={beam} ({src}) =====")
        out["systems"][r] = {"decoder": {"alpha": a, "beta": b, "beam_width": beam,
                                         "source": src}}
        for sname, rows in sets.items():
            log(f"[bench] {r} on {sname} ...")
            out["systems"][r][sname] = run_our_model(rows, d, args.lm, device, a, b, beam)

    wsets = {"both": ("dev-clean", "l2-arctic"), "dev-clean": ("dev-clean",),
             "l2-arctic": ("l2-arctic",), "none": ()}[args.whisper_on]
    for wm in [w for w in args.whisper.split(",") if w.strip()]:
        tag = wm.split("/")[-1]
        out["whisper"][tag] = {}
        for sname in wsets:
            log(f"[bench] {tag} on {sname} ...")
            out["whisper"][tag][sname] = run_whisper(sets[sname], wm, device)
    if not wsets:
        log("[bench] Whisper skipped. Its dev-clean numbers already exist in "
            "whisper_bench.ipynb -- same model, same set, same number.")

    # Timestamped: a second run never destroys the first. `latest.json` is a
    # convenience copy, and it is the ONLY thing that gets overwritten.
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p_stamped = od / f"benchmark_{stamp}.json"
    p_latest = od / "benchmark_latest.json"
    p_stamped.write_text(json.dumps(out, indent=2))
    p_latest.write_text(json.dumps(out, indent=2))
    log(f"\n[bench] written: {p_stamped}")

    # Mirrors under CLEAR/Phase 1/ASR-300/benchmark/ -- a place that is about the
    # comparison, not about any one run, because this file is about all three.
    sync_to_drive([p_stamped, p_latest], "benchmark")

    log("\n" + "=" * 78)
    hdr = f"{'system':<34}" + "".join(f"{s:>21}" for s in sets)
    log(hdr)
    for r, v in out["systems"].items():
        row = f"{r:<34}"
        for s in sets:
            g = v[s]["greedy"]["wer"] * 100
            k = v[s].get("kenlm", {}).get("wer")
            row += f"{g:>9.2f}/{(k * 100 if k else float('nan')):>10.2f}"
        log(row + "   (greedy/+KenLM WER%)")
    for t, v in out["whisper"].items():
        row = f"{'whisper-' + t.replace('whisper-', ''):<34}"
        for s in sets:
            row += f"{(v[s]['wer'] * 100 if s in v else float('nan')):>9.2f}{'':>11}"
        log(row)
    log("=" * 78)

    # kenlm aborts during teardown after everything is written; skip destructors.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
'''

# Written and run in the SAME cell on purpose. Every other module lives in the
# big %%writefile cell, and a stale on-disk copy there has now cost debugging
# time three times (the bfloat16 fix ran against an old file whose line numbers
# did not match). Self-contained means that cannot happen here.
_bench_path = asr_dir / "bench_all.py"
_bench_path.write_text(BENCH_SRC, encoding="utf-8")

import hashlib as _hl
print(f"[src] bench_all.py sha1:{_hl.sha1(_bench_path.read_bytes()).hexdigest()[:8]}",
      flush=True)

_out_dir = runs_dir.parent / "benchmark"
_cmd = [
    str(py_bin), str(_bench_path),
    "--runs", "baseline_100h", "run_ws_9_10_11_12_full300h",
    "--runs-dir", str(runs_dir),
    "--lm", str(lm_dir / "3-gram.pruned.1e-7.arpa"),
    "--whisper-on", "l2-arctic",
    "--out-dir", str(_out_dir),
    # "--limit", "500",          # uncomment for a ~20 min dry run
]
print("=== FINAL BENCHMARK: 100h vs 300h vs Whisper ===", flush=True)
print(" ".join(_cmd), flush=True)

_bp = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, bufsize=1)
for _line in _bp.stdout:
    print(_line, end="", flush=True)
_bp.wait()

_latest = _out_dir / "benchmark_latest.json"
if _latest.is_file():
    _d = json.loads(_latest.read_text())
    _rows = ["| system | decoder | dev-clean greedy | dev-clean +KenLM | "
             "L2-ARCTIC greedy | L2-ARCTIC +KenLM |", "|---|---|---|---|---|---|"]
    for _r, _v in _d["systems"].items():
        _dc, _l2 = _v["dev-clean"], _v["l2-arctic"]
        _p = _v["decoder"]
        _rows.append(
            f"| `{_r}` | a={_p['alpha']} b={_p['beta']} beam={_p['beam_width']} "
            f"| {100 * _dc['greedy']['wer']:.2f} "
            f"| **{100 * _dc.get('kenlm', {}).get('wer', float('nan')):.2f}** "
            f"| {100 * _l2['greedy']['wer']:.2f} "
            f"| **{100 * _l2.get('kenlm', {}).get('wer', float('nan')):.2f}** |")
    for _t, _v in _d["whisper"].items():
        _rows.append(f"| {_t} | — | — | — "
                     + (f"| {100 * _v['l2-arctic']['wer']:.2f} | — |"
                        if "l2-arctic" in _v else "| — | — |"))
    bench_msg = mo.md(
        f"✓ **Benchmark complete** — {_d['sets']}\n\n" + "\n".join(_rows)
        + f"\n\nWritten to `{_latest}` (plus a timestamped copy) and mirrored to "
          "Drive under `CLEAR/Phase 1/ASR-300/benchmark/`. Each system decoded "
          "with its OWN tuned parameters; Whisper ran once and is shared.")
else:
    bench_msg = mo.md(f"❌ Benchmark produced no output (exit {_bp.returncode}). "
                      "The log above ends at the failure.")
bench_msg
