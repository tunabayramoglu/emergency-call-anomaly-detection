# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=4.44",
#     "datasets>=2.20",
#     "peft>=0.11",
#     "jiwer",
#     "soundfile",
#     "huggingface_hub",
#     "numpy",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
#
# Phase-1b - overnight ablation, marimo interface.
#
# TWO FILES are required and both must sit in the SAME folder.
#   overnight_nb.py   <- this file (the interface)
#   aug_night_v2.py     <- the engine (augmentation, training, HF)
#
# molab   : upload both, open this file, run the cells in order
# local   : marimo edit overnight_nb.py

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Overnight ablation")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Phase-1b — night ablation

        ## Flow

        1. **Base** - pull the best checkpoint from HF, or train from scratch if there is none
        2. **Ablation** - every axis warm starts from the base, **independent**, short run
        3. **Combo** - the winning axes are merged and run together
        4. **Final** - the winning setup, a long run from scratch

        > **Why does `X_control` exist?** It continues the base with the same short budget and
        > nothing added. Without it every axis gain is confounded with the *"a few more epochs"*
        > effect. The `d-clean` column measures against **control**, not against BASE.

        > **What to expect.** The job of augmentation is to lower the `tel8k/clean` ratio.
        > It does not move the W/C ratio (WER divided by CER), only the language model does.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## 0 · Load the engine""")
    return


@app.cell
def _(mo):
    import importlib
    import os
    import sys
    from pathlib import Path

    _here = Path(__file__).parent if "__file__" in dir() else Path.cwd()
    _cands = [_here, Path.cwd(), Path("/marimo"), Path("/marimo/notebooks")]
    _found = next((p for p in _cands if (p / "aug_night_v2.py").exists()), None)

    if _found is None:
        E = None
        _msg = mo.md(
            f"""
            **`aug_night_v2.py` was not found.**

            Places searched:
            ```
            {chr(10).join(str(p) for p in _cands)}
            ```
            Upload the engine file next to this notebook, then run this cell again.
            """
        )
    else:
        if str(_found) not in sys.path:
            sys.path.insert(0, str(_found))
        import aug_night_v2 as E

        importlib.reload(E)
        _msg = mo.md(
            f"Engine loaded -> `{_found / 'aug_night_v2.py'}` - "
            f"**{len(E.ORDER)}** axes defined"
        )
    _msg
    return E, Path, importlib, os, sys


@app.cell
def _(E, mo):
    mo.stop(E is None, mo.md("*Cannot continue until the engine is loaded.*"))
    import torch

    _has = torch.cuda.is_available()
    mo.md(
        f"""
    | check | result |
    |---|---|
    | GPU | {torch.cuda.get_device_name(0) if _has else "NONE - falls back to CPU silently"} |
    | VRAM | {f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB" if _has else "-"} |
    | torch | {torch.__version__} |
    | bf16 | {torch.cuda.is_bf16_supported() if _has else "-"} |
    | data | `{E._envp("ECAD_DATA_ROOT", "./data")}` |
    | runs | `{E._envp("ECAD_OUT_ROOT", "./runs")}` |
    | cache | `{E._envp("ECAD_CACHE_ROOT", "./cache")}` |
    """
    )
    return (torch,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · Hugging Face credentials

        The token is entered **here**. It is never written into the code, the command line or the logs.
        When running in the background it is passed to the subprocess as an environment
        variable, so it does not appear in `ps` output.

        Token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        → type **Write**. *A read token can pull the base, but the push at the end of the night will fail.*

        If you added `HF_TOKEN` to molab's Secrets section you can leave this field empty.
        """
    )
    return


@app.cell
def _(mo, os):
    hf_token_ui = mo.ui.text(
        label="HF token (if empty, HF_TOKEN and huggingface-cli login are tried)",
        kind="password", full_width=True, placeholder="hf_...",
    )
    hf_repo_ui = mo.ui.text(
        label="Repo (user/repo)",
        value=os.environ.get("ECAD_HF_REPO", ""),
        placeholder="tuna/clear-phase1-runs", full_width=True,
    )
    hf_type_ui = mo.ui.dropdown(
        ["dataset", "model"], value="dataset", label="Repo type",
    )
    mo.vstack([hf_token_ui, hf_repo_ui, hf_type_ui])
    return hf_repo_ui, hf_token_ui, hf_type_ui


@app.cell
def _(mo):
    check_btn = mo.ui.run_button(label="Verify identity and write access")
    check_btn
    return (check_btn,)


@app.cell
def _(E, check_btn, hf_repo_ui, hf_token_ui, hf_type_ui, mo):
    # We do NOT use mo.stop. The plan and run cells below depend on hf_runs but
    # they must still work from the defaults even when verification is skipped.
    if not check_btn.value:
        hf_ok, hf_runs, hf_msg = False, [], mo.md("*Waiting for verification.*")
    else:
        E.set_token(hf_token_ui.value)
        E.set_repo_type(hf_type_ui.value)
        _repo = hf_repo_ui.value.strip() or None
        with mo.redirect_stdout():
            hf_ok = E.hf_preflight(_repo, E.REPO_TYPE)
            hf_runs = E.hf_list_runs(_repo, E.REPO_TYPE) if hf_ok else []
            if hf_runs:
                print(f"[HF] runs in the repo ({len(hf_runs)}): {', '.join(hf_runs)}")
            elif hf_ok:
                print("[HF] the repo is empty, the base will be trained from scratch.")
        hf_msg = mo.md(
            "**Ready** - results will be uploaded to the repo after every run."
            + (f"<br>Base candidates: `{'`, `'.join(hf_runs)}`" if hf_runs else "")
            if hf_ok
            else "**Push is off** - the night still runs but the results stay in the local "
            "`runs/` folder only. The molab container is deleted at the end of the session, "
            "and that folder goes with it."
        )
    hf_msg
    return hf_msg, hf_ok, hf_runs


@app.cell
def _(mo):
    mo.md(r"""## 2 · Budget and plan""")
    return


@app.cell
def _(hf_runs, mo):
    hours_ui = mo.ui.slider(1, 12, value=6, step=0.5, label="Budget (hours)",
                            show_value=True)
    base_ep_ui = mo.ui.slider(10, 60, value=30, step=5,
                              label="Base epochs (unused when pulled from HF)",
                              show_value=True)
    abl_ep_ui = mo.ui.slider(4, 30, value=12, step=1, label="Ablation epoch",
                             show_value=True)
    spe_ui = mo.ui.number(30, 600, value=130, step=10,
                          label="Epoch time estimate (s), refined as it is measured")
    base_run_ui = mo.ui.dropdown(
        ["(best CER)"] + list(hf_runs), value="(best CER)",
        label="Base run",
    )
    mo.vstack([hours_ui, base_ep_ui, abl_ep_ui, spe_ui, base_run_ui])
    return abl_ep_ui, base_ep_ui, base_run_ui, hours_ui, spe_ui


@app.cell
def _(E, mo):
    axes_ui = mo.ui.multiselect(
        options=list(E.ORDER), value=list(E.ORDER),
        label="Axes to run (priority order is preserved, the deadline cuts from the end)",
        full_width=True,
    )
    combo_ui = mo.ui.switch(label="Run COMBO at the end (merge the winners)",
                            value=True)
    mo.vstack([axes_ui, combo_ui])
    return axes_ui, combo_ui


@app.cell
def _(E, abl_ep_ui, axes_ui, base_ep_ui, base_run_ui, hf_runs, hours_ui, mo, spe_ui):
    _have_base = bool(hf_runs) or (
        E._envp("ECAD_OUT_ROOT", "./runs") / E.BASE_RUN / "head.pt"
    ).exists()
    _sel = [a for a in E.ORDER if a in axes_ui.value]
    _p = E.plan_night(hours_ui.value, base_ep_ui.value, abl_ep_ui.value,
                      spe_ui.value, _have_base, len(_sel))
    _fit, _skip = _sel[: _p["n_fit"]], _sel[_p["n_fit"]:]

    mo.md(
        f"""
    ### Plan

    | stage | time | note |
    |---|---|---|
    | base | {"**0 min**" if _have_base else f"**{_p['base_min']:.0f} min**"} | {"ready from HF or locally" if _have_base else f"{base_ep_ui.value} epochs from scratch"} |
    | ablation | **{_p['abl_min']:.0f} min** × {_p['n_fit']} | {abl_ep_ui.value} epochs/axis |
    | combo | **{_p['combo_min']:.0f} min** | winners merged |
    | **total** | **{_p['est_total_min'] / 60:.1f} h** | budget {hours_ui.value:.1f} h |

    **Fits ({len(_fit)}):** `{'`, `'.join(_fit) or '-'}`

    {f"**Does not fit ({len(_skip)}):** `" + "`, `".join(_skip) + "`" if _skip else "All selected axes fit."}

    *Base: `{base_run_ui.value}`*
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · Run

        There are two options and the difference matters.

        | | background *(recommended)* | inside the cell |
        |---|---|---|
        | if the notebook closes | **the run continues** | the run dies |
        | output | a log file, read with the button below | live in the cell |
        | if the molab session drops | continues while the container lives | ends |

        Use **background** for a six-hour run. Do not rely on a browser tab staying open for
        six hours. When the molab connection drops an in-cell run dies and you have to start
        it over from the beginning.
        """
    )
    return


@app.cell
def _(mo):
    bg_btn = mo.ui.run_button(label="Start in the background (recommended)")
    fg_btn = mo.ui.run_button(label="Run in this cell (dies if the connection drops)")
    mo.hstack([bg_btn, fg_btn], justify="start", gap=1)
    return bg_btn, fg_btn


@app.cell
def _(
    E, Path, abl_ep_ui, axes_ui, base_ep_ui, base_run_ui, bg_btn, combo_ui,
    hf_repo_ui, hf_token_ui, hf_type_ui, hours_ui, mo, os, spe_ui, sys,
):
    mo.stop(not bg_btn.value, mo.md("*Waiting to start the background run.*"))
    import subprocess

    _out = E._envp("ECAD_OUT_ROOT", "./runs")
    _out.mkdir(parents=True, exist_ok=True)
    LOG_PATH = _out / "gece.log"

    # The token goes into the environment, NOT into argv, so it never shows up in `ps` or the logs.
    _env = dict(os.environ)
    if hf_token_ui.value.strip():
        _env["HF_TOKEN"] = hf_token_ui.value.strip()
        _env["HUGGING_FACE_HUB_TOKEN"] = hf_token_ui.value.strip()

    _engine = Path(E.__file__).parent / "aug_night_v2.py"
    _cmd = [
        sys.executable, str(_engine), "--night",
        "--hours", str(hours_ui.value),
        "--base-epochs", str(base_ep_ui.value),
        "--abl-epochs", str(abl_ep_ui.value),
        "--sec-per-epoch", str(spe_ui.value),
        "--hf-repo-type", hf_type_ui.value,
        "--only", ",".join(a for a in E.ORDER if a in axes_ui.value),
    ]
    if hf_repo_ui.value.strip():
        _cmd += ["--hf-repo", hf_repo_ui.value.strip()]
    if base_run_ui.value != "(best CER)":
        _cmd += ["--hf-run", base_run_ui.value]
    if not combo_ui.value:
        _cmd.append("--no-combo")

    with open(LOG_PATH, "a") as _f:
        _f.write(f"\n{'=' * 70}\n### start {__import__('time').ctime()}\n")
        _proc = subprocess.Popen(
            _cmd, env=_env, stdout=_f, stderr=subprocess.STDOUT,
            start_new_session=True,  # the process survives even if the notebook dies
            cwd=str(_engine.parent),
        )

    mo.md(
        f"""
    **Started** - PID `{_proc.pid}`

    ```
    {" ".join(_cmd)}
    ```

    Log: `{LOG_PATH}` - follow it with the **Refresh log** button below.
    You can close the notebook, the run keeps going.

    To stop it, in a new cell: `import os, signal; os.killpg({_proc.pid}, signal.SIGTERM)`
    """
    )
    return LOG_PATH, subprocess


@app.cell
def _(
    E, abl_ep_ui, axes_ui, base_ep_ui, base_run_ui, combo_ui, fg_btn,
    hf_repo_ui, hf_token_ui, hf_type_ui, hours_ui, mo, spe_ui,
):
    mo.stop(not fg_btn.value, mo.md("*Waiting to start the in-cell run.*"))
    E.set_token(hf_token_ui.value)
    E.set_repo_type(hf_type_ui.value)
    with mo.redirect_stdout():
        fg_result = E.run_night(
            hours=hours_ui.value,
            base_epochs=base_ep_ui.value,
            abl_epochs=abl_ep_ui.value,
            sec_per_epoch=spe_ui.value,
            hf_repo=hf_repo_ui.value.strip() or None,
            hf_run=None if base_run_ui.value == "(best CER)" else base_run_ui.value,
            only=[a for a in E.ORDER if a in axes_ui.value],
            do_combo=combo_ui.value,
        )
    fg_result
    return (fg_result,)


@app.cell
def _(mo):
    mo.md(r"""## 4 · Monitoring""")
    return


@app.cell
def _(mo):
    tail_btn = mo.ui.run_button(label="↻ Refresh the log")
    tail_n = mo.ui.slider(20, 400, value=60, step=20, label="last N lines",
                          show_value=True)
    mo.hstack([tail_btn, tail_n], justify="start", gap=1)
    return tail_btn, tail_n


@app.cell
def _(E, mo, tail_btn, tail_n):
    tail_btn
    _log = E._envp("ECAD_OUT_ROOT", "./runs") / "gece.log"
    if not _log.exists():
        mo.md("*No log yet, the background run has not been started.*")
    else:
        _lines = _log.read_text(errors="replace").splitlines()[-tail_n.value:]
        mo.md(f"`{_log}` - last {len(_lines)} lines\n\n```\n" + "\n".join(_lines) + "\n```")
    return


@app.cell
def _(mo):
    rep_btn = mo.ui.run_button(label="Refresh the results table (works while a run is going)")
    rep_btn
    return (rep_btn,)


@app.cell
def _(E, mo, rep_btn):
    rep_btn
    import json as _json

    _out = E._envp("ECAD_OUT_ROOT", "./runs")
    _rows = []
    for _p in sorted(_out.glob("*/summary.json")):
        try:
            _s = _json.loads(_p.read_text())
        except Exception:
            continue
        _f = _s.get("final", {})
        _rows.append({
            "run": _s.get("run", _p.parent.name),
            "valCER": (_s.get("best_cer") or float("nan")) * 100,
            "clean W": _f.get("clean", {}).get("wer", float("nan")) * 100,
            "tel W": _f.get("tel", {}).get("wer", float("nan")) * 100,
            "tel8k W": _f.get("tel8k", {}).get("wer", float("nan")) * 100,
            "tel8k/clean": _s.get("tel8k_over_clean") or float("nan"),
            "ep": _s.get("epochs_done", 0),
            "note": _s.get("stopped") or "",
        })

    if not _rows:
        results_md = mo.md("*No `summary.json` yet.*")
    else:
        _ctrl = next(
            (r for r in _rows if r["run"] == f"{E.ABL_PREFIX}control"), None
        )
        _hdr = "| run | valCER | clean W | tel W | tel8k W | tel8k/clean | d-clean | ep |"
        _sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
        _body = []
        for r in sorted(_rows, key=lambda x: (x["run"] != E.BASE_RUN, x["run"])):
            _d = (f"{r['clean W'] - _ctrl['clean W']:+.2f}"
                  if _ctrl and r["clean W"] == r["clean W"] else "—")
            _body.append(
                f"| `{r['run']}` | {r['valCER']:.2f} | {r['clean W']:.2f} | "
                f"{r['tel W']:.2f} | {r['tel8k W']:.2f} | {r['tel8k/clean']:.2f} | "
                f"{_d} | {r['ep']} |"
            )
        results_md = mo.md(
            "\n".join([_hdr, _sep] + _body)
            + "\n\n**Rule:** take the axis that lowers the `tel8k/clean` ratio while keeping "
              "`d-clean` within +5% relative. Compare against `X_control`, not against `BASE`."
        )
    results_md
    return (results_md,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5 · Winners and final

        When the ablation finishes, look at the winners and then start the final run.
        **The final runs from scratch**, not from a warm start. If your observation that it
        plateaus at 50 epochs holds, the final number should come from a clean run.
        """
    )
    return


@app.cell
def _(mo):
    win_btn = mo.ui.run_button(label="Compute the winners")
    final_ep_ui = mo.ui.slider(20, 100, value=50, step=5, label="Final epoch",
                               show_value=True)
    final_btn = mo.ui.run_button(label="Start the FINAL run (background)")
    mo.vstack([win_btn, mo.hstack([final_ep_ui, final_btn], justify="start", gap=1)])
    return final_btn, final_ep_ui, win_btn


@app.cell
def _(E, mo, win_btn):
    mo.stop(not win_btn.value, mo.md("*Waiting to compute the winners.*"))
    with mo.redirect_stdout():
        winners, ctrl = E.pick_winners(E._envp("ECAD_OUT_ROOT", "./runs"))
    mo.md(
        f"**Winners:** `{'`, `'.join(w[0] for w in winners) or 'none'}`"
        if winners else
        "No winner. No axis beat control without hurting clean WER. "
        "That is a result too, this setup does not benefit from augmentation at this budget."
    )
    return ctrl, winners


@app.cell
def _(
    E, Path, final_btn, final_ep_ui, hf_repo_ui, hf_token_ui, hf_type_ui, mo, os, sys,
):
    mo.stop(not final_btn.value, mo.md("*Waiting to start the final run.*"))
    import subprocess as _sp

    _out = E._envp("ECAD_OUT_ROOT", "./runs")
    _log = _out / "final.log"
    _env = dict(os.environ)
    if hf_token_ui.value.strip():
        _env["HF_TOKEN"] = hf_token_ui.value.strip()
        _env["HUGGING_FACE_HUB_TOKEN"] = hf_token_ui.value.strip()
    _engine = Path(E.__file__).parent / "aug_night_v2.py"
    _cmd = [sys.executable, str(_engine), "--stage", "final",
            "--final-epochs", str(final_ep_ui.value),
            "--hf-repo-type", hf_type_ui.value]
    if hf_repo_ui.value.strip():
        _cmd += ["--hf-repo", hf_repo_ui.value.strip()]
    with open(_log, "a") as _f:
        _p = _sp.Popen(_cmd, env=_env, stdout=_f, stderr=_sp.STDOUT,
                       start_new_session=True, cwd=str(_engine.parent))
    mo.md(f"Final started - PID `{_p.pid}` - log `{_log}`")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Notes

        - **The cache is shared.** `_cache_key()` is byte-for-byte the same as in `sweep_v2.py`,
          so the existing `audio.i16` is reused and nothing is decoded again.
        - **Independent axes.** Not a cumulative chain. If the deadline cuts the run short,
          whatever was measured stays valid. COMBO picks up the interactions.
        - **The `backbone.eval()` fix.** v2 called `eval()` unconditionally inside the training
          loop, and HF only applies masking while `self.training` is set, which meant
          SpecAugment had never run. It is now conditional.
        - **`layerdrop` must stay 0.** The layer it skips can be exactly the
          `hidden_state` that weighted-sum needs.
        - **KenLM is not in the ablation.** It is a fixed offset, it does not change the ranking,
          and it costs decode time. Enable it once, after the winner is chosen.
        - **Zero extra dependencies.** Augmentation is pure numpy FFT. No torchaudio,
          scipy, librosa or audiomentations needed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
