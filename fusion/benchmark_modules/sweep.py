"""
sweep.py — equal-budget hyperparameter sweep.

Deliverable #3 of the extension. Applies to `late` and `intermediate` ONLY,
per the spec — `early` and `text_only_finetuned` get NO sweep and run at
standard fine-tuning defaults, which is recorded explicitly in
`ASYMMETRIC_SWEEP_NOTE` below so the asymmetry is stated on a slide rather
than silently assumed. (`film.py` and `attn.py`
also use this module for their sweep, at the SAME grid/budget as
`intermediate`, per their own module docstrings — they are new intermediate-
style methods, not part of the mandatory late/intermediate comparison, but
"equal footing" for them means the same grid size, not a bigger one.)

Grid (identical size for every method swept — 3 x 3 x 2 = 18 configs):
    lr      in {3e-4, 1e-3, 3e-3}
    hidden  in {128, 256, 512}   (late: text-branch hidden size; intermediate
                                  / film / attn: the fc1 width after fusion)
    dropout in {0.1, 0.3}

Selection: winning config = highest VAL macro-F1 (using one search seed).
Then the winner is re-run at 3 seeds and reported on TEST — never select on
test.

These methods train on frozen embeddings in seconds (no encoder fine-tuning
involved), so the sweep runs on the FULL training split — no subsampling.
`_run_late_core` / `_run_intermediate_core` / `_run_film_core` / `_run_attn_core`
already train on `splits["train"]` directly; this module does not subsample it.
"""

from __future__ import annotations

import itertools
import numpy as np
from sklearn.metrics import f1_score

from common import make_result

ASYMMETRIC_SWEEP_NOTE = (
    "Sweep coverage is intentionally asymmetric: late/intermediate/film/attn "
    "(frozen-embedding methods, seconds per run) get an 18-config grid search "
    "selected on val macro-F1. early and text_only_finetuned (fine-tune the "
    "encoder, minutes per run) get NO sweep and run at the standard "
    "fine-tuning defaults (AdamW lr=2e-5, batch 32, 4 epochs) specified in "
    "the original contract. State this on the slide: an early/text_only_ft "
    "win over a swept late/intermediate is a comparison of a tuned baseline "
    "against an untuned one, in early's favour if anything — so it is a "
    "conservative, not inflated, comparison."
)

LR_GRID = [3e-4, 1e-3, 3e-3]
HIDDEN_GRID = [128, 256, 512]
DROPOUT_GRID = [0.1, 0.3]


def sweep_grid() -> list[dict]:
    """The 18 (lr, hidden, dropout) configs, identical for every swept method."""
    return [
        {"lr": lr, "hidden": hidden, "dropout": dropout}
        for lr, hidden, dropout in itertools.product(LR_GRID, HIDDEN_GRID, DROPOUT_GRID)
    ]


def _core_val_macro_f1(core_result: dict) -> float:
    return float(f1_score(core_result["y_val"], core_result["val_pred"],
                          average="macro", zero_division=0))


def sweep_method(
    core_fn,
    hparam_names: dict,
    splits: dict,
    emb: dict,
    device: str,
    encoder_name: str,
    search_seed: int = 0,
    eval_seeds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    """
    Generic equal-budget sweep runner.

    `core_fn(splits, emb, seed, device, encoder_name, **kwargs) -> dict` must
    be one of `_run_late_core` / `_run_intermediate_core` / `_run_film_core` /
    `_run_attn_core` (or anything with that shape: returns a dict with
    y_val/val_pred/y_test/test_pred/test_logits/val_logits keys).

    `hparam_names` maps the grid's generic keys {"lr","hidden","dropout"} to
    the specific keyword `core_fn` expects — e.g. for late's core,
    {"lr": "lr", "hidden": "text_hidden", "dropout": "text_dropout"}, since
    late's hidden size is a "text_hidden" kwarg, not "hidden".

    Runs the full 18-config grid ONCE at `search_seed`, selects by val
    macro-F1, then re-runs the winning config at every seed in `eval_seeds`
    and returns those as the reportable test results.

    Returns:
        {
          "best_config": {"lr":.., "hidden":.., "dropout":..},
          "best_val_macro_f1": float,
          "all_configs": [{"config":.., "val_macro_f1":..}, ...],   # all 18, for the appendix
          "test_runs": [ (y_true, y_pred, seed), ... ],             # one per eval_seed, on the winner
        }
    """
    all_configs = []
    best_val_f1 = -1.0
    best_config = None

    for cfg in sweep_grid():
        kwargs = {hparam_names[k]: v for k, v in cfg.items()}
        result = core_fn(splits, emb, search_seed, device, encoder_name, **kwargs)
        val_f1 = _core_val_macro_f1(result)
        all_configs.append({"config": dict(cfg), "val_macro_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = dict(cfg)

    test_runs = []
    kwargs = {hparam_names[k]: v for k, v in best_config.items()}
    for seed in eval_seeds:
        result = core_fn(splits, emb, seed, device, encoder_name, **kwargs)
        test_runs.append((result["y_test"], result["test_pred"], seed))

    return {
        "best_config": best_config,
        "best_val_macro_f1": best_val_f1,
        "all_configs": all_configs,
        "test_runs": test_runs,
    }


def sweep_late(splits, emb, device, encoder_name, search_seed=0, eval_seeds=(0, 1, 2)) -> dict:
    """Sweep `late`'s text branch: hidden -> text_hidden, dropout -> text_dropout."""
    from late import _run_late_core
    return sweep_method(
        _run_late_core,
        {"lr": "lr", "hidden": "text_hidden", "dropout": "text_dropout"},
        splits, emb, device, encoder_name, search_seed, eval_seeds,
    )


def sweep_intermediate(splits, emb, device, encoder_name, search_seed=0, eval_seeds=(0, 1, 2)) -> dict:
    """Sweep `intermediate`'s fc1 width/dropout/lr."""
    from inter import _run_intermediate_core
    return sweep_method(
        _run_intermediate_core,
        {"lr": "lr", "hidden": "hidden", "dropout": "dropout"},
        splits, emb, device, encoder_name, search_seed, eval_seeds,
    )


def sweep_film(splits, emb, device, encoder_name, search_seed=0, eval_seeds=(0, 1, 2)) -> dict:
    """Sweep `intermediate_film`'s fc1 width/dropout/lr, same grid as intermediate."""
    from film import _run_film_core
    return sweep_method(
        _run_film_core,
        {"lr": "lr", "hidden": "hidden", "dropout": "dropout"},
        splits, emb, device, encoder_name, search_seed, eval_seeds,
    )


def sweep_attn(splits, emb, device, encoder_name, search_seed=0, eval_seeds=(0, 1, 2)) -> dict:
    """Sweep `intermediate_attn`'s fc1 width/dropout/lr, same grid as intermediate."""
    from attn import _run_attn_core
    return sweep_method(
        _run_attn_core,
        {"lr": "lr", "hidden": "hidden", "dropout": "dropout"},
        splits, emb, device, encoder_name, search_seed, eval_seeds,
    )


def sweep_results_to_records(method: str, encoder: str, variant: str, sweep_out: dict) -> list[dict]:
    """Convert a `sweep_*` output's `test_runs` into the flat result-dict
    format (`common.make_result`) used everywhere else, so the
    sweep winner slots directly into `results_table` / `plot_results` next
    to the unswept methods."""
    return [
        make_result(method, encoder, variant, seed, y_true, y_pred)
        for y_true, y_pred, seed in sweep_out["test_runs"]
    ]


def json_safe(obj):
    """Recursively convert numpy types so a sweep result can be json.dump'd.

    `sweep_method` returns `test_runs` as raw numpy arrays because the driver
    wants them in memory; that makes the dict itself un-serialisable. Rather
    than degrading the in-memory return value, convert at the point of writing.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def sweep_summary(sweep_out: dict) -> dict:
    """A compact, JSON-safe view of one sweep: the winner, the full config
    ranking (useful as a slide appendix), and per-seed test scores on the
    winner — with the raw prediction arrays dropped."""
    from common import compute_metrics

    return {
        "best_config": json_safe(sweep_out["best_config"]),
        "best_val_macro_f1": float(sweep_out["best_val_macro_f1"]),
        "all_configs": json_safe(sweep_out["all_configs"]),
        "n_configs": len(sweep_out["all_configs"]),
        "winner_test_per_seed": [
            {"seed": int(seed), **{k: v for k, v in compute_metrics(y_true, y_pred).items()
                                   if k != "confusion"}}
            for y_true, y_pred, seed in sweep_out["test_runs"]
        ],
    }
