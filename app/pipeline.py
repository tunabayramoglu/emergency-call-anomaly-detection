"""
deployable inference pipeline (UI-free core).

    audio ──► [frozen mHuBERT-147] ──► ASR LoRA + WS + CTC ──► transcript ─┐
                     (ONE copy)     └─► SER LoRA + WS + head ──► emotion ──┤
                                                                          ▼
                                              intermediate_attn fusion (BERT tokens)
                                                                          ▼
                                                     normal / borderline / anomaly

Built against DEMO_BRIEF.md, with every value checked against the artefacts on
disk rather than taken on trust. Where the brief and the files disagree, the
files win and the disagreement is recorded below.

Load-bearing notes — read before editing
----------------------------------------
1.  ONE backbone, TWO adapters. LoRA stays *unmerged* and is injected twice
    under two adapter names, so a single frozen mHuBERT serves both tasks.
    `asr/eval_asr.py::load_our_model` is deliberately NOT reused: it builds its
    own backbone, which would give two copies and falsify the project's
    architectural claim (brief §1). The LoRA config, head shape, vocabulary and
    decode semantics here are byte-equivalent to it.

2.  EMOTION ORDER — the brief is wrong here, deliberately overridden.
    Brief §3 lists the SER classes as
        [neutral, confusion, fear, panic, urgency, distress]
    That is the *fusion* one-hot order (`common.EMOTIONS`). The SER
    head's own output order (`train_ser.CLASSES`) is
        [neutral, distress, fear, urgency, panic, confusion]
    Same six labels, different index space. `ser/config.json` records only
    `n_cls: 6`, so nothing on disk contradicts this. Crossing the boundary by
    index silently mislabels 4 of 6 emotions — three of them high-risk — and
    raises no error. Everything here maps BY NAME.

3.  Two forward passes are unavoidable. LoRA changes the layer computation, so
    ASR's hidden states are not SER's. The saving is memory, not time.

4.  ASR is the 300 h run (`run_ws_9_10_11_12_full300h`, ws=9,10,11,12).
    Decoder parameters come from `asr/lm_params_clean.json` (alpha 0.6,
    beta 0.0, beam 50) — pyctcdecode's defaults are not these.

5.  bert-base-uncased is *uncased*, so its tokenizer lowercases input. The ASR
    head emits uppercase; that mismatch is therefore neutralised for free.
    Punctuation still differs from the training text and is not corrected.

Usage
-----
    python pipeline.py --self-test
    python pipeline.py clip.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------

# train_ser.py CLASSES — the order of the SER head's output logits.
SER_CLASSES = ["neutral", "distress", "fear", "urgency", "panic", "confusion"]

# common.py EMOTIONS — the order of the fusion head's 6-dim
# one-hot input. Confirmed against the checkpoint's meta["labels"] ordering
# convention and the benchmark module source.
FUSION_EMOTIONS = ["neutral", "confusion", "fear", "panic", "urgency", "distress"]

FUSION_LABELS = ["normal", "borderline", "anomaly"]

# SER logit index -> fusion one-hot index. See note 2.
SER_TO_FUSION = [FUSION_EMOTIONS.index(c) for c in SER_CLASSES]

# ablation_engine.py / eval_asr.py CHARS. Must NOT be extended: KenLM was built on
# LibriSpeech-normalised text and an unseen symbol collapses the beam.
ASR_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

# train_ser.py HIGH_RISK — the binarisation behind the 86.0% risk-detector figure.
HIGH_RISK_EMOTIONS = {"distress", "fear", "urgency", "panic"}

SAMPLE_RATE = 16_000
TEXT_ENCODER_IDS = {"bert": "bert-base-uncased",
                    "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
TOKENIZER_MAX_LENGTH = 64          # encoders.embed_texts_tokenwise


def build_vocab() -> dict[str, int]:
    """Byte-identical to eval_asr.build_vocab()."""
    v = {c: i for i, c in enumerate(ASR_CHARS)}
    v["|"], v["[UNK]"], v["[PAD]"] = len(v), len(v) + 1, len(v) + 2
    return v


def ctc_greedy(ids, i2c: dict[int, str], blank: int, unk: int) -> str:
    """Byte-identical to eval_asr.greedy_decode()."""
    return "".join(
        i2c.get(k, "") for k, _ in groupby(ids) if k not in (blank, unk)
    ).replace("|", " ").strip()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


@dataclass
class Paths:
    """Layout expected by DEMO_BRIEF.md §2.

        <root>/
          asr/     config.json  adapter.pt  head.pt  lm_params_clean.json
          ser/     config.json  adapter.pt  head.pt
          fusion/  WINNER_intermediate_attn_bert_full_p2_seed1.pt
          lm/      3-gram.pruned.1e-7.arpa          (optional)
    """

    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "models")

    def __post_init__(self):
        # Callers pass strings (CLI args, subprocess snippets); coerce once here
        # rather than failing later with "unsupported operand str / str".
        self.root = Path(self.root)

    @property
    def asr(self) -> Path:
        return self.root / "asr"

    @property
    def ser(self) -> Path:
        return self.root / "ser"

    @property
    def fusion_dir(self) -> Path:
        return self.root / "fusion"

    @property
    def lm_dir(self) -> Path:
        return self.root / "lm"

    def fusion_checkpoints(self) -> list[Path]:
        """The brief notes that `demo_prefetch.py` globs `BEST_*.pt` while the
        file is named `WINNER_*.pt`. Accept either, plus a bare name, so the
        naming mismatch cannot be the thing that breaks demo day."""
        if not self.fusion_dir.exists():
            return []
        seen, out = set(), []
        for pat in ("WINNER_*.pt", "BEST_*.pt", "*.pt"):
            for p in sorted(self.fusion_dir.glob(pat)):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    # A HubertModel checkpoint is ~360 MB; anything far below that is a
    # truncated extraction, not a smaller model.
    MIN_BACKBONE_BYTES = 300 * 2**20

    def backbone_candidates(self) -> list[Path]:
        """Places an extracted mHuBERT-147 plausibly lives, in preference order.

        `mHuBERT-147.zip` wraps its contents in a `mHuBERT-147/` directory, so
        unzipping it "here" produces `<somewhere>/mHuBERT-147/mHuBERT-147/`.
        Each candidate is therefore also probed one level down rather than
        demanding the files be moved.
        """
        app = self.root.parent
        roots = [self.root / "backbone", app / "mHuBERT-147", app.parent / "mHuBERT-147"]
        out = []
        for r in roots:
            out.append(r)
            out.append(r / "mHuBERT-147")
        return out

    def backbone(self) -> Path | None:
        """A locally extracted mHuBERT-147. Preferred over the hub: no download,
        and no dependence on the HF cache surviving until demo day.

        A truncated weights file is skipped rather than accepted. Accepting one
        would suppress the download and then fail inside `from_pretrained` with
        an error that names neither the file nor the cause — so a half-finished
        extraction in the first candidate must not shadow a complete one later
        in the list.
        """
        for d in self.backbone_candidates():
            if not (d / "config.json").is_file():
                continue
            for n in ("model.safetensors", "pytorch_model.bin"):
                f = d / n
                if f.is_file() and f.stat().st_size >= self.MIN_BACKBONE_BYTES:
                    return d
        return None

    def lm_params(self) -> Path | None:
        p = self.asr / "lm_params_clean.json"
        return p if p.exists() else None

    def arpa(self) -> Path | None:
        if not self.lm_dir.exists():
            return None
        for p in sorted(self.lm_dir.glob("*.arpa")) + sorted(self.lm_dir.glob("*.bin")):
            return p
        return None

    def missing(self) -> list[str]:
        out = []
        for d, files in ((self.asr, ("adapter.pt", "head.pt", "config.json")),
                         (self.ser, ("adapter.pt", "head.pt", "config.json"))):
            for f in files:
                if not (d / f).exists():
                    out.append(f"{d.name}/{f}")
        if not self.fusion_checkpoints():
            out.append("fusion/WINNER_*.pt")
        return out


# --------------------------------------------------------------------------
# Backbone: one frozen mHuBERT, two LoRA adapters
# --------------------------------------------------------------------------


def _lora_layers(cfg: dict) -> list[int]:
    """peft's 0-indexed `layers_to_transform`.

    The two configs express the span differently and both must work:
      asr/config.json  ->  "lora_layers": [1, 2, ..., 12]
      ser/config.json  ->  "lora_lo": 1, "lora_hi": 12
    """
    if "lora_layers" in cfg:
        return [int(i) - 1 for i in cfg["lora_layers"]]
    lo, hi = int(cfg.get("lora_lo", 1)), int(cfg.get("lora_hi", 12))
    return [i - 1 for i in range(lo, hi + 1)]


def _remap_adapter_keys(sd: dict, name: str) -> dict:
    """Checkpoints were saved from a single-adapter injection, so their LoRA
    keys carry peft's default adapter name. Rewrite them onto `name`."""
    out = {}
    for k, v in sd.items():
        if ".default." in k and ("lora_A" in k or "lora_B" in k or "lora_embedding" in k):
            k = k.replace(".default.", f".{name}.")
        out[k] = v
    return out


def _set_active(model, name: str) -> None:
    from peft.tuners.tuners_utils import BaseTunerLayer

    for m in model.modules():
        if isinstance(m, BaseTunerLayer):
            m.set_adapter(name)


class SharedBackbone:
    """Frozen mHuBERT-147 carrying an ASR adapter and a SER adapter."""

    BACKBONE_ID = "utter-project/mHuBERT-147"

    def __init__(self, asr_cfg: dict, ser_cfg: dict, device: str = "cpu",
                 local_dir: Path | None = None):
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import HubertModel

        names = {c["backbone"] for c in (asr_cfg, ser_cfg) if c.get("backbone")}
        if len(names) > 1:
            raise RuntimeError(
                f"ASR and SER name different backbones {sorted(names)}; they cannot "
                "share one copy and the architectural claim would be false.")
        name = names.pop() if names else self.BACKBONE_ID
        source = str(local_dir) if local_dir else name

        # No SpecAugment kwargs, and eval() below: masking is a training-time
        # regulariser and would make the same clip give different answers.
        try:
            bb = HubertModel.from_pretrained(source, attn_implementation="sdpa")
        except Exception:
            bb = HubertModel.from_pretrained(source)

        for spec, adapter in ((asr_cfg, "asr"), (ser_cfg, "ser")):
            lc = LoraConfig(
                r=int(spec.get("lora_r", 16)),
                lora_alpha=int(spec.get("lora_alpha", 32)),
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                layers_to_transform=_lora_layers(spec),
            )
            bb = inject_adapter_in_model(lc, bb, adapter_name=adapter)

        bb = bb.to(device).eval()
        for p in bb.parameters():
            p.requires_grad = False

        self.model = bb
        self.device = device
        self.backbone_id = name
        self.loaded_from = "local" if local_dir else "huggingface"
        self._active: str | None = None

    def load_adapter(self, path: Path, name: str) -> int:
        """Load a saved LoRA-only state dict onto adapter `name`.

        A partial match is an error: a half-loaded adapter produces fluent
        garbage rather than failing.
        """
        import torch

        sd = _remap_adapter_keys(torch.load(path, map_location=self.device), name)
        own = set(self.model.state_dict().keys())
        hit = sum(1 for k in sd if k in own)
        if hit != len(sd):
            raise RuntimeError(
                f"{path.name}: {hit}/{len(sd)} tensors matched the injected "
                f"'{name}' adapter. Inspect the key naming before trusting "
                "anything downstream.")
        self.model.load_state_dict(sd, strict=False)
        return hit

    def stack(self, wav: np.ndarray, ws, adapter: str):
        """Forward `wav` with `adapter` live; return the WS input [1,T,n_ws,hid]."""
        import torch

        if self._active != adapter:
            _set_active(self.model, adapter)
            self._active = adapter

        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None, :]
        x = x.to(self.device)
        am = torch.ones(x.shape, dtype=torch.long, device=self.device)
        with torch.no_grad():
            o = self.model(x, attention_mask=am, output_hidden_states=True)
            # .float() is not optional if this ever runs under autocast(bfloat16):
            # numpy has no bfloat16 and raises "unsupported ScalarType".
            return torch.stack([o.hidden_states[L] for L in ws], 2).float()

    def feat_lengths(self, n_samples: int) -> int:
        return int(self.model._get_feat_extract_output_lengths(n_samples))


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------


def make_asr_head(cfg: dict, vocab_size: int, device: str):
    """Replica of eval_asr.load_our_model's inner Head."""
    import torch
    import torch.nn as nn

    n, dim = len(cfg["ws"]), int(cfg.get("hid", 768))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(n))
            self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Dropout(0.0),
                                     nn.Linear(dim, vocab_size))

        def weights(self):
            return self.layer_w.softmax(0)

        def forward(self, x):                       # x: [B,T,N,D]
            w = self.layer_w.softmax(0)
            return self.net((x * w[None, None, :, None]).sum(2))

    return Head().to(device).eval()


def make_ser_head(cfg: dict, device: str):
    """Replica of train_ser.SerHead."""
    import torch
    import torch.nn as nn

    hid = int(cfg.get("hid", 768))
    pool = cfg.get("pool", "meanstd")
    feat_dim = hid * 2 if pool == "meanstd" else hid
    hidden_dim = int(cfg.get("hidden_dim", 256))
    n_cls = int(cfg.get("n_cls", 6))
    n_ws = len(cfg["ws"])

    class SerHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_w = nn.Parameter(torch.zeros(n_ws))
            self.net = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(0.0), nn.Linear(hidden_dim, n_cls))

        def weights(self):
            return self.layer_w.softmax(0)

        def forward(self, hs):                      # hs: [B,T,n_ws,hid]
            w = self.layer_w.softmax(0)
            blend = (hs * w[None, None, :, None]).sum(2)
            mu = blend.mean(1)
            f = torch.cat([mu, blend.std(1)], -1) if pool == "meanstd" else mu
            return self.net(f)

    return SerHead().to(device).eval()


def make_attn_fusion_head(state_dict: dict, device: str, num_heads: int = 4):
    """Replica of attn._AttnFusionModel.

    `text_dim` and `hidden` are read off the checkpoint. `num_heads` cannot be:
    `MultiheadAttention`'s parameter shapes are identical for any head count
    that divides `embed_dim`. 4 is the module default and what the brief states;
    a wrong value here changes the attention pattern **silently**.
    """
    import torch
    import torch.nn as nn

    required = {"emotion_embedding.weight", "query_proj.weight", "attn.in_proj_weight",
                "fc1.weight", "fc2.weight", "fc3.weight"}
    missing = required - set(state_dict)
    if missing:
        raise RuntimeError(
            f"checkpoint is not an intermediate_attn head — missing {sorted(missing)}. "
            "Point --models at the WINNER_intermediate_attn_* file.")

    text_dim = int(state_dict["query_proj.weight"].shape[0])
    hidden = int(state_dict["fc1.weight"].shape[0])
    if text_dim % num_heads:
        raise RuntimeError(f"text_dim {text_dim} is not divisible by num_heads {num_heads}")

    class AttnFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.emotion_embedding = nn.Linear(6, 32, bias=False)
            self.query_proj = nn.Linear(32, text_dim)
            self.attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads,
                                              batch_first=True)
            self.fc1 = nn.Linear(text_dim, hidden)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.0)          # inference: always off
            self.fc2 = nn.Linear(hidden, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 3)

        def forward(self, emo_onehot, token_states, attn_mask):
            q = self.query_proj(self.emotion_embedding(emo_onehot)).unsqueeze(1)
            attended, _ = self.attn(q, token_states, token_states,
                                    key_padding_mask=(attn_mask == 0))
            x = self.relu1(self.fc1(attended.squeeze(1)))
            return self.fc3(self.relu2(self.fc2(self.dropout1(x))))

    m = AttnFusion().to(device)
    m.load_state_dict({k: v for k, v in state_dict.items() if k != "_metadata"})
    return m.eval(), text_dim


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class Result:
    transcript: str
    decoder: str                      # greedy | kenlm | manual
    emotion: str
    emotion_probs: dict[str, float]
    voice_risk: str                   # high | low
    verdict: str
    verdict_probs: dict[str, float]
    timings: dict[str, float]
    decoder_params: dict | None = None   # alpha/beta/beam when kenlm ran

    @property
    def decoder_label(self) -> str:
        """What actually decoded this utterance, with the parameters it used.

        Kept separate from `decoder` so the bare name stays comparable, and so a
        greedy fallback is never mistaken for a tuned KenLM run — the two differ
        by half the word error rate.
        """
        p = self.decoder_params
        if not p:
            return self.decoder
        return (f"{self.decoder} α{p['alpha']:g} β{p['beta']:g} "
                f"beam {p['beam_width']}")

    def as_text(self) -> str:
        ep = ", ".join(f"{k} {v:.2f}" for k, v in
                       sorted(self.emotion_probs.items(), key=lambda kv: -kv[1]))
        vp = ", ".join(f"{k} {v:.2f}" for k, v in
                       sorted(self.verdict_probs.items(), key=lambda kv: -kv[1]))
        t = "  ".join(f"{k}={v:.2f}s" for k, v in self.timings.items())
        return (f"transcript ({self.decoder_label}): {self.transcript}\n"
                f"emotion: {self.emotion}  (voice_risk={self.voice_risk})\n"
                f"  {ep}\n"
                f"verdict: {self.verdict.upper()}\n"
                f"  {vp}\n"
                f"[{t}]")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class Pipeline:
    def __init__(self, paths: Paths | None = None, device: str = "cpu",
                 use_kenlm: bool = True, num_heads: int = 4, log=print):
        self.paths = paths or Paths()
        self.device = device
        self.log = log

        # Cheapest check first, and before any heavy import: a half-installed
        # environment should still be told the real problem is missing weights.
        missing = self.paths.missing()
        if missing:
            raise FileNotFoundError(
                "Missing model files under " + str(self.paths.root) + ":\n  "
                + "\n  ".join(missing)
                + "\n\nSee app/README.md for the expected layout.")

        import torch

        t0 = time.time()
        self.asr_cfg = json.loads((self.paths.asr / "config.json").read_text())
        self.ser_cfg = json.loads((self.paths.ser / "config.json").read_text())

        self.backbone = SharedBackbone(self.asr_cfg, self.ser_cfg, device,
                                       local_dir=self.paths.backbone())
        n_asr = self.backbone.load_adapter(self.paths.asr / "adapter.pt", "asr")
        n_ser = self.backbone.load_adapter(self.paths.ser / "adapter.pt", "ser")
        log(f"[backbone] one frozen {self.backbone.backbone_id} "
            f"({self.backbone.loaded_from}) · asr adapter {n_asr} tensors · "
            f"ser adapter {n_ser} tensors")

        self.vocab = build_vocab()
        self.i2c = {v: k for k, v in self.vocab.items()}
        self.blank, self.unk = self.vocab["[PAD]"], self.vocab["[UNK]"]

        self.asr_head = make_asr_head(self.asr_cfg, len(self.vocab), device)
        self.asr_head.load_state_dict(
            torch.load(self.paths.asr / "head.pt", map_location=device))
        self.ser_head = make_ser_head(self.ser_cfg, device)
        self.ser_head.load_state_dict(
            torch.load(self.paths.ser / "head.pt", map_location=device))
        log(f"[asr] {self.asr_cfg.get('run', '?')} · ws={list(self.asr_cfg['ws'])}")
        log(f"[ser] {self.ser_cfg.get('run', '?')} · ws={list(self.ser_cfg['ws'])}")

        # --- fusion head -------------------------------------------------
        ckpt_path = self.paths.fusion_checkpoints()[0]
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.fusion_meta = dict(payload.get("meta", {}))
        method = self.fusion_meta.get("method")
        if method and method != "intermediate_attn":
            raise RuntimeError(
                f"{ckpt_path.name} is a '{method}' head; this pipeline implements "
                "intermediate_attn only. Use WINNER_intermediate_attn_*.pt "
                "(DEMO_BRIEF.md §7 rules out `early`).")
        self.fusion_head, self.text_dim = make_attn_fusion_head(
            payload["state_dict"], device, num_heads=num_heads)
        self.fusion_meta.setdefault("checkpoint", ckpt_path.name)
        self.fusion_meta["num_heads_assumed"] = num_heads
        log(f"[fusion] {ckpt_path.name} · {method} / "
            f"{self.fusion_meta.get('encoder')} · regime="
            f"{self.fusion_meta.get('regime')} · text_dim={self.text_dim}")

        # --- text encoder: per-token last_hidden_state, NOT pooled --------
        from transformers import AutoModel, AutoTokenizer

        enc = self.fusion_meta.get("encoder", "bert")
        model_id = TEXT_ENCODER_IDS.get(enc, enc)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = AutoModel.from_pretrained(model_id).to(device).eval()
        got = int(self.text_model.config.hidden_size)
        if got != self.text_dim:
            raise RuntimeError(
                f"text encoder {model_id} is {got}-dim but the fusion head expects "
                f"{self.text_dim}. meta['encoder'] and the weights disagree.")
        log(f"[text] {model_id} · tokenwise last_hidden_state · "
            f"max_length={TOKENIZER_MAX_LENGTH}")

        # --- decoder ------------------------------------------------------
        self.decoder = None
        self.decoder_name = "greedy"
        self.lm_params: dict = {}
        if use_kenlm:
            self._try_kenlm()

        log(f"[ready] {time.time() - t0:.1f}s · device={device} · "
            f"decoder={self.decoder_name}")

    # -- KenLM ------------------------------------------------------------

    def _try_kenlm(self) -> None:
        """Tuned decoder parameters come from lm_params_clean.json. pyctcdecode's
        defaults (0.5 / 1.0 / 100) are NOT these, and the reported WER was
        measured with the tuned values."""
        arpa = self.paths.arpa()
        if arpa is None:
            self.log("[kenlm] no .arpa under lm/ — greedy decoding")
            return

        pp = self.paths.lm_params()
        if pp is None:
            self.log("[kenlm] lm_params_clean.json missing — refusing to guess "
                     "alpha/beta; greedy decoding")
            return
        params = json.loads(pp.read_text())
        alpha, beta = float(params["alpha"]), float(params["beta"])
        beam = int(params["beam_width"])

        if Path(params.get("lm_path", "")).name not in ("", arpa.name):
            self.log(f"[kenlm] WARNING lm_params was tuned against "
                     f"{Path(params['lm_path']).name} but lm/ holds {arpa.name}; "
                     "alpha is not transferable between language models")

        try:
            from pyctcdecode import build_ctcdecoder
        except Exception as exc:
            # An .arpa on disk means the user asked for KenLM, so a fallback is
            # a thwarted intention, not a default. Say so at a volume that
            # survives Gradio's startup output.
            self.log(f"[kenlm] !! FALLING BACK TO GREEDY: pyctcdecode is not "
                     f"installed ({exc}).")
            self.log(f"[kenlm] !! {arpa.name} is on disk but cannot be used. "
                     "The .arpa is data; the decoder is code.")
            self.log("[kenlm] !! Fix: pip install pyctcdecode pypi-kenlm")
            return
        try:
            self.decoder = build_ctcdecoder(self._pyctc_labels(),
                                            kenlm_model_path=str(arpa),
                                            alpha=alpha, beta=beta)
            self.decoder_name = "kenlm"
            self.beam_width = beam
            self.lm_params = {"alpha": alpha, "beta": beta, "beam_width": beam,
                              "lm": arpa.name}
            self.log(f"[kenlm] {arpa.name} · alpha={alpha} beta={beta} beam={beam}")
        except Exception as exc:
            self.log(f"[kenlm] !! FALLING BACK TO GREEDY: building the decoder from "
                     f"{arpa.name} failed ({type(exc).__name__}: {exc}).")
            self.log("[kenlm] !! If this names the `kenlm` module, install it: "
                     "pip install pypi-kenlm  (needs Python <= 3.11 on Windows)")
            self.decoder = None

    def _pyctc_labels(self) -> list[str]:
        labels = [""] * len(self.vocab)
        for ch, i in self.vocab.items():
            labels[i] = {"|": " ", "[PAD]": "", "[UNK]": "⁇"}.get(ch, ch)
        return labels

    # -- channels ---------------------------------------------------------

    def run_asr(self, wav: np.ndarray) -> tuple[str, str]:
        import torch

        hs = self.backbone.stack(wav, self.asr_cfg["ws"], "asr")
        with torch.no_grad():
            logits = self.asr_head(hs)[0].float()
        n = min(int(logits.shape[0]), self.backbone.feat_lengths(len(wav)))
        logits = logits[:n]
        if self.decoder is not None:
            lp = logits.log_softmax(-1).float().cpu().numpy()
            return self.decoder.decode(lp, beam_width=self.beam_width).strip(), "kenlm"
        ids = logits.argmax(-1).cpu().numpy().tolist()
        return ctc_greedy(ids, self.i2c, self.blank, self.unk), "greedy"

    def run_ser(self, wav: np.ndarray) -> tuple[str, dict[str, float]]:
        import torch

        hs = self.backbone.stack(wav, self.ser_cfg["ws"], "ser")
        with torch.no_grad():
            probs = self.ser_head(hs).float().softmax(-1)[0].cpu().numpy()
        return SER_CLASSES[int(probs.argmax())], {
            c: float(p) for c, p in zip(SER_CLASSES, probs)}

    def encode_text(self, text: str):
        """Per-token `last_hidden_state` + attention mask, tokenised exactly as
        `encoders.embed_texts_tokenwise` did. Not CLS-pooled, not
        normalised — the attention head consumes the token sequence."""
        import torch

        enc = self.tokenizer([text], max_length=TOKENIZER_MAX_LENGTH, truncation=True,
                             padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.text_model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.float(), mask

    def run_fusion(self, transcript: str, emotion: str) -> tuple[str, dict[str, float]]:
        import torch

        if emotion not in SER_CLASSES:
            raise ValueError(f"unknown emotion {emotion!r}")
        onehot = np.zeros((1, 6), np.float32)
        # Cross the SER/fusion boundary through the table the tests pin, not
        # through a second lookup that happens to agree with it. TC-05 and
        # TC-05b assert SER_TO_FUSION; before this line used it, both could pass
        # while the live path was permuted.
        onehot[0, SER_TO_FUSION[SER_CLASSES.index(emotion)]] = 1.0
        tokens, mask = self.encode_text(transcript)
        with torch.no_grad():
            logits = self.fusion_head(
                torch.from_numpy(onehot).to(self.device), tokens, mask)
            probs = logits.float().softmax(-1)[0].cpu().numpy()
        return FUSION_LABELS[int(probs.argmax())], {
            l: float(p) for l, p in zip(FUSION_LABELS, probs)}

    # -- end to end -------------------------------------------------------

    def __call__(self, wav: np.ndarray, sr: int = SAMPLE_RATE,
                 emotion_override: str | None = None,
                 text_override: str | None = None) -> Result:
        wav = prepare_audio(wav, sr)
        timings = {}

        t = time.time()
        if text_override is None:
            transcript, decoder = self.run_asr(wav)
        else:
            transcript, decoder = text_override, "manual"
        timings["asr"] = time.time() - t

        t = time.time()
        if emotion_override is None:
            emotion, eprobs = self.run_ser(wav)
        else:
            emotion = emotion_override
            eprobs = {c: float(c == emotion) for c in SER_CLASSES}
        timings["ser"] = time.time() - t

        t = time.time()
        verdict, vprobs = self.run_fusion(transcript, emotion)
        timings["fusion"] = time.time() - t
        timings["audio"] = len(wav) / SAMPLE_RATE

        return Result(transcript=transcript, decoder=decoder, emotion=emotion,
                      decoder_params=self.lm_params if decoder == "kenlm" else None,
                      emotion_probs=eprobs,
                      voice_risk="high" if emotion in HIGH_RISK_EMOTIONS else "low",
                      verdict=verdict, verdict_probs=vprobs, timings=timings)


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def prepare_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 in [-1, 1] at 16 kHz.

    Training fed `int16 / 32768.0` with no further normalisation, so inference
    must not normalise either.
    """
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[1] <= wav.shape[0] else 0)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / 32768.0
    wav = wav.astype(np.float32)
    if sr != SAMPLE_RATE:
        n = int(round(len(wav) * SAMPLE_RATE / sr))
        wav = np.interp(np.linspace(0, len(wav) - 1, n),
                        np.arange(len(wav)), wav).astype(np.float32)
    if len(wav) < SAMPLE_RATE // 2:
        wav = np.pad(wav, (0, SAMPLE_RATE // 2 - len(wav)))
    return wav


class AudioDecodeError(RuntimeError):
    """Raised when no available decoder can read a file. Carries the fix."""


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode any audio file to (samples, sample_rate).

    `soundfile` (libsndfile) is tried first: it reads wav, flac, ogg and — since
    libsndfile 1.1, which recent wheels bundle — mp3. It does NOT read AAC in an
    MP4 container, which is what `.m4a` normally is, and what phones and
    Windows Voice Recorder produce by default.

    PyAV is tried second when installed, which covers m4a/aac/mp4. If neither
    works the error names the file and the two ways out, rather than surfacing
    libsndfile's "Format not recognised" with no context.
    """
    path = Path(path)
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return wav, sr
    except Exception as sf_exc:
        try:
            return _load_with_av(path)
        except ImportError:
            raise AudioDecodeError(
                f"Cannot decode {path.name}: soundfile reported "
                f"{type(sf_exc).__name__}, and PyAV is not installed.\n"
                f"'{path.suffix}' is most likely AAC, which libsndfile does not read.\n"
                "Either convert the clip to .wav or .mp3, or: pip install av"
            ) from sf_exc
        except Exception as av_exc:
            raise AudioDecodeError(
                f"Cannot decode {path.name}: soundfile said {sf_exc}; "
                f"PyAV said {av_exc}. Convert the clip to .wav."
            ) from av_exc


def _load_with_av(path: Path) -> tuple[np.ndarray, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sr = int(stream.codec_context.sample_rate)
        chunks = []
        for frame in container.decode(stream):
            a = frame.to_ndarray()
            # PyAV gives (channels, samples) for planar formats and
            # (1, samples*channels) interleaved otherwise.
            if a.ndim == 2 and a.shape[0] > 1:
                a = a.mean(axis=0)
            chunks.append(np.asarray(a, dtype=np.float32).reshape(-1))
    if not chunks:
        raise RuntimeError("no audio frames decoded")
    wav = np.concatenate(chunks)
    if np.issubdtype(wav.dtype, np.integer) or np.abs(wav).max() > 1.5:
        wav = wav / 32768.0
    return wav.astype(np.float32), sr


# --------------------------------------------------------------------------
# Self-test — invariants that need no model files
# --------------------------------------------------------------------------


def self_test() -> int:
    fails = []

    def chk(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    print("label spaces")
    chk("same six emotions", set(SER_CLASSES) == set(FUSION_EMOTIONS))
    chk("orders genuinely differ (the trap is real)", SER_CLASSES != FUSION_EMOTIONS)
    chk("SER_TO_FUSION is a permutation", sorted(SER_TO_FUSION) == list(range(6)))
    chk("mapping is by name",
        all(FUSION_EMOTIONS[SER_TO_FUSION[i]] == SER_CLASSES[i] for i in range(6)))
    chk("identity mapping would be wrong for 4 of 6",
        sum(1 for i in range(6) if SER_TO_FUSION[i] != i) == 4,
        f"differs at {[SER_CLASSES[i] for i in range(6) if SER_TO_FUSION[i] != i]}")

    print("ctc vocabulary")
    v = build_vocab()
    chk("30 symbols", len(v) == 30, f"got {len(v)}")
    chk("blank is [PAD]", v["[PAD]"] == 29)
    chk("charset unchanged", set(v) == set(ASR_CHARS) | {"|", "[UNK]", "[PAD]"})

    print("greedy decode")
    i2c = {i: c for c, i in v.items()}
    ids = [v["H"], v["H"], v["E"], v["[PAD]"], v["E"], v["|"], v["Y"], v["O"]]
    got = ctc_greedy(ids, i2c, v["[PAD]"], v["[UNK]"])
    chk("collapses repeats, drops blanks, '|'->space", got == "HEE YO", f"got {got!r}")

    print("lora spans")
    chk("asr style (explicit list)",
        _lora_layers({"lora_layers": list(range(1, 13))}) == list(range(12)))
    chk("ser style (lo/hi inclusive)",
        _lora_layers({"lora_lo": 1, "lora_hi": 12}) == list(range(12)))

    print("audio")
    a = prepare_audio(np.zeros(8000, np.int16), 8000)
    chk("resamples 8k->16k", len(a) == 16000, f"got {len(a)}")
    chk("float32 output", a.dtype == np.float32)
    chk("stereo -> mono", prepare_audio(np.zeros((100, 2), np.float32), 16000).ndim == 1)

    print()
    if fails:
        print(f"{len(fails)} check(s) FAILED: {', '.join(fails)}")
        return 1
    print("all checks passed")
    return 0


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Emergency-call anomaly inference pipeline")
    ap.add_argument("wav", nargs="?", help="audio file to analyse")
    ap.add_argument("--self-test", action="store_true",
                    help="offline invariant checks (no model files needed)")
    ap.add_argument("--models", default=None, help="path to models/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-kenlm", action="store_true")
    ap.add_argument("--num-heads", type=int, default=4,
                    help="MultiheadAttention head count; not recoverable from the "
                         "checkpoint, 4 is the trained value")
    ap.add_argument("--text", default=None, help="bypass ASR with this transcript")
    ap.add_argument("--emotion", default=None, choices=SER_CLASSES,
                    help="bypass SER with this emotion")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.wav:
        ap.error("give an audio file, or --self-test")

    paths = Paths(Path(args.models)) if args.models else Paths()
    pipe = Pipeline(paths, device=args.device, use_kenlm=not args.no_kenlm,
                    num_heads=args.num_heads)
    wav, sr = load_wav(args.wav)
    print()
    print(pipe(wav, sr, emotion_override=args.emotion, text_override=args.text).as_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
