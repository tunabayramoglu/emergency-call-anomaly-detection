"""
interactive demo (Gradio).

Thin UI over pipeline.Pipeline.  All model logic lives in the pipeline;
this file only renders.  Run:

    python app.py                 # http://127.0.0.1:7860
    python app.py --no-kenlm      # skip the LM if it will not install

The 2x2 panel mirrors _Staj/meeting/D1_fusion_matrix.png so the demo and the
slide show the same object.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import gradio as gr

from pipeline import (FUSION_LABELS, HIGH_RISK_EMOTIONS, SER_CLASSES,
                            AudioDecodeError, Paths, Pipeline, load_wav)

CLIPS_DIR = Path(__file__).resolve().parent / "clips"

# libsndfile reads these without help. mp3 needs libsndfile >= 1.1, which recent
# `soundfile` wheels bundle.
NATIVE_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".au"}

# AAC in an MP4 container — what phones and Windows Voice Recorder produce by
# default. libsndfile cannot read it; `load_wav` falls back to PyAV if it is
# installed, and says so clearly if it is not. Listed anyway: a file silently
# missing from the dropdown is worse than one that explains itself when clicked.
FALLBACK_EXTENSIONS = {".m4a", ".mp4", ".aac", ".wma"}

CLIP_EXTENSIONS = NATIVE_EXTENSIONS | FALLBACK_EXTENSIONS

VERDICT_STYLE = {
    "normal":     ("#1a7f37", "NORMAL"),
    "borderline": ("#9a6700", "BORDERLINE"),
    "anomaly":    ("#cf222e", "ANOMALY"),
}


def _bars(probs: dict[str, float], highlight: str | None = None) -> str:
    rows = []
    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        w = max(1.0, v * 100)
        bold = "font-weight:600;" if k == highlight else ""
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;{bold}'>"
            f"<span style='width:96px;font-size:13px'>{html.escape(k)}</span>"
            f"<span style='flex:0 0 220px;background:#e6e6e6;height:10px;border-radius:5px'>"
            f"<span style='display:block;width:{w:.1f}%;background:#4c78a8;height:10px;"
            f"border-radius:5px'></span></span>"
            f"<span style='font-size:12px;color:#555'>{v:.3f}</span></div>")
    return "".join(rows)


def render(result, note: str = "") -> str:
    colour, label = VERDICT_STYLE[result.verdict]
    rtf = result.timings["asr"] + result.timings["ser"] + result.timings["fusion"]
    rtf = rtf / max(result.timings["audio"], 1e-6)
    # Colours are set explicitly on every block. Inheriting them meant the
    # transcript rendered white-on-white under some Gradio themes.
    return f"""
<div style='font-family:system-ui,sans-serif;color:#111'>
  <div style='font-size:26px;font-weight:700;color:{colour};margin-bottom:2px'>{label}</div>
  <div style='margin-bottom:14px'>{_bars(result.verdict_probs, result.verdict)}</div>

  <div style='font-size:12px;text-transform:uppercase;color:#666;letter-spacing:.05em'>
    transcript <span style='text-transform:none'>({html.escape(result.decoder_label)})</span></div>
  <div style='font-family:ui-monospace,monospace;background:#f6f8fa;color:#111;
              padding:8px;border:1px solid #e2e5e9;border-radius:6px;
              margin:4px 0 14px'>{html.escape(result.transcript) or "<i style='color:#888'>(empty)</i>"}</div>

  <div style='font-size:12px;text-transform:uppercase;color:#666;letter-spacing:.05em'>
    voice emotion &mdash; <b style='color:#111'>{html.escape(result.emotion)}</b>
    (voice_risk {html.escape(result.voice_risk)})</div>
  <div style='margin:4px 0 14px'>{_bars(result.emotion_probs, result.emotion)}</div>

  <div style='font-size:11px;color:#777'>
    asr {result.timings['asr']:.2f}s &middot; ser {result.timings['ser']:.2f}s &middot;
    fusion {result.timings['fusion']:.2f}s &middot; audio {result.timings['audio']:.2f}s
    &middot; RTF {rtf:.3f}</div>
  {f"<div style='font-size:11px;color:#a33;margin-top:6px'>{html.escape(note)}</div>" if note else ""}
</div>"""


def find_clips() -> list[Path]:
    """Any audio file in clips/, whatever it is called. Names carry no meaning
    to the pipeline. Sorted for a stable dropdown order."""
    if not CLIPS_DIR.exists():
        return []
    return sorted(p for p in CLIPS_DIR.iterdir()
                  if p.suffix.lower() in CLIP_EXTENSIONS)


def build(pipe: Pipeline) -> gr.Blocks:
    clips = find_clips()

    def analyse(audio, note=""):
        if audio is None:
            return "<i>No audio.</i>"
        try:
            if isinstance(audio, tuple):        # gradio numpy mode: (sr, data)
                sr, data = audio
                return render(pipe(data, sr), note)
            wav, sr = load_wav(audio)
            return render(pipe(wav, sr), note)
        except AudioDecodeError as exc:
            # A traceback in the console is no use to someone standing in front
            # of a room. Show the reason and the fix where they are looking.
            return (f"<div style='font-family:system-ui;color:#cf222e;font-weight:600;"
                    f"margin-bottom:6px'>Could not read that file</div>"
                    f"<pre style='white-space:pre-wrap;font-size:12px;color:#111;"
                    f"background:#f6f8fa;border:1px solid #e2e5e9;border-radius:6px;"
                    f"padding:8px'>{html.escape(str(exc))}</pre>")

    with gr.Blocks(title="Emergency-call anomaly detector") as demo:
        gr.Markdown("## multimodal emergency-call anomaly detector")
        gr.Markdown(
            "The anomaly is a **mismatch between how something is said and what is said**. "
            "Calm voice + catastrophic content, or panicked voice + harmless content, "
            "both signal risk.")

        with gr.Tab("Prepared clips"):
            if clips:
                pick = gr.Dropdown([p.name for p in clips], label="clip",
                                   value=clips[0].name)
                player = gr.Audio(type="filepath", label="preview",
                                  value=str(clips[0]))
                out1 = gr.HTML()
                pick.change(lambda n: str(CLIPS_DIR / n), pick, player)
                gr.Button("Analyse", variant="primary").click(
                    lambda n: analyse(str(CLIPS_DIR / n)), pick, out1)
            else:
                gr.Markdown(
                    f"No audio found in `{CLIPS_DIR}`.\n\n"
                    f"Drop files there &mdash; {', '.join(sorted(CLIP_EXTENSIONS))} "
                    "are accepted, and the filenames are up to you.")

        with gr.Tab("Microphone"):
            mic = gr.Audio(sources=["microphone"], type="numpy", label="record")
            out2 = gr.HTML()
            gr.Button("Analyse", variant="primary").click(
                lambda a: analyse(a, "live audio — SER is noisier here than on the "
                                     "held-out academic split"), mic, out2)

        with gr.Tab("Fusion only (no audio)"):
            txt = gr.Textbox(label="transcript", value="THERE'S A FIRE IN THE BUILDING")
            emo = gr.Dropdown(SER_CLASSES, label="emotion", value="neutral")
            out3 = gr.HTML()

            def fuse(t, e):
                verdict, probs = pipe.run_fusion(t, e)
                colour, label = VERDICT_STYLE[verdict]
                return (f"<div style='font-family:system-ui;font-size:24px;font-weight:700;"
                        f"color:{colour}'>{label}</div>"
                        f"<div style='margin-top:8px'>{_bars(probs, verdict)}</div>"
                        f"<div style='margin-top:10px;font-size:12px;color:#666'>"
                        f"voice_risk = {'high' if e in HIGH_RISK_EMOTIONS else 'low'}</div>")

            gr.Button("Fuse", variant="primary").click(fuse, [txt, emo], out3)

    return demo


def build_parser() -> argparse.ArgumentParser:
    """Separated from main() so tests can exercise it without launching a server.

    There are deliberately no --alpha/--beta/--beam flags. Decoder parameters
    live in lm_params_clean.json, tuned for this checkpoint; a CLI override
    would let someone silently decode with values the reported WER was never
    measured at.
    """
    ap = argparse.ArgumentParser(prog="app")
    ap.add_argument("--models", default=None, help="path to models/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-kenlm", action="store_true")
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = Paths(args.models) if args.models else Paths()
    pipe = Pipeline(paths, device=args.device, use_kenlm=not args.no_kenlm,
                    num_heads=args.num_heads)
    build(pipe).launch(server_port=args.port, share=args.share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
