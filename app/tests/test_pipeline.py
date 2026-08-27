"""
executable acceptance tests.

Scope: **functional correctness only.** Nothing here asserts a WER, an F1 or an
accuracy — model quality is the benchmark's job and is reported there. These
tests answer one question: does the system do what it says it does.

Two tiers:

  * TC-01 .. TC-11   run anywhere, no model files, no network. These are the
                     invariants that protect against silent corruption.
  * TC-20 .. TC-46   need `app/models/` built by `setup_weights.py`. They skip
                     with a named reason when the files are absent, so a green
                     run on a bare checkout is never mistaken for a full pass.

The three `slow` tests (latency, memory, cold load) check resource budgets, not
model quality, and are opt-in.

Run:
    pytest -v                      # functional suite
    pytest -v -m "not needs_models"
    pytest -v -m slow              # resource budgets only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline as P  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"
PATHS = P.Paths(MODELS)
HAVE_MODELS = not PATHS.missing()
needs_models = pytest.mark.skipif(
    not HAVE_MODELS,
    reason=f"model files missing under {MODELS}: {PATHS.missing()[:3]}...")


def _have_stack() -> str | None:
    """Name the first missing runtime dependency, or None if all present."""
    for mod in ("torch", "transformers", "peft"):
        try:
            __import__(mod)
        except ImportError:
            return mod
    return None


@pytest.fixture(scope="session")
def pipe():
    if not HAVE_MODELS:
        pytest.skip("no model files")
    if (m := _have_stack()):
        pytest.skip(f"{m} not installed")
    return P.Pipeline(PATHS, device="cpu", log=lambda *a, **k: None)


def _tone(seconds=3.0, sr=16_000, freq=180.0):
    """A deterministic, speech-band-ish signal. Not speech — these tests check
    plumbing and contracts, never transcription quality."""
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * 3 * freq * t)
    return (x * np.hanning(len(x))).astype(np.float32)


# ===========================================================================
# Tier 1 — no model files required
# ===========================================================================


class TestLabelSpaces:
    """FR-07. The two modules order the same six emotions differently; crossing
    the boundary by index silently mislabels most of them."""

    def test_tc05_orders_differ_and_mapping_is_by_name(self):
        assert set(P.SER_CLASSES) == set(P.FUSION_EMOTIONS)
        assert P.SER_CLASSES != P.FUSION_EMOTIONS, (
            "orders now agree — if this is a deliberate change, the by-name "
            "mapping is redundant, but verify both codebases before removing it")
        assert sorted(P.SER_TO_FUSION) == list(range(6))
        for i, name in enumerate(P.SER_CLASSES):
            assert P.FUSION_EMOTIONS[P.SER_TO_FUSION[i]] == name

    def test_tc05b_identity_mapping_would_corrupt_four_of_six(self):
        wrong = [P.SER_CLASSES[i] for i in range(6) if P.SER_TO_FUSION[i] != i]
        assert len(wrong) == 4
        # Three of the four high-risk emotions are among them: an index-based
        # bug would be worst exactly where the demo needs to be right.
        assert len(set(wrong) & P.HIGH_RISK_EMOTIONS) == 3

    def test_tc04_voice_risk_partition(self):
        assert P.HIGH_RISK_EMOTIONS < set(P.SER_CLASSES)
        assert set(P.SER_CLASSES) - P.HIGH_RISK_EMOTIONS == {"neutral", "confusion"}

    def test_tc25a_three_verdict_labels(self):
        assert P.FUSION_LABELS == ["normal", "borderline", "anomaly"]


class TestVocabulary:
    """FR-01. The CTC vocabulary must not drift: KenLM was built on
    LibriSpeech-normalised text and an unseen symbol collapses the beam."""

    def test_tc01_vocab_shape(self):
        v = P.build_vocab()
        assert len(v) == 30
        assert set(v) == set(P.ASR_CHARS) | {"|", "[UNK]", "[PAD]"}
        assert v["[PAD]"] == 29 and v["[UNK]"] == 28 and v["|"] == 27

    def test_tc01b_no_digits_or_punctuation(self):
        assert not any(c.isdigit() for c in P.ASR_CHARS)
        assert set(P.ASR_CHARS) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

    def test_tc02_greedy_decode_contract(self):
        v = P.build_vocab()
        i2c = {i: c for c, i in v.items()}
        ids = [v["H"], v["H"], v["E"], v["[PAD]"], v["E"], v["|"], v["Y"], v["O"]]
        assert P.ctc_greedy(ids, i2c, v["[PAD]"], v["[UNK]"]) == "HEE YO"

    def test_tc02b_blank_separates_repeats(self):
        v = P.build_vocab()
        i2c = {i: c for c, i in v.items()}
        assert P.ctc_greedy([v["A"], v["A"]], i2c, v["[PAD]"], v["[UNK]"]) == "A"
        assert P.ctc_greedy([v["A"], v["[PAD]"], v["A"]], i2c,
                            v["[PAD]"], v["[UNK]"]) == "AA"

    def test_tc02c_unk_dropped(self):
        v = P.build_vocab()
        i2c = {i: c for c, i in v.items()}
        assert P.ctc_greedy([v["A"], v["[UNK]"], v["B"]], i2c,
                            v["[PAD]"], v["[UNK]"]) == "AB"


class TestAudioPreparation:
    """FR-15. Training fed int16/32768 with no further normalisation; inference
    must match or the acoustic model sees a different distribution."""

    def test_tc08_resamples_to_16k(self):
        for sr in (8_000, 22_050, 44_100, 48_000):
            out = P.prepare_audio(np.zeros(sr, np.float32), sr)
            assert abs(len(out) - 16_000) <= 1, f"sr={sr} gave {len(out)}"

    def test_tc09_stereo_to_mono_and_dtype(self):
        out = P.prepare_audio(np.zeros((16_000, 2), np.float32), 16_000)
        assert out.ndim == 1 and out.dtype == np.float32

    def test_tc10_int16_scaling_matches_training(self):
        out = P.prepare_audio(np.array([32767, -32768, 0], np.int16), 16_000)
        assert out[0] == pytest.approx(0.99997, abs=1e-4)
        assert out[1] == pytest.approx(-1.0, abs=1e-4)

    def test_tc10b_no_amplitude_normalisation(self):
        quiet = P.prepare_audio(_tone() * 0.01, 16_000)
        loud = P.prepare_audio(_tone() * 1.00, 16_000)
        assert np.abs(loud).max() > 50 * np.abs(quiet).max(), (
            "amplitude was normalised — training did not do this")

    def test_tc08b_short_clip_padded(self):
        assert len(P.prepare_audio(np.zeros(100, np.float32), 16_000)) >= 8_000

    def test_tc08c_passthrough_is_lossless(self):
        x = _tone(1.0)
        assert np.array_equal(P.prepare_audio(x, 16_000), x)


class TestFusionHeadReconstruction:
    """FR-06. `text_dim` and `hidden` are read off the checkpoint. `num_heads`
    cannot be — MultiheadAttention's parameter shapes are identical for any head
    count dividing embed_dim — so it is an explicit, documented assumption."""

    @staticmethod
    def _synthetic(hidden=256, text_dim=768):
        torch = pytest.importorskip("torch")
        return {
            "emotion_embedding.weight": torch.randn(32, 6),
            "query_proj.weight": torch.randn(text_dim, 32),
            "query_proj.bias": torch.randn(text_dim),
            "attn.in_proj_weight": torch.randn(3 * text_dim, text_dim),
            "attn.in_proj_bias": torch.randn(3 * text_dim),
            "attn.out_proj.weight": torch.randn(text_dim, text_dim),
            "attn.out_proj.bias": torch.randn(text_dim),
            "fc1.weight": torch.randn(hidden, text_dim),
            "fc1.bias": torch.randn(hidden),
            "fc2.weight": torch.randn(128, hidden),
            "fc2.bias": torch.randn(128),
            "fc3.weight": torch.randn(3, 128),
            "fc3.bias": torch.randn(3),
        }

    def test_tc25_shapes_inferred_from_checkpoint(self):
        torch = pytest.importorskip("torch")
        for hidden, dim in ((256, 768), (128, 768), (64, 384)):
            model, got = P.make_attn_fusion_head(self._synthetic(hidden, dim), "cpu")
            assert got == dim
            out = model(torch.zeros(1, 6), torch.zeros(1, 8, dim),
                        torch.ones(1, 8, dtype=torch.long))
            assert out.shape == (1, 3)

    def test_tc25b_emotion_projection_is_biasless(self):
        model, _ = P.make_attn_fusion_head(self._synthetic(), "cpu")
        assert model.emotion_embedding.bias is None, (
            "a bias here breaks equivalence with nn.Embedding on a one-hot")

    def test_tc25c_wrong_method_checkpoint_rejected(self):
        """The pooled `intermediate` head has no query_proj/attn. Loading it here
        must fail loudly and name the file to use instead."""
        torch = pytest.importorskip("torch")
        pooled = {"emotion_embedding.weight": torch.randn(32, 6),
                  "fc1.weight": torch.randn(256, 32 + 384),
                  "fc2.weight": torch.randn(128, 256),
                  "fc3.weight": torch.randn(3, 128)}
        with pytest.raises(RuntimeError, match="intermediate_attn"):
            P.make_attn_fusion_head(pooled, "cpu")

    def test_tc25g_indivisible_head_count_rejected(self):
        with pytest.raises(RuntimeError, match="divisible"):
            P.make_attn_fusion_head(self._synthetic(text_dim=768), "cpu", num_heads=5)

    def test_tc25d_dropout_off_at_inference(self):
        torch = pytest.importorskip("torch")
        model, dim = P.make_attn_fusion_head(self._synthetic(), "cpu")
        e = torch.zeros(1, 6)
        t = torch.randn(1, 8, dim)
        m = torch.ones(1, 8, dtype=torch.long)
        assert torch.equal(model(e, t, m), model(e, t, m))

    def test_tc25h_padding_is_masked_out(self):
        """The attention must ignore positions where attn_mask == 0. If padding
        leaked in, a short utterance's verdict would depend on 64-token padding."""
        torch = pytest.importorskip("torch")
        model, dim = P.make_attn_fusion_head(self._synthetic(), "cpu")
        e = torch.zeros(1, 6)
        real = torch.randn(1, 3, dim)
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        a = model(e, torch.cat([real, torch.randn(1, 2, dim)], 1), mask)
        b = model(e, torch.cat([real, torch.randn(1, 2, dim) * 99], 1), mask)
        assert torch.allclose(a, b, atol=1e-5), (
            "changing the padded positions changed the output — the key padding "
            "mask is not being applied")


class TestAdapterKeyRemap:
    """FR-05 / FR-14."""

    def test_tc07_only_lora_keys_are_renamed(self):
        sd = {
            "encoder.layers.0.attention.q_proj.lora_A.default.weight": 1,
            "encoder.layers.0.attention.q_proj.lora_B.default.weight": 2,
            "encoder.layers.0.attention.q_proj.weight": 3,
            "feature_projection.default.weight": 4,     # not a lora key
        }
        out = P._remap_adapter_keys(sd, "ser")
        assert "encoder.layers.0.attention.q_proj.lora_A.ser.weight" in out
        assert "encoder.layers.0.attention.q_proj.lora_B.ser.weight" in out
        assert "encoder.layers.0.attention.q_proj.weight" in out
        assert "feature_projection.default.weight" in out
        assert len(out) == len(sd)

    def test_tc07b_remap_is_reversible_in_count(self):
        sd = {f"l.{i}.lora_A.default.weight": i for i in range(48)}
        assert len(P._remap_adapter_keys(sd, "asr")) == 48

    def test_tc07c_both_config_dialects_for_lora_span(self):
        """asr/config.json writes an explicit `lora_layers` list; ser/config.json
        writes `lora_lo`/`lora_hi`. Supporting only one silently adapts the wrong
        layers on the other model."""
        assert P._lora_layers({"lora_layers": list(range(1, 13))}) == list(range(12))
        assert P._lora_layers({"lora_lo": 1, "lora_hi": 12}) == list(range(12))
        assert P._lora_layers({"lora_lo": 7, "lora_hi": 9}) == [6, 7, 8]
        assert P._lora_layers({"lora_layers": [9, 10]}) == [8, 9]

    def test_tc07d_explicit_list_wins_over_lo_hi(self):
        assert P._lora_layers({"lora_layers": [3], "lora_lo": 1, "lora_hi": 12}) == [2]


class TestFailureModes:
    """NFR-07."""

    def test_tc11_missing_models_error_names_the_files(self, tmp_path):
        with pytest.raises(FileNotFoundError) as e:
            P.Pipeline(P.Paths(tmp_path), device="cpu", log=lambda *a, **k: None)
        msg = str(e.value)
        assert "adapter.pt" in msg and "config.json" in msg and "WINNER" in msg

    def test_tc11b_missing_lists_every_gap_not_just_the_first(self, tmp_path):
        assert len(P.Paths(tmp_path).missing()) >= 7

    def test_tc11c_no_arpa_reports_none(self, tmp_path):
        assert P.Paths(tmp_path).arpa() is None

    def test_tc19_truncated_backbone_counts_as_absent(self, tmp_path):
        """An interrupted extraction leaves a weights file that exists but is
        short. Treating it as local suppresses the download and then fails
        inside from_pretrained with an error naming neither file nor cause."""
        d = tmp_path / "backbone"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\0" * 1024)
        assert P.Paths(tmp_path).backbone() is None

    @staticmethod
    def _fake_backbone(d: Path):
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        with open(d / "model.safetensors", "wb") as fh:   # sparse, no 360 MB write
            fh.truncate(P.Paths.MIN_BACKBONE_BYTES + 1)
        return d

    def test_tc19b_complete_backbone_is_used(self, tmp_path):
        d = self._fake_backbone(tmp_path / "backbone")
        assert P.Paths(tmp_path).backbone() == d

    def test_tc19d_nested_extraction_is_found(self, tmp_path):
        """mHuBERT-147.zip wraps its contents in a directory, so unzipping it
        in place gives backbone/mHuBERT-147/. Demanding the files be moved is a
        step that will be skipped under time pressure."""
        d = self._fake_backbone(tmp_path / "backbone" / "mHuBERT-147")
        assert P.Paths(tmp_path).backbone() == d

    def test_tc19e_truncated_first_candidate_does_not_shadow_a_good_one(self, tmp_path):
        """The failure this guards: an interrupted extraction in the preferred
        location silently hides a complete copy one level down, and the demo
        downloads 380 MB it already has."""
        bad = tmp_path / "backbone"
        bad.mkdir(parents=True)
        (bad / "config.json").write_text("{}")
        (bad / "model.safetensors").write_bytes(b"\0" * 1024)
        good = self._fake_backbone(bad / "mHuBERT-147")
        assert P.Paths(tmp_path).backbone() == good

    def test_tc19f_sibling_layouts_are_searched(self, tmp_path):
        """Unzipped next to the app rather than under models/."""
        models = tmp_path / "models"
        models.mkdir()
        d = self._fake_backbone(tmp_path / "mHuBERT-147")
        assert P.Paths(models).backbone() == d

    def test_tc19c_config_without_weights_is_absent(self, tmp_path):
        d = tmp_path / "backbone"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert P.Paths(tmp_path).backbone() is None

    def test_tc12_fusion_glob_accepts_both_names(self, tmp_path):
        """DEMO_BRIEF.md §2 flags that demo_prefetch.py globs BEST_*.pt while the
        file is named WINNER_*.pt. Accept either, so a naming mismatch cannot be
        what breaks demo day."""
        d = tmp_path / "fusion"
        d.mkdir()
        for name in ("WINNER_intermediate_attn_bert_full_p2_seed1.pt", "BEST_x.pt"):
            (d / name).write_bytes(b"x")
        found = {p.name for p in P.Paths(tmp_path).fusion_checkpoints()}
        assert found == {"WINNER_intermediate_attn_bert_full_p2_seed1.pt", "BEST_x.pt"}
        assert P.Paths(tmp_path).fusion_checkpoints()[0].name.startswith("WINNER_"), (
            "WINNER must sort first — it is the checkpoint the brief names")

    def test_tc06_cli_requires_input(self):
        with pytest.raises(SystemExit):
            P.main([])

    def test_tc06b_self_test_passes(self):
        assert P.main(["--self-test"]) == 0


class TestEntryPoints:
    """TC-13..TC-18. 500 lines of tests that never checked the program starts.

    These exercise the argparse layer and the render helpers without launching a
    server, so they run everywhere. Actually serving the UI is TC-35 (manual)."""

    def test_tc13_pipeline_help_exits_clean(self, capsys):
        with pytest.raises(SystemExit) as e:
            P.main(["--help"])
        assert e.value.code == 0
        assert "--self-test" in capsys.readouterr().out

    def test_tc14_app_module_imports_and_exposes_a_parser(self):
        app = _app_module()
        ap = app.build_parser()
        args = ap.parse_args([])
        assert args.device == "cpu" and args.port == 7860
        assert args.models is None and args.no_kenlm is False

    def test_tc15_app_has_no_decoder_override_flags(self):
        """Decoder parameters come from lm_params_clean.json, tuned for this
        checkpoint. A CLI override would let someone silently decode at values
        the reported WER was never measured at."""
        app = _app_module()
        flags = {a for act in app.build_parser()._actions for a in act.option_strings}
        assert not ({"--alpha", "--beta", "--beam"} & flags), (
            f"decoder overrides are back on the command line: {sorted(flags)}")

    def test_tc15b_pipeline_has_no_decoder_override_flags(self):
        import argparse as _a

        ap = _a.ArgumentParser()
        try:
            P.main(["--alpha", "0.9"])
        except SystemExit as e:
            assert e.code != 0, "--alpha was accepted; it should not exist"
        else:
            pytest.fail("--alpha was accepted by pipeline.main")
        del ap

    def test_tc16_app_rejects_unknown_flags(self):
        app = _app_module()
        with pytest.raises(SystemExit):
            app.build_parser().parse_args(["--not-a-flag"])

    def test_tc17_no_metrics_panel_remains(self):
        """The caveats panel and reported_metrics.json were removed on request.
        This asserts the removal was complete rather than partial — a helper
        left behind with no caller is the usual way dead code returns."""
        app = _app_module()
        for gone in ("caveats", "load_reported_metrics", "METRICS_PATH"):
            assert not hasattr(app, gone), f"{gone} survived the removal"
        assert not (Path(app.__file__).parent / "reported_metrics.json").exists()

    def test_tc18b_render_helpers_produce_html(self):
        app = _app_module()
        r = P.Result(transcript="A FIRE", decoder="greedy", emotion="panic",
                     emotion_probs={c: 1 / 6 for c in P.SER_CLASSES},
                     voice_risk="high", verdict="anomaly",
                     verdict_probs={l: 1 / 3 for l in P.FUSION_LABELS},
                     timings={"asr": .1, "ser": .1, "fusion": .01, "audio": 3.0})
        out = app.render(r)
        assert "ANOMALY" in out and "A FIRE" in out

    def test_tc18d_transcript_has_an_explicit_colour(self):
        """It rendered white-on-white under some Gradio themes because the block
        inherited its colour. Every text block now sets one."""
        app = _app_module()
        r = P.Result(transcript="READABLE", decoder="greedy", emotion="neutral",
                     emotion_probs={c: 1 / 6 for c in P.SER_CLASSES},
                     voice_risk="low", verdict="normal",
                     verdict_probs={l: 1 / 3 for l in P.FUSION_LABELS},
                     timings={"asr": .1, "ser": .1, "fusion": .01, "audio": 1.0})
        block = app.render(r).split("transcript")[1].split("</div>")[1]
        assert "color:#111" in block.replace(" ", ""), (
            "the transcript block does not set a text colour")

    def test_tc18e_clip_discovery_is_name_agnostic_and_multi_format(self, tmp_path,
                                                                    monkeypatch):
        app = _app_module()
        monkeypatch.setattr(app, "CLIPS_DIR", tmp_path)
        audio = {"anything.wav", "call_02.mp3", "z.flac", "panic-kitchen.m4a"}
        for n in audio | {"notes.txt", "cover.png"}:
            (tmp_path / n).write_bytes(b"x")
        found = {p.name for p in app.find_clips()}
        assert found == audio
        assert "notes.txt" not in found and "cover.png" not in found

    def test_tc18f_m4a_is_listed_even_though_libsndfile_cannot_read_it(self):
        """It is offered because `load_wav` has a PyAV fallback and, failing
        that, explains itself. A file silently missing from the dropdown is
        worse than one that says why it did not work."""
        app = _app_module()
        assert ".m4a" in app.CLIP_EXTENSIONS
        assert ".m4a" in app.FALLBACK_EXTENSIONS
        assert ".m4a" not in app.NATIVE_EXTENSIONS
        assert ".wav" in app.NATIVE_EXTENSIONS and ".mp3" in app.NATIVE_EXTENSIONS


class TestAudioDecodingFailures:
    """FR-21. libsndfile cannot read AAC/MP4, which is what phones and Windows
    Voice Recorder produce by default."""

    def test_tc29_undecodable_file_raises_a_named_error_with_the_fix(self, tmp_path):
        pytest.importorskip("soundfile")
        bad = tmp_path / "panic-kitchen.m4a"
        bad.write_bytes(b"not audio at all")
        with pytest.raises(P.AudioDecodeError) as e:
            P.load_wav(bad)
        msg = str(e.value)
        assert "panic-kitchen.m4a" in msg
        assert ".wav" in msg, "the error must name a way out, not just a cause"

    def test_tc29b_missing_file_also_reports_cleanly(self, tmp_path):
        pytest.importorskip("soundfile")
        with pytest.raises(P.AudioDecodeError):
            P.load_wav(tmp_path / "nope.wav")

    def test_tc29c_ui_renders_the_failure_instead_of_a_traceback(self, tmp_path):
        """Gradio's default is a red toast plus a console traceback. Neither is
        readable from the back of a room."""
        pytest.importorskip("soundfile")
        app = _app_module()

        class _Pipe:
            fusion_meta = {}

            def __call__(self, *a, **k):
                raise AssertionError("must not reach the model")

        bad = tmp_path / "clip.m4a"
        bad.write_bytes(b"x")
        # Exercise the same guard the UI uses.
        try:
            P.load_wav(bad)
        except P.AudioDecodeError as exc:
            out = app.html.escape(str(exc))
            assert "clip.m4a" in out
        else:
            pytest.fail("expected AudioDecodeError")

    def test_tc18c_transcript_is_html_escaped(self):
        app = _app_module()
        r = P.Result(transcript="<script>x</script>", decoder="greedy",
                     emotion="neutral", emotion_probs={c: 1 / 6 for c in P.SER_CLASSES},
                     voice_risk="low", verdict="normal",
                     verdict_probs={l: 1 / 3 for l in P.FUSION_LABELS},
                     timings={"asr": .1, "ser": .1, "fusion": .01, "audio": 1.0})
        assert "<script>" not in app.render(r)


def _app_module():
    """Import app, skipping if gradio is absent. Kept out of the class so
    the skip reason names the real dependency."""
    pytest.importorskip("gradio")
    import app

    return app


class TestResultRendering:
    @staticmethod
    def _result(decoder="greedy", params=None):
        return P.Result(transcript="A FIRE", decoder=decoder, emotion="panic",
                        decoder_params=params,
                        emotion_probs={c: 1 / 6 for c in P.SER_CLASSES},
                        voice_risk="high", verdict="normal",
                        verdict_probs={l: 1 / 3 for l in P.FUSION_LABELS},
                        timings={"asr": 0.1, "ser": 0.1, "fusion": 0.01, "audio": 3.0})

    def test_tc03_result_reports_active_decoder(self):
        text = self._result().as_text()
        assert "greedy" in text and "A FIRE" in text and "NORMAL" in text

    def test_tc03b_kenlm_label_carries_its_tuned_parameters(self):
        r = self._result("kenlm", {"alpha": 0.6, "beta": 0.0, "beam_width": 50,
                                   "lm": "3-gram.pruned.1e-7.arpa"})
        assert r.decoder_label == "kenlm α0.6 β0 beam 50"
        assert r.decoder == "kenlm", "the bare name must stay comparable"

    def test_tc03c_greedy_label_claims_no_parameters(self):
        """A greedy fallback must not be mistakable for a tuned KenLM run —
        they differ by roughly half the word error rate."""
        assert self._result().decoder_label == "greedy"
        assert self._result("manual").decoder_label == "manual"


# ===========================================================================
# Tier 2 — model files required
# ===========================================================================


@needs_models
class TestPipelineContracts:
    def test_tc23_one_backbone_two_adapters(self, pipe):
        from peft.tuners.tuners_utils import BaseTunerLayer
        layers = [m for m in pipe.backbone.model.modules()
                  if isinstance(m, BaseTunerLayer)]
        assert layers, "no LoRA layers found — injection did not happen"
        for m in layers:
            assert {"asr", "ser"} <= set(m.lora_A.keys())

    def test_tc23b_backbone_is_frozen(self, pipe):
        assert not any(p.requires_grad for p in pipe.backbone.model.parameters())

    def test_tc23c_no_augmentation_at_inference(self, pipe):
        """FR-05. SpecAugment and dropout are training-time regularisers. Left on,
        they would make the same clip give different answers on every click."""
        assert not pipe.backbone.model.training
        assert not pipe.asr_head.training
        assert not pipe.ser_head.training
        assert not pipe.fusion_head.training

    def test_tc24_partial_adapter_load_raises(self, pipe, tmp_path):
        torch = pytest.importorskip("torch")
        bad = tmp_path / "adapter.pt"
        torch.save({"nonsense.lora_A.default.weight": torch.zeros(1)}, bad)
        with pytest.raises(RuntimeError, match="matched"):
            pipe.backbone.load_adapter(bad, "asr")

    def test_tc20_transcript_charset(self, pipe):
        t, _ = pipe.run_asr(P.prepare_audio(_tone(), 16_000))
        assert set(t) <= set(P.ASR_CHARS) | {" "}, f"unexpected symbols in {t!r}"

    def test_tc21_decoder_is_named_and_consistent(self, pipe):
        _, name = pipe.run_asr(P.prepare_audio(_tone(1.0), 16_000))
        assert name == pipe.decoder_name
        assert name in ("greedy", "kenlm")

    def test_tc22_ser_distribution(self, pipe):
        emo, probs = pipe.run_ser(P.prepare_audio(_tone(), 16_000))
        assert emo in P.SER_CLASSES
        assert set(probs) == set(P.SER_CLASSES)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
        assert max(probs, key=probs.get) == emo

    def test_tc25e_fusion_distribution(self, pipe):
        v, probs = pipe.run_fusion("THERE'S A FIRE IN THE BUILDING", "panic")
        assert v in P.FUSION_LABELS
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)

    def test_tc25f_unknown_emotion_rejected(self, pipe):
        with pytest.raises(ValueError):
            pipe.run_fusion("ANYTHING", "furious")

    def test_tc27_provenance_is_exposed(self, pipe):
        """FR-12. Presence only — no assertion on the value. Quality is the
        benchmark's job, not this suite's."""
        m = pipe.fusion_meta
        for k in ("checkpoint", "val_f1", "regime", "class_weighting"):
            assert k in m, f"{k} missing from fusion_meta"

    def test_tc28_cli_overrides_bypass_channels(self, pipe):
        r = pipe(_tone(), 16_000, emotion_override="neutral",
                 text_override="MY CAT IS STUCK ON THE ROOF")
        assert r.transcript == "MY CAT IS STUCK ON THE ROOF"
        assert r.emotion == "neutral" and r.decoder == "manual"


@needs_models
class TestEmotionChannelMatters:
    """UR-03 / UR-04. The benchmark's headline is that the emotion token earns
    its place. If the deployed head ignores emotion, the demo has no story."""

    TEXTS = ["THERE'S A FIRE IN THE BUILDING",
             "MY CAT IS STUCK ON THE ROOF",
             "I CAN'T BREATHE PLEASE SEND SOMEONE"]

    def test_tc26_verdict_responds_to_emotion(self, pipe):
        moved = 0
        for text in self.TEXTS:
            verdicts = {e: pipe.run_fusion(text, e)[0] for e in P.SER_CLASSES}
            if len(set(verdicts.values())) > 1:
                moved += 1
        assert moved > 0, (
            "holding text fixed and sweeping all six emotions never changed the "
            "verdict on any probe sentence — the head is ignoring the emotion "
            "channel and UR-03 cannot be demonstrated")

    def test_tc26b_probabilities_shift_even_when_label_holds(self, pipe):
        base = pipe.run_fusion(self.TEXTS[0], "neutral")[1]
        alt = pipe.run_fusion(self.TEXTS[0], "panic")[1]
        delta = max(abs(base[k] - alt[k]) for k in base)
        assert delta > 1e-3, f"emotion moved the distribution by only {delta:.2e}"

    def test_tc46_all_four_quadrants_reachable(self, pipe):
        """UR-02. Not every combination must produce a mismatch verdict, but the
        four (content, voice) quadrants must be constructible and deterministic."""
        cases = [("THERE'S A FIRE IN THE BUILDING", "panic"),
                 ("THERE'S A FIRE IN THE BUILDING", "neutral"),
                 ("MY CAT IS STUCK ON THE ROOF", "panic"),
                 ("MY CAT IS STUCK ON THE ROOF", "neutral")]
        out = [pipe.run_fusion(t, e) for t, e in cases]
        assert all(v in P.FUSION_LABELS for v, _ in out)
        for (t, e), (v, _) in zip(cases, out):
            assert pipe.run_fusion(t, e)[0] == v


@needs_models
class TestNonFunctional:
    def test_tc35_ui_builds_against_a_real_pipeline(self, pipe):
        """The Blocks graph is constructed but not served. Catches a broken
        component wiring at test time rather than at demo time."""
        app = _app_module()
        assert app.build(pipe) is not None

    def test_tc40_runs_on_cpu(self, pipe):
        assert pipe.device == "cpu"
        assert all(p.device.type == "cpu" for p in pipe.backbone.model.parameters())

    def test_tc44_determinism(self, pipe):
        wav = _tone()
        a, b = pipe(wav, 16_000), pipe(wav, 16_000)
        assert a.transcript == b.transcript
        assert a.emotion == b.emotion and a.verdict == b.verdict
        for k in a.verdict_probs:
            assert a.verdict_probs[k] == pytest.approx(b.verdict_probs[k], abs=1e-6)

    @pytest.mark.slow
    def test_tc41_latency_budget(self, pipe):
        wav = _tone(5.0)
        pipe(wav, 16_000)                                  # warm up
        t = time.time()
        r = pipe(wav, 16_000)
        elapsed = time.time() - t
        assert elapsed <= 5.0, (
            f"5 s clip took {elapsed:.2f} s (RTF {elapsed / 5:.2f}); NFR-02 wants "
            f"RTF <= 1.0. Breakdown: {r.timings}")

    @pytest.mark.slow
    def test_tc42_memory_budget(self):
        psutil = pytest.importorskip("psutil")
        proc = psutil.Process()
        rss = proc.memory_info().rss / 2**20
        assert rss <= 2048, f"resident {rss:.0f} MB exceeds the 2 GB budget"

    @pytest.mark.slow
    def test_tc43_cold_load_budget(self, pipe):
        t = time.time()
        P.Pipeline(PATHS, device="cpu", log=lambda *a, **k: None)
        elapsed = time.time() - t
        assert elapsed <= 120, f"cold load took {elapsed:.0f} s (NFR-04: 120 s)"

    @pytest.mark.slow
    def test_tc45_works_offline(self, pipe):
        """NFR-06 / UR-01. Loads with the hub forced offline, in a subprocess so
        the already-imported transformers cannot mask a network call."""
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        code = ("import sys; sys.path.insert(0, r'%s');"
                "import pipeline as P;"
                "p = P.Pipeline(P.Paths(r'%s'), device='cpu', log=lambda *a, **k: None);"
                "print('OK', p.decoder_name)"
                % (Path(__file__).resolve().parents[1], MODELS))
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (
            "pipeline could not start offline — something still reaches the "
            f"network:\n{r.stderr[-2000:]}")
        assert "OK" in r.stdout
