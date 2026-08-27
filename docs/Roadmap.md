# Emergency Call Intelligence System — PoC Roadmap
**Author:** Tuna  
**Contact:** tunabayram35@gmail.com  
**Scope:** Proof of Concept — Multimodal Speech Emotion Recognition (SER)  
**Primary Language:** English first, European languages in expansion  
**Last Updated:** July 2026

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Model Selection](#3-model-selection)
4. [Dataset Inventory](#4-dataset-inventory)
5. [Training Pipeline](#5-training-pipeline)
6. [Data Augmentation Strategy](#6-data-augmentation-strategy)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Known Limitations & Risks](#8-known-limitations--risks)
9. [Future Roadmap](#9-future-roadmap)
10. [Glossary](#10-glossary)

---

## 1. Project Overview

### Goal
Build a Proof of Concept (PoC) AI system that processes emergency call audio in real time and detects the emotional state of the caller using a multimodal deep learning approach.

### Scope (PoC)
- **Single modality input:** Voice only (no GPS, no metadata for PoC)
- **Primary task:** Speech Emotion Recognition (SER)
- **Language:** English first, European languages in expansion
- **Deployment target:** On premises, CPU only. Call recordings cannot leave the
  institution, so the system runs locally rather than behind a remote API.
- **Latency budget:** Faster than the audio it processes. The measured targets are
  in `app/REQUIREMENTS.md` NFR-02 and NFR-03, a 5 s clip in under 5 s and under
  2 GB resident on a laptop CPU with no GPU.

### Out of Scope for PoC
- Real-time operator dashboard
- GIS / location metadata integration
- Accent Identification (AID) fine-tuning
- Audio Event Detection (AED)
- French language training

---

## 2. Architecture Summary

### Pipeline Flow
```
Incoming call audio (16kHz)
        ↓
Preprocessing (resample, channel split)
        ↓
VAD — Silero-VAD (silence removal)
        ↓
┌─────────────────────────────────┐
│   mHuBERT-147 (frozen backbone) │
└──────────────┬──────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
ASR head (CTC)      Mel Spectrogram
    ↓                     ↓
Transcript           MaxArea Pooling2D
    ↓                     ↓
Text Encoder         Reshape (time × freq)
(XLM-RoBERTa)            ↓
    ↓               Modality Dropout
Modality Dropout    (p=0.1, min_active=1)
(p=0.1, min_active=1)    ↓
    ↓                     Q
   KV                     │
    └──────────┬──────────┘
               ▼
      Cross-Attention Layer
      + KV Cache (streaming)
               ↓
      Global Average Pooling
               ↓
      Dropout (p=0.3)
               ↓
      Dense Layer (ELU/ReLU)
               ↓
      Dropout (p=0.3)
               ↓
      Output Layer (Softmax)
               ↓
      SER Output (5 classes)
```

### Emotion Taxonomy (Emergency Context)
| Class | Description | Maps from academic labels |
|---|---|---|
| Distress | Caller in pain or suffering | High arousal, negative valence |
| Panic | Acute fear, loss of control | Very high arousal, negative |
| Confusion | Disoriented, unclear situation | Low arousal, negative valence |
| Urgency | Time-critical, pressing | High arousal, mixed valence |
| Neutral | Baseline, calm | Low arousal, neutral valence |

### Key Architectural Decisions

**Why mel spectrogram as Q (not mHuBERT internal representations)?**  
mHuBERT was trained with a speech-centric SSL objective — its internal representations encode phonological and linguistic content but ignore background acoustic events. The mel spectrogram is computed independently of mHuBERT via a fixed mathematical transform, providing genuine modality orthogonality. Background sounds (sirens, crowd noise, environmental context) that influence emotional interpretation are captured in the spectrogram but not in mHuBERT representations.

**Why cross-attention (intermediate fusion)?**  
Cross-attention allows each modality to inform the other — the acoustic signal can query the text for semantic context, and the temporal structure of the audio (Q) is preserved in the output. Early fusion loses cross-modal interaction depth. Late fusion loses internal relationship learning. Intermediate fusion is the best balance of expressiveness and trainability.

**Why KV caching for streaming?**  
Emergency calls are continuous streams. Processing the full conversation each step is O(N²) in compute. KV caching computes new token K and V values once and caches them — new audio chunks attend against the full conversation history at O(N) cost. Full context is preserved without quadratic cost explosion.

**Why modality dropout (min_active=1)?**  
ASR errors are expected in noisy, distressed, accented emergency calls. Modality dropout forces the model to learn to make predictions from either branch alone, making it robust when ASR fails. The min_active=1 constraint ensures both branches are never dropped simultaneously, preventing uninformative zero-signal training steps.

**Why differential learning rates?**  
Cross-attention has a harder learning task (fusing two modalities) than the classifier (mapping a vector to 5 classes). Training both at the same rate risks the classifier overfitting before cross-attention has stabilised. Cross-attention trains at 1e-4, classifier at 1e-3.

---

## 3. Model Selection

### Core Backbone
| Model | Role | Params | License | Status |
|---|---|---|---|---|
| `utter-project/mHuBERT-147` | SSL backbone | ~95M | CC-BY-NC 4.0 | Frozen — download only |
| `naver/mHuBERT-147-ASR-fr` | French ASR reference | ~95M + CTC | Research | Load as reference implementation |

### Supporting Models
| Model | Role | Params | License | Status |
|---|---|---|---|---|
| Silero-VAD | Voice Activity Detection | ~1M | MIT | Load and run — no training |
| XLM-RoBERTa base | Text encoder for KV branch | 278M | MIT | Frozen — no training needed |
| `Jzuluaga/accent-id-commonaccent_xlsr-en-english` | AID (parallel, future) | ~300M | CC-BY 4.0 | Phase 2 |

### Full Model Capability Table
| Model | Params | Train Type | Languages | LID | AID | SER | ASR | AED | VAD |
|---|---|---|---|---|---|---|---|---|---|
| `facebook/mms-300m` | 300M | SSL backbone | 1,406 | * | *_2 | * | * | *_1 | * |
| `facebook/mms-1b` | 1B | SSL backbone | 1,406 | * | *_2 | * | * | *_1 | * |
| `facebook/mms-1b-all` | 1B + 2.5M/lang | Fine-tuned | ASR:1,107 / LID:4,017 | + (sep.) | *_2 | * | + | *_1 | * |
| `microsoft/wavlm-large` | 316M | SSL backbone | EN only | - | *_2 | * | * | *_1 | * |
| `facebook/wav2vec2-xls-r-300m` | 300M | SSL backbone | 128 langs | * | *_2 | * | * | *_1 | * |
| `utter-project/mHuBERT-147` | ~95M | SSL backbone | 147 langs | * | *_2 | * | * | *_1 | * |
| `FunAudioLLM/SenseVoiceSmall` | ~225M | Fine-tuned | 5 langs | + | *_2 | + | + | + (partial) | + |
| `FunAudioLLM/SenseVoiceLarge` | ~1B | Fine-tuned | 50+ langs | + | *_2 | + | + | + (partial) | + |
| `openai/whisper-large-v3` | 1,550M | Fine-tuned | 99 langs | + | *_2 | * | + | *_1 | + |
| `openai/whisper-large-v3-turbo` | 809M | Fine-tuned | 99 langs | + | *_2 | * | + | *_1 | + |
| `Qwen/Qwen2-Audio-7B-Instruct` | 7B | Fine-tuned | 8+ langs | + | - | + (prompt) | + | + (prompt) | - |
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | Fine-tuned | 52 langs | + | *_2 | * | + | - | + |
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | Fine-tuned | 52 langs | + | *_2 | * | + | - | + |
| `nvidia/parakeet-tdt-0.6b-v3` | 600M | Fine-tuned | 25 European langs | + | - | - | + | - | - |

**Legend:**
- `+` — Available out of the box
- `-` — Architecturally not possible
- `*` — Achievable with head training / modification
- `*_1` — Achievable with CNN or LoRA-only adaptation (risk of conflicting with speech heads)
- `*_2` — Achievable but requires extensive accent-labelled dataset

**Disclaimer:** Models such as `Jzuluaga/accent-id-commonaccent_xlsr-en-english` provide AID as `+` but as a standalone parallel model, not integrated into the backbone above.

---

## 4. Dataset Inventory

### LID — Language Identification
| Dataset | Size | Languages | License | Access |
|---|---|---|---|---|
| VoxLingua107 | 6,628 hrs | 107 languages | CC-BY 4.0 | HuggingFace / OpenSLR |
| FLEURS | ~12 hrs/lang | 102 languages | CC-BY 4.0 | `google/fleurs` HuggingFace |
| Mozilla Common Voice 18 | EN ~3,000 hrs, FR ~900 hrs | 100+ languages | CC-0 | HuggingFace |

### AID — Accent Identification
| Dataset | Size | Accents | License | Access |
|---|---|---|---|---|
| CommonAccent | ~1,200 hrs | 16 EN accents incl. African, Canadian, Indian | CC-BY 4.0 | `DTU54DL/common-accent` HuggingFace |
| African Accented French (OpenSLR 57) | ~22 hrs | Cameroonian FR, Gabonese FR | CC-BY 4.0 | `openslr.org/57` |
| AfriSpeech-200 | 200+ hrs | 120 African accents (EN) | CC-BY 4.0 | `tobiolatunji/afrispeech-200` HuggingFace |
| L2-ARCTIC | ~11.2 hrs | Non-native EN: Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese | CC-BY 4.0 | `KoelLabs/L2Arctic` HuggingFace |

### SER — Speech Emotion Recognition
| Dataset | Size | Language | Emotions | License | Access |
|---|---|---|---|---|---|
| RAVDESS | ~1 hr, 24 actors | EN | 8 emotions incl. fear | CC-BY-NC-SA 4.0 | Zenodo |
| CREMA-D | 7,442 clips, 91 actors | EN | 6 emotions incl. fear | ODbL | GitHub |
| IEMOCAP | ~12 hrs, 10 actors | EN | Anger, happiness, sadness, neutral + more | Research (free registration) | USC SAIL Lab |
| MSP-Podcast | 237+ hrs, 1,500+ speakers | EN (naturalistic) | 8 emotions + valence/arousal | Research (free registration) | UT Dallas |
| CaFE | ~6 hrs, 12 actors | Canadian French | 7 emotions incl. fear | CC-BY 4.0 | Zenodo |
| EmoNet-Voice Big | 5,000 hrs synthetic | EN, DE, ES, FR | 40 emotion categories incl. distress/pain | CC-BY 4.0 | HuggingFace |

**Critical note on domain mismatch:** All acted SER datasets (RAVDESS, CREMA-D, CaFE) contain performed emotions recorded in studio conditions. Real emergency calls contain spontaneous, physiologically genuine distress. This is the primary data risk for the PoC — the model may learn to recognise performed distress, not real distress. MSP-Podcast (naturalistic) and EmoNet-Voice (synthetic but diverse) partially address this. Phase 4 active learning loop is the long-term solution.

### ASR — Automatic Speech Recognition
| Dataset | Size | Languages | License | Access |
|---|---|---|---|---|
| LibriSpeech train-clean-100 | 100 hrs | EN | CC-BY 4.0 | `openslr/librispeech_asr` HuggingFace |
| LibriSpeech full (960h) | 960 hrs | EN | CC-BY 4.0 | Same |
| MLS French | ~1,096 hrs | FR | CC-BY 4.0 | `facebook/multilingual_librispeech` HuggingFace |
| FLEURS | ~12 hrs/lang | 102 languages | CC-BY 4.0 | `google/fleurs` HuggingFace |

### VAD — Voice Activity Detection
| Dataset | Size | Content | License | Access |
|---|---|---|---|---|
| MUSAN | ~60 hrs | Music, speech, noise | CC-BY 4.0 | OpenSLR 17 |
| LibriVAD | 15 GB / 150 GB / 1.5 TB | Explicit speech/silence labels | Open | arXiv:2512.17281 |
| AMI Meeting Corpus | 100 hrs | Multi-speaker naturalistic | CC-BY 4.0 | `edinburghcstr/ami` HuggingFace |

### AED — Audio Event Detection
| Dataset | Size | Emergency-relevant classes | License | Access |
|---|---|---|---|---|
| UrbanSound8K | 8,732 clips | Siren, gun shot, engine | Mixed CC | urbansounddataset.weebly.com |
| FSD50K | 108 hrs, 200 classes | Sirens, alarms, screaming, crying | CC-BY (mixed) | Zenodo / HuggingFace |
| ESC-50 | 2,000 clips | Rain, wind, fire, crying | CC-BY | GitHub |

### Noise Augmentation
| Dataset | Purpose | License | Access |
|---|---|---|---|
| MUSAN | Generic noise injection | CC-BY 4.0 | OpenSLR 17 |
| DEMAND | Room impulse responses (RIR) | CC-BY-SA 3.0 | Zenodo |
| RIR Noise (OpenSLR 28) | Room impulse responses | Apache 2.0 | openslr.org/28 |

---

## 5. Training Pipeline

### Phase 1 — ASR Head Training
**Goal:** Build English ASR capability on mHuBERT-147

| Item | Detail |
|---|---|
| Backbone | `utter-project/mHuBERT-147` — fully frozen |
| Head | CTC head — single linear layer, randomly initialised |
| Vocabulary | 29 tokens (26 chars + space + apostrophe + blank) |
| Training data | LibriSpeech `train-clean-100` (100h) for PoC |
| Loss function | CTC loss (Connectionist Temporal Classification) |
| Hardware | Single A100-40GB (Google Colab Pro+) |
| Estimated time | 8–12 hours |
| Target metric | WER < 10% on LibriSpeech `test-clean` |
| Reference implementation | `naver/mHuBERT-147-ASR-fr` CTC_model.py |
| Framework | HuggingFace `run_speech_recognition_ctc.py` |
| Output | Trained CTC head → freeze both mHuBERT + CTC head |

**Note:** French ASR is already available at `naver/mHuBERT-147-ASR-fr` — no training needed for French in Phase 1.

---

### Phase 2 — Transcript Generation for SER Datasets
**Goal:** Enrich SER datasets with auto-generated transcripts

| Item | Detail |
|---|---|
| Models used | mHuBERT-147 (frozen) + CTC head (frozen) |
| Mode | Inference only — no training |
| Input datasets | CREMA-D, IEMOCAP, MSP-Podcast, RAVDESS |
| Process | Run each audio file through ASR pipeline, save transcript alongside audio and emotion label |
| Output | Augmented SER datasets with (audio, transcript, emotion label) tuples |
| Storage format | JSON per sample: `{audio_path, transcript, emotion_label, confidence}` |
| Why offline? | Generating transcripts once is 50× cheaper than generating on-the-fly per training batch per epoch |

---

### Phase 3 — SER Fusion Training
**Goal:** Train the multimodal cross-attention SER model

| Item | Detail |
|---|---|
| Frozen components | mHuBERT-147, CTC head, XLM-RoBERTa text encoder, spectrogram transform |
| Trained components | Cross-attention layer, GAP, Dense layer, Classifier |
| Training data | Phase 2 output (audio + transcript + emotion label) |
| Primary datasets | CREMA-D (91 actors, diversity) + MSP-Podcast (naturalistic) |
| Loss function | Cross-entropy |
| Optimiser | AdamW |
| Learning rates | Cross-attention: 1e-4 · Dense+Classifier: 1e-3 |
| Batch size | 32 samples |
| Epochs | 30–50 (early stopping on validation loss) |
| Modality dropout | p=0.1, min_active=1 on both branches |
| Hardware | Single A100-40GB |
| Estimated time | 4–8 hours |
| Target metric | Weighted F1 > 0.65 on held-out test set |
| Output | Trained SER fusion model → ready for PoC |

**Augmentation applied during Phase 3 training:**

| Augmentation | Tool | Purpose |
|---|---|---|
| Codec simulation | ffmpeg G.711 µ-law encode/decode | Simulate telephony degradation |
| Noise injection | MUSAN + audiomentations | Background noise robustness |
| Room impulse response | DEMAND + pyroomacoustics | Acoustic environment simulation |
| Speed perturbation | audiomentations (rate 0.9–1.1×) | Speaking rate variation |
| Pitch shift | audiomentations (±2 semitones) | Vocal range variation |
| Gain perturbation | audiomentations (±6 dB) | Volume level variation |
| Random SNR | 5–20 dB range | Variable noise conditions |

---

### Phase 4 — Active Learning Loop
**Goal:** Continuously improve model on real-world data

| Item | Detail |
|---|---|
| Input | Scraped audio from emergency/crisis scenes in public domain films |
| Process | Divide audio into 5s chunks → run Phase 3 model → flag low-confidence predictions (max softmax < 0.65) → human validation of flagged samples |
| Uncertainty metric | Uncertainty sampling — flag when max class probability < 0.65 |
| Human annotation | Annotators review flagged samples only → assign emotion label from 5-class taxonomy |
| Annotation protocol | 2D valence-arousal wheel → map to 5 classes post-hoc · Target inter-annotator agreement κ ≥ 0.7 |
| Output | Human-validated samples + corrected annotations |
| Re-training | Add validated samples to Phase 3 training set → retrain fusion model |
| Loop frequency | Every 200–500 new validated samples |
| Long-term goal | Progressively shift training distribution from acted → naturalistic emergency speech |

**Note on film data:** Use public domain or research-licensed content only. Film emergency/crisis scenes provide closer approximation to real distress than studio-recorded acted datasets, while avoiding the ethical and privacy issues of real emergency call recordings.

---

## 6. Data Augmentation Strategy

### Why Augmentation is Critical
Model trains on clean, studio-recorded acted speech (CREMA-D, RAVDESS). Deployment encounters:
- Telephony codec degradation (G.711 / G.722)
- Background environmental noise
- Variable recording conditions
- Physiologically altered voice (panic, crying, hyperventilation)

Without augmentation, the model will perform significantly worse in production than on the test set.

### Augmentation Pipeline
```python
# Conceptual augmentation chain per training sample
audio
  → codec simulation (G.711 encode/decode via ffmpeg)     # telephony degradation
  → RIR convolution (random room from DEMAND)              # room acoustics
  → noise injection (MUSAN, SNR 5-20dB)                   # background noise
  → gain perturbation (±6dB)                              # volume variation
  → speed perturbation (0.9-1.1x)                         # speaking rate
  → pitch shift (±2 semitones)                            # vocal variation
  → resample to 16kHz                                     # ensure standard format
  → mel spectrogram computation                           # spectrogram branch
  → mHuBERT-147 inference                                 # text branch
```

### Emergency-Specific Augmentation Targets
| Real-world condition | Augmentation technique |
|---|---|
| Telephony codec (G.711) | ffmpeg µ-law encode → decode → resample |
| Packet loss | Random frame zeroing (0–5% of frames) |
| Background siren | Overlay UrbanSound8K siren clips at SNR 10-20dB |
| Crowd/panic noise | Overlay FSD50K crowd clips |
| Wind / outdoor | Overlay ESC-50 wind clips |
| Caller crying | No augmentation available — data gap |
| Hyperventilation / breathing | No augmentation available — data gap |
| Moving vehicle | Overlay engine noise from UrbanSound8K |

---

## 7. Evaluation Strategy

### Metrics
| Metric | Target | Why |
|---|---|---|
| Weighted F1 | > 0.65 | Handles class imbalance (neutral dominates) |
| Per-class recall on Distress/Panic | > 0.70 | Most critical classes for emergency context |
| WER (ASR head) | < 10% clean, < 20% noisy | Text branch quality |
| Inference latency | < 2s per 5s chunk on T4/A100 | Real-time requirement |

### Evaluation Protocol
- 80/10/10 train/validation/test split per dataset
- Cross-corpus evaluation — train on CREMA-D, test on MSP-Podcast (measures domain generalisation)
- Noisy evaluation — test set with codec simulation applied (measures real-world robustness)
- Ablation study — audio only vs text only vs fusion (measures contribution of each modality)

### Ablation Study Design
| Configuration | Purpose |
|---|---|
| Audio branch only (no KV) | Baseline acoustic SER |
| Text branch only (no Q) | Baseline text SER |
| Late fusion (separate models + weighted average) | Compare vs cross-attention |
| Full cross-attention fusion | Your proposed architecture |

---

## 8. Known Limitations & Risks

### Critical Risks

**1. Acted vs naturalistic domain mismatch**
All primary SER training data is acted (studio-recorded, performed emotions). Real emergency callers exhibit physiologically genuine distress with different acoustic properties. Model may fail to generalise. Mitigation: MSP-Podcast (naturalistic) + active learning loop (Phase 4).

**2. ASR errors propagate to SER**
In noisy, accented, or distressed speech, ASR WER increases. Bad transcripts degrade the text branch. A 30% WER can cause up to 10% SER accuracy drop. Mitigation: modality dropout trains the model to survive without reliable text.

**3. No real emergency call data**
CEMO (real French SAMU emergency calls) is the closest dataset but is privacy-restricted and not publicly available. All training data is approximations. Mitigation: Phase 4 active learning on film data, long-term pursuit of real-call data access.

**4. Québécois French accent gap**
No large open training corpus for Québécois French exists. Common Voice FR filtered by "canada" locale tag is the only option — noisy and low volume. Mitigation: synthetic TTS augmentation or targeted data collection.

**5. mHuBERT-147 licence**
CC-BY-NC 4.0 — non-commercial. Acceptable for PoC and research. Commercial deployment requires licence negotiation with NAVER LABS Europe or switching to Apache 2.0 backbone (XLS-R 300M).

### Minor Risks
- KV cache VRAM growth on very long calls — mitigate with cache eviction after 10 minutes
- Spectrogram reshape vs flatten — must preserve time dimension for meaningful cross-attention
- Modality dropout both branches simultaneously — prevented by min_active=1 constraint
- Copyright on film data in Phase 4 — use public domain or research-licensed content only

---

## 9. Future Roadmap

### Phase 2 Production (post-PoC)
- Expand to French — add MLS French ASR training, CaFE SER data, EmoNet-Voice FR
- Add AID — CommonAccent ECAPA-TDNN in parallel for accent detection
- Add AED — UrbanSound8K + FSD50K fine-tuned AST for siren/alarm detection
- Codec robustness — full G.711 augmentation pipeline
- Session-level context — hierarchical model over chunk-level outputs for emotion trajectory tracking

### Phase 3 Production (multilingual expansion)
- Add German, Spanish, Italian — MLS data, CommonAccent DE/ES/IT, EMO-DB
- Evaluate zero-shot transfer from EN/FR model before training language-specific adapters
- MMS-1B-all adapter strategy for new languages (~100 fine-tuning steps per language)
- Timestamp-based emotion recognition — preserve time dimension through architecture for per-frame emotion curve

### Phase 4 Production (full system)
- Multimodal fusion with metadata — GPS, weather, cell tower data via GIS lookup
- Operator dashboard — real-time emotion visualisation and escalation flagging
- Speaker diarization — separate agent and caller channels
- Active learning loop maturation — automated pipeline with human-in-the-loop annotation

---

## 10. Glossary

| Term | Definition |
|---|---|
| LID | Language Identification — detecting which language is being spoken |
| AID | Accent Identification — detecting which accent variant of a language is being spoken |
| SER | Speech Emotion Recognition — detecting emotional state from voice signal |
| ASR | Automatic Speech Recognition — converting speech to text (STT) |
| AED | Audio Event Detection — identifying non-speech sounds (sirens, alarms) |
| VAD | Voice Activity Detection — detecting speech vs silence segments |
| SSL | Self-Supervised Learning — training on unlabelled audio without task-specific labels |
| CTC | Connectionist Temporal Classification — loss function for sequence-to-sequence tasks without alignment |
| WER | Word Error Rate — percentage of incorrect words in ASR output |
| CER | Character Error Rate — percentage of incorrect characters (used for CJK languages) |
| RTFx | Real Time Factor — how many times faster than real time the model processes audio |
| MoE | Mixture of Experts — architecture with multiple specialist sub-networks and a routing mechanism |
| MTL | Multi-Task Learning — training one model on multiple tasks simultaneously |
| LoRA | Low-Rank Adaptation — parameter-efficient fine-tuning using small trainable adapter matrices |
| PEFT | Parameter-Efficient Fine-Tuning — umbrella term for LoRA, adapters, prefix tuning |
| RIR | Room Impulse Response — acoustic fingerprint of a physical space used for augmentation |
| GAP | Global Average Pooling — collapsing a sequence into a single vector by averaging |
| KV Cache | Key-Value Cache — storing computed attention keys and values for efficient streaming inference |
| GIS | Geographic Information System — software for spatial data analysis and terrain labelling |
| Modality Dropout | Zeroing an entire input branch during training to improve robustness to missing modalities |
| Cross-Attention | Transformer mechanism where Q comes from one modality and K, V from another |
| Intermediate Fusion | Combining modalities at the representation level via cross-attention (between early and late fusion) |
| Domain Mismatch | Performance gap between training data distribution and real deployment conditions |
| Active Learning | Training strategy where the model flags uncertain samples for human annotation |
| Uncertainty Sampling | Active learning strategy that flags samples where max class probability is below a threshold |
