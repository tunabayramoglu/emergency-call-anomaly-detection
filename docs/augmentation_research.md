# Augmentation Methods & Background Noise Datasets for SER
## Research findings from literature and open datasets

---

## 1. Augmentation Methods from SER Literature

### 1.1 Audio-Level Augmentations

| Method | Description | SER Usage | Source |
|--------|-------------|-----------|--------|
| **Noise injection** | Add background noise at various SNRs (typically 5-20dB) | Standard, widely used | Multiple SER papers |
| **Speed perturbation** | Stretch/squeeze audio by 0.9×-1.1× | Standard in ASR and SER | Ko et al. 2015 (Interspeech) |
| **Pitch shifting** | Change pitch without changing duration | Common | Multiple papers |
| **Vocal Tract Length Perturbation (VTLP)** | Warp frequency axis to simulate different speaker vocal tracts | Growing in SER | Padi et al. 2021 |
| **Volume/amplitude change** | Random gain scaling | Basic preprocessing | Multiple |
| **Time shifting** | Shift waveform left/right | Basic | Multiple |
| **RIR convolution** | Convolve with room impulse response for reverberation | Common for robust SER | OpenSLR 28 |
| **Mixup** | Linear combination of two samples (α mix ratio) | +0.54% to +2.6% UAR | Latif et al. 2020, Malik et al. 2023 |
| **Resampling** | Change sample rate, then convert back (information loss) | Less common | Tao et al. 2022 |

### 1.2 Spectrogram-Level Augmentations

| Method | Description | SER Usage | Source |
|--------|-------------|-----------|--------|
| **SpecAugment** | Mask random time bins (max T=20) + frequency bins (max F=8) | Growing in SER | Park et al. 2019 (Interspeech), adapted by Malik et al. 2023 |
| **Frequency masking** | Mask contiguous frequency bands | Subset of SpecAugment | Multiple |
| **Time masking** | Mask contiguous time frames | Subset of SpecAugment | Multiple |

### 1.3 Generative Augmentation Methods

| Method | Description | Results | Open Source? | Source |
|--------|-------------|---------|-------------|--------|
| **GAN-based (vanilla GAN)** | Generate emotional feature vectors | +0.87% UAR (IEMOCAP) | Partial | Sahu et al. 2018 (Interspeech) |
| **Conditional GAN** | Condition on emotion class | Better than vanilla GAN | No | Sahu et al. 2018 |
| **CycleGAN** | Emotion style transfer between emotion classes | +0.89% UAR | No | Bao et al. 2019 |
| **StarGAN** | Multi-domain emotion conversion | Comparable to CycleGAN | Partial | Rizos et al. 2020 (ICASSP) |
| **Mixup-GAN** | GAN + mixup augmentation | +0.54% UAR | No | Latif et al. 2020 (Interspeech) |
| **Diffusion (IDDPM)** | Generate synthetic emotional mel-spectrograms conditioned on BERT text embeddings | +2.6% UAR (best in class) | No — domain dead | Malik et al. 2023 (Interspeech) |
| **EmoAug (style transfer)** | Diffusion-based style transfer to enhance emotional expression | Novel, no code released | No | Qu et al. 2022 (arXiv) |
| **TargetSEC** | Latent diffusion for emotion style embeddings | Latest (2025) | No | arXiv 2025 |

### 1.4 Key Findings from the Survey Paper

Source: Avci et al. 2025, *"A Comprehensive Analysis of Data Augmentation Methods for Speech Emotion Recognition"* (IEEE Access)

- SpecAugment is the most consistently effective single augmentation for SER
- Noise injection at moderate SNR (10-15dB) improves robustness without degrading emotion signal
- Speed perturbation is dataset-dependent — some emotions degrade at extreme speeds
- Combined augmentations (noise + SpecAugment + speed) outperform any single method
- Diffusion-based generation gives best quality synthetic data but requires significant compute
- GANs face convergence issues on small emotional corpora

---

## 2. Open Background Noise Datasets

### 2.1 Primary Datasets

#### MUSAN — ~109 hours
| Detail | Value |
|--------|-------|
| **URL** | https://www.openslr.org/17/ |
| **License** | CC-BY 4.0 |
| **Sample rate** | 16kHz |
| **Subsets** | Music (42h), Speech (60h babble), Noise (6h: white/pink/brown, ambient, office, fan, etc.) |
| **Content** | Clean isolated noise recordings, music tracks, single speaker and babble speech |
| **SER relevance** | BABBLE subset is the most useful — simulates crowd/background conversations. Noise subset is clean but limited |
| **Safe classes** | All noise + babble subsets. Music subset optional (less common in 911) |

#### DEMAND — ~6 hours
| Detail | Value |
|--------|-------|
| **URL** | https://zenodo.org/record/1227121 |
| **License** | CC-BY-SA 3.0 |
| **Sample rate** | 48kHz (can be downsampled) |
| **Subsets** | 18 environments across 6 categories: Domestic (DKITCHEN, DLIVING, DWASHING), Nature (NFIELD, NRIVER, NWIND), Office (OOFFICE, OHALLWAY), Public (PCAFETERIA, PSTATION, PTRAFFIC, PBUS), Street (SCAR, SSQUARE, STRAFFIC), Transportation (TBUS, TCAR, TMETRO) |
| **SER relevance** | Highly relevant — real recorded environments including car interiors, street traffic, cafeteria babble. These directly simulate 911 call acoustic conditions |
| **Safe classes** | All 18 environments are safe. Specifically 911-relevant: PCAFETERIA (babble), PSTATION (public noise), STRAFFIC (road), TCAR, TBUS (vehicle interiors) |

#### Room Impulse Response (RIR) — OpenSLR 28 — ~14 hours
| Detail | Value |
|--------|-------|
| **URL** | https://www.openslr.org/28/ |
| **License** | Apache 2.0 |
| **Subsets** | Simulated RIRs (small, medium, large rooms), real RIRs (from recorded spaces), isotropic noise, point-source noise |
| **Content** | Convolution with these simulates room acoustics — natural reverb |
| **SER relevance** | Useful for making clean academic clips sound like they were recorded in a room rather than a studio. Phone calls have minimal reverb, so use sparingly |
| **Safe classes** | All |

#### FSDnoisy18k — ~42.5 hours
| Detail | Value |
|--------|-------|
| **URL** | https://zenodo.org/records/2529934 |
| **License** | CC-BY 4.0 |
| **Content** | 42.5h of audio across 20 sound event classes, sourced from Freesound |
| **SER relevance** | Real sound events. Must carefully filter classes to avoid semantic leakage |
| **Safe classes** (recommended) | Acoustic guitar, Bass guitar, Cough, Cow, Crow, Double bass, Fan, Finger snapping, Fireworks, Flute, Glockenspiel, Goblet drum, Harmonica, Hi-hat, Keyboard, Microwave oven, Organ, Acoustic piano, Saxophone, Scissors, Shatter, Sigh, Squeak, Tambourine, Tearing, Violin, Wind (from FSD50k classification) |
| **Avoid for SER** | Alarm, Bark, Baby cry, Car alarm, Car horn, Crash, Crying, Dog, Engine, Explosion, Fire engine siren, Fire truck, Glass, Gunshot, Police siren, Scream, Siren, Smoke detector |
| **Filtering needed** | Yes — must curate the ~42h down to ~15h of safe ambient sounds |

### 2.2 Secondary/Supplementary Datasets

| Dataset | Hours | License | Link | Notes |
|---------|-------|---------|------|-------|
| **ESC-50** | ~3h | CC-BY | GitHub | 50 environmental sound classes, clean labels. Good for supplementing specific noise types |
| **UrbanSound8K** | ~9h | CC-BY | GitHub | 10 urban sound classes. Safe: street music, children playing (use with caution), drilling, jackhammer. Avoid: gunshot, siren, dog bark |
| **WHAM! Noise** | ~20h | CC-BY | wham.whisper.ai | Generated from environmental noises. Used for source separation but usable as noise source |
| **VCTK Noise** | ~4h | CC-BY-SA | Datashare (Edinburgh) | Clean noise recordings intended for speech enhancement |
| **EARSet** | ~3h | CC-BY | various | Emotional ambient recordings — use with caution (emotional bias possible) |

### 2.3 Safe vs Avoid Classification Logic

**Filtering principle:** *Noise types should be uncorrelated with the target emotion class.*

| Safe | Risky / Avoid |
|------|---------------|
| White/pink/brown noise, Room tone, HVAC hum | Gunshots, explosions, Sirens (fire, police, ambulance) |
| Traffic (distant/continuous), Crowd babble (cafeteria) | Car alarms, horns, Screaming, crying |
| Office/indoor ambient, Footsteps, door sounds | Laughter, moaning, Dog barking, growling |
| Kitchen sounds, Outdoor wind, rain | Glass breaking, crashing, Baby crying |
| Vehicle interior (car, bus, train), Radio static | Engine racing, tire screech, Alarms, beeping |
| Fan, appliance hum, Music (instrumental, ambient) | Angry shouting, arguing, Music with emotional vocals |

### 2.4 Recommended Noise Pool (curated from above)

| Noise type | Source dataset | Subset/class |
|------------|---------------|-------------|
| White noise | MUSAN | noise/free/white_noise |
| Pink noise | MUSAN | noise/free/pink_noise |
| Brown noise | MUSAN | noise/free/brown_noise |
| Crowd babble | MUSAN | speech/librivox (multiple speakers mixed) |
| Cafeteria | DEMAND | PCAFETERIA |
| Street traffic | DEMAND | STRAFFIC |
| Car interior | DEMAND | TCAR |
| Bus interior | DEMAND | TBUS |
| Office ambient | DEMAND | OOFFICE |
| Park/nature | DEMAND | NFIELD |
| Kitchen | DEMAND | DKITCHEN |
| Fan noise | MUSAN | noise/free/fan |
| Room reverb | RIR OpenSLR 28 | small, medium, large rooms |

### 2.5 Telephone Channel Simulation (NEW — Domain Critical)

The single most important augmentation for matching academic audio to 911 call conditions is **telephone channel simulation.** Real 911 calls pass through PSTN, VoIP, or cellular networks with various speech codecs — academic recordings are pristine studio audio. The mismatch is massive.

**Codec Simulation Approach** (from Vu et al., APSIPA 2019):

The paper collected 27 audio codecs, categorized by distortion level:

| Category | Codecs | MOS Range | FFmpeg command outline |
|----------|--------|-----------|----------------------|
| **Highly distorted** | GSM-FR (06.10), GSM-HR (06.20), AMR-NB (4.75-5.9kbps), G.729a | 3.5-3.9 | `-c:a libgsm` / `-c:a libopencore_amrnb -b:a 4.75k` |
| **Medium distorted** | AMR-NB (7.4-12.2kbps), G.723.1, G.726 (16-24kbps) | 3.9-4.1 | `-c:a libopencore_amrnb -b:a 7.95k` / `-c:a g726` |
| **Minor distorted** | G.711 (A-law, μ-law), G.722, G.726 (32-40kbps), Opus (16-32kbps) | 4.1-4.4 | `-c:a pcm_alaw` / `-c:a libopus -b:a 16k` |
| **Mixed** | Random selection from full codec list | — | Random choice each sample |

Results from the paper: **Highly distorted codecs reduced WER by 7.28-12.78%** on real telephony test sets compared to clean-trained baselines. The codec-augmented model was significantly closer to real telephony conditions.

**Implementation via torchaudio:**

```python
import torchaudio

# Apply GSM codec simulation (mobile phone quality)
augmented = torchaudio.functional.apply_codec(waveform, sample_rate, "gsm")

# Apply G.711 μ-law (US landline standard)
augmented = torchaudio.functional.apply_codec(waveform, sample_rate, "ulaw")

# In torchaudio 2.0+
from torchaudio.io import CodecConfig
augmented = torchaudio.functional.apply_codec(waveform, 16000, "amrnb", 
                                                config=CodecConfig(bit_rate=4750))
```

**Note:** `torchaudio.functional.apply_codec()` is GPU-compatible and differentiable. This means codec simulation can be integrated directly into the training DataLoader, not just pre-processing.

**Domain Distortion Pipeline (proposed):**

```
Clean academic audio
  ├──► [50%] Downsample 16kHz → 8kHz (telephone bandwidth, cuts 3.4-8kHz)
  ├──► [50%] Apply random codec from pool:
  │         ├──► Mobile: AMR-NB at 4.75-12.2kbps (worst quality)
  │         ├──► Landline: G.711 μ-law or A-law
  │         ├──► VoIP: Opus at 5.5-16kbps or G.729
  │         └──► Radio: GSM full-rate or half-rate
  ├──► [30%] Simulate packet loss (drop random frames with interpolation)
  ├──► [20%] Add background noise (from safe pool §2.4)
  └──► [20%] Add RIR reverb (light — phone audio has minimal reverb)
```

This produces audio that sounds like it was recorded over a phone line, which is exactly the domain of 911 calls.

---

## 3. Domain-Specific Findings for Emergency Call SER

### 3.1 911 Call Acoustic Conditions

From the literature and real data analysis:

| Condition | Real 911 | Academic datasets | Can be simulated? |
|-----------|----------|-------------------|-------------------|
| **Narrow bandwidth** | 300-3400Hz (telephone) | 0-8000Hz (studio) | ✓ Downsample 16→8kHz |
| **Speech codec** | G.711, AMR-NB, GSM | None (PCM WAV) | ✓ Codec simulation |
| **Background noise** | Car, traffic, crowd, wind | Silence | ✓ Noise injection |
| **Phone distortion** | Clipping, compression | Clean | ✓ Dynamic range compression |
| **Cross-talk** | Dispatcher + caller overlapping | Single speaker | ✗ Complex to simulate |
| **Emotional speech** | Real stress, crying | Acted | ✗ Hard to simulate |
| **Non-stationary noise** | Sirens passing, doors | None | ✓ Via sound event mixing |

### 3.2 Emergency Call SER Challenges (from literature)

| Challenge | Reference | Impact | Potential Mitigation |
|-----------|-----------|--------|---------------------|
| **Clean-trained models fail on real calls** | Zhu-Zhou et al. 2022 | Error rate increases from 25.57% to 79.13% in worst case | Multi-condition training with noise + codec |
| **Text-based features improve urgency detection** | Abi Kanaan et al. 2023 | Text features provide complementary robustness | Fusion (our Stage 5 design) |
| **Phone-like ASR degradation** | Vu et al. 2019 | WER increases 7-12% on telephony data | Codec-aware augmentation |
| **Speech enhancement before SER** | Triantafyllopoulos et al. 2019 | Enhancement can remove emotional cues | Prefer augmentation over enhancement |
| **Short utterances degrade SER** | Wijayasingha et al. | Segments <1s lose emotion signal | Academic clips run 1-2 s, so this applies to us |

### 3.3 Recommended Augmentation Strategy for Each Dataset

| Dataset | Quality | Augmentation needed | Approach |
|---------|---------|-------------------|----------|
| CREMA-D (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| RAVDESS (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| TESS (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| SAVEE (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| JL-Corpus (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| ASVP-ESD (studio) | Clean | Heavy | Codec + noise + SpecAugment + simplex |
| Kaggle Emerg (phone) | Some noise | Light | Simplex only, light SpecAugment |

---

## 4. Augmentation Design Logic

### 4.1 Three Independent Augmentation Dimensions

| Dimension | Transform | Applied to | Parameters | Rationale |
|-----------|-----------|-----------|------------|-----------|
| **Simplex distortion** | Speed, pitch, volume | Raw waveform (time domain) | Speed: 0.9-1.1×, Pitch: ±2 semitones, Volume: ±6dB | Speaker and recording variation — no new acoustic conditions |
| **Noise injection** | Add background noise at controlled SNR | Waveform (before mel) | SNR: 5-20dB (random), noise type: random draw from safe pool (see 2.4) | Simulates real-world 911 acoustic environments |
| **Codec simulation** | Telephony codecs, downsampling | Raw waveform | Random codec from pool (see 2.5), 8kHz downsampling | MATCHES DOMAIN — academic → telephone acoustic |
| **SpecAugment** | Frequency + time masking | Mel spectrogram | F: max 8 bins, T: max 20 frames | Robustness to partial signal loss or spectral corruption |

### 4.2 Per-Class Augmentation Rates

| Class | Hours | Augmentation multiplier | Strategy |
|-------|-------|------------------------|----------|
| panic | 31.0h (30.6%) | 1× (none) | Standard simplex + codec + noise + SpecAugment |
| neutral | 26.3h (26.0%) | 1× (none) | Standard |
| urgency | 19.9h (19.6%) | 1× (none) | Standard |
| distress | 17.3h (17.1%) | 1× (none) | Standard |
| **fear** | **3.4h (3.4%)** | **3× (oversample)** | **Heavier SpecAugment, lower SNR noise (5-10dB), stronger codec** |
| **confusion** | **3.5h (3.5%)** | **3× (oversample)** | **Heavier SpecAugment, lower SNR noise (5-10dB), stronger codec** |

### 4.3 Augmentation Pipeline Order

```
Raw audio (academic)
  │
  ├──► Step 1: Simplex distortion (speed ±10%, pitch ±2 semitones, volume ±6dB)
  │         Randomly select 0-3 transforms, each applied with 50% probability
  │
  ├──► Step 2: Codec simulation (DOMAIN-CRITICAL)
  │         Random: GSM, AMR-NB (4.75-12.2k), G.711, Opus (5.5-16k)
  │         Or downsample 16→8kHz (telephone bandwidth)
  │         Applied with 50-80% probability
  │
  ├──► Step 3: Noise injection
  │         Randomly select one noise clip from safe pool (see §2.4)
  │         Scale to target SNR (random: 5-20dB, tail classes: 5-10dB)
  │         Applied with 80% probability
  │
  ├──► Step 4: RIR convolution (optional — light reverb)
  │         Applied with 20-30% probability (phone calls are dry)
  │
  ├──► Step 5: Convert to log-mel spectrogram
  │
  ├──► Step 6: SpecAugment
  │         Time mask: max T=20 (or T=30 for tail classes)
  │         Freq mask: max F=8 (or F=12 for tail classes)
  │         Applied with 80% probability
  │
  └──► Step 7: Feed into fusion transformer

Raw audio (already-degraded sources)
  │
  ├──► Step 1: Light simplex (speed ±5%, pitch ±1 semitone, volume ±3dB)
  │
  └──► Step 2: Convert to log-mel → feed into fusion
         (no codec, no noise, no SpecAugment — real distortion already present)
```

---

## 5. Implementation Notes

### 5.1 Recommended Libraries

| Library | Functions | Why |
|---------|-----------|-----|
| `torchaudio` | `apply_codec`, `sox_effects`, `functional` | Codec simulation + speed/pitch/volume. Native PyTorch, GPU-compatible |
| `torchaudio-augmentations` | `AugmentModule` | Pre-built augmentation pipeline with composable transforms |
| `librosa` | `effects.pitch_shift`, `effects.time_stretch` | Simplex distortion (CPU, used in preprocessing) |
| `audiomentations` | `Compose`, `AddBackgroundNoise`, etc. | Audio augmentation library |
| `ffmpeg` (via subprocess or `imageio-ffmpeg`) | Codec encoding/decoding | For codecs not supported by torchaudio |
| `numpy` | Noise generation, SNR computation | White/pink/brown noise, random selection |

### 5.2 SNR Computation

```python
def add_noise_at_snr(speech, noise, target_snr_db):
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    current_snr = 10 * np.log10(speech_power / (noise_power + 1e-10))
    scaling_factor = 10 ** ((current_snr - target_snr_db) / 20)
    scaled_noise = noise * scaling_factor
    return speech + scaled_noise
```

### 5.3 Codec Simulation (via torchaudio)

```python
import torchaudio
from torchaudio.io import CodecConfig
import random

def apply_random_codec(waveform: torch.Tensor, sample_rate: int = 16000):
    """Apply random telephony codec to simulate phone channel."""
    codec = random.choices(
        ["gsm", "ulaw", "amrnb", "opus"],
        weights=[0.3, 0.3, 0.25, 0.15]
    )[0]
    
    if codec == "gsm":
        # GSM full-rate (13 kbps) — mobile calls
        return torchaudio.functional.apply_codec(waveform, sample_rate, "gsm")
    elif codec == "ulaw":
        # G.711 μ-law — US landline
        return torchaudio.functional.apply_codec(waveform, sample_rate, "ulaw")
    elif codec == "amrnb":
        # AMR-NB at low bitrate — degraded mobile
        bitrate = random.choice([4750, 5150, 5900, 6700, 7400, 7950])
        return torchaudio.functional.apply_codec(
            waveform, sample_rate, "amrnb",
            config=CodecConfig(bit_rate=bitrate)
        )
    elif codec == "opus":
        # Opus VoIP codec at low bitrate
        bitrate = random.choice([5500, 7700, 9500, 12500, 16000])
        return torchaudio.functional.apply_codec(
            waveform, sample_rate, "opus",
            config=CodecConfig(bit_rate=bitrate)
        )
```

### 5.4 Noise Dataset Preparation

Each noise dataset's selected classes should be:
1. Downloaded and extracted once (not at training time)
2. Concatenated into a single noise pool with labeled source
3. Split into random-access chunks (~2-10s segments matching typical segment length)
4. Cached as a single tensor or numpy file for fast random access during training

---

## 6. References

### 6.1 Academic References

| Reference | Title | Venue | Year |
|-----------|-------|-------|------|
| Ko et al. | Audio Augmentation for Speech Recognition | Interspeech | 2015 |
| Park et al. | SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition | Interspeech | 2019 |
| Sahu et al. | On Enhancing Speech Emotion Recognition using Generative Adversarial Networks | Interspeech | 2018 |
| Bao et al. | CycleGAN-based Emotion Style Transfer as Data Augmentation for Speech Emotion Recognition | ICASSP | 2019 |
| Latif et al. | Augmenting Generative Adversarial Networks for Speech Emotion Recognition | Interspeech | 2020 |
| Malik et al. | A Preliminary Study on Augmenting Speech Emotion Recognition using a Diffusion Model | Interspeech | 2023 |
| Avci et al. | A Comprehensive Analysis of Data Augmentation Methods for Speech Emotion Recognition | IEEE Access | 2025 |
| Tao et al. | Strong Generalized Speech Emotion Recognition Based on Effective Data Augmentation | MDPI Sensors | 2022 |
| Rizos et al. | StarGAN for Emotional Speech Conversion | ICASSP | 2020 |
| Vu et al. | Audio Codec Simulation based Data Augmentation for Telephony Speech Recognition | APSIPA | 2019 |
| Zhu-Zhou et al. | Robust Multi-Scenario Speech-Based Emotion Recognition System | MDPI Sensors | 2022 |
| Triantafyllopoulos et al. | Towards Robust Speech Emotion Recognition Using Deep Residual Networks | Interspeech | 2019 |
| Wijayasingha et al. | Robustness to Noise for Speech Emotion Classification using CNNs | CHASE | 2020 |
| Abi Kanaan et al. | A methodology for emergency calls severity prediction | HAL Science | 2023 |
| Gil-Pita et al. | Speech Emotion Recognition Using Transfer Learning and Adaptive Timeframe Segmentation | IEEE ICSPCC | 2025 |

### 6.2 Dataset Sources

| Dataset | Purpose in Pipeline | License | Access URL |
|---------|-------------------|---------|-----------|
| MUSAN | Generic noise + babble injection | CC-BY 4.0 | https://www.openslr.org/17/ |
| DEMAND | Real-world environmental noise (traffic, vehicles, public) | CC-BY-SA 3.0 | https://zenodo.org/record/1227121 |
| RIR OpenSLR 28 | Room impulse response for reverberation | Apache 2.0 | https://www.openslr.org/28/ |
| FSDnoisy18k | Ambient sound events (curated safe subset) | CC-BY 4.0 | https://zenodo.org/records/2529934 |
| ESC-50 | Supplemental environmental sounds | CC-BY | GitHub (karoldvl/ESC-50) |
| UrbanSound8K | Urban noise (curated safe subset) | CC-BY | GitHub (marcobank/UrbanSound8K) |
| AudioSet | Cherry-picked noise types (sparingly) | CC-BY 4.0 | https://research.google.com/audioset/ |
| SEAME corpus | Codec simulation training baseline (reference only) | Research use | NTU Singapore |
