# 🎙️ XTTS-v2 Investigation Report
**Indic Speech Dubbing Pipeline - TTS Stability Research**

> **Project:** Indic Speech Dubbing & QC Pipeline
> **Component:** XTTS-v2 Voice Cloning Engine
> **Status:** Stable Configuration Identified (MVP)
> **Date:** July 2026

---

## 🎯 Objective

The objective of this investigation was to establish a stable, reproducible, production-ready XTTS-v2 configuration for the Indic Speech Dubbing Pipeline.

The primary issues observed were:

- inconsistent synthesized audio durations
- occasional gibberish/noise outputs
- unstable decoding behavior
- dependency incompatibilities
- lack of deterministic inference

The goal was to understand why these failures occurred instead of simply masking them.

---

## 💻 Environment

### Hardware
- **Google Colab**
- **GPU:** Tesla T4

### Python
- Python 3.11

---

## 📦 Stable Dependency Matrix

After extensive experimentation, the following dependency combination proved to be the most stable.

```text
Python             3.11
torch              2.5.1+cu121
torchaudio         2.5.1+cu121
transformers       4.40.0
tokenizers         0.19.1
TTS                0.22.0
numpy              1.26.4
pandas             1.5.3
librosa            0.10.1
ffmpeg             latest
```

> ⚠️ **Note:** This dependency matrix should be treated as the known-good baseline.

---

## ⚠️ Initial Problem

A simple sentence:
> *"Hello, my name is Ayush. I am testing XTTS version two."*

produced dramatically different outputs on every run.

### Example

**Run 1**
- 4 seconds
- Excellent quality

**Run 2**
- 7 seconds
- Acceptable

**Run 3**
- 14 seconds
- Speech continued far too long

**Run 4**
- 5 seconds
- Good

**Same:**
- text
- speaker
- parameters
- environment

Only the generated speech differed.

---

## 🔬 Phase 1: Dependency Stabilization

Initially multiple dependency combinations were attempted.

**Problems included:**
- incompatible transformers versions
- tokenizer mismatches
- CUDA incompatibilities
- runtime crashes
- noisy outputs

After multiple rebuilds the environment shown above became stable.

---

## 🔬 Phase 2: Reference Audio Validation

**Reference recording:**
`reference.wav`

**Properties:**
- **Duration:** 24.47 seconds

The model loaded successfully.

No checkpoint corruption was found.

**Verified cache:**
- `config.json`
- `model.pth`
- `vocab.json`
- `speakers_xtts.pth`
- `hash.md5`

---

## 🔬 Phase 3: Inspecting XTTS Internal APIs

Instead of guessing, the internal XTTS implementation was inspected.

The following functions were analyzed.

```text
tts_to_file()
      ↓
tts()
      ↓
Synthesizer.tts()
      ↓
tts_model.synthesize()
      ↓
full_inference()
      ↓
model.inference()
      ↓
GPT.generate()
```

This investigation showed that `tts_to_file()` does NOT directly call `inference()`

instead:

```text
tts_to_file
      ↓
Synthesizer.tts
      ↓
synthesize
      ↓
full_inference
      ↓
model.inference
```

Understanding this call chain was essential before modifying generation parameters.

---

## 🔬 Phase 4: Conditioning Parameter Investigation

**Default XTTS values:**
- `gpt_cond_len = 30`
- `gpt_cond_chunk_len = 6`
- `max_ref_len = 30`

Since the reference audio itself was only **24.47 seconds**, XTTS was conditioning on almost the entire recording.

This produced inconsistent speaker conditioning.

### Experiments

**Test 1**
- `gpt_cond_len = 6`

**Result:** Large quality improvement.

**Test 2**
- `max_ref_len = 10`

**Result:** Cleaner voice cloning.

**Test 3**
- `gpt_cond_len = 8`
- `gpt_cond_chunk_len = 4`
- `max_ref_len = 10`

**Result:** Most consistent conditioning.

This became the preferred configuration.

---

## 🔬 Phase 5: Investigating max_new_tokens

The suspicion was that GPT decoding continued for too many audio tokens.

The following experiment was performed.

- `max_new_tokens = 120`

**Observed:**
- Output duration became consistent.
- **However:** Speech became slightly robotic.

### Additional Benchmark

Multiple token limits:
- `100`
- `110`
- `120`
- `130`
- `140`
- `150`

were tested.

The model always produced GPT latent tensors matching:
`(1, max_new_tokens, 1024)`

showing that the generation length was directly constrained.

---

## 🔬 Phase 6: Investigating Sampling

Inspection of `model.inference()` revealed:
- `temperature`
- `top_k`
- `top_p`
- `do_sample`
- `hf_generate_kwargs`

were forwarded into `transformers.generate()`.

This suggested that sampling randomness might be responsible.

### Experiment

**Default:**
- `do_sample=True`

**Repeated runs:**
- `4 sec`
- `7 sec`
- `14 sec`
- `5 sec`

The randomness persisted.

---

## 🔬 Phase 7: Greedy Decoding

The next experiment disabled sampling.

- `do_sample=False`

**Result:**
Five consecutive runs produced:
- `5 sec`
- `5 sec`
- `5 sec`
- `5 sec`
- `5 sec`

This demonstrated that **sampling was the root cause of duration instability.**

### Investigation of Greedy Output

To understand why greedy decoding remained stable, the generated latent tensor size was inspected.

`gpt_latents.shape`

**Output:**
`(1, 116, 1024)`

**Meaning:**
The model naturally terminated after **116 audio tokens** without requiring any artificial cutoff.

### 🚨 Important Discovery

Greedy decoding naturally stopped before `120` tokens.

Therefore `max_new_tokens=120` was not responsible for stability.

Instead `do_sample=False` was responsible.

---

## 🗑️ Why max_new_tokens was Removed

Initially `max_new_tokens=120` appeared to solve duration inconsistency.

However further investigation showed:
Greedy decoding already stopped naturally after **116 tokens**.

Therefore `max_new_tokens` added no benefit during normal inference.

It only serves as a safety limit if generation ever fails to terminate.

**For MVP:** it was removed.

---

## ⚠️ Warnings Observed

When `do_sample=False` the following parameters become inactive.
- `temperature`
- `top_k`
- `top_p`

Transformers correctly emits warnings.

These parameters should simply be omitted.

---

## 🔍 Root Cause Analysis

The instability was not caused by:
- corrupted checkpoints
- incompatible dependencies
- GPU
- preprocessing
- reference audio loading
- tokenizer issues

The instability originated from **stochastic autoregressive decoding** inside the GPT component of XTTS.

Sampling occasionally failed to predict the EOS token at the correct point, allowing generation to continue much longer than intended.

---

## 🛠️ Stable Production Configuration

The recommended XTTS configuration for the MVP is:

```python
tts.tts_to_file(
    text=text,
    speaker_wav="reference.wav",
    language="en",

    gpt_cond_len=8,
    gpt_cond_chunk_len=4,
    max_ref_len=10,

    do_sample=False,

    repetition_penalty=5.0,

    split_sentences=False,
)
```

### Production Notes

**Current recommendations:**
- ✅ Python 3.11
- ✅ Torch 2.5.1
- ✅ TTS 0.22.0
- ✅ Transformers 4.40.0
- ✅ Greedy decoding
- ✅ Short reference conditioning
- ✅ 10 second maximum reference

**Do not use:**
- `temperature`
- `top_p`
- `top_k`

when `do_sample=False`

---

## 💡 Lessons Learned

1. Dependency stability is critical before debugging model behavior.
2. XTTS conditioning length significantly affects cloning quality.
3. Shorter conditioning windows improved consistency.
4. Sampling (`do_sample=True`) introduced non-deterministic duration and quality.
5. Greedy decoding (`do_sample=False`) eliminated duration instability.
6. `max_new_tokens` is not a quality improvement mechanism; it is only a safeguard against runaway generation.
7. Inspecting the XTTS source code (`tts_to_file` → `synthesize` → `full_inference` → `inference` → `generate`) was essential for understanding parameter propagation and identifying the true cause of the issue.

---

## 🎉 Final Outcome

The investigation produced a stable, deterministic XTTS-v2 baseline suitable for the Indic Speech Dubbing MVP.

### Achievements

- ✔ Identified a known-good dependency matrix.
- ✔ Eliminated random-duration synthesis.
- ✔ Verified the XTTS inference call chain.
- ✔ Established optimal conditioning parameters (`gpt_cond_len=8`, `gpt_cond_chunk_len=4`, `max_ref_len=10`).
- ✔ Determined that stochastic sampling was the primary source of instability.
- ✔ Established `do_sample=False` as the production baseline for deterministic speech generation.

This configuration should serve as the reference implementation for the TTS stage of the dubbing pipeline until future model evaluations justify revisiting decoder settings or adopting a newer XTTS release.