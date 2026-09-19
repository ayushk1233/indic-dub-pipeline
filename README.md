# Indic Speech Dubbing & QC Pipeline

A video in one language goes in; the same video, dubbed into an Indic language
and still in the original speaker's voice, comes out.

The interesting problem is not translation and it is not cloning. It is that
**Hindi takes about 1.2x as long to speak as the English it replaces**, while
dubbing requires it to fit the original timing anyway. Most of this repository
is about closing that gap — choosing translations that fit, and absorbing what
is left over during assembly.

Everything here is measured rather than assumed. [FINDINGS.md](FINDINGS.md) is
the record, including [§14](FINDINGS.md), a list of things this project
believed and then disproved.

## Status

| leg | state |
|---|---|
| **en → hi** (dubbing) | **Measured.** CER 0.052, identity at 93% of the calibrated scale, pace 1.00x natural |
| **hi → hi** | **Measured.** CER 0.055 against a human floor of 0.084 — at the floor |
| en → en, hi → en | **Not possible yet.** The base model cannot generate English (FINDINGS §4). This is what fine-tuning is for |

"At the floor" means the content detector can no longer separate the model
from the speaker's own recording. It does not mean the product is finished —
identity, naturalness and accent are separately unmeasured, and
[FINDINGS §3](FINDINGS.md) explains why each needs its own floor before any
number from it can be read.

**Deliberately not built yet:** HTTP endpoints and any frontend. Those come
after fine-tuning, so that what gets shown is the finished model rather than
this one.

## Model

**IndicF5** (MIT, 0.4B, 24 kHz). XTTS-v2 remains in the tree as a calibrated
baseline and **cannot ship** — CPML, non-commercial, Coqui is defunct, and
fine-tuned checkpoints inherit the licence (FINDINGS §9).

The shipping configuration is [FINDINGS §1](FINDINGS.md), and three of its
settings are load-bearing in ways that fail silently if changed:

- a **10 s** reference clip, because IndicF5 clips a longer one internally and
  never shortens its transcript to match (§5d)
- a reference transcript **lowercased and stripped of punctuation** (§5c) —
  and it should also be in Devanagari (§16e), which is blocked on
  transliteration
- duration set from the target slot via `fix_duration`, **not** from the
  model's byte ratio, which over-allocates by 2.46x (§5a)

## Running it

Python 3.11. The venv was created at an older path, so call its interpreter
directly rather than activating it.

```bash
./venv/bin/python -m pytest tests/ -v
```

The pipeline stops at the GPU boundary with a bundle zip:

```bash
./venv/bin/python -m src.cli --input test.mp4 --job-id demo --target-lang hi --candidates 6
```

Carry `artifacts/demo/tts_bundle.zip` to a GPU host, run the worker, bring the
result back, and resume:

```bash
python -m colab.indicf5_worker --bundle /kaggle/working/tts_bundle
```

```bash
./venv/bin/python -m src.cli --input test.mp4 --job-id demo --from-stage import
```

`--candidates 1` turns length control off. [colab/kaggle.md](colab/kaggle.md)
covers the GPU host; every trap in it fails in a way that looks like a
different problem.

## Architecture

Stages run in order — preprocess (FFmpeg), ASR (Faster-Whisper), translation
(IndicTrans2), synthesis (IndicF5), assembly, remux. Everything except
synthesis runs locally on CPU.

Stages hand off through files under `artifacts/<job_id>/`, not in-memory
objects, and timestamps are always absolute against the source media so
alignment survives translation and synthesis.

**The GPU boundary is a file contract, not a network call.** The local side
writes a versioned bundle; a worker on the GPU host reads it and writes
`output/synthesis_result.json` after every segment, so a timeout or one bad
segment still leaves salvageable work. `ExternalExecutionBackend` is the
general contract — Colab and Kaggle are transports, not the architecture.

[CLAUDE.md](CLAUDE.md) documents the stage contract, the backend/processor
split, and the pins that break things silently when bumped.

## Where the numbers come from

Every content number is a character error rate from Whisper, reported next to
a floor measured by transcribing the speaker's own recording of the same
sentence. A CER alone is unreadable, and a speaker-similarity cosine alone is
worse — [FINDINGS §2](FINDINGS.md) sets out the calibration this project uses
and why a raw cosine cannot be read as a percentage of anything.

Two of this project's findings were caught by ear and by no metric. Listening
is part of the protocol, not a courtesy.
