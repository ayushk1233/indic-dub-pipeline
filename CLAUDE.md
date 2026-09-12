# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Indic Speech Dubbing & QC Pipeline: a video in one language goes in, the same video dubbed into an
Indic language comes out. Stages run in order — preprocess (FFmpeg), ASR (Faster-Whisper),
translation (IndicTrans2), synthesis (XTTS-v2), assembly, remux. Everything except synthesis runs
locally on CPU; synthesis runs on an external GPU.

The problem the project exists around, measured rather than assumed: Hindi takes about 1.2x as long
to speak as the English it replaces, while dubbing requires it to fit the original timing anyway.
Most of the interesting code is about closing that gap — choosing translations that fit, and
absorbing what is left over during assembly.

## Commands

Everything runs from the repo root with the local virtualenv (Python 3.11). The venv was created at
an older path, so invoke its interpreter directly rather than relying on `activate`:

```bash
./venv/bin/python -m pytest tests/ -v
```

Single test:

```bash
./venv/bin/python -m pytest tests/test_preprocessing.py::test_build_segments -v
```

Tests use relative paths (`test.mp4`, `config/pipeline.yaml`) and write into `artifacts/`, so they
only pass when run from the repo root. There is no linter, formatter, or build step configured.

Run the whole pipeline:

```bash
./venv/bin/python -m src.cli --input test.mp4 --job-id demo --target-lang hi --candidates 6
```

It stops at the GPU boundary with a bundle zip. After synthesis comes back, resume with
`--from-stage import`. `--candidates 1` turns length control off.

Measure speaking rates and fit the duration model:

```bash
./venv/bin/python -m src.data.measure --languages hi --limit 300
./venv/bin/python -m src.eval.duration_model --languages en hi --limit 300
```

`requirements.txt` is pinned, and the non-obvious pins carry their reasoning inline. The one to know
about: **transformers is held below 4.47** because IndicTrans2's `trust_remote_code` modelling code
indexes `past_key_values` as legacy tuples, and newer versions always hand `generate()` a Cache
object. Transformers 5.x additionally removed `transformers.onnx`, which that same code imports.
Bumping it silently breaks translation at decode time, not at import time. Colab dependencies are
pinned separately in [colab/requirements.txt](colab/requirements.txt) and are deliberately different.

## Architecture

### Stage contract

Every stage implements [PipelineStage](src/stages/base.py): `validate_input(path) -> bool` and
`run(input_path, job_id, cfg) -> StageResult`. Orchestration code should only ever talk to that
interface. `StageResult` carries a status enum, an output path, and a free-form `metrics` dict, all
defined in [src/orchestrator/models.py](src/orchestrator/models.py).

Only preprocessing implements `PipelineStage`. ASR, translation and TTS are processors that
[PipelineRunner](src/pipeline/runner.py) drives directly, because they need richer per-stage inputs
than `run(input_path, job_id, cfg)` provides. The database and serving layers are still empty
placeholder files.

### Backend/processor split

Each of ASR, translation, and TTS follows the same three-file shape inside its package:

- `backend.py` — abstract inference interface (`load()` plus one inference method).
- `<vendor>_backend.py` — the concrete model implementation, holding all model-specific details.
- `processor.py` — orchestrates the backend over a whole job and returns typed Pydantic results.

Model-specific knowledge stays behind the backend boundary. For example the FLORES language tags
(`hin_Deva`, `tam_Taml`) live only in [indictrans2_backend.py](src/stages/translation/indictrans2_backend.py);
everything upstream uses plain two-letter codes.

### Data flow between stages

Stages hand off through files on disk under `artifacts/<job_id>/`, not in-memory objects.
Preprocessing writes `audio.wav`, `chunks/chunk_NNNN.wav`, and `manifest.json`. The ASR processor
reads that manifest path and derives `job_id` from the parent directory name. Timestamps are always
absolute against the source media: the ASR processor adds each chunk's `start_ts` to the per-chunk
offsets Whisper returns, and every downstream model carries `segment_id`, `chunk_id`, `start_ts`, and
`end_ts` so alignment survives translation and synthesis.

Note that [ManifestWriter](src/stages/preprocessing/manifest.py) emits a bare JSON list, while
[ChunkManifest](src/stages/preprocessing/models.py) describes a richer object. The typed models were
added first and the writer has not been migrated, so the two do not currently agree.

### Segmentation

[Segmenter](src/stages/preprocessing/segmentation.py) finds speech by inverting FFmpeg's
`silencedetect` output, parsed out of stderr with regexes. Segments shorter than 0.25 s are dropped.
When silence detection yields nothing usable, `preprocess.py` falls back to fixed 30 s windows with
1 s overlap. Those thresholds are class constants and are deliberately not in `pipeline.yaml` yet.

### The GPU boundary

XTTS needs CUDA, which the local machine does not have, so TTS is split across a file-based
boundary rather than a network call. The local side builds a `SynthesisRequest` and
[BundleExporter](src/stages/tts/bundle/exporter.py) writes a versioned bundle directory
(`manifest.json`, `request/synthesis_request.json`, `request/reference.wav`, empty `output/` and
`logs/`) which can be zipped and carried anywhere. On the GPU side,
[XTTSWorker](colab/xtts_worker.py) reads the bundle, checks `bundle_version == "1.0"`, runs GPU
preflight checks, loads XTTS, and synthesizes.

Treat Colab as one transport, not as the architecture. `ExternalExecutionBackend` in
[src/stages/tts/backend.py](src/stages/tts/backend.py) is the general contract: export a request,
import a result. `ColabTTSBackend.synthesize()` raises `NotImplementedError` on purpose, because
execution happens elsewhere.

`XTTSWorker.run()` writes `output/synthesis_result.json` after every segment, so a Colab timeout
or one bad segment still leaves salvageable work on disk. A segment that raises is recorded with
`status="failed"` rather than aborting the run. The loop itself has not yet been executed against a
real GPU.

### XTTS configuration

[colab/xtts.md](colab/xtts.md) records why the current XTTS settings were chosen. The short version:
random output durations came from stochastic decoding, not from checkpoints or dependencies, and
`do_sample=False` fixes it. Preferred conditioning is `gpt_cond_len=8`, `gpt_cond_chunk_len=4`,
`max_ref_len=10`. Do not pass `temperature`, `top_k`, or `top_p` alongside `do_sample=False`. `synthesize_segment()`
passes `INFERENCE_PARAMS` and no sampling parameters. One unresolved tension from the end of that
document: it reports that greedy decoding degrades with short reference audio, while
`CONDITIONING_PARAMS` caps `max_ref_len` at 10s and the runner picks the longest chunk, which for the
test clip is 7.5s. Nobody has measured which effect wins.
Read that document before changing any inference parameter.

## Conventions

- Config comes from [config/pipeline.yaml](config/pipeline.yaml), loaded with `yaml.safe_load` and
  passed to stages as a plain dict. TTS device is `cuda` even though the rest is `cpu`, because
  synthesis executes externally.
- Every cross-stage data structure is a Pydantic model. Evaluation metrics in `src/eval/` are plain
  dataclasses instead.
- Imports are absolute from `src.`, so commands must run from the repo root.
- After each unit of work, append an entry to [PROGRESS.md](PROGRESS.md) in the existing format:
  step number, phase/day, what was completed, a verification line, and any deviations. Deviations are
  used to record intentionally deferred work, so state what was deferred and why.
