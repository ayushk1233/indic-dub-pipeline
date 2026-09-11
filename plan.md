# Indic Dubbing Pipeline: Assessment and Roadmap to First Output

## Context

The repository contains 57 logged build steps, 69 commits, and roughly 2,100 lines of
carefully layered code across four pipeline stages. It has never run end to end. There is no
entrypoint of any kind in the working tree or anywhere in git history: no `__main__` block, no
CLI, no notebook, no shell script. Twelve files are zero bytes, including the orchestrator, the
database layer, the bundle importer, and the remux stage. Nothing in the repo ever calls
`load()` on an ASR or translation backend, so both would raise `RuntimeError("Model not
loaded.")` on first use by any caller.

The abstractions are good. The problem is that they were built downward, layer by layer,
without a spine connecting them. This plan builds the spine.

Scope decided with the user: Hindi only, prototype heading toward a real system, and the
overriding goal is a first watchable dubbed video.

## How the pipeline stands today

**What works.** Preprocessing is real, tested, and runs. It extracts normalized audio, finds
speech by inverting FFmpeg silence detection, cuts chunks, and writes a manifest. Seven tests
pass. The backend/processor split across ASR, translation, and TTS is the best thing in the
codebase: abstract interface, vendor implementation, and a processor that loops the backend
over a job. Model-specific detail stays behind the boundary. The bundle format that carries
work to a GPU is a sound piece of design, and the research in `colab/xtts.md` is genuine
engineering rather than parameter roulette.

**What is missing.** The dubbing step of the dubbing pipeline does not exist. No code compares
a synthesized segment's duration against its slot, no code time-stretches, no code assembles a
track, and no code muxes audio back onto video. `SynthesizedSegment.duration` is a field that
nothing populates. The GPU round trip is open at both ends: the worker's `write_result` raises
`NotImplementedError`, and the importer meant to read its output is an empty file.

**What is stranded.** The declared dependency list describes an architecture that was never
built. FastAPI, Celery, Redis, SQLAlchemy, and ONNX Runtime are installed and imported by
nothing. The two libraries the code genuinely needs, PyTorch and Transformers, are absent from
the list. Locally, PyTorch is a broken install with a missing shared library, so the
translation stage cannot even be imported.

**The constraint nobody wrote down.** The translation layer declares thirteen Indic target
languages. XTTS-v2 speaks exactly one of them. The pipeline is architecturally committed to a
breadth its voice engine cannot deliver. Confirm the supported-language list on the Colab side
before locking this in, then either scope to Hindi or change the TTS engine.

## Bottlenecks, ranked by what they actually cost

The ordering below is deliberate. The first item dwarfs the rest, and the measurement work
that would rank items three through five properly cannot happen until item one is fixed.

1. **Nothing runs.** This is not a figure of speech. The critical path from video to dubbed
   video is broken in at least five separate places, and no single command exercises any two
   stages in sequence. Every performance number below is an estimate, because the pipeline has
   never produced a measurement.

2. **The human in the GPU loop.** Synthesis runs by hand in Colab: upload, run cells,
   download. That is minutes to tens of minutes of wall clock per iteration, it is
   unreproducible, and the notebook was never committed. It dominates every other cost by an
   order of magnitude and it blocks iteration on the timing problem, which needs many cheap
   synthesis runs.

3. **ASR pays a full encoder pass per chunk.** Whisper pads every input to a thirty-second mel
   window. The current processor hands it chunks that are often one or two seconds long, so a
   1.4-second segment costs the same `large-v3` encoder pass as a full window. On the 24-second
   test clip that is six padded windows for 24 seconds of speech. `BatchedInferencePipeline` is
   installed and unused, and it also does its own voice-activity segmentation, which makes the
   chunk loop unnecessary rather than merely slow.

4. **Translation runs one 1B-parameter beam-search pass per segment on CPU.** Batch size one,
   five beams, no batching anywhere. Batching and dropping to greedy decoding are independent
   multipliers, and greedy output is shorter, which helps the timing problem downstream.

5. **FFmpeg process spawns, a minor cost worth knowing.** I measured it rather than guessing.

   | Operation | Cost |
   |---|---|
   | One FFmpeg process spawn | 51 ms |
   | Full decode of the 24 s test audio | 37 ms |

   Process startup costs more than decoding the entire file. Preprocessing spawns four
   processes plus one per segment, so a ten-minute video with 200 segments spends about ten
   seconds in process startup alone. Real, but an order of magnitude below items three and
   four. Do not start here.

**Correctness defects that will bite.** The silence-detection regex cannot match a negative
number, and FFmpeg emits a negative first marker whenever audio begins in silence, which is
most real video. I reproduced the failure: silences at -0.003 to 1.2 and 5.5 to 6.75 parse into
the single inverted pair (5.5, 1.2), producing overlapping speech segments. The test clip
happens not to start in silence, so this is invisible today. Separately, the Colab worker names
its output files by chunk identifier, but many ASR segments share one chunk, so synthesized
audio silently overwrites itself; and `run()` calls a dictionary method on a Pydantic model,
which crashes immediately.

## Decisions

**Wrap the remaining stages in the existing `PipelineStage` contract rather than replacing it.**
The interface looks too narrow for downstream stages, but it is not. Each stage has one primary
input artifact and everything else derives from the job identifier and the fixed artifacts
layout. Each stage's output path becomes the next stage's input path, and that chain is the
entire runner. The stages are thin adapters over the processors that already exist.

**Fit duration with a three-tier cascade, computed locally after synthesis returns.** This is
the central unsolved design problem, so the reasoning matters. Hindi typically runs ten to
thirty percent longer than English, XTTS cannot be told to hit a duration, and every segment
must land at its original start time.

- Tier one, absorb the gap. Most segments fit without processing, because a normal
  inter-sentence pause is exactly the slack a thirty percent expansion needs. Free, and no
  artifacts.
- Tier two, time-stretch with FFmpeg `atempo` up to about 1.25. It is a time-domain,
  pitch-preserving stretcher, better on speech than a phase vocoder, and it adds no new Python
  dependency to a project whose dependency situation is already a liability.
- Tier three, allow bounded drift. Cap the stretch, let the segment run long, push the next
  segment later by at most a quarter second, and reset accumulated drift at the next real pause.
  Drift is therefore bounded by the distance to the next pause and never compounds across a video.

Rejected: controlling length at translation time, because IndicTrans2 has no length control and
the only levers truncate or degrade meaning; the XTTS speed parameter, because the required
ratio is unknowable until after synthesis, which means a second manual GPU round trip; and
`librosa` time-stretching, because it is a phase vocoder and pulls in four packages that are
not installed.

**Assemble the track as a NumPy buffer, not an FFmpeg filtergraph.** Allocate silence for the
full video duration at the synthesis sample rate, add each fitted segment at its exact sample
offset with short fades to kill clicks, and write one WAV. This avoids the `amix` normalization
trap, scales to hundreds of segments without an unwieldy command line, and is unit-testable
without spawning FFmpeg: you can assert a segment landed at an exact sample index.

**Keep the bundle abstraction, delete the premature scaffolding.** The external-execution
boundary is correct and should survive. The FastAPI, Celery, Redis, SQLAlchemy, and ONNX files
should be deleted rather than left empty. Twelve zero-byte files with plausible names are a
large part of why this repo reads as finished when it has never executed. An empty file asserts
that something exists; a missing file asserts nothing, and git remembers either way.

## Roadmap

### Phase 0: make it runnable

Repair the broken PyTorch install and pin a matched set of `torch`, `transformers`,
`tokenizers`, `sentencepiece`, and `numpy` below version 2. Rewrite `requirements.txt` to
describe the architecture that exists: add the machine-learning dependencies, drop the seven
that nothing imports.

Fix the silence-detection regex in `src/stages/preprocessing/segmentation.py` to accept
negative numbers, and pair markers defensively instead of zipping two independent lists. Wrap
the body of `run()` in `src/stages/preprocess.py` so FFmpeg failures return a failed stage
result instead of escaping the contract.

Exit criterion: the translation backend loads and translates one English sentence to Hindi in
this virtual environment.

### Phase 1: wire the local half

Add `src/pipeline/paths.py` holding a small frozen dataclass that resolves every artifact path
for a job against an absolute artifacts root. This single change fixes the CWD-relative paths
that the manifest currently stores verbatim, and it lets tests point at a temporary directory
instead of writing into the repository.

Add `src/pipeline/runner.py` that walks a list of stages, threads each output path into the next
input path, records results into the `Job` model that already exists and is unused in
`src/orchestrator/models.py`, and persists job state after every stage so a crash is resumable.

Add `src/cli.py` with arguments for input, target language, job identifier, reference audio,
config, and stage range. With the runner, this is the file that converts a pile of components
into a program.

Wrap ASR, translation, and TTS export as `PipelineStage` implementations that delegate to the
existing processors without changing their signatures. Make `load()` idempotent on both
backends and call it from the processors, so the missing-load defect is fixed at the source
rather than at every future call site. Stages take a constructed backend as a constructor
argument so models load once per process, not once per job.

Add an awaiting-external status to `StageStatus` so the manual GPU hand-off is an honest state
rather than a crash or a silent stop.

Exit criterion: one command turns `test.mp4` into a bundle zip and halts cleanly at the GPU
boundary.

### Phase 2: close the GPU loop

Fix `colab/xtts_worker.py`: read segments off the Pydantic model rather than calling a
dictionary method, name output files by the globally unique segment identifier rather than the
shared chunk identifier, honor the requested sample rate instead of a hardcoded constant, and
implement the result writer.

Apply the project's own research. Greedy decoding and the repetition penalty belong on the
inference call; the conditioning-length parameters belong on the conditioning-latents call,
where the worker currently passes none. Passing them to inference silently does nothing. Remove
the temperature argument, which is inert under greedy decoding and contradicts the documented
finding.

Write the synthesis loop with per-segment error handling and incremental result writing, so one
bad segment or a Colab timeout does not cost the whole run. Record the exact inference
parameters in the result file. Given that this project's central lesson was a parameter
silently changing the output, that record belongs in the artifact.

Write `src/stages/tts/bundle/importer.py`: unzip, validate the bundle version against a set
rather than an equality check, rewrite remote paths to local ones, reject any path escaping the
bundle root, re-derive every duration locally rather than trusting the remote value, and
reconcile against the request so a missing segment surfaces as a reported hole rather than a
mysterious silence.

Derive the voice reference from the speaker's own audio by concatenating the longest, highest
confidence chunks to about fifteen seconds. The current five-second reference is exactly the
short-reference case that the project's own research notes sounds bad under greedy decoding.

Commit the Colab notebook so the manual step is four cells rather than a remembered ritual.

Exit criterion: synthesized audio and a result file come back and import cleanly with every
requested segment accounted for.

### Phase 3: the dub

Add timing models and `src/stages/assemble.py` implementing the three-tier cascade and the
NumPy track assembly described above. Put every threshold in the config file under a timing
block; these are the knobs that will actually get tuned. Write a timeline artifact recording per
segment what happened: how much slack was available, what tempo was applied, where it landed,
and how much drift accumulated. That artifact is the quality-control data this project is named
for.

Implement `src/stages/remux.py` as a stream copy of the video with the new audio encoded to AAC.
Pad or truncate the assembled track to the exact video duration rather than relying on FFmpeg's
shortest flag, so a short final segment cannot clip the video.

Exit criterion: a watchable dubbed video file exists. This is the milestone that matters.

### Phase 4: make it fast and honest

Replace the per-chunk ASR loop with the batched pipeline over the whole audio file, with voice
activity detection enabled from config, which it currently ignores. This one edit removes the
padding waste, the overlap double-transcription in the fixed-window fallback, and the
first-chunk language detection problem together. Timestamps come back already absolute, so the
offset arithmetic disappears.

Filter Whisper hallucinations in the same edit. Three fields on the transcript segment model
already carry the signal and nothing reads them. Without this you will synthesize a phantom
"thank you" over silence and debug it through three downstream stages.

Add batch translation, drop to greedy decoding, and make the model and beam count
config-driven so a smaller distilled model can be tried against the 1B.

Collapse the FFmpeg spawns: run silence detection in the same pass that writes the audio, reuse
the probe result that the validator currently discards, and emit all chunks in one invocation.

Reconcile the manifest writer with the typed chunk models it contradicts, give the tests
teardown against a temporary directory, and delete the premature scaffolding.

Then implement the evaluation harness over the metrics modules that already exist and the new
timing data. That is the quality-control half of the project title, and it is cheap once timing
data exists.

## Verification

Each phase has an exit criterion above; those are the real gates. End to end:

```bash
./venv/bin/python -m src.cli --input test.mp4 --target-lang hi --job-id demo
```

This should stop at the GPU boundary with a bundle zip. After running the Colab notebook and
returning the zip:

```bash
./venv/bin/python -m src.cli --job-id demo --from-stage tts_import
```

This should produce a dubbed video. Watch it. Check that speech starts where the original
speaker starts, that no segment is obviously rushed, and that the end has not drifted.

Regression coverage that does not need a GPU: the silence parser against a negative first
marker, the timing cascade against a synthetic synthesis result with known durations, and the
track assembler asserting a segment lands at an exact sample offset. The existing preprocessing
tests should be repointed at a temporary artifacts root so they stop writing into the repository.

## Out of scope

Replacing the manual Colab step with a hosted GPU service, which becomes attractive immediately
after the first end-to-end run and changes nothing else because the backend abstraction already
exists. Proper IndicTrans2 preprocessing through the official toolkit, which matters only for
target scripts other than Devanagari. A second TTS engine for the twelve Indic languages XTTS
cannot speak. Retaining background music through source separation. An HTTP API, a job
database, and distributed task execution, none of which should exist before the pipeline has
run once.

## Note on process

The repository convention is a `PROGRESS.md` entry per unit of work with a deviations line. The
deviations from these phases are worth writing plainly, because they are the honest record of
what "Verification: PASSED" meant for the steps that built components which had never been
executed together.
