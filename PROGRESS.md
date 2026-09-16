# Progress Log

## Step 2
- Phase/Day: Day 0
- Completed:
  - Initialized Git repository
  - Added .gitignore
  - Added README.md
  - Created initial commit
- Verification: PASSED
- Deviations:
  - PROGRESS.md was introduced in Step 3 instead of Step 2 to correct an oversight before implementation begins.


  ## Step 4
- Phase/Day: Day 0
- Completed:
  - Created Python 3.11 virtual environment
  - Activated virtual environment
  - Verified python executable
  - Verified pip executable
- Verification: PASSED
- Deviations:
  - None


## Step 5
- Phase/Day: Day 0
- Completed:
  - Created repository scaffold
  - Created package structure
  - Added placeholder modules
- Verification: PASSED
- Deviations:
  - None

## Step 7
- Phase/Day: Day 0
- Completed:
  - Added pipeline configuration
- Verification: PASSED
- Deviations:
  - TTS device intentionally set to CUDA because synthesis will execute in Colab during Phase 3.


## Step 8
- Phase/Day: Day 0
- Completed:
  - Implemented StageStatus
  - Implemented StageResult
  - Implemented Job
- Verification: PASSED
- Deviations:
  - Used Field(default_factory=dict) instead of mutable dictionary defaults while preserving the specification interface.


## Step 9
- Phase/Day: Day 0 / Phase 1 Foundation
- Completed:
  - Implemented abstract PipelineStage interface
- Verification: PASSED
- Deviations:
  - None

## Step 10
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented FFmpegPreprocessStage
  - Implemented lightweight validate_input()
- Verification: PASSED
- Deviations:
  - ffprobe-based validation intentionally deferred to the next atomic step.

## Step 11
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Added ffprobe-based media validation
  - Added audio stream verification
  - Added duration verification
- Verification: PASSED
- Deviations:
  - None

## Step 12
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented FFmpeg audio extraction
  - Implemented audio normalization
  - Implemented StageResult output
- Verification: PASSED
- Deviations:
  - Silence detection and manifest generation intentionally deferred to the next atomic step.

## Step 13
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented silence detection helper using FFmpeg silencedetect
- Verification: PASSED
- Deviations:
  - Integration into preprocessing pipeline deferred to the next atomic step.

## Step 14
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented speech segment boundary generation
- Verification: PASSED
- Deviations:
  - Audio chunk generation deferred to the next atomic step.

## Step 15
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented speech segment extraction
- Verification: PASSED
- Deviations:
  - Manifest generation deferred to the next atomic step.

## Step 16
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Implemented manifest generation helper
- Verification: PASSED
- Deviations:
  - Manifest integration into run() deferred to the next atomic step.

## Step 17
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Integrated preprocessing pipeline into run()
  - Generated chunk manifest as pipeline output
- Verification: PASSED
- Deviations:
  - Fixed-window fallback intentionally deferred to the next atomic step.

## Step 18
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Added minimum speech segment duration filtering
- Verification: PASSED
- Deviations:
  - Threshold temporarily implemented in code (0.25 s). It will be moved into configuration in a later configuration-refinement step.

## Step 19
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Added typed manifest models
- Verification: PASSED
- Deviations:
  - Existing JSON manifest intentionally unchanged. Models introduced first to preserve backward compatibility.

## Step 20
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Created preprocessing package
- Verification: PASSED
- Deviations:
  - No behavior changes. Refactoring infrastructure only.

## Step 21
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Extracted validation logic into preprocessing/validator.py
  - FFmpegPreprocessStage now delegates validation
- Verification: PASSED
- Deviations:
  - None

## Step 22
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Extracted audio normalization into preprocessing/audio.py
  - Extracted audio duration helper
- Verification: PASSED
- Deviations:
  - None

## Step 23
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Introduced AudioProcessor service
  - FFmpegPreprocessStage now delegates audio extraction to AudioProcessor
- Verification: PASSED
- Deviations:
  - None

## Step 24
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Moved preprocessing models into preprocessing package
- Verification: PASSED
- Deviations:
  - None

## Step 25
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Introduced Segmenter service
  - Moved silence detection, segment generation, and chunk extraction into Segmenter
  - FFmpegPreprocessStage now delegates segmentation
- Verification: PASSED
- Deviations:
  - None

## Step 26
- Phase/Day: Phase 1 / Refactoring
- Completed:
  - Introduced ManifestWriter service
  - FFmpegPreprocessStage delegates manifest generation
- Verification: PASSED
- Deviations:
  - None

## Step 27
- Phase/Day: Phase 1 / Testing
- Completed:
  - Added preprocessing unit test suite
- Verification: PASSED
- Deviations:
  - End-to-end integration tests intentionally deferred to the next step.

## Step 28
- Phase/Day: Phase 1 / Day 1
- Completed:
  - Added fixed-window segmentation fallback
- Verification: PASSED
- Deviations:
  - Chunk size temporarily hardcoded (30 s / 1 s overlap). Will become configurable in a later configuration refinement.

## Step 29
- Phase/Day: Phase 1 / Integration Testing
- Completed:
  - Added preprocessing integration tests
- Verification: PASSED
- Deviations:
  - MP4 integration test uses existing preprocessing pipeline; dedicated no-audio video fixture will be added later.

## Step 30
- Phase/Day: Phase 1 / Integration Testing
- Completed:
  - Added concurrent preprocessing integration test
- Verification: PASSED
- Deviations:
  - Uses ThreadPoolExecutor because Celery orchestration has not yet been introduced.

## Step 31
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Created ASR package scaffold
- Verification: PASSED
- Deviations:
  - No implementation yet. Package scaffold only.

## Step 32
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added typed ASR transcript models
- Verification: PASSED
- Deviations:
  - Models defined before Faster-Whisper integration to establish a stable contract.

## Step 33
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added ASRProcessor abstract interface
- Verification: PASSED
- Deviations:
  - Faster-Whisper implementation intentionally deferred until interface is established.

## Step 34
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added inference backend interface
- Verification: PASSED
- Deviations:
  - Faster-Whisper implementation deferred until backend abstraction is established.

## Step 35
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added FasterWhisperBackend implementation
- Verification: PASSED
- Deviations:
  - Backend only loads the model and exposes transcribe(). Integration into the processor is deferred.

## Step 36
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Implemented ASRProcessor using FasterWhisperBackend
- Verification: PASSED
- Deviations:
  - Uses first transcription result for language detection.

## Step 37
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added configurable ASR language hint
- Verification: PASSED
- Deviations:
  - Default language set to "en" for deterministic MVP behavior.

## Step 38
- Phase/Day: Phase 2 / Day 2
- Completed:
  - Added ASR evaluation metrics (WER, CER, confidence, average segment duration)
- Verification: PASSED
- Deviations:
  - Uses manually supplied reference transcript for MVP evaluation.

## Step 39
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Created Translation package scaffold
- Verification: PASSED
- Deviations:
  - No translation implementation yet.

## Step 40
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added typed translation models
- Verification: PASSED
- Deviations:
  - Models defined before IndicTrans2 integration to establish a stable contract.

## Step 41
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added TranslationBackend abstract interface
- Verification: PASSED
- Deviations:
  - IndicTrans2 implementation deferred until backend contract is established.

## Step 42
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added IndicTrans2Backend implementation
- Verification: PASSED
- Deviations:
  - Backend encapsulates model-specific language tags.

## Step 43
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented TranslationProcessor
  - Connected TranscriptResult to TranslationBackend
- Verification: PASSED
- Deviations:
  - Sequential translation for MVP. Batching will be added later if needed.

## Step 44
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added translation quality metrics
- Verification: PASSED
- Deviations:
  - Uses structural metrics only; reference-based metrics (BLEU/COMET) are deferred.

## Step 45
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added TTS request and response models
- Verification: PASSED
- Deviations:
  - XTTS execution deferred to Google Colab per project constraints.

## Step 46
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented TTSProcessor
  - Added synthesis request JSON export
- Verification: PASSED
- Deviations:
  - XTTS execution intentionally deferred to Colab.

## Step 47
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added TTS backend abstraction
  - Added ExternalExecutionBackend interface
- Verification: PASSED
- Deviations:
  - Google Colab treated as an implementation, not part of the architecture.

## Step 48
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented ColabTTSBackend
  - Added request export and result import
- Verification: PASSED
- Deviations:
  - synthesize() intentionally raises NotImplementedError until external GPU execution completes.

## Step 49
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Created bundle exchange scaffold
  - Introduced artifact-based GPU boundary
- Verification: PASSED
- Deviations:
  - Replaced Colab-specific workflow with transport-agnostic bundle architecture.

## Step 50
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added bundle manifest models
  - Introduced versioned bundle contract
- Verification: PASSED
- Deviations:
  - None

## Step 51
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented BundleExporter
  - Bundle directory generation
  - Manifest generation
- Verification: PASSED
- Deviations:
  - ZIP packaging deferred to a later transport utility.

## Step 52
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Created XTTSWorker lifecycle skeleton
  - Defined worker execution flow
- Verification: PASSED
- Deviations:
  - XTTS inference intentionally deferred to next step.

## Step 53
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added GPU preflight validator
  - Added runtime diagnostics before XTTS initialization
- Verification: PASSED
- Deviations:
  - None

## Step 54
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented XTTS model initialization
  - Integrated GPU preflight validation
- Verification: PASSED
- Deviations:
  - Speaker embedding intentionally deferred.

## Step 55
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added bundle ZIP packaging
  - Standardized portable GPU artifact
- Verification: PASSED
- Deviations:
  - Bundle signing/checksums deferred to a future version.

## Step 56
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Added deterministic XTTS model caching
  - Prevented redundant GPU loading
- Verification: PASSED
- Deviations:
  - None

## Step 57
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented XTTS speaker embedding computation
  - Added caching for GPT conditioning latent and speaker embedding
- Verification: PASSED
- Deviations:
  - None

## Step 58
- Phase/Day: Phase 3 / Day 3
- Completed:
  - Implemented single segment XTTS synthesis
  - Configured output wav saving via torchaudio
- Verification: PASSED
- Deviations:
  - None
## Step 59
- Phase/Day: Phase 4 / Evaluation
- Completed:
  - Fixed XTTSWorker: greedy decoding and repetition penalty on inference(), conditioning
    lengths on get_conditioning_latents() where they take effect, temperature removed
  - Fixed output filenames to use segment_id instead of the shared chunk_id
  - Implemented write_result() and a full run() loop with per-segment error isolation
  - Added speaker-similarity and GPT-token capture to the synthesis result
  - Added per-stage eval metrics: preprocess_metrics, tts_metrics, translation feasibility,
    reference-free ASR signals
  - Implemented eval/harness.py as a QC report aggregator with a CLI
  - Added colab/run_bundle.ipynb to replace the exploratory notebook
  - Added 12 eval regression tests (19 total, all passing)
- Verification: PASSED (pytest tests/ -q -> 19 passed; harness run against
  artifacts/test_mp4_job reproduces the measured pace figures)
- Deviations:
  - save_segment() was folded into synthesize_segment() rather than implemented separately,
    since the duration and sample count come free from the inference output.
  - Translation feasibility uses heuristic natural speaking rates per language, not measured
    ones. They are defined in one table in translation_metrics.py and should be replaced with
    figures measured from real speech when available.
  - Round-trip intelligibility is implemented but not wired into the harness, because it needs
    a loaded ASR backend and would make the report depend on model download.
  - The bundle importer and the local synthesis-result path remain unwritten, so the harness
    reads the bundle directory directly for now.
## Step 60
- Phase/Day: Phase 5 / Environment repair
- Completed:
  - Repaired the local torch install, which was missing libtorch_cpu.dylib and had made the
    translation backend unrunnable on this machine
  - Established the working dependency set and pinned it: torch 2.14.0, transformers 4.46.3
  - Rewrote requirements.txt with versions and the reasoning behind the non-obvious pins,
    and removed fastapi/uvicorn/celery/redis/sqlalchemy/onnxruntime, which nothing imported
  - Fixed the silence-detection parser: one regex pass pairing markers in order, accepting
    the negative silence_start FFmpeg emits when audio opens in silence
- Verification: PASSED (IndicTrans2 loads and translates "Hello everyone, welcome to the
  show." to Hindi locally; pytest tests/ -q -> 22 passed)
- Deviations:
  - transformers is pinned below 4.47 rather than kept current. IndicTrans2's remote code
    indexes past_key_values as legacy tuples, and newer transformers always passes generate()
    a Cache object, so decoding raises AttributeError. Transformers 5.x additionally removed
    transformers.onnx, which the same remote code imports at module scope. Both were
    reproduced before pinning.

## Step 61
- Phase/Day: Phase 5 / Measurement
- Completed:
  - Added src/data/ with a FLEURS loader and a measurement CLI. FLEURS is n-way parallel, so
    one sentence id yields both the English and the Hindi recording of the same content
  - Loader never decodes audio: duration comes from the num_samples field, which avoids the
    torchcodec dependency entirely
  - Added src/eval/duration_model.py: ridge regression from text features to spoken seconds,
    scored against the constant-rate baseline on a deterministic held-out split
  - Feature set separates base characters from combining marks, because Devanagari vowel
    signs attach to a preceding consonant rather than adding a syllable
- Verification: PASSED (measured en 13.36 cps and hi 10.88 cps against the guessed 15.0 and
  13.0; pytest -> 19 new tests covering the arithmetic and the fallback behaviour)
- Deviations:
  - A fitted model that fails to beat the constant baseline on held-out data is never used;
    predict() falls back to the constant. This is deliberate: the comparison is the point of
    the module, and a model that loses should not be shipped just because it exists.
  - FLEURS clips carry recording silence, so the measured characters-per-second understates
    articulation rate. That overhead is what the model's intercept captures, and both numbers
    are reported rather than one being silently corrected.

## Step 62
- Phase/Day: Phase 5 / Length-controlled translation
- Completed:
  - Added translate_candidates() to the translation backend contract, with diverse beam
    search in the IndicTrans2 backend so candidates differ meaningfully rather than being
    near-duplicates
  - Added src/stages/translation/length_control.py: scores each candidate on predicted
    duration against its budget and on semantic fidelity, then selects under a stated policy
  - Added src/stages/translation/fidelity.py, LaBSE cosine similarity between source and
    candidate
  - Added src/stages/translation/language_check.py after measuring that LaBSE cannot do the
    job alone
- Verification: PASSED (diverse beam search found a faithful Hindi translation 33% shorter
  than the beam-search favourite: 46 chars to 31; pytest -> 21 new tests)
- Deviations:
  - A third selection axis was added that the plan did not anticipate. Diverse beam search
    returned Maithili for a Hindi request, and LaBSE scored that candidate 0.888 against the
    English source while correct Hindi candidates scored ~0.86 — it ranked the wrong-language
    output best. LaBSE is trained to be language-agnostic, so this is it working as designed,
    not a bug. Fidelity and target-language correctness are therefore orthogonal checks.
    Off-language candidates are now disqualified outright rather than scored down.
  - The language check is a closed-class function-word heuristic, not a language identifier.
    It is scoped to Devanagari, the only script in our target set where several candidate
    languages compete. A production system should use a real LID model such as NLLB's, which
    labels Hindi, Maithili and Bhojpuri separately.

## Step 63
- Phase/Day: Phase 5 / Closing the loop
- Completed:
  - Implemented src/stages/tts/bundle/importer.py: version check, path-escape rejection for
    both zip members and result paths, durations re-derived from the audio rather than
    trusted, and partial runs reported rather than raised
  - Wired ColabTTSBackend.import_result() to it
  - Implemented src/stages/assemble.py: a four-tier fitting cascade (trim, absorb the
    following pause, time-stretch within the 1.25x ceiling, bounded drift returned at the
    next real pause), assembling into one NumPy buffer
  - Implemented src/stages/remux.py with video stream copy and the output length pinned to
    the video's own duration, so audio cannot drift against picture
  - Added src/pipeline/paths.py, src/pipeline/runner.py and src/cli.py: one command from
    video to dubbed video, resumable from any stage, stopping cleanly at the GPU boundary
- Verification: PASSED (pytest tests/ -q -> 99 passed, including an end-to-end test from
  translation to a dubbed mp4 whose audio and video durations agree within 0.1s)
- Deviations:
  - Time-stretching shells out to FFmpeg's atempo once per stretched segment. That is a
    process spawn per segment, which the bottleneck analysis flagged as costly. It is behind
    one function and can be replaced with an in-process implementation without touching the
    cascade.
  - ASR and translation are driven by the runner directly rather than through PipelineStage
    wrappers. The empty service.py files are still empty. The runner needs richer per-stage
    inputs than run(input_path, job_id, cfg) provides, so wrapping them would have meant
    changing the stage contract, which is a larger decision than this step should make.

## Step 64
- Phase/Day: Phase 5 / Measurement results
- Completed:
  - Measured speaking rates from the full FLEURS validation split: English 13.13 cps over 394
    sentences, Hindi 10.81 cps over 239. The code had guessed 15.0 and 13.0
  - Measured English-to-Hindi duration expansion at 1.191x median over 332 n-way-parallel
    pairs, interquartile range 0.95x to 1.49x
  - Replaced the guessed entries in NATURAL_CPS with the measured ones, labelling every
    remaining entry ESTIMATED so a reader can tell which verdicts rest on evidence
  - Added disk caching to the FLEURS loader after a language download timed out mid-run
  - Fitted the duration model: it beat the constant-rate baseline by 1.5% on English and
    3.8% on Hindi in held-out mean absolute error
- Verification: PASSED (artifacts/measurements/speaking_rates.json and duration_model.json;
  pytest tests/ -q -> 99 passed after the constants changed)
- Deviations:
  - Only English and Hindi are measured. Tamil, Bengali and Marathi were attempted in the
    same run and Marathi's download stalled on an HF CDN timeout, taking the run down with
    it. The measurement CLI now survives one language failing, and the loader caches, so the
    remaining languages are a re-run rather than a rebuild.
  - Both guesses were too high, which means the feasibility check had been flattering itself.
    Every verdict in the committed baseline report was optimistic.

## Step 65
- Phase/Day: Phase 5 / Controlled comparison
- Completed:
  - Ran the pipeline end to end on test.mp4 twice from the same transcript, once with length
    control off and once with six candidates per segment
  - Length control removed 13.5% of translated characters (408 to 353), cutting aggregate
    expansion from 1.17x to 1.01x and mean required pace from 1.50x to 1.35x
  - Three segments moved out of the "impossible" band; mean fidelity cost was 0.016
  - 4 of 34 generated candidates were off-language and rejected by the language check
- Verification: PASSED (harness run against artifacts/baseline_demo and artifacts/lc_demo,
  same ASR output, same segmentation, only the translation policy differing)
- Deviations:
  - The fitted duration model changed nothing. All 7 segments selected the same candidate
    with it as with the measured constant, because candidate ranking depends only on relative
    order and both predictors order candidates of one sentence identically. The model is kept
    because it is better calibrated in absolute terms, which matters for feasibility
    thresholds, but it is not currently earning its place in selection and that should be
    said plainly rather than implied by its presence.
  - Its fitted intercept had to be excluded from prediction. Trained on FLEURS studio takes
    of 40 to 270 characters, it learned 1.54s of recording silence that does not exist in
    synthesized speech on a timeline, and with it eight characters of Hindi predicted 2.61
    seconds. The held-out score said "model wins" while the model was wrong for the task: a
    training-versus-deployment distribution mismatch, not a bad fit.
  - Length control alone does not make the dub fit. Four segments remain infeasible and two
    impossible, so the assembly cascade still has to finish the job. That was expected: the
    measured expansion is in articulation rate, not text length, and no amount of shortening
    reaches the part of the gap that comes from Hindi simply being slower per character.

## Step 66 — Phase 1 (evaluation): a controlled recording session, and the fixtures from it

- Completed: a two-take recording session replaces the single contaminated reference the
  project had been measuring against, and the derived audio is committed so a fresh Colab
  runtime reaches it with `git pull` instead of a 200 MB upload.
  - `scripts/build_fixtures.py` cuts three files from each take: the conditioning reference
    (25s, through the same Segmenter and `build_reference` path the pipeline uses at export),
    every speech span joined, and the room tone at its recorded gain.
  - `colab/four_arm.py` runs en->en, hi->hi, en->hi and hi->en from the same speaker, and
    calibrates the similarity scale before using any of it.
  - `colab/voice_experiment.ipynb` drives the whole thing from a clean runtime.
- Verification: PASSED. 110 tests still green. The new takes measure 30.7 dB and 31.4 dB SNR
  against the old reference's 17.1 dB — a 21 dB drop in noise floor. ASR word error rate
  against the scripted text is 7.9% after number normalization, and three of the remaining
  nine errors are filler words the speaker actually said, so true ASR error is near 5%.
- Deviations:
  - The similarity metric was uncalibrated for this project's entire history. Measuring it
    properly puts the ceiling at 0.922 (the speaker against himself) and the floor at 0.124
    (58 different real speakers), a span of about 0.80. The seven-configuration conditioning
    sweep in step 64 spanned 0.046, which is 5.8% of the measurement range. That sweep
    reported a winner it had no power to detect, and conditioning is therefore held at the
    shipped XTTS-v2 config values here rather than at the value it picked.
  - English cloning scored *below* Hindi (0.466 against 0.511) on the old reference. That is
    the opposite of the prediction that cross-lingual transfer was the bottleneck, and it is
    why the four-arm design exists: no single-arm experiment could have caught it.
  - The cross-language ceiling is still unmeasured. Speaker embeddings shift between
    languages even for a real person, so `en -> hi` synthesis has never been scored against a
    ceiling it could actually reach. `colab/four_arm.py` measures it from the real Hindi take
    before scoring anything synthetic.
  - `build_fixtures.py` filters speech spans by level rather than trusting silence detection
    alone. At -45 dB the recorded room tone fragments into transients, and the gaps between
    them survive as speech spans made entirely of noise. The pipeline's own -30 dB threshold
    does not hit this, so no pipeline behaviour was changed.
  - Room tone is written un-normalized on purpose, since gain applied to it would destroy the
    noise-floor measurement it exists to carry. The gain applied to the other two files is
    recorded in `fixtures/metadata.json` instead.

## Step 67 — Phase 4 (voice diagnosis) — repair the notebook's cell encoding

- Completed: `colab/voice_experiment.ipynb` was written with each cell's `source` array
  holding bare lines and no line terminators. nbformat treats that array as fragments to
  concatenate, so Colab rendered every multi-line cell as a single line: the setup cell
  became `%cd /content!git clone ...`, which fails with
  `[Errno 2] No such file or directory: '/content!git clone ...'`. Rebuilt every cell with
  `splitlines(keepends=True)`, and bumped `nbformat_minor` from 0 to 4.
- Verification: PASSED. All 21 cells re-checked; no source entry before the last now lacks a
  trailing newline. The setup cell reads back as six separate lines.
- Deviations:
  - The restart note pointed at cell numbers ("continue from cell 4", "do not re-run cell 2")
    that did not match the cells it meant, and would not survive the notebook being rebuilt
    by hand in a fresh Colab. It names sections instead now.

## Step 68 — Phase 4 (voice diagnosis) — the four-arm result

- Completed: ran `colab/voice_experiment.ipynb` on a T4 against the new recording session.
  Calibration on the clean takes: floor 0.095 (58 real strangers), same-language ceiling 0.968,
  and — measured here for the first time — a cross-language ceiling of 0.896, the speaker's own
  Hindi scored against his own English anchor with no synthesis involved.

  | arm | greedy | sampled | position |
  |---|---|---|---|
  | `en -> en` | 0.502 | 0.486 | 47% |
  | `hi -> hi` | 0.719 | 0.689 | 72% |
  | `en -> hi` | 0.713 | 0.718 | **78%** (cross-language scale) |
  | `hi -> en` | 0.553 | 0.532 | 53% |

- Verification: PASSED. Placed on the same same-language scale as step 66, the production path
  `en -> hi` moved from 48% to 71% purely by re-recording the reference, while `en -> en` moved
  from 42% to 47%. The arm-to-arm gap of 0.22 is three to ten times the within-arm spread
  (0.021 to 0.077 across three sentences), so it is a real effect and not the kind of
  sub-noise difference the step-64 sweep mistook for a result.
- Deviations:
  - The cross-lingual hypothesis is dead. Holding the spoken language fixed and swapping the
    reference language changes similarity by -0.011 (speaking Hindi) and +0.048 (speaking
    English) — both inside the within-arm spread, and the second one favours the *foreign*
    reference. An English reference clones into Hindi as well as a Hindi reference does.
    Voice conversion was the planned fix for a cross-lingual gap; there is no gap to fix, so
    that work is dropped rather than deferred.
  - What predicts the score is the language being *spoken*, not the language being cloned
    from. Speaking English scores ~0.52 from either reference; speaking Hindi scores ~0.71
    from either. XTTS-v2's English decoder overwrites speaker identity, which is the same
    mechanism as the American accent reported in step 66, measured a second way.
  - Synthesis runs 1.20x to 1.23x faster than the FLEURS rates the duration model is fitted
    on, consistently across every arm. The speaker himself is only 1.07x to 1.13x fast, so
    this is the synthesizer, not the reference. `src/eval/duration_model.py` therefore
    predicts slot durations about 20% longer than XTTS actually delivers, which biases
    length control toward translations shorter than they need to be. Not yet fixed; the
    model should be recalibrated against synthesized audio, since XTTS is what fills the slot.
  - Greedy decoding beat sampling on three of four arms and tied on the fourth, agreeing with
    the determinism finding in `colab/xtts.md` for an unrelated reason.

## Step 69 — Phase 4 (model choice) — reference transcripts and the IndicF5 comparison

- Completed:
  - `scripts/transcribe_fixtures.py` writes `fixtures/reference_text.json`, an ASR transcript
    of each 25-second reference clip. IndicF5 conditions on reference audio together with
    what was said in it, so a comparison against XTTS needs the transcript of exactly the
    audio each model is handed — not the scripted text, since the reference is spans cut from
    the middle of the take and the speaker did not read the script word for word.
  - `colab/indicf5_check.py` runs XTTS-v2 and IndicF5 over the same two references and the
    same seven Hindi sentences, scored by the same speaker encoder against the same anchors
    and the same floor as the four-arm run.
  - `colab/voice_experiment.ipynb` gains sections 9 to 11, and IndicF5's install moves into
    the existing setup cell so one runtime restart covers both stacks.
- Verification: PASSED. `scripts.transcribe_fixtures` run twice produces byte-identical
  output. Both transcripts end cleanly at the clip boundary with no hallucinated tail.
- Deviations:
  - The transcription does not go through `FasterWhisperBackend`. Whisper's default decoding
    retries at rising temperatures when a decode trips the compression-ratio guard, and these
    clips trip it because they end mid-sentence. Those retries sample, so two runs of the
    first version of this script disagreed about the Hindi tail — one closed on
    `झाल झाल झाल झाल` and the other on `अजय को` four times. A committed fixture that changes
    under its own re-run is not a fixture, so this script pins `temperature=0.0` and
    `condition_on_previous_text=False` rather than changing the pipeline's own ASR settings.
  - `strip_repetition_tail` survives as a backstop even though greedy decoding stopped the
    loops at source and it no longer fires on either clip. It handles repeated phrases up to
    four words, not just repeated single tokens, because the two observed hallucinations were
    a repeated word and a repeated pair.
  - Seven sentences per arm rather than the four-arm run's three, and the standard error is
    reported. The within-arm spread there was 0.02 to 0.08, and a model difference worth
    acting on could be 0.05, so three samples could not have resolved one.
  - The XTTS arm is re-run inside this script rather than compared against the recorded
    four-arm numbers, so both models are measured in one process against one calibration.

## Step 70 — Phase 4 (model choice) — the numpy downgrade that broke XTTS silently

- Completed: pinned `numpy>=2.1,<3` in `colab/requirements.txt` and reordered the notebook's
  setup cell to install IndicF5 first and this repo's pins second, so our constraints are the
  ones that survive the resolver. Added `pip check` to the setup cell.
- Verification: PASSED by diagnosis rather than by test. Installing IndicF5 alongside the XTTS
  stack made `import TTS` fail with `cannot import name 'GPT2PreTrainedModel' from
  'transformers'`, while `transformers.__version__` reported 4.57.6 — inside the existing pin.
  Probing the namespace showed `GPT2Config` present but `GPT2PreTrainedModel`, `GPT2LMHeadModel`
  and `BertModel` absent, and importing the module directly raised the real error:
  `module 'numpy.dtypes' has no attribute 'StringDType'`. That attribute arrived in numpy 2.0;
  IndicF5 had pulled numpy back to 1.x.
- Deviations:
  - The failure mode is worth recording on its own. transformers' lazy loader catches the
    `AttributeError` raised while building its torch-backed classes and drops those names from
    the namespace instead of propagating it. The result is that `import transformers` succeeds,
    the version string looks correct, and the only symptom is a missing name reported three
    layers downstream inside a different package. A version pin cannot catch this, because the
    version was never wrong — the dependency underneath it was.
  - Two hypotheses were wrong before the diagnostic ran: that transformers had dropped the
    top-level re-export in a patch release, and that torch registration had failed wholesale.
    `is_torch_available()` returning True while model classes were missing ruled out both.
  - Whether f5-tts tolerates numpy 2 is unverified from here. If it declares `numpy<2` the
    resolver cannot satisfy both, and the fallback is to run the two models in separate
    kernels: `/content` survives a restart, so XTTS can synthesize and save its anchors,
    IndicF5 can synthesize into the same directory after a restart, and a third pass on the
    XTTS stack can score everything together.

## Step 71 — Phase 4 (model choice) — narrow the numpy pin to the only window that exists

- Completed: changed the Colab numpy pin from `>=2.1,<3` to `>=2.1,<2.3`, and trimmed the
  setup cell's `pip check` to the packages that gate the run.
- Verification: PASSED by diagnosis. The wider pin resolved to numpy 2.5.3, which fixed
  transformers but broke the next link in the chain: `import TTS` failed with
  `Numba needs NumPy 2.2 or less. Got NumPy 2.5`. Numba arrives through librosa and enforces
  its ceiling at import rather than at install.
- Deviations:
  - Three packages constrain numpy in opposite directions: transformers 4.57 needs >= 2.0,
    numba needs < 2.3, and f5-tts declares <= 1.26.4. The first two leave exactly
    2.1 <= numpy < 2.3; f5-tts cannot be satisfied alongside them at all. Its declaration is
    deliberately overridden rather than honoured, so pip prints an f5-tts conflict on every
    install and that line is expected output. Whether the pin reflects a real incompatibility
    or an inherited ceiling is now an empirical question, and running IndicF5 on numpy 2.2 is
    what answers it.
  - Colab's own `pip check` output runs to roughly forty lines about preinstalled packages
    that have nothing to do with this pipeline, which buried the one line that mattered on the
    first attempt. The setup cell now greps it down to numpy, numba, transformers and tts.
  - If IndicF5 does fail on numpy 2.2, the fallback is separate kernels rather than a
    resolvable environment: `/content` survives a restart, so each model can synthesize in an
    environment built for it and a final pass on the XTTS stack can score both sets together.

## Step 72 — Phase 4 (model choice) — meta tensors, and a ceiling per clip length

- Completed:
  - `load_indicf5()` passes `low_cpu_mem_usage=False` and then asserts that no parameter or
    buffer is left on the meta device.
  - Replaced the single-point calibration in `colab/indicf5_check.py` with a ceiling measured
    at four clip lengths, and placed every clip against the point nearest its own duration.
    Added a LENGTH section reporting each model above and below 8 seconds, and a guard that
    flags clips peaking below 0.01.
- Verification: PASSED. The placement logic was exercised against a synthetic falling ceiling:
  nearest-bucket selection picks correctly at 4.0s, 6.0s, 13.6s and 60s, and a clip scoring
  0.799 at 13.6s places at 84% while one scoring 0.557 at 4.87s places at 60%.
- Deviations:
  - IndicF5 loaded every weight into a meta tensor. transformers now builds models on the meta
    device and fills them afterwards; IndicF5's remote code constructs its Vocos vocoder inside
    its own `__init__`, which inherits the meta context but not the fill, so each parameter
    copied as a no-op and torch said so eighty times. The run only failed because the code
    called `.to("cuda")` afterwards. Without that call the model would have loaded cleanly and
    synthesized noise, and noise scored against a speaker anchor is indistinguishable from a
    model that clones badly — it would have been written down as a result. The explicit meta
    check exists so that failure is loud rather than plausible.
  - Speaker similarity is strongly length-dependent. Across the 14 XTTS clips of this run,
    clip duration correlated with similarity at r = +0.82; clips of 8 seconds or more averaged
    0.740 and shorter ones 0.627. The four-arm run used the three longest sentences and scored
    them against a ceiling built from 17-second thirds, so its 78% was measured at a length the
    production pipeline never sees. On all seven sentences the same arm is 70%, and on the
    short ones alone 61%.
  - This is the same class of error as scoring cross-lingual synthesis against a same-language
    ceiling, which step 66 already corrected once: a ceiling measured under conditions the
    thing being judged does not share. The fix is the same — measure the ceiling under matched
    conditions rather than adjusting the score.
  - The 14 segments the local pipeline exported from `english.mov` average under 5 seconds, so
    production sits in the bucket where both the metric and, on this evidence, the model are
    weakest. Whether IndicF5 degrades the same way is the question the next run answers.

## Step 73 — Phase 4 (model choice) — load IndicF5 outside transformers' meta context

- Completed: `colab/indicf5_check.py` now builds IndicF5's remote class directly — read
  `auto_map["AutoModel"]` from the config, fetch the class with
  `get_class_from_dynamic_module`, instantiate it against the config, and load
  `model.safetensors` on top — with `AutoModel.from_pretrained` kept as a fallback. Both
  routes are checked for meta tensors before and after the device move.
- Verification: PARTIAL, and honestly so. The duration-matched calibration ran and behaved as
  designed: the cross-language ceiling falls from 0.887 at 16.7s to 0.779 at 3.6s, and XTTS's
  short-segment penalty shrank from 16 points to 9 once each clip was placed against a ceiling
  measured at its own length. The IndicF5 loader itself is unverified — the repository is
  gated, so it cannot be exercised from this machine, and the next Colab run is the test.
- Deviations:
  - `low_cpu_mem_usage=False` did not fix the meta-tensor failure, because the exception is
    raised from inside the remote `__init__` rather than from the weight loading that flag
    governs. transformers runs that `__init__` under an empty-weights context; IndicF5 builds
    its Vocos vocoder there and calls `.to(device)` on it, which cannot copy out of meta. No
    argument to `from_pretrained` reaches that, so the constructor has to be called directly.
  - The checkpoint-match check reports missing keys but only refuses on unexpected keys or on
    more than half the parameters missing. Vocos is downloaded separately by the remote
    `__init__` and lives under an attribute name that cannot be inspected from here, so its
    keys are legitimately absent; failing on any missing key would have rejected a correct
    load. Refusing on a name mismatch still catches the case that matters, which is a model
    left at random initialization that runs fine and sounds wrong.
  - Short-clip similarity was partly a measurement artifact and partly real. Against a single
    long-clip ceiling the gap looked like 0.627 against 0.740; against length-matched ceilings
    it is 71% against 80%. The metric was exaggerating the penalty, but a penalty remains.

## Step 74 — Phase 4 (model choice) — IndicF5 wins on identity, and its duration model is broken

- Completed: ran the comparison on a T4. IndicF5 loaded via the direct route and beat XTTS-v2
  decisively on the production case.

  | model | arm | sim | position | cps vs natural |
  |---|---|---|---|---|
  | xtts | en_ref | 0.656 | 77% | 1.06x |
  | xtts | hi_ref | 0.695 | 74% | 1.04x |
  | indicf5 | en_ref | 0.800 | **97%** | 0.87x |
  | indicf5 | hi_ref | 0.713 | 81% | **2.07x** |

  Added a 10-second reference fixture and its transcript (`*_reference_short.wav`,
  `english_short` / `hindi_short` in `reference_text.json`), split the arm lists per model, and
  added a PACE section that recovers how much reference audio IndicF5 actually used.
- Verification: PASSED for identity. The production gap is +0.144 with a combined standard
  error of 0.031, which is 4.6 standard errors on n=7 per arm. IndicF5 also held up where it
  mattered most: on clips under 8 seconds it placed at 86% against XTTS's 71%, so the
  short-segment weakness that worried us is XTTS's, not the metric's alone. 110 tests green.
- Deviations:
  - IndicF5's `hi_ref` arm returned every sentence at 0.48x the duration natural Hindi needs,
    with a standard deviation of 0.02 across seven sentences. That consistency identified the
    mechanism: the model sets generated length from the UTF-8 byte ratio of generated to
    reference transcript, scaled by the reference audio's duration. Inverting it recovers the
    reference length it actually used — 13.7s (sd 0.1) for English and 12.4s (sd 0.0) for
    Hindi, against 25s clips. It clips the audio but keeps the whole transcript.
  - The deeper fault is that the byte ratio assumes one script. Devanagari is three bytes per
    character and Latin one, so an English reference implies 0.075 s/byte where generating
    Hindi needs 0.035 — an overstatement of about 2.15x. The `en_ref` arm's apparently healthy
    0.87x was luck: clipping 25s to 13.7s divided by 1.83 and very nearly cancelled the script
    inflation. Two errors of opposite sign, not a working duration model.
  - This matters more than it would for most projects, because output duration is the problem
    this pipeline exists around. A model that cannot be asked for a length cannot be dropped
    into the length-control stage unchanged.
  - The next run is a prediction rather than an observation: with unclipped 10s references the
    byte ratio predicts 2.17x natural for `en_ref_10s` and 1.08x for `hi_ref_10s`. Those
    numbers are printed beside the observed ones. If they land, the duration behaviour is
    fully characterized; if the implied reference length comes back near 10.5s, clipping is
    ruled out as well.
  - The 25s English arm is retained as a control so the bug and its correction appear in the
    same report rather than being asserted from a previous one.
  - A Hindi reference is the configuration whose pacing is correct by construction, which
    bears on the fixed-roster question raised in step 66: recording each speaker once in Hindi
    would make the production path `hi -> hi`, where the duration arithmetic is self-consistent.
    Unverified until the next run.

## Step 75 — Phase 4 (model choice) — check what the model actually said

- Completed: added an intelligibility check to `colab/indicf5_check.py`. Every generated clip
  is transcribed back with Whisper and aligned against the sentence it was given, reporting
  character error rate together with the insertion and deletion rates it decomposes into. A
  CONTENT section reports each arm and prints the worst inserted speech verbatim; the verdict
  now states whether the winning arm is clean before discussing identity at all; `listen()`
  flags clips inline and prints what the transcriber heard.
- Verification: PASSED. The alignment was exercised against seven constructed cases. A matra
  difference scores 0.018 and passes; gibberish appended, prepended and inserted mid-sentence
  are all caught by the insertion rate at 0.418, 0.309 and 0.182; a sentence truncated to a
  quarter is caught by the deletion rate at 0.564 and is not confused with the gibberish
  cases. 110 tests green.
- Deviations:
  - The listener reported audible gibberish between the intended words on every `en_ref_25s`
    clip — the arm that placed at 97% of scale, the best of any arm in the run. Speaker
    similarity reads timbre, so nonsense in the right voice outscores clean speech in a
    slightly wrong one. No identity metric can catch this, and the project had no content
    check at all, so the highest number in the report was the least usable audio.
  - The likely cause is specific and testable. F5-TTS generates `[ref_text + gen_text]` as one
    continuation of the reference audio and strips the reference by length. On `en_ref_25s`
    the model logged `Audio is over 15s, clipping short` and used 13.7s of audio while still
    receiving the whole 25s transcript, so roughly eleven seconds of reference text had no
    matching audio and got spoken. `hi_ref_10s`, where transcript and audio agree, was clean
    by ear. The CONTENT section measures this rather than leaving it as an inference.
  - A single error rate cannot do this job. Gibberish appended to an otherwise correct
    sentence scores 0.31, which sits below any threshold loose enough to tolerate Whisper's
    own Hindi error, so the first version of this check would have passed the broken arm.
    Insertions and deletions are judged separately, which also separates the two failures:
    extra speech the model invented against speech it never finished.
  - Character rather than word error rate. Hindi word boundaries move under ASR — compounds
    split, matras attach differently — so a word rate reports differences a listener would not
    call errors.
  - Whisper is loaded through transformers rather than faster-whisper. faster-whisper needs
    ctranslate2, and this environment already required pinning numpy into a one-minor-version
    window to keep transformers, numba and f5-tts from breaking each other. This adds no new
    dependency.
