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
