# Indic Dubbing Pipeline

A video in one language goes in. The same video comes out dubbed into an Indic
language — **still in the original speaker's voice**, and still matching the
original timing.

```
  myvideo.mp4  (English speaker)   ->   artifacts/demo/dubbed.mp4  (same person, Hindi)
```

Nothing is re-recorded and no voice actor is involved. The speaker's own voice
is cloned from a few seconds of the source audio.

---

## The part that is actually hard

It is tempting to think this is translation plus text-to-speech. It is not, and
the reason is a measurement:

> **Hindi takes about 1.19× as long to say as the English it replaces.**
> Measured over 332 sentence-aligned pairs. Meanwhile the *character* count
> ratio is 0.989 — essentially identical.

So the expansion is not in the text. It is in **articulation rate**: Hindi uses
the same number of characters and needs a fifth more time to speak them. That
means you cannot fix it by shortening the translation alone, and you cannot fix
it by speeding up the audio alone — past about 1.25× speech stops sounding like
speech.

Most of this repository is about closing that gap: choosing translations that
fit the time available, and absorbing whatever is left over during assembly.

---

## How you run it

Three commands. Two on your laptop, one notebook in between, because synthesis
needs a GPU and the rest does not.

```
   ┌─ your laptop ──────────┐   ┌─ Kaggle / Colab ──┐   ┌─ your laptop ──────┐
   │  preprocess            │   │                    │   │  assemble          │
   │  transcribe            │   │  clone the voice   │   │  remux             │
   │  translate + fit       │──>│  speak the text    │──>│                    │
   │  cut a voice reference │   │                    │   │                    │
   │  => tts_bundle.zip     │   │  => result.zip     │   │  => dubbed.mp4     │
   └────────────────────────┘   └────────────────────┘   └────────────────────┘
```

The bundle is a plain directory of JSON and WAV files. Nothing phones home,
nothing streams, and you can carry it on a USB stick if you like.

### 0. Install

Needs **Python 3.11** and **ffmpeg** on your PATH.

```bash
git clone -b test https://github.com/ayushk1233/indic-dub-pipeline.git
cd indic-dub-pipeline

python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

ffmpeg -version    # must print something
```

Check it works before going further:

```bash
./venv/bin/python -m pytest tests/ -q
```

> All commands below use `./venv/bin/python` rather than activating the venv,
> and run **from the repo root** — imports are absolute from `src.`.

### 1. First local run

```bash
./venv/bin/python -m src.cli \
  --input myvideo.mp4 \
  --job-id demo \
  --target-lang hi \
  --candidates 6
```

This transcribes, translates, picks the translation that fits each slot, cuts a
voice reference, and stops at the GPU boundary. First run downloads Whisper and
IndicTrans2 — a few GB, once.

**You get:** `artifacts/demo/tts_bundle.zip`, and a message telling you to take
it to a GPU.

<br>

> **What happens with the reference transcript, and why you may see a notice**
>
> Voice cloning conditions on a clip of the speaker **and a transcript of that
> clip**, as one sequence. The clip is cut from your source video, so on an
> English → Hindi job it is English audio — and the transcript has to say those
> English words, in the target script, because a script boundary between the
> reference text and the generated text puts invented speech at the head of a
> clip.
>
> That is transliteration, not translation, and it now happens automatically:
>
> ```
> [export]
>   reference transcript transliterated into the target script:
>     heard  So let me tell you what this project actually does. You get a video...
>     using  सो लेट मी टेल यू वट दिस प्राजेक्ट ऐक्चवली डज़. यू गेट अ विडीओ...
>     (pass --reference-text to override with a reviewed one)
> ```
>
> It is printed, not hidden, because it is a generated approximation of
> somebody's speech and you should be able to see what the model will be told
> the clip says. The pronunciations come from CMUdict and are American, so
> `project` becomes प्राजेक्ट where an Indian speaker is closer to प्रोजेक्ट.
> The consonant skeleton and syllable count are what the alignment needs, and
> those are right.
>
> **If a word matters and comes out wrong**, write the transcript yourself and
> pass it — a reviewed transcript always wins:
>
> ```bash
> --reference-text "सो लेट मी टेल यू व्हाट दिस प्रोजेक्ट ऐक्चुअली डज़ ..."
> ```
>
> Or point the flag at a `.txt`, or a `.json` with a `devanagari` key.
>
> **Dubbing Hindi into Hindi** never triggers any of this: the transcript is
> already in the target script and is passed through untouched.

### 2. The notebook

Open whichever you prefer — they do the same job:

| | |
|---|---|
| [`notebooks/kaggle_synthesis.ipynb`](notebooks/kaggle_synthesis.ipynb) | 30 free GPU hours/week, bundle attached as a Dataset |
| [`notebooks/colab_synthesis.ipynb`](notebooks/colab_synthesis.ipynb) | direct file upload, shorter free sessions |

Both walk you through it cell by cell, say what output to expect, and mark the
one point where you **must restart the runtime**. You will need a free Hugging
Face token, and you must accept the terms on the
[IndicF5 model page](https://huggingface.co/ai4bharat/IndicF5) once — it is a
gated repo.

**You get:** `demo_result.zip`.

### 3. Second local run

```bash
unzip -o ~/Downloads/demo_result.zip -d artifacts/demo/tts_bundle/output/

./venv/bin/python -m src.cli \
  --input myvideo.mp4 \
  --job-id demo \
  --target-lang hi \
  --from-stage import
```

**You get:** `artifacts/demo/dubbed.mp4`.

### What ends up on disk

```
artifacts/demo/
├── audio.wav                  extracted 16 kHz mono
├── chunks/                    one wav per detected speech span
├── manifest.json              chunk -> timestamp map
├── transcript.json            what was said, with absolute timestamps
├── translation.json           what will be said instead
├── translation_candidates.json   every translation considered, and why one won
├── reference.wav              the voice the model clones
├── tts_bundle/                what goes to the GPU, and what comes back
├── timeline.json              where each clip landed and what it cost
├── dubbed.wav                 the assembled track
└── dubbed.mp4                 ← the deliverable
```

`translation_candidates.json` and `timeline.json` are worth opening. They are the record of
every decision the pipeline made.

---

## How it works

### Stage by stage

**1 · Preprocess.** ffmpeg extracts a 16 kHz mono track. Speech is found by
running `silencedetect` and **inverting** it — the gaps between silences are the
speech. Spans under 0.25 s are dropped as clicks. If silence detection finds
nothing usable it falls back to fixed 30 s windows with 1 s overlap.

**2 · Transcribe.** Faster-Whisper, per chunk. The detail that matters is
timestamp arithmetic: Whisper returns offsets relative to each chunk, and the
processor adds that chunk's start time back, so every timestamp downstream is
absolute against the source. That is what lets a segment still know where it
belongs after translation and synthesis have changed everything else about it.

**3 · Translate, and make it fit.** This is where the 1.19× problem is
attacked. With `--candidates 6`, for every segment the pipeline:

- generates six genuinely different translations via diverse beam search
- predicts how long each would take **to speak** — from a fitted duration model
  over characters, base characters, combining marks and word count, not a
  character count
- scores each for semantic fidelity against the source
- keeps the one that fits the slot **without losing the meaning**

The budget is not the segment's own slot. It is the slot *plus the pause that
follows*, because speech can legitimately run into a pause. The same arithmetic
is used by the fitting cascade later, so the two never disagree.

The refusal is the important half: a candidate that fits but has lost the
meaning is rejected in favour of one that overruns. Assembly can stretch audio;
it cannot restore meaning.

**4 · Cut a voice reference.** The single input that decides whether cloning
works. Three things are deliberate:

- **10 seconds**, because the model silently clips a reference longer than 15 s
  without shortening its transcript to match — which hands it a transcript
  describing audio it cannot hear
- **re-cut from the source at 24 kHz**, not reused from the 16 kHz ASR chunk. A
  16 kHz clip gets upsampled and arrives with nothing above 8 kHz, which is
  where much of what makes a voice recognisable lives
- **level-normalised** to −1 dBFS, because the speaker encoder is not
  level-invariant

Spans are chosen longest-first (long uninterrupted speech conditions better
than the same seconds chopped up), then replayed in time order so the clip
sounds like continuous speech.

**5 · The GPU boundary.** Instead of a network call, the pipeline writes a
versioned bundle directory and zips it. Three guards run **before** it is
written, all of them refusals rather than warnings, because each fault they
catch is invisible to every other metric:

| guard | refuses | because |
|---|---|---|
| unspeakable text | a digit that survived normalisation | the model renders a digit as noise |
| reference script | a transcript in the wrong script | a script boundary invents speech at the head of the clip |
| reference length | a transcript too long for its audio | the model cannot align them and the generation collapses |

A bundle costs a GPU round trip to test. These cost a second.

**6 · Synthesis.** The worker loads the model once and walks the bundle. Two
levers are set explicitly rather than left to the model:

- **duration is forced per segment.** Left alone, the model allocates length
  from the reference's UTF-8 byte ratio, which across a script boundary
  over-allocates by more than 2× and fills the surplus with invented speech.
  The forced duration is the **total including the reference**, which is the
  single most expensive thing to get wrong here.
- **one chunk per segment**, because a chunked generation prepends the
  reference to every chunk.

The result file is written after *every* segment, so a session timeout leaves
salvageable work rather than nothing. A segment that raises is recorded as
failed and the run continues — with one exception: if the instrumentation is
not bound, the run aborts, because then every clip after it is wrong the same
way.

**7 · Assemble.** Synthesis returns clips that are roughly the right words and
rarely the right length. Four tiers, cheapest damage first:

| tier | what it does |
|---|---|
| **trim** | cut the trailing silence the decoder added. Free, and often enough alone |
| **absorb** | let the clip run into the pause that follows, if nothing collides |
| **stretch** | speed it up — but never past 1.25×, where speech stops sounding like speech |
| **drift** | let it run long, push what follows, and hand the lateness back at the next real pause |

Built as one NumPy buffer rather than an ffmpeg filtergraph, for two reasons:
ffmpeg's `amix` renormalises levels and would silently undo the gain work, and a
buffer lets a test assert that a clip landed at an exact sample index.

**8 · Remux.** The dubbed track goes back onto the original video, pinned to the
**video stream's** duration rather than the container's — QuickTime `.mov` files
report a container length longer than either stream, which otherwise shows up as
a sync failure on a file that is perfectly in sync.

### Text the model can actually say

Small, and the difference between a clean dub and an unusable one.

The model **cannot pronounce a digit**. A run where all 30 segments returned, every
clip landed on its forced duration, and assembly reported no stretching at all
still shipped audible gibberish — on exactly the segments containing numerals:

```
asked  लगभग 20% अधिक समय      heard  लगभग लतक अधिक समय
asked  उसी 3 सेकंड में         heard  उसी इदस सेकंड में
```

Nine of the ten digit-bearing segments were corrupted. Every timing metric read
green throughout. So numbers are spelled out before they reach the model —
Indian grouping for Hindi (`2,50,000` → दो लाख पचास हज़ार), years read as years
(`1947` → उन्नीस सौ सैंतालीस), long runs read digit by digit like a phone
number, decimals and digit grouping handled.

This happens **before** candidate scoring, not after, because `20%` is three
characters where `बीस प्रतिशत` is eleven — expanding after selection would mean
every number had been timed against text the model never speaks.

### Options

| flag | what it does |
|---|---|
| `--input` | source video or audio |
| `--job-id` | names the artifact directory |
| `--target-lang` | two-letter target code, default `hi` |
| `--source-lang` | two-letter source code. `--source-lang hi --target-lang hi` is the voice-cloning leg and skips translation |
| `--candidates` | translations per segment. `1` turns length control off and is much faster; `6` is the real thing |
| `--reference-text` | the reference transcript in the target script — literal text, or a path to a `.json`/`.txt` |
| `--reference-audio` | supply your own voice reference instead of cutting one |
| `--from-stage` | resume: `preprocess` `asr` `translate` `export` `import` `assemble` `remux` `report` |
| `--no-fidelity` | skip semantic scoring. Faster, avoids a 1.8 GB download, decides on duration alone |
| `--artifacts-root` | where job directories are written |

Every stage writes a file and reads the previous one's, so any stage can be the
starting point.

---

## When it goes wrong

### Install and dependencies

**`ImportError` during translation, or `'NoneType' object has no attribute 'shape'`**
`transformers` is too new. The translation model ships custom code that indexes
its cache as legacy tuples, and newer versions always pass a `Cache` object.
`requirements.txt` pins `transformers==4.46.3` for exactly this. Do not bump it
without re-running an end-to-end translation.

**The two requirements files disagree — is that a mistake?**
No, it is deliberate. `requirements.txt` is the laptop side and pins
`transformers<4.47` for the translation model. `requirements-gpu.txt` is the GPU
side, which never loads that model and needs `>=4.57`. Never install both into
one environment.

**A red `pip` conflict about `f5-tts` requiring `numpy<=1.26.4`**
Expected on the GPU side, and not a failure. That ceiling is inherited from an
older scientific stack; librosa's numba needs a *newer* numpy. We override it
deliberately and the model runs correctly on numpy 2.2.

**Model classes mysteriously missing on the GPU side**
Almost always a stale `numpy`, and it fails *silently*: with numpy 1.x,
`transformers` catches the error raised while building its torch-backed classes
and quietly drops those names. `import transformers` succeeds, the version
string looks right, and the only symptom is a missing class reported from
somewhere unrelated. Check `numpy.__version__` is ≥ 2.1 — if it is not, you did
not restart the runtime after installing.

**`ffmpeg: command not found`**
`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian/Ubuntu. Both
`ffmpeg` and `ffprobe` must be on PATH; the tests need them too.

### Transcription

**The transcript is confident and completely wrong**
You set the source language wrong. Whisper does not error on a mismatch — asked
for English on Hindi audio, it produces fluent, confident English words that
were never said. Set `--source-lang` to match your video.

**Indian-accented English is misheard**
The known weak spot. Real example: `fit` transcribed as `feet`, which then
survived translation as a fluent, correctly-pronounced *wrong word* — no
downstream check can catch that, because nothing is malformed. If your video has
domain vocabulary, expect to check `artifacts/<job>/transcript.json` before
letting it run on.

**Segments are far too long or too short**
Silence detection did not find usable boundaries and the 30 s fallback kicked
in. Check `manifest.json`. Very quiet recordings and continuous background music
both cause it.

**A repeated tail, or the same phrase over and over**
Whisper conditions on its own previous output and will continue a repetition
once one starts. Visible in `transcript.json` as an obviously duplicated line.

### Export and synthesis

**`ReferenceScriptMismatch`**
The automatic transliteration did not run or did not help. The run writes
`artifacts/<job>/reference_text.todo.json` with the transcript it heard and an
empty field — fill in the same words in the target script and re-run with
`--from-stage export --reference-text <that file>`.

**`TransliterationUnavailable`, or an NLTK download error on the first run**
`g2p-en` needs an NLTK tagger, fetched once at first use. Behind a proxy or
offline, pre-fetch it:

```bash
./venv/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```

If you cannot install `g2p-en` at all, nothing else breaks — the export refuses
the cross-script transcript and hands you the template above to fill in by hand.

**`UnspeakableText: N segment(s) still contain digits`**
A number no rule could expand reached the export. The message names the segment
and the number. Edit the text in `translation.json` and resume with
`--from-stage export`.

**`OSError: You are trying to access a gated repo`**
You are signed in but have not accepted the model terms. Open the
[IndicF5 page](https://huggingface.co/ai4bharat/IndicF5) while signed in, accept,
and re-run the notebook cell.

**The token is set but the notebook cannot read it**
Creating a secret and granting it to a notebook are two separate switches. On
Colab, open 🔑 and toggle **Notebook access**. On Kaggle, tick the box next to
the secret under **Add-ons → Secrets**.

### Assembly and output

**`remux failed: audio and video durations differ`**
Usually a truncated or partially-imported result. Check that
`tts_bundle/output/` holds one wav per segment plus `synthesis_result.json`.

**Some segments are silent in the dub**
Those segments failed on the GPU. `timeline.json` marks them `missing`, and
`tts_bundle/logs/errors.log` says why. Re-running the notebook on the same
bundle retries them.

**It sounds rushed**
Look at `timeline.json` for how many segments came back `stretched` or
`drifted`. A lot of either means the translations were too long for their slots —
raise `--candidates`, which gives length control more to choose from.

**It sounds fine but says the wrong thing**
The failure mode worth knowing about: every timing number can read perfect while
the audio is wrong. Listen before you trust a report. The notebooks include a
listen cell for this reason.

---

## What is in the repo

```
src/
  cli.py                 the one entry point
  pipeline/runner.py     the spine — walks the stages in order
  stages/
    preprocess*          ffmpeg, silence detection, chunking
    asr/                 Faster-Whisper
    translation/         IndicTrans2, length control, language check
    tts/bundle/          the GPU boundary: export guards, import checks
    assemble.py          the fitting cascade
    remux.py             back onto the video
    reference.py         cutting the voice reference
  text/                  numbers, loanwords, what the model can say
  eval/                  duration model, QC harness, metrics
colab/indicf5_worker.py  what runs on the GPU
notebooks/               the two notebooks
tests/                   219 tests, all offline, no GPU
```

**Supported today:** any source language Whisper handles → Hindi, and
Hindi → Hindi as pure voice cloning. Other Indic targets are wired end to end
but their speaking-rate constants are estimates rather than measurements, so
treat the timing verdicts as unvalidated. **English output is not supported** —
the voice model does not generate English.

### Tests

```bash
./venv/bin/python -m pytest tests/ -q
```

219 tests, all offline on CPU. [TESTS.md](TESTS.md) catalogues every one: what
it asserts and which failure it was written after. Most of them exist because
this pipeline's characteristic failure is a run that completes, writes plausible
files, prints a clean report, and is wrong.

---

## Roadmap

**Output register: Hinglish or pure Hindi, chosen by the user.**
Today the register is not a setting — it is a side effect of which leg ran. An
English source goes through a translation model trained largely on government
and news text, so it reaches for the formal word every time: परियोजना for
*project*, व्याख्यान for *lecture*, साक्षात्कार for *interview*. A Hindi source
never translates at all, so you get the speaker's own words — and real people say
*project*, *लेक्चर*, *इंटरव्यू*. The plan is to make that a choice rather than an
accident, on the translation path where it belongs.

**Diarization, and cloning each speaker.**
A video with more than one person in it should come back dubbed with more than
one voice. Right now there is no concept of a speaker anywhere: one reference
clip is cut from the whole recording and every segment is cloned from it, so a
two-person interview is dubbed entirely in whoever spoke longest. The work is a
speaker label that attaches at transcription time and survives to synthesis, one
reference per speaker, and a bundle that carries a set of references instead of
one.

**Better transliteration.** The reference transcript is now transliterated
automatically, from CMUdict pronunciations. Those are American, so the output is
a phonetic approximation rather than the spelling an Indian speaker would use.
A model trained on Indian English would do better; the one that exists
(`ai4bharat-transliteration`) depends on fairseq, which does not build on
Python 3.11.
