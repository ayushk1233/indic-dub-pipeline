# Voice cloning findings

Everything here was measured on this repo's fixtures unless it says otherwise. It exists so a
later session does not re-derive it, and so a claim can be checked against the run that produced
it rather than against memory. Read it before proposing anything about speaker similarity,
reference clips, or IndicF5's pacing.

Produced by `colab/four_arm.py` and `colab/indicf5_check.py`, against
`fixtures/{english,hindi}_reference*.wav` and `fixtures/scripted_text.json`.

---

## 1. A cosine similarity is meaningless without its own floor and ceiling

A raw speaker-embedding cosine cannot be read as a percentage of anything. Three separate
calibration errors were made and corrected during this work, and each one moved a headline number
by ten points or more.

**Floor** — 0.095, the mean cosine of the speaker against 58 XTTS studio speakers. That is what
"a stranger" scores, and it is not zero.

**Ceiling** — the speaker against himself, no synthesis anywhere in the measurement.

**Cross-language ceiling** — a real human's embedding shifts between languages. Synthesis of
`en -> hi` must be scored against *his real Hindi versus his real English*, not against a
same-language ceiling. Ignoring this made cross-lingual results look far worse than they were.

**Duration-matched ceiling** — embeddings from short audio are noisier and regress toward the
population mean. Measured r = +0.82 between clip duration and similarity. A ceiling measured on
17 s thirds must not be applied to 5 s clips.

| clip length | same-language ceiling | cross-language ceiling |
|---|---|---|
| 16.7 s | 0.968 | 0.887 |
| 8.3 s | 0.923 | 0.857 |
| 5.0 s | 0.881 | 0.825 |
| 3.6 s | 0.847 | 0.779 |

The ceiling falls 0.108 from 17 s to 3.6 s. Most of the apparent short-segment collapse in earlier
runs was the ruler, not the model. The real 14-segment bundle from `english.mov` averages under
5 s, which is the least reliable bucket.

`position = 100 * (score - floor) / (ceiling_for_this_clip_length - floor)`, averaged per clip —
never the mean score placed on one ceiling.

---

## 2. The four-arm result: cross-lingual transfer is not the bottleneck

XTTS-v2, clean re-recorded references, three sentences per arm.

| arm | greedy | sampled | position |
|---|---|---|---|
| `en -> en` | 0.502 | 0.486 | 47% |
| `hi -> hi` | 0.719 | 0.689 | 72% |
| `en -> hi` | 0.713 | 0.718 | **78%** (cross-language scale) |
| `hi -> en` | 0.553 | 0.532 | 53% |

Holding the spoken language fixed and swapping only the reference language moves the score by
−0.011 (speaking Hindi) and +0.048 (speaking English). Both sit inside the within-arm spread of
0.021–0.077, and the second one *favours the foreign reference*.

**What predicts the score is the language being spoken, not the language cloned from.** Speaking
English lands near 0.52 from either reference; speaking Hindi lands near 0.71 from either.
XTTS-v2's English decoder overwrites speaker identity — the same mechanism as its American accent,
measured a second way.

This retracted an earlier prediction of mine that a good English clone would prove cross-lingual
transfer was the problem, and it removed voice conversion from the shortlist of fixes.

Two further results from the same run:

- Re-recording the reference moved production `en -> hi` from 48% to 71% on the same scale, while
  `en -> en` moved only 42% to 47%. Recording quality is worth more than any parameter touched so far.
- Greedy decoding beat sampling on three of four arms and tied on the fourth. `do_sample=False`
  stays.

An earlier seven-configuration conditioning sweep was **statistically meaningless** and should not
be cited: the scale spans about 0.87 and the whole sweep spanned 0.046. Conditioning is held at
the shipped XTTS-v2 values.

---

## 3. The 90% target is inside the ruler's own noise

| target | raw cosine needed | gap from current 0.713 |
|---|---|---|
| 85% of scale | 0.776 | +0.063 |
| **90% of scale** | **0.816** | +0.103 |

The cross-language ceiling's four readings were 0.923 / 0.914 / 0.862 / 0.886, a span of 0.061.
**90% of scale (0.816) sits below the lowest reading of the human against himself.** It is a target
inside the measurement error of the instrument. **85% is the defensible target.**

The two complaints that prompted the target have different causes and different fixes:

- *"it matches the voice about 70%"* — timbre. This is what the cosine reads.
- *"I would not emphasize like this"* — speaking style. **The cosine is nearly blind to it.**

Zero-shot cloning copies vocal tract shape from the reference but takes delivery, emphasis and
phrasing from the model's own priors. Both are cured by speaker-specific training, which XTTS-v2's
licence forbids.

---

## 4. Licensing decides this before quality does

- **XTTS-v2 — CPML, non-commercial.** Coqui is defunct, so nobody can now grant commercial terms,
  and fine-tuned checkpoints inherit the licence. Whatever it scores, it cannot ship. It stays as
  a calibrated baseline only.
- **IndicF5 — MIT.** 0.4B parameters, 1417 hours of Indian speech, clones from a reference clip
  plus its transcript.

---

## 5. IndicF5 versus XTTS-v2

Seven Hindi sentences per arm, same sentences, same seed, same speaker encoder scoring both.

| model | arm | sim | se | position | cps | vs natural | gen |
|---|---|---|---|---|---|---|---|
| xtts | en_ref | 0.656 | 0.028 | 77% | 11.5 | 1.06x | 4.1 s |
| xtts | hi_ref | 0.695 | 0.026 | 74% | 11.3 | 1.04x | 4.2 s |
| indicf5 | en_ref_10s | 0.805 | 0.015 | 92% | 5.1 | **0.47x** | 39.0 s |
| indicf5 | **hi_ref_10s** | **0.785** | 0.011 | **86%** | 11.1 | **1.03x** | 21.9 s |
| indicf5 | en_ref_25s | 0.800 | 0.014 | 97% | 9.4 | 0.87x | 33.9 s |

Production case gap: **+0.149, combined standard error 0.031 — 4.6 standard errors.**

Length behaviour: XTTS scores 71% under 8 s and 80% at or above; IndicF5 scores 92% and 91%. **No
length degradation on IndicF5**, which matters because real dubbing segments are short.

---

## 6. The highest-scoring arm was speaking gibberish

`indicf5 en_ref_25s` scored **97%, the best of any arm**, and on listening it inserts invented
speech between the intended words. This was caught by ear, not by any metric in the project.

**Speaker similarity reads timbre only.** Nonsense in exactly the right voice outscores clean
speech in a slightly wrong one. The project had no content check at all until commit `bc3b66b`.

The check now in `colab/indicf5_check.py` transcribes every generated clip back with
`whisper-large-v3-turbo` and aligns it to the intended text with a Levenshtein alignment that
**counts insertions and deletions separately**. A single error rate is not enough: appended
gibberish scores CER 0.309, which passes a 0.40 threshold. Thresholds are `extra > 0.15`
(invented speech), `missing > 0.25` (cut short), `cer > 0.35`. Verified against seven constructed
cases, including a matra difference at 0.018 and a truncation at 0.564.

**Any future TTS comparison must gate on content before it reports identity.**

---

## 6b. IndicF5 cannot generate English, and the encoder does not care

Measured 2026-09-17, seven sentences, English reference, English output. IndicF5 declares eleven
Indian languages and English is not among them. The output is not accented English — it is not
English:

| asked | heard |
|---|---|
| "Seven were impossible, and we had to rewrite them." | `Sraindari ansu alwe atcho rureshi chong.` |
| "Last week the system processed forty-seven segments." | `Also, this is the process of making` ×95 |
| "Thirty-one fit perfectly." | `This is my surface fatigue.` |

**That arm scored 0.754, 84% of scale — and one clip reached 0.851 at 91%, higher than anything
the working Hindi arms produced.** This is the third time in this project that the best-looking
number came from broken audio, so it is now a rule rather than an observation:

> A speaker-similarity score is meaningless until the clip has been shown to say the right words.
> Never report identity before content.

Whisper loops on audio it cannot parse, fluently, so nothing downstream notices. A 3.97 s clip
transcribed to 2074 characters — 522 characters per second — and the report printed it as a
48 second prefix on a four second clip. `transcript_impossible()` now refuses to score any
transcript above 40 cps.

Four detectors are needed and none subsumes another:

| failure | what catches it |
|---|---|
| invented words | `extra` |
| speech before the sentence starts | `lead` (free-start alignment) |
| clip cut short | `missing` |
| **right length, right rhythm, wrong words** | **`cer` only** |
| babble filling an over-long slot | pace, `got/natural` |
| ASR degeneration | `transcript_impossible` |

The fourth row is why `cer` is not redundant: the sentence above has no inserted span, no missing
span and no prefix. Every positional check passes and not one word is correct.

---

## 7. What an English reference actually changes inside IndicF5

Read from `f5_tts/infer/utils_infer.py` in the public IndicF5 repo, not inferred. An English
reference changes three things at once, and two of them apply even to a short clip.

**a. Duration allocation is computed in UTF-8 bytes.**

```
duration = ref_audio_len + ref_audio_len / ref_text_bytes * gen_text_bytes / speed
```

Latin is 1 byte per character, Devanagari is 3. So an English reference prices a Hindi character
at roughly three times its real cost.

| reference | seconds per byte |
|---|---|
| English (Latin) | 0.0749 |
| Hindi (Devanagari) | 0.0348 |

F5-TTS is an in-filling model: it is told the total duration up front and must produce exactly
that many frames. Over-allocation leaves surplus time that has to be filled with something.

Implied reference length, recovered by inverting the ratio:

| arm | actual ref | implied ref | predicted pace | observed |
|---|---|---|---|---|
| en_ref_10s | 10.5 s | 10.3 s | 2.17x | **2.13x** |
| hi_ref_10s | 10.5 s | 9.4 s | 1.08x | **0.97x** |
| en_ref_25s | 25.5 s | **13.7 s** | 2.13x pre-clip | 0.87x |

**The `en_ref_25s` arm's healthy-looking 0.87x was luck.** Clipping 25 s down to 13.7 s divided the
allocation by 1.83 and nearly cancelled the 2.15x script inflation. Two errors of opposite sign,
not a working duration model.

**b. Chunking is also computed in bytes.**

```
max_chars = ref_text_bytes / ref_seconds * (25 - ref_seconds)
```

A 10.5 s English reference permits about 200 bytes of generated text — roughly 67 Devanagari
characters. The Hindi reference permits about 450. Real sentences run past 130 characters, so the
**same sentence takes a different code path depending only on the reference language**: split at
punctuation, generated independently, cross-faded back at 0.15 s.

**c. Reference truncation, on clips over 15 s only.**

`preprocess_ref_audio_text` clips audio over 15 s and **never truncates `ref_text` to match**. The
model is told that 13.7 s of audio contains 25 s of transcript. Confirmed by the library's own log
line `Audio is over 15s, clipping short.`, printed only on the 25 s arm.

### The experiment that separated them

Six arms, same seven sentences, same seed, one variable at a time
(`colab/indicf5_diagnose.py`, run 2026-09-17).

| arm | asked/natural | got/natural | extra | cer | bad clips |
|---|---|---|---|---|---|
| `en10_base` | 2.15 | 2.15 | 0.444 | 0.514 | 4 / 7 |
| **`en10_speed`** | **1.00** | **0.99** | **0.037** | **0.108** | **0 / 7** |
| `en10_one` | 2.15 | 2.15 | 0.574 | 0.737 | **6 / 7** |
| `en10_both` | 1.00 | 1.00 | 0.049 | 0.114 | 0 / 7 |
| `hi10_base` | 1.06 | 0.97 | 0.011 | 0.091 | 0 / 7 |
| `en25_base` | 1.16 | 1.15 | 0.223 | 0.438 | 3 / 7 |

Correlations across all 42 clips: invented speech against over-allocation **r = +0.58**; against
chunk count **r = −0.21**; over-allocation against chunk count **r = +0.02**, so the two mechanisms
were cleanly separated rather than confounded.

**Duration over-allocation is the cause.** Correcting it alone takes an English reference from four
bad clips out of seven to none, and to the same content quality as the Hindi-reference arm.

**Chunking is a consequence of the same fault, not an independent one.** `max_chars` exists to keep
`reference + generated` inside F5-TTS's 25 s training window. With the duration wrong, the longest
sentence asked for 27.8 s of generation on top of a 10.3 s reference — 38 s, half again past the
window the model was ever trained on, which is why `en10_one` was the worst arm of the six. With
the duration corrected, the same sentence needs 13.4 s and totals 23.7 s, so **no split is required
at all** and the cross-fade seam disappears with it. `en10_both` is the correct configuration, not
`en10_speed`.

This was corrected by ear after the run. The metric could not separate them — extra 0.037 against
0.049 — but on the 134-character sentence `en10_speed` splits into two chunks at roughly 5.9 s, and
that is exactly where a listener reports gibberish in the 4–6 s region of an otherwise clean clip.
A 0.15 s seam is far too short to move a character error rate.

Two details that confirm the mechanism rather than merely fitting it:

- **`got/natural` tracks `asked/natural` to two decimals in every arm.** The model produces exactly
  the duration it is handed. It is not drifting or running on; it is filling a slot that is too big.
- **Short sentences suffer most.** At `en10_base` the 42–52 character sentences scored extra 1.10,
  1.02 and 0.76, while the 115–145 character ones scored 0.02–0.16 — those were the ones chunking
  happened to split. Real dubbing segments are short, so this is the worst possible distribution.

### The content check has a blind spot: Whisper is a fluency prior

On `en10_base` the 134-character sentence scores extra 0.04 — apparently clean — and is audibly
full of gibberish. The transcribe-back check catches invented **words**; it does not catch invented
**sound**. Whisper is trained to emit fluent text and discards non-lexical babble rather than
transcribing it, so garbled audio can round-trip to a clean transcript. On the short sentences the
babble happened to be lexical enough to transcribe (`वीर्ड उदे एंस पसे एड़ शे एंड़...`, extra 1.10)
and the check fired; on the long ones it did not.

**What caught it instead was pace.** That clip reads `got/natural = 2.11`. Because the model fills
exactly the slot it is given, a pace ratio away from 1.0 is itself a content-integrity signal, and
on this arm it was the reliable one. Neither check is sufficient alone:

| failure | caught by `extra` | caught by pace |
|---|---|---|
| invented words | yes | sometimes |
| non-lexical babble filling a slot | **no** | yes |
| cross-fade seam at a chunk boundary | no | no |

The seam has no automatic detector yet. The fix is to remove the seam rather than measure it — with
the duration corrected, nothing needs splitting.

The four clips whose conditions were untouched between `en10_base` and `en10_one` returned
byte-identical scores, which is the control on the experiment itself.

**The 25 s arm is a genuinely separate fault.** It sits at 1.16x — pacing is nearly correct — and
still invents speech on three clips, one of them cut 60% short. Its measured 0.0412 s/byte lands
near Hindi's 0.0377 purely because clipping shortened the audio while the transcript stayed whole,
which quantifies the luck described above. Correcting duration will not fix it; the transcript has
to be truncated to match the clipped audio, or the reference kept under 15 s.

`speed` and `fix_duration` are both exposed by `infer_batch_process`, so both (a) and the
allocation half of (c) are correctable from outside the model.

`colab/indicf5_diagnose.py` separates (a) from (b) by varying one at a time. It patches
`chunk_text` and `infer_batch_process` as module globals in `f5_tts.infer.utils_infer` and wraps
`model.sample`, so the durations it reports are the durations the sampler was actually given
rather than arithmetic reproduced from the source. `tests/test_indicf5_diagnose.py` proves the
interception works against a pre-bound import, which is the assumption the whole measurement rests
on.

---

## 7b. The prefix is punctuation in the reference transcript

Measured 2026-09-17, seven sentences, English reference, corrected duration, single chunk.

| reference transcript | clips with a prefix | extra | cer |
|---|---|---|---|
| `So let me tell you what this project actually does. You get a video, lecture and...` | 1 of 7 | 0.049 | 0.114 |
| `so let me tell you what this project actually does you get a video lecture and...` | **0 of 7** | 0.037 | 0.108 |

Lowercasing and removing punctuation from the reference transcript removes the prefix. Nothing
else changed: same clip, same sentences, same corrected duration, same seed.

**This was found by accident and the first report of it was wrong.** The second arm was labelled
`en_deva` and was supposed to be Devanagari — Whisper's `language` argument is a hint, not a
constraint, and asked to read ten seconds of English "in Hindi" it returned English in Roman
letters. The verdict named script. 132 Latin characters and 132 Devanagari characters are
indistinguishable in a report; the byte counts are not, and are now printed for every variant,
with any arm under 80% Devanagari dropped rather than run under that name.

The mechanism is consistent with everything else here. `infer_batch_process` hands the model
`ref_text + gen_text` as one sequence and strips exactly `ref_audio_len` frames with no alignment
check behind the slice. Punctuation the speaker did not pause for is text the model has to place
somewhere, and what it cannot fit inside the conditioned frames is spoken at the start of the kept
region — which is where `पेंट केगे उसे` came from.

**Consequence: IndicXlit stays out of the pipeline.** Transliterating the reference transcript was
the fallback if punctuation had not been the cause. It is not needed, and the fix is one call to
the normaliser that already exists for scoring.

**Strength of evidence.** One flagged clip going to zero is thin on its own. It is corroborated
by `extra` and `cer` improving across all seven clips rather than only the flagged one, and it is
worth acting on regardless because stripping punctuation costs nothing and removes text the model
demonstrably cannot place.

---

## 8. Standing conclusion

**Superseded 2026-09-17.** The previous conclusion was that `indicf5 hi_ref_10s` is the only
shippable configuration, because it was the only one that was simultaneously accurate, correctly
paced and licensable — and that production would therefore have to record every speaker in Hindi.

That constraint came from the byte-ratio fault, not from the model. With the duration corrected,
a **10 s English reference** is clean (0 bad clips of 7) and correctly paced (1.00x natural), at
the same content quality as the Hindi arm. Recording speakers in Hindi is no longer required.

**Measured 2026-09-17.** The corrected English reference ties the Hindi one on identity:

| arm | sim | position | got/natural | prefix | cer |
|---|---|---|---|---|---|
| `en_hi` (English reference, corrected) | 0.768 | **93%** | 1.00 | 0.2 s | 0.114 |
| `hi_hi` (Hindi reference, control) | 0.785 | 86% | 0.97 | 0.0 s | 0.091 |

A difference of −0.017 at 0.7 standard errors — indistinguishable. Both were judged good by ear.
**The shipping configuration is a 10 s English reference with the duration corrected**, which is
what production supplies anyway.

The 92% the broken `en_ref_10s` arm scored is not carried forward anywhere: correcting the
duration changes what the model generates, so that number describes a configuration that no
longer exists.

**Open:** one clip in seven still opens with a 14-character prefix (`पेंट केगे उसे`), down from
three seconds on every clip. Keep the reference under 15 s until the transcript-truncation fault
is fixed.

---

## 9. Speaking rates and the duration model

Measured from FLEURS: English **13.13 cps** (n=394), Hindi **10.81 cps** (n=239).

**Open problem, identified and not yet fixed.** XTTS synthesis runs 1.20–1.23x faster than the
FLEURS human rates that `src/eval/duration_model.py` is fitted on, while the speaker himself is
only 1.07–1.13x fast. The model therefore predicts slots about 20% longer than XTTS actually
delivers, which biases length control toward over-short translations. The duration model should be
recalibrated against synthesized audio, not against FLEURS humans.

---

## 10. Fine-tuning: when, and on what

**Fine-tune when:** accent preservation is a product requirement; the speaker roster is fixed;
the content is code-mixed Hinglish; or the target language is undertrained in the base model.

**Do not fine-tune when:** the reference is noisy (fix the recording); identity holds in-language
but collapses cross-language (that pointed at voice conversion, and section 2 ruled it out here);
the audio does not fit the slot (that is translation length, not the voice); or there is under
about 10 minutes of clean speech per speaker.

**Tier 1 — Indic-native and permissive:** IndicF5 (MIT), F5-TTS (MIT), IndicParler-TTS
(Apache 2.0).
**Tier 2:** CosyVoice 2 (Apache 2.0), StyleTTS2 (MIT), Chatterbox (MIT), Orpheus (Apache 2.0).
**Cheaper path:** RVC (about 10 min per speaker, under an hour on a free T4), OpenVoice v2, seed-vc.
**Benchmark against but do not fine-tune:** XTTS-v2 (licence dead end), Sarvam Bulbul, ElevenLabs.
**Other stages:** IndicWhisper or a Whisper fine-tune for ASR (try `initial_prompt` first),
IndicConformer, an IndicTrans2 fine-tune for MT.

**The long pole is data.** Fine-tuning wants 20–30 minutes of clean speech per speaker. About 110
seconds currently exist.

---

## 11. Recording protocol

This produced the single largest measured improvement in the project, so it is written down
precisely.

Both languages, same speaker, same session, same microphone. A small soft room. 15–20 cm off-axis.
Never Bluetooth. 44.1 or 48 kHz. Peaks around −6 dB. **Explain to a friend, do not read** — read
speech gives the wrong prosody to clone from. Leave 10 s of silence, speak, then 5 s of silence.

Measured result: old reference 17.1 dB SNR; new English 30.7 dB; new Hindi 31.4 dB. A 21 dB drop
in noise floor. ASR word error rate 7.9% normalized, with `fit` heard as `feet` the one real
content error.

---

## 12. Environment traps that cost real time

**numpy is squeezed from three sides on the Colab side**, and getting it wrong fails *silently*.
transformers 4.57 needs >= 2.0, numba (via librosa) needs < 2.3, f5-tts declares <= 1.26.4. The
first two leave exactly `2.1 <= numpy < 2.3`; f5-tts cannot be satisfied alongside them and its
declaration is deliberately overridden, so pip prints a conflict for it on every install and that
is expected output. Install `colab/requirements.txt` **after** IndicF5 so these constraints survive.

Why it is silent: numpy 1.x makes transformers' lazy loader raise an `AttributeError` while
building its torch-backed classes; the loader **catches it and drops those names**. `import
transformers` then succeeds, the version string is correct, `is_torch_available()` still returns
True, and the only symptom is `cannot import name 'GPT2PreTrainedModel'` reported from inside
coqui-tts. A version pin cannot catch this, because the version was never wrong.

**IndicF5 loads entirely onto the meta device** under `from_pretrained`, because the failure is
raised inside the remote `__init__` that transformers runs under an empty-weights context. No
`from_pretrained` flag reaches it — `low_cpu_mem_usage=False` does not work. The fix is direct
instantiation via `AutoConfig` plus `get_class_from_dynamic_module`. **The `.to("cuda")` call is
what caught this**: without it the model would have loaded cleanly and synthesized noise, which
scores like a model that clones badly and would have been written down as a result.
`_assert_materialized()` now refuses a model with any parameter still on meta.

**Whisper transcription of the fixtures is nondeterministic** at default settings and hallucinates
a repeated tail. `scripts/transcribe_fixtures.py` deliberately bypasses `FasterWhisperBackend`,
pins `temperature=0.0` and `condition_on_previous_text=False`, and strips repeated phrases up to
four words long. Verified byte-identical across two runs.
