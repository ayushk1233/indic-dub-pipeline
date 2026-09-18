# Voice cloning findings

English-to-Hindi dubbing: what was measured, what it means, and what turned out to be wrong.

Every number here came from a run against this repo's fixtures, and the run that produced it is
named. Read this before proposing anything about speaker similarity, reference clips, or pacing —
several of the obvious moves have already been tried and measured, and three of the conclusions in
here replaced earlier ones that were confidently wrong.

Produced by `colab/four_arm.py`, `colab/indicf5_check.py`, `colab/indicf5_diagnose.py`,
`colab/indicf5_english.py` and `colab/vocab_check.py`.

---

## 1. The shipping configuration

Settled 2026-09-17.

| | |
|---|---|
| model | **IndicF5** (MIT) |
| reference clip | 10 s, **English**, under the 15 s internal clipping threshold |
| reference transcript | **lowercased, punctuation stripped** |
| duration | set from the target slot, **not** from the byte ratio |
| chunking | single chunk |
| identity | 0.768 — **93% of the calibrated scale** |
| pace | 1.00x natural Hindi |
| content | clean across 7 sentences, 0 prefixes, cer 0.114 |

Against the Hindi-reference control at 0.785 and 86%, that is a difference of −0.017 at 0.7
standard errors: indistinguishable. Both were judged good by ear.

**What this buys.** No per-speaker Hindi recording session. No transliteration dependency. No
licence dead end. The reference is an English clip, which is what production supplies anyway.

**Do not** use XTTS-v2 (see §8), do not ask IndicF5 for English output (§3), do not hand it a
reference over 15 s (§4d), and do not quote the 92% that the pre-fix `en_ref_10s` arm scored —
correcting the duration changed what the model generates, so that number describes a
configuration that no longer exists.

---

## 2. A similarity score is meaningless without its own floor and ceiling

A raw speaker-embedding cosine cannot be read as a percentage of anything. Three separate
calibration errors were made here and each moved a headline number by ten points or more.

**Floor — 0.095.** The speaker against 58 XTTS studio speakers. That is what a stranger scores,
and it is not zero.

**Ceiling — the speaker against himself**, no synthesis anywhere in the measurement.

**Cross-language ceiling.** A real human's embedding shifts between languages, so `en -> hi`
synthesis must be scored against *his real Hindi versus his real English*, not a same-language
ceiling. Ignoring this made cross-lingual results look far worse than they were.

**Duration-matched ceiling.** Embeddings from short audio are noisier and regress toward the
population mean. Measured r = +0.82 between clip duration and similarity, so a ceiling measured on
17 s pieces must not be applied to 5 s clips.

| clip length | same-language | cross-language |
|---|---|---|
| 16.7 s | 0.968 | 0.887 |
| 8.3 s | 0.923 | 0.857 |
| 5.0 s | 0.881 | 0.825 |
| 3.6 s | 0.847 | 0.779 |

The ceiling falls 0.108 from 17 s to 3.6 s. Most of the apparent short-segment collapse in earlier
runs was the ruler, not the model. Real dubbing segments matter here: the 14-segment bundle from
`english.mov` averages under 5 s, the least reliable bucket.

```
position = 100 * (score - floor) / (ceiling_at_this_clip_length - floor)
```

averaged per clip — never the mean score placed on one ceiling. Implemented in
`colab/speaker_scale.py`, arithmetic under test in `tests/test_speaker_scale.py`.

---

## 3. Never report identity before content

**The rule, learned three times:**

> A speaker-similarity score is meaningless until the clip has been shown to say the right words.
> The encoder reads timbre and nothing else, so nonsense in exactly the right voice outscores
> clean speech in a slightly wrong one.

Three times in this project the best-looking number came from broken audio:

1. `indicf5 en_ref_25s` scored **97% of scale, the highest of any arm**, while inserting invented
   speech between the intended words. Caught by ear, not by any metric — the project had no
   content check at all until then.
2. `en10_base` scored extra 0.04 — apparently clean — on a clip audibly full of gibberish.
3. `en_en` scored **0.754 at 84%, one clip at 0.851 and 91%**, higher than anything the two
   *working* Hindi arms produced, on audio that is not English at all (§3b).

### 3a. The detectors, and why none subsumes another

| failure | what catches it |
|---|---|
| invented words | `extra` — insertions over intended length |
| speech before the sentence starts | `lead` — free-start alignment |
| clip cut short | `missing` — deletions |
| **right length, right rhythm, wrong words** | **`cer` only** |
| babble filling an over-long slot | pace, `got/natural` |
| ASR degeneration | `transcript_impossible()` |

Thresholds: `extra > 0.15`, `missing > 0.25`, `cer > 0.35`, `lead > 0.05`.

`lead` is tighter than `extra` because the failures are not comparable: scattered insertions worth
12% of a sentence are tolerable, three contiguous seconds before it starts are not — and three
seconds at the front of a twelve second clip scores exactly 0.12 on a total rate.

The fourth row is why `cer` is not redundant. "Seven were impossible, and we had to rewrite them."
came back as `Sraindari ansu alwe atcho rureshi chong.` — no inserted span, no missing span, no
prefix, not one correct word. Every positional check passes.

A single error rate is not enough either: gibberish appended to an otherwise correct sentence
scores CER 0.309, below any threshold loose enough to tolerate Whisper's own Hindi error. The
alignment counts insertions and deletions separately for exactly this reason. Verified against
seven constructed cases including a matra difference at 0.018 and a truncation at 0.564.

### 3b. Two blind spots in the content check itself

**Whisper is a fluency prior.** It emits well-formed text and discards non-lexical babble rather
than transcribing it, so garbled audio can round-trip to a clean transcript. On `en10_base` the
134-character sentence scored extra 0.04 and was audibly gibberish. What caught it was **pace**:
`got/natural = 2.11`. Because the model fills exactly the slot it is handed, a pace ratio away
from 1.0 is itself a content signal.

**Whisper loops on audio it cannot parse**, fluently, so nothing downstream notices. A 3.97 s clip
transcribed to 2074 characters of "the process of making" repeated — 522 characters per second —
and the report printed it as a 48-second prefix on a four-second clip.
`transcript_impossible()` now refuses to score any transcript above 40 cps. Natural speech is
10.8 cps in Hindi and 13.1 in English; the fastest thing this project has synthesized was 22.9.

---

## 4. IndicF5 cannot generate English

Measured 2026-09-17, seven sentences, English reference, English output. IndicF5 declares eleven
Indian languages — Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi,
Tamil, Telugu — trained on Rasa, IndicTTS, LIMMITS and IndicVoices-R. English is not among them.

The output is not accented English. It is not English:

| asked | heard |
|---|---|
| "Seven were impossible, and we had to rewrite them." | `Sraindari ansu alwe atcho rureshi chong.` |
| "Last week the system processed forty-seven segments." | `Also, this is the process of making` ×95 |
| "Thirty-one fit perfectly." | `This is my surface fatigue.` |

Latin is nevertheless the largest script in its custom vocabulary (1501 of 2545 tokens) and the
English transcript tokenizes at 100% coverage, so a vocabulary gap is **not** the explanation.

If English output is ever needed it will not come from this model **in Latin script**. §4b
qualifies this: the same model produces intelligible English when the same English is spelled in
Devanagari.

### 4a. What a missing vocabulary token costs, and what it does not

Measured 2026-09-18 on Colab, `colab/vocab_check.py::check_arms`, against the transliteration
probe's two arms.

**An out-of-vocabulary character does not raise. It maps to index 0, and index 0 is the space.**
Nothing warns, no audio goes missing, and the model speaks a pause where the character was. That
is the worst possible failure for this project's purposes, because in a transcribe-back report a
pause inside a word is indistinguishable from the model being unable to say the word — which is
the one thing a transliteration arm exists to measure.

The 2545-token vocabulary breaks down as Latin 1501, Hangul 140, **Devanagari 104**, Bengali 82,
Kannada 79, Oriya 75, Malayalam 73, Telugu 72, Gujarati 70, Cyrillic 66. 104 Devanagari tokens is
fewer than the Hangul it will never use, and it is still enough: every Devanagari row below
tokenizes at 100% apart from punctuation.

| text | coverage | missing |
|---|---|---|
| `english_short` reference transcript | 135/135 | none |
| `hindi_short` reference transcript | 104/104 | none |
| `latin` arm, 6 of 7 sentences | 100% | none |
| `latin[1]` | 126/128 | em dash ×2 |
| `deva_hand` arm, 6 of 7 sentences | 100% | none |
| `deva_hand[1]` | 118/120 | em dash ×2 |

**Both predictions about which character would fail were wrong.** The hyphen in `फोर्टी-सेवन` and
`थर्टी-वन` was flagged as the unmeasured risk — no Devanagari line in any existing fixture contains
one — and it is in vocabulary. The em dash is not, and no one had thought to check it, because it
had been in the shipping gen text all along.

**A missing token is judged by position, not by Unicode category.** The em dash in `video — a
lecture` already stands between spaces, so substituting a space turns a pause into a slightly
different pause, which is what an em dash is for. `fixtures/scripted_text.json`'s Hindi gen text
carries **six em dashes**, and that run scored 93% of scale and was judged clean by ear — so this
is measured, not argued, and a rule that blocked on any missing token would have refused a run
already known to be good.

The opposite case is not "letters". An apostrophe is punctuation too, and losing it turns `isn't`
into `isn t` — a pause inside a word, the failure above exactly. `stands_alone(char, text)` tests
whether every occurrence is flanked by whitespace, over the whole string, because the substitution
is global. Only word-internal gaps block.

One asymmetry worth knowing: the tokenizer inserts a space after a **Latin** hyphen (`Thirty-one`
is 25 characters and 26 tokens) but not after a Devanagari one (`थर्टी-वन` is 23 and 23). Both are
in vocabulary, so it changes nothing today.

### 4b. English spelled in Devanagari works — and overshoots the accent

Heard 2026-09-18 on Colab, `colab/indicf5_xlit_probe.py`, 28 clips, seven sentences, `latin`
against `deva_hand` at three seeds. **This is an ear result and only an ear result.** The run's
content metrics were all nan for the reason in §13, so there is no CER, no seed spread, and no
confirmation that the `latin` control failed. Synthesis itself was sound: all 28 clips were asked
for their own slot within 21 ms, and none came back off-script.

**The hypothesis holds.** Hand-transliterated English in Devanagari comes back as intelligible,
recognisably Indian-accented English in the cloned voice. The route exists.

**The accent is not the speaker's.** On an informal 1-to-10 scale where 1 is American or British
and 10 is heavily Indian, the speaker places his own reference recording at about **6** and the
generated output at about **9** — "the hard t, d especially". The voice is cloned; the accent is
imposed. Two consequences, and the second is the one that matters:

- A listener comparing dubbed output against the same person's real speech hears a different
  accent in the right timbre. Nothing in §2's identity scale can see this — it reads timbre, and
  the timbre is correct.
- **It does not vary with the speaker.** Someone whose English sits at 4 or 5 would be expected to
  come back at the same 9, because nothing in the current configuration carries that speaker's
  accent into the generation. Accent would be a constant the system imposes rather than a property
  of the person being dubbed, which is the opposite of what dubbing is for.

Where the 9 comes from is not yet measured, and the two candidates have opposite consequences:

| candidate | why it is plausible | if true |
|---|---|---|
| **The orthography.** Devanagari forces a choice English does not make: `ट`/`ड` are retroflex, English `/t/` `/d/` are alveolar, and Hindi's plain stops are unaspirated where English aspirates word-initially. `लेट`, `सिस्टम`, `फिट` all spell the hard stop the speaker is hearing. Post-vocalic `र` in `लॉन्गर` and `वर्ड्स` forces full rhoticity. | Accent is a dial on the transliterator and moving it is cheap — but it is one dial for everyone, and it does not track the speaker. |
| **The reference pair is script-mismatched.** `ref_text` is Latin, `gen_text` is Devanagari. §4 shows the model has no learned mapping from Latin orthography to English phones, so the reference transcript is close to useless as an alignment anchor: the model has audio at accent 6 and no text it can use to learn what that audio *is*. | Accent transfers in-context from the reference, adapts per speaker for free, and the fix is to give the reference its own Devanagari transcript. |

The discriminating test is one arm: transliterate the reference transcript into Devanagari, change
nothing else. Until it runs, neither candidate is a finding.

This is also the point where an accent metric stops being optional
(`TRANSLITERATION_PLAN.md` §3c). The right anchor is not a generic classifier verdict but
`fixtures/en_speaker/*.wav` — the same sentences, the same speaker, his own accent — scored the way
§2 scores identity: against his own floor and ceiling rather than against an absolute.

---

## 5. What an English reference changes inside IndicF5

Read from `f5_tts/infer/utils_infer.py`, not inferred from output. Four distinct faults were found
here, in this order, and only the first two were ever guessed correctly.

### 5a. Duration is allocated in UTF-8 bytes — the primary fault

```
duration = ref_audio_len + ref_audio_len / ref_text_bytes * gen_text_bytes / speed
```

Latin is 1 byte per character, Devanagari is 3, so an English reference prices a Hindi character
at roughly three times its real cost.

| reference | seconds per byte |
|---|---|
| English (Latin) | 0.0749 |
| Hindi (Devanagari) | 0.0348 |

F5-TTS is an **in-filling** model: it is told the total duration up front and must produce exactly
that many frames. Over-allocation leaves surplus time that has to be filled with something.

Implied reference length, recovered by inverting the ratio:

| arm | actual ref | implied ref | predicted pace | observed |
|---|---|---|---|---|
| en_ref_10s | 10.5 s | 10.3 s | 2.17x | **2.13x** |
| hi_ref_10s | 10.5 s | 9.4 s | 1.08x | **0.97x** |
| en_ref_25s | 25.5 s | **13.7 s** | 2.13x pre-clip | 0.87x |

The 25 s arm's healthy-looking 0.87x was **luck**: clipping 25 s to 13.7 s divided the allocation
by 1.83 and nearly cancelled the 2.15x script inflation. Two errors of opposite sign, not a
working duration model.

**The six-arm experiment** (`colab/indicf5_diagnose.py`, same sentences, same seed, one variable
at a time):

| arm | asked/natural | got/natural | extra | cer | bad clips |
|---|---|---|---|---|---|
| `en10_base` | 2.15 | 2.15 | 0.444 | 0.514 | 4 / 7 |
| **`en10_speed`** | **1.00** | **0.99** | **0.037** | **0.108** | **0 / 7** |
| `en10_one` | 2.15 | 2.15 | 0.574 | 0.737 | **6 / 7** |
| `en10_both` | 1.00 | 1.00 | 0.049 | 0.114 | 0 / 7 |
| `hi10_base` | 1.06 | 0.97 | 0.011 | 0.091 | 0 / 7 |
| `en25_base` | 1.16 | 1.15 | 0.223 | 0.438 | 3 / 7 |

Correlations across all 42 clips: invented speech against over-allocation **r = +0.58**; against
chunk count **r = −0.21**; over-allocation against chunk count **r = +0.02**, so the two were
cleanly separated rather than confounded.

Two details confirm the mechanism rather than merely fitting it:

- **`got/natural` tracks `asked/natural` to two decimals in every arm.** The model produces
  exactly the duration it is handed. It is not drifting; it is filling a slot that is too big.
- **Short sentences suffer most.** At `en10_base` the 42–52 character sentences scored extra 1.10,
  1.02 and 0.76 while the 115–145 character ones scored 0.02–0.16. Real dubbing segments are
  short, so this is the worst possible distribution.

The four clips whose conditions were untouched between `en10_base` and `en10_one` returned
identical scores, which is the control on the experiment itself.

### 5b. Chunking is a consequence, not a cause

```
max_chars = ref_text_bytes / ref_seconds * (25 - ref_seconds)
```

Also in bytes. A 10.5 s English reference permits about 200 bytes — roughly 67 Devanagari
characters — against about 450 for the Hindi one, so sentences past 130 characters get split and
cross-faded only on the English arm.

But `max_chars` exists to keep `reference + generated` inside F5-TTS's **25 s training window**.
With the duration wrong, the longest sentence asked for 27.8 s of generation on a 10.3 s
reference — 38 s, half again past the window — which is why `en10_one` was the *worst* arm of the
six. With the duration corrected the same sentence totals 23.7 s, so nothing needs splitting and
the seam disappears with it.

Chunking neither causes the babbling nor prevents it. Single chunk, always, once duration is right.

### 5c. Punctuation in the reference transcript — the residual prefix

With duration corrected and a single chunk, one clip in seven still opened with 14 characters of
invented speech (`पेंट केगे उसे`). Measured 2026-09-17:

| reference transcript | clips with a prefix | extra | cer |
|---|---|---|---|
| `So let me tell you what this project actually does. You get a video, lecture and...` | 1 of 7 | 0.049 | 0.114 |
| `so let me tell you what this project actually does you get a video lecture and...` | **0 of 7** | 0.037 | 0.108 |

Lowercasing and removing punctuation removes the prefix. Nothing else changed.

Consistent with the mechanism: `infer_batch_process` hands the model `ref_text + gen_text` as one
sequence and strips exactly `ref_audio_len` frames with **no alignment check behind the slice**.
Punctuation the speaker did not pause for is text the model must place somewhere, and what does
not fit inside the conditioned frames is spoken at the start of the kept region.

**Consequence: transliteration stays out of the pipeline.** IndicXlit was the fallback if
punctuation had not been the cause. The fix is one call to the normaliser that already exists for
scoring.

*Strength of evidence:* one flagged clip going to zero is thin. It is corroborated by `extra` and
`cer` improving across all seven clips rather than only the flagged one, and worth acting on
regardless because it costs nothing and removes text the model demonstrably cannot place.

### 5d. Reference truncation over 15 s — still open

`preprocess_ref_audio_text` clips audio over 15 s and **never truncates `ref_text` to match**, so
the model is told that 13.7 s of audio contains 25 s of transcript. Confirmed by the library's own
log line `Audio is over 15s, clipping short.`, printed only on the 25 s arm.

`en25_base` sits at 1.16x — pacing nearly right — and still invents speech on three clips with one
cut 60% short. Correcting duration does not touch it. **The shipping configuration avoids it by
keeping the reference under 15 s.**

### 5e. The levers

`speed` and `fix_duration` are both exposed by `infer_batch_process` and neither requires touching
the model. `fix_duration` is the better one for this project: dubbing already knows each segment's
slot, so setting it closes the fit problem and the babbling problem with the same call.

`colab/indicf5_diagnose.py` measures from inside the library — it patches `chunk_text` and
`infer_batch_process` as module globals and wraps `model.sample`, so reported durations are the
ones the sampler was actually given. `tests/test_indicf5_diagnose.py` proves the interception
works against a pre-bound import, which is the assumption the whole measurement rests on, and the
report refuses to print its tables if no call was intercepted.

---

## 6. IndicF5 versus XTTS-v2

Seven Hindi sentences per arm, same sentences, same seed, same speaker encoder scoring both.

| model | arm | sim | se | position | cps | vs natural |
|---|---|---|---|---|---|---|
| xtts | en_ref | 0.656 | 0.028 | 77% | 11.5 | 1.06x |
| xtts | hi_ref | 0.695 | 0.026 | 74% | 11.3 | 1.04x |
| indicf5 | en_ref_10s | 0.805 | 0.015 | 92% | 5.1 | 0.47x |
| indicf5 | hi_ref_10s | 0.785 | 0.011 | 86% | 11.1 | 1.03x |
| indicf5 | en_ref_25s | 0.800 | 0.014 | 97% | 9.4 | 0.87x |

Production-case gap **+0.149 at 4.6 standard errors**. The IndicF5 rows predate the duration fix,
so their pacing is wrong and two of them were speaking gibberish (§3). The identity ordering
survived the fix; see §1 for the post-fix numbers.

Length behaviour: XTTS scores 71% under 8 s and 80% at or above. IndicF5 scores 92% and 91% —
**no length degradation**, which matters because real segments are short.

---

## 7. The four-arm result: cross-lingual transfer is not the bottleneck

XTTS-v2, clean re-recorded references, three sentences per arm.

| arm | greedy | sampled | position |
|---|---|---|---|
| `en -> en` | 0.502 | 0.486 | 47% |
| `hi -> hi` | 0.719 | 0.689 | 72% |
| `en -> hi` | 0.713 | 0.718 | **78%** |
| `hi -> en` | 0.553 | 0.532 | 53% |

Holding the spoken language fixed and swapping only the reference language moves the score by
−0.011 and +0.048 — both inside the within-arm spread of 0.021–0.077, and the second *favours the
foreign reference*.

**What predicts the score is the language being spoken, not the language cloned from.** Speaking
English lands near 0.52 from either reference; speaking Hindi near 0.71 from either. XTTS-v2's
English decoder overwrites speaker identity — the same mechanism as its American accent, measured
a second way. This removed voice conversion from the shortlist of fixes.

Also from that run:

- **Re-recording the reference moved production `en -> hi` from 48% to 71%** on the same scale,
  while `en -> en` moved only 42% to 47%. Recording quality was worth more than any parameter
  touched before or since.
- Greedy beat sampling on three of four arms and tied on the fourth. `do_sample=False` stays.

---

## 8. The 90% target is inside the ruler's own noise

| target | raw cosine needed | gap from 0.713 |
|---|---|---|
| 85% of scale | 0.776 | +0.063 |
| **90% of scale** | **0.816** | +0.103 |

The cross-language ceiling's four readings were 0.923 / 0.914 / 0.862 / 0.886, a span of 0.061.
**0.816 sits below the lowest reading of the human against himself.** 85% is the defensible target.

The two complaints that prompted it have different causes:

- *"it matches the voice about 70%"* — timbre. This is what the cosine reads.
- *"I would not emphasize like this"* — speaking style. **The cosine is nearly blind to it.**

Zero-shot cloning copies vocal tract shape from the reference but takes delivery, emphasis and
phrasing from the model's own priors. Both are cured by speaker-specific training.

---

## 9. Licensing decides this before quality does

- **XTTS-v2 — CPML, non-commercial.** Coqui is defunct, nobody can now grant commercial terms, and
  fine-tuned checkpoints inherit the licence. Whatever it scores, it cannot ship. It remains a
  calibrated baseline only.
- **IndicF5 — MIT.** 0.4B parameters, 1417 hours of Indian speech, clones from a reference clip
  plus its transcript. Gated on Hugging Face: the account needs to accept the terms at
  `huggingface.co/ai4bharat/IndicF5`, and a valid token on an account without access fails
  identically to no token.

---

## 10. Speaking rates and the duration model

Measured from FLEURS: English **13.13 cps** (n=394), Hindi **10.81 cps** (n=239).

**Open problem.** XTTS synthesis runs 1.20–1.23x faster than the FLEURS human rates that
`src/eval/duration_model.py` is fitted on, while the speaker himself is only 1.07–1.13x fast. The
model therefore predicts slots about 20% longer than synthesis actually delivers, biasing length
control toward over-short translations. It should be recalibrated against synthesized audio.

---

## 11. Fine-tuning: when, and on what

**Fine-tune when:** accent preservation is a product requirement; the speaker roster is fixed; the
content is code-mixed Hinglish; or the target language is undertrained in the base model.

**Do not fine-tune when:** the reference is noisy (fix the recording — §12); identity holds
in-language but collapses cross-language (that pointed at voice conversion, and §7 ruled it out
here); the audio does not fit the slot (that is translation length, not the voice); or there is
under about 10 minutes of clean speech per speaker.

**Tier 1 — Indic-native and permissive:** IndicF5 (MIT), F5-TTS (MIT), IndicParler-TTS (Apache 2.0).
**Tier 2:** CosyVoice 2 (Apache 2.0), StyleTTS2 (MIT), Chatterbox (MIT), Orpheus (Apache 2.0).
**Cheaper path:** RVC (about 10 min per speaker, under an hour on a free T4), OpenVoice v2, seed-vc.
**Benchmark against but do not fine-tune:** XTTS-v2 (licence), Sarvam Bulbul, ElevenLabs.
**Other stages:** IndicWhisper or a Whisper fine-tune for ASR (try `initial_prompt` first),
IndicConformer, an IndicTrans2 fine-tune for MT.

**The long pole is data.** Fine-tuning wants 20–30 minutes of clean speech per speaker. About 110
seconds currently exist.

---

## 12. Recording protocol

This produced the single largest measured improvement in the project.

Both languages, same speaker, same session, same microphone. A small soft room. 15–20 cm off-axis.
Never Bluetooth. 44.1 or 48 kHz. Peaks around −6 dB. **Explain to a friend, do not read** — read
speech gives the wrong prosody to clone from. Leave 10 s of silence, speak, then 5 s of silence.

Measured: old reference 17.1 dB SNR; new English 30.7 dB; new Hindi 31.4 dB — a 21 dB drop in
noise floor. ASR word error rate 7.9% normalized, with `fit` heard as `feet` the one real content
error.

---

## 13. Environment traps that fail silently

**numpy is squeezed from three sides.** transformers 4.57 needs >= 2.0, numba (via librosa) needs
< 2.3, f5-tts declares <= 1.26.4. The first two leave exactly `2.1 <= numpy < 2.3`; f5-tts cannot
be satisfied alongside them and its declaration is deliberately overridden, so pip prints a
conflict for it on every install and **that is expected output**. Install `colab/requirements.txt`
**after** IndicF5 so these constraints survive.

Why it is silent: numpy 1.x makes transformers' lazy loader raise an `AttributeError` while
building its torch-backed classes, and the loader **catches it and drops those names**. `import
transformers` then succeeds, the version string is correct, `is_torch_available()` still returns
True, and the only symptom is `cannot import name 'GPT2PreTrainedModel'` reported from inside
coqui-tts. **A version pin cannot catch this** — the version was never wrong. Check by importing
the class, not by reading `__version__`.

**IndicF5 loads entirely onto the meta device** under `from_pretrained`, because the failure is
raised inside the remote `__init__` that transformers runs under an empty-weights context. No
`from_pretrained` flag reaches it; `low_cpu_mem_usage=False` does not work. The fix is direct
instantiation via `AutoConfig` plus `get_class_from_dynamic_module`. **The `.to("cuda")` call is
what caught it** — without it the model would have loaded cleanly and synthesized noise, which
scores like a model that clones badly and would have been written down as a result.
`_assert_materialized()` now refuses a model with any parameter still on meta.

**Whisper's `language` argument is a hint, not a constraint.** Asked to read ten seconds of English
"in Hindi" it returned English in Roman letters. An arm was labelled `en_deva` on that basis and a
verdict named script. 132 Latin characters and 132 Devanagari characters are indistinguishable in a
report; the byte counts are not. Always print characters against bytes, and gate on
`devanagari_fraction()`.

**Whisper transcription of the fixtures is nondeterministic** at default settings and hallucinates
a repeated tail. `scripts/transcribe_fixtures.py` bypasses `FasterWhisperBackend`, pins
`temperature=0.0` and `condition_on_previous_text=False`, and strips repeated phrases up to four
words. Verified byte-identical across two runs.

**faster-whisper and transformers spell the same decode parameter differently**, and pinning it in
the wrong dialect voids a whole run without failing it. faster-whisper takes `condition_on_previous_text`;
transformers takes `condition_on_prev_tokens`. `generate()` declares `**kwargs`, so the wrong name
is not rejected where it is written — it is forwarded to the model's forward and raises there, once
per clip, inside a per-row `except`. Measured 2026-09-18: the 28-clip transliteration probe ran to
completion, wrote its audio, confirmed `fix_duration` had been applied, and printed a full report in
which **every content number was nan**. Because nan compares False against every threshold, each arm
showed zero bad clips and the `latin` negative control came back "clean" — a report that reads as a
result. The exception text was being stored in the transcript field, and it is Latin, so it also
cleared the script gate. `scripts/transcribe_fixtures.py` is faster-whisper and keeps the other
spelling correctly; the two files disagree on purpose.
`tests/test_asr_decode_params.py` now checks every key against the installed signature.

**An out-of-vocabulary character is spoken as a pause and nothing raises**, because it maps to
index 0 and index 0 is the space. It cannot be caught by looking at the audio, the logs or the
score — only by tokenizing the text against `vocab.txt` first. §4a has the rule for which gaps
matter; `colab/vocab_check.py::check_arms` runs it before synthesis.

**Kaggle:** `HF_TOKEN` in the secrets panel is stored and never read — Colab's panel is wired into
`huggingface_hub` and Kaggle's is not, so a gated download returns a 401 that looks like a
permissions failure. `/kaggle/working` survives a kernel restart but **not a container
replacement**, and the next container restores it only from the last saved version's output. See
`colab/kaggle.md`.

---

## 14. What turned out to be wrong

Kept because a later session finding these cited elsewhere needs to know they do not hold.

| claim | what actually happened |
|---|---|
| A good English clone would prove cross-lingual transfer is the bottleneck | Reference language changes the score by −0.011 / +0.048, inside noise. The *spoken* language is what predicts it (§7). |
| The seven-configuration conditioning sweep found something | The scale spans 0.87 and the whole sweep spanned 0.046. Statistically meaningless; do not cite it. |
| Gibberish came from the 25 s transcript/audio mismatch | That is one fault, and it does not touch the 10 s arm. The primary cause was duration over-allocation (§5a). |
| Chunking was protective | It was a consequence of over-allocation. `en10_one` was worst because it blew past the 25 s training window (§5b). |
| The residual prefix was a cross-fade seam | It survived a single chunk. It was punctuation (§5c). |
| The prefix was a vocabulary gap — the model has no Latin tokens | 100% coverage; Latin is the largest script in the vocabulary (§4). |
| Whisper in Hindi mode gives a Devanagari transcript | It returned Latin, and the arm built on it was mislabelled (§13). |
| `en_en` would discriminate the prefix | IndicF5 cannot generate English, so its transcribe-back is unreadable (§4). |
| The hyphen in `फोर्टी-सेवन` is the character to check for vocabulary coverage | It is in vocabulary. The em dash is not, and nobody thought to check it because it was already in the shipping gen text (§4a). |
| Any missing vocabulary token invalidates the row | It would have refused the shipping `en -> hi` text, which carries six em dashes and scored 93% of scale. Position decides it, not category (§4a). |

Two of these were caught by ear rather than by any metric, and one of them — *"for every
en_ref_25s audio there is gibberish in between"* — is what started the investigation that produced
§5 entirely.

---

## 15. Open

- **Wire IndicF5 in as a pipeline TTS backend.** Needs `reference_text` on `SynthesisSegment`, a
  bundle version bump, the transcript normaliser applied to the reference, and `fix_duration` from
  each segment's real slot.
- **Recalibrate `src/eval/duration_model.py`** against synthesis rather than FLEURS humans (§10).
- **§5d**, reference transcript truncation over 15 s. Avoided, not fixed.
- Whisper `initial_prompt` with domain vocabulary for the Indian-accented-English ASR weakness.
- Trimming silence from reference spans before joining them.
