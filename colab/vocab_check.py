"""
Does IndicF5 have a vocabulary entry for the English reference transcript?

Three mechanisms have now been ruled in or out for the babbling on an English
reference. Duration over-allocation was real and is fixed. Chunking was a
consequence of it, not a cause. Neither explains what is left: with a single
chunk and a correct duration, generation still opens with about three seconds
of invented speech before the intended sentence starts.

That residue is not random. Transcribed back it reads as garbled English —
`एंड़` for "and", `शे` for "speech" — which is the reference transcript, not the
sentence asked for. infer_batch_process conditions on the reference audio,
hands the model `ref_text + gen_text` as one sequence, and then strips exactly
`ref_audio_len` frames off the front:

    generated = generated[:, ref_audio_len:, :]

There is no alignment check behind that slice. It assumes the model finished
speaking ref_text within the conditioned frames. If the model did not — if it
could not align the reference transcript to the reference audio — the remainder
is spoken at the start of the kept region, which is exactly what is heard.

The cheapest reason the model would fail to align Latin text is that it has no
tokens for it. IndicF5 was retrained on eleven Indian languages and ships its
own vocabulary. If Latin characters are absent, every character of a 135-char
English transcript is an unknown token, the model is conditioned on 10s of
audio it has no text for, and spillover is not a bug but the only possible
outcome.

This takes two seconds and no GPU, so it runs before any more synthesis.

    import colab.vocab_check as vc
    vc.main()
"""

import glob
import json
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"

# HF_HOME moves on some hosts, so the cache is searched rather than assumed.
VOCAB_TAIL = "hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"


def cache_roots():
    import os

    seen, roots = set(), []
    for candidate in (os.environ.get("HF_HOME"),
                      os.environ.get("HUGGINGFACE_HUB_CACHE"),
                      str(Path.home() / ".cache" / "huggingface"),
                      "/root/.cache/huggingface",
                      "/kaggle/working/.cache/huggingface"):
        if candidate and candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)
    return roots


def find_vocab(pattern=None):
    patterns = [pattern] if pattern else [
        str(Path(root) / VOCAB_TAIL) for root in cache_roots()]

    for candidate in patterns:
        matches = sorted(glob.glob(candidate))
        if matches:
            return Path(matches[-1])

    raise FileNotFoundError(
        f"no vocab.txt under any of {patterns} — load the model once first, "
        "or pass the path printed by load_indicf5 as `vocab :`")


def read_vocab(path):
    """
    One token per line, and the blank first line is a real token — it is the
    space. Splitting on newlines rather than using readlines() keeps it.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def script_of(char):
    if char == " ":
        return "space"
    try:
        name = unicodedata.name(char)
    except ValueError:
        return "unnamed"
    return name.split(" ")[0]


def report(label, text, vocab):
    from f5_tts.model.utils import convert_char_to_pinyin

    tokens = convert_char_to_pinyin([text])[0]
    known = [t for t in tokens if t in vocab]
    unknown = [t for t in tokens if t not in vocab]
    multi = [t for t in tokens if len(t) > 1]

    print(f"\n--- {label}: {len(text)} chars -> {len(tokens)} tokens")
    print(f"    in vocabulary   {len(known):>4} / {len(tokens)} "
          f"({100 * len(known) / max(len(tokens), 1):.1f}%)")
    print(f"    missing         {len(unknown):>4}")
    print(f"    multi-character {len(multi):>4}"
          + ("   (a token longer than one character is a tokenizer mismatch, "
             "not a vocabulary gap)" if multi else ""))

    if unknown:
        counts = Counter(unknown)
        shown = ", ".join(f"{t!r}x{n}" for t, n in counts.most_common(12))
        print(f"    missing tokens  {shown}")
        scripts = Counter(script_of(t[0]) for t in unknown if t)
        print(f"    by script       {dict(scripts)}")

    print(f"    first 24        {tokens[:24]}")
    return len(unknown), len(tokens)


def main():
    path = find_vocab()
    vocab = set(read_vocab(path))
    print(f"vocabulary {path}")
    print(f"{len(vocab)} tokens")

    scripts = Counter(script_of(t[0]) for t in vocab if t)
    print(f"scripts    {dict(scripts.most_common(10))}")

    text = json.loads((FIXTURES / "reference_text.json").read_text(encoding="utf-8"))
    results = {}
    for key in ("english_short", "hindi_short"):
        results[key] = report(key, text[key]["text"], vocab)

    print("\n" + "=" * 70)
    english_missing = results["english_short"][0]
    hindi_missing = results["hindi_short"][0]

    if english_missing and not hindi_missing:
        print("The English reference transcript is partly outside the model's")
        print("vocabulary and the Hindi one is not. The model is conditioned on")
        print("ten seconds of audio whose transcript it cannot read, so it")
        print("cannot know where the reference ends — and infer_batch_process")
        print("cuts at ref_audio_len regardless. The spillover is explained and")
        print("no amount of duration correction will remove it. The fix is to")
        print("give the reference a transcript the model can tokenize.")
    elif not english_missing:
        print("Both transcripts tokenize cleanly, so the spillover is not a")
        print("vocabulary gap. The next test holds the English audio fixed and")
        print("moves only the script: transliterate the transcript into")
        print("Devanagari and see whether the prefix survives.")
    else:
        print("Both transcripts have missing tokens, so coverage does not")
        print("separate the arms and this was not the discriminating test.")

    return results


def check_arms(arms=("deva_hand",)):
    """
    The same question asked of the text the probe *generates*, not conditions on.

    main() above covers the two reference transcripts, which is what the
    spillover investigation needed. The transliteration probe hands the model
    new text in a script no fixture has exercised, and a gap there fails in a
    quieter way than spillover does: an unknown token maps to index 0, index 0
    is the space, and a space is spoken as a pause. No exception, no warning,
    no missing audio — just a word that comes out wrong. In a report that is
    indistinguishable from IndicF5 being unable to say the word at all, which
    is the one thing the probe exists to measure.

    The hyphen in फोर्टी-सेवन and थर्टी-वन is the specific character to watch:
    no Devanagari line in any existing fixture contains one, so nothing
    measured so far says whether it is in vocab. If it is not, it degrades to
    the spaced form, which is benign — but that is worth knowing rather than
    assuming.

    Run before synthesis. Two seconds, no GPU.

        import colab.vocab_check as vc
        vc.check_arms()
    """
    vocab = set(read_vocab(find_vocab()))

    sentences = json.loads(
        (FIXTURES / "sentences" / "fixture7.json").read_text(encoding="utf-8"))
    english = {s["id"]: s["text"] for s in sentences["sentences"]}

    texts = [(f"latin[{i}]", english[i]) for i in sorted(english)]
    for arm in arms:
        data = json.loads((FIXTURES / "xlit" / f"{arm}.json")
                          .read_text(encoding="utf-8"))
        texts += [(f"{arm}[{s['id']}]", s["devanagari"])
                  for s in data["sentences"]]

    gaps = {}
    for label, text in texts:
        missing, _ = report(label, text, vocab)
        if missing:
            gaps[label] = missing

    print("\n" + "=" * 70)
    if not gaps:
        print("Every arm tokenizes fully. Whatever the probe hears is the")
        print("model's handling of the text, not a vocabulary gap.")
    else:
        print("MISSING TOKENS — do not synthesize these rows yet:")
        for label, missing in gaps.items():
            print(f"    {label:<16} {missing} token(s) map to index 0, the space")
        print("A row listed here cannot be scored for content: the pause it")
        print("produces looks exactly like the model failing at the word.")
    return gaps
