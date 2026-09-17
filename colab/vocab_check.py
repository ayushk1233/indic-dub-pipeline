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

VOCAB_GLOB = ("/root/.cache/huggingface/hub/models--ai4bharat--IndicF5/"
              "snapshots/*/checkpoints/vocab.txt")


def find_vocab(pattern=VOCAB_GLOB):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no vocab.txt under {pattern} — load the model once first, or "
            "pass the path printed by load_indicf5 as `vocab :`")
    return Path(matches[-1])


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
