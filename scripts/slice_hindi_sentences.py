"""
Cut the speaker's real Hindi reading into one clip per fixture7_hi sentence.

The Hindi configuration of scripts/slice_sentences.py. Same speaker, same
paragraph, same session as the English take, so the hi -> hi probe gets slots,
a same-language identity ceiling and a Whisper content floor from tape that
already exists.

Two things differ from the English run and both matter:

  - **The Whisper floor should be expected to sit higher.** FINDINGS §12 puts
    word error at 7.9% normalized on his English. Whisper is weaker on Hindi,
    so a Hindi synthesis CER must be read against the Hindi floor and never
    against the English one or against zero.
  - **Characters per second is not comparable across the two takes.**
    Devanagari writes a syllable in fewer characters than Latin does, so the
    Hindi rate is lower for reasons that have nothing to do with how fast he
    speaks. slots.json records natural_cps per take so the comparison stays
    inside one script.

    ./venv/bin/python -m scripts.slice_hindi_sentences
"""

from scripts.slice_sentences import HI, run

TAKE = HI
SOURCE = HI.source
SENTENCES = HI.sentences
OUTPUT_DIR = HI.output_dir
SLOTS = HI.slots
LANGUAGE = HI.language


def main() -> None:
    run(HI)


if __name__ == "__main__":
    main()
