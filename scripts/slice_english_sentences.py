"""
Cut the speaker's real English reading into one clip per fixture7 sentence.

The machinery moved to scripts/slice_sentences.py when hi -> hi needed the
same three outputs from the Hindi take of the same paragraph. This module is
the English configuration of it and the name every existing reference uses —
FINDINGS §4c, TRANSLITERATION_PLAN, and the refusal message in
colab/indicf5_xlit_probe.py all name it. The re-exports below are what
tests/test_slice_english_sentences.py imports.

    ./venv/bin/python -m scripts.slice_english_sentences
"""

from scripts.slice_sentences import (  # noqa: F401  (re-exported)
    EN,
    MIN_MATCH,
    PAD_S,
    _key,
    locate,
    reference_window,
    run,
    words_of,
)

TAKE = EN
SOURCE = EN.source
SENTENCES = EN.sentences
OUTPUT_DIR = EN.output_dir
SLOTS = EN.slots
LANGUAGE = EN.language


def main() -> None:
    run(EN)


if __name__ == "__main__":
    main()
