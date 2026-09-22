# run with ./venv/bin/python (src.text and g2p_en live there)
from prep.xlit_freeze import normalize_en


def test_normalize_en_lowercases_spells_numbers_strips_punctuation():
    assert normalize_en("Hello, World! I have 3 cats.") == "hello world i have three cats"
