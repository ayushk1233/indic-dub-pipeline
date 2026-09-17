"""
The transcribe-back check scored a clip clean at 0.12 while it opened with
three seconds of speech the speaker never said. A total insertion rate cannot
see a failure that is contiguous and positional, so leading_extra measures the
prefix directly. These tests pin what it must report, in both scripts.
"""

from colab.indicf5_check import leading_extra, normalize, score_text


def test_a_clean_transcript_has_no_prefix():
    assert leading_extra("abc def", "abc def") == 0


def test_a_prefix_is_reported_in_characters():
    assert leading_extra("abc", "xyabc") == 2


def test_the_real_failure(): 
    """
    The clip the listener caught: en10_both on the 134-character sentence,
    which opened with three seconds of invented speech.
    """
    intended = ("आप इसे एक वीडियो देते हैं कोई लेक्चर कोई इंटरव्यू कुछ भी "
                "जिसमें साफ आवाज हो")
    heard = "पेंट केगे उसे " + intended

    assert leading_extra(intended, heard) == len("पेंट केगे उसे ")


def test_the_prefix_is_invisible_to_the_total_rate():
    """
    The reason this function exists: extra stays under its threshold while the
    clip is unusable.
    """
    intended = ("आप इसे एक वीडियो देते हैं कोई लेक्चर कोई इंटरव्यू कुछ भी "
                "जिसमें साफ आवाज हो और यह वही वीडियो हिंदी में बोलता हुआ वापस देता है")
    scored = score_text(intended, "पेंट केगे उसे " + intended)

    assert scored["extra"] < 0.15
    assert scored["lead_chars"] == len(normalize("पेंट केगे उसे ")) + 1


def test_trailing_speech_is_not_counted_as_a_prefix():
    """Only the start is free. Speech that runs on is already an insertion."""
    assert leading_extra("abc", "abcxyz") == 0


def test_a_prefix_and_a_suffix_report_only_the_prefix():
    assert leading_extra("abc", "xyabcz") == 2


def test_substitutions_inside_the_sentence_are_not_a_prefix():
    assert leading_extra("abcdef", "abcXef") == 0


def test_a_truncated_clip_has_no_prefix():
    assert leading_extra("abcdef", "abc") == 0


def test_an_empty_intended_text_does_not_divide_by_zero():
    assert leading_extra("", "anything") == 0
