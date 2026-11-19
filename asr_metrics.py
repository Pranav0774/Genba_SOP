"""Shared helpers for Japanese speech-recognition checks: text normalisation and character error counts."""
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# Punctuation and spacing are ignored, as in cer_check.py. The long-vowel mark ー is a real character and is kept.
PUNCTUATION = re.compile(r"[\s、。，．,.!！?？「」『』（）()\[\]【】・…‥:：;；\"'“”‘’〜~]")
TRANSCRIPT_STAMP = re.compile(r"\[\d+(?:\.\d+)?s\s*->\s*\d+(?:\.\d+)?s\]")
GROUND_TRUTH_STAMP = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*[–\-~]\s*\d{1,2}:\d{2}(?::\d{2})?)?\]")


def normalize(text):
    """Text as compared for CER: NFKC (full-width -> half-width), lower case, no punctuation or spaces."""
    return PUNCTUATION.sub("", unicodedata.normalize("NFKC", text).lower())


def clean_file_text(text):
    """Remove headers and time stamps from a transcript or ground-truth file (same rules as cer_check.py,
    plus [MM:SS–MM:SS] range stamps, which cer_check.py leaves in)."""
    text = re.sub(r"^(TIMESTAMPED|Source:|TRANSCRIPT\s*$).*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Speaker \d+ \(\d{2}:\d{2}\)", "", text)
    text = TRANSCRIPT_STAMP.sub("", text)
    text = GROUND_TRUTH_STAMP.sub("", text)
    return text


def char_errors(reference, hypothesis):
    """(substitutions, deletions, insertions) turning the normalised reference into the hypothesis."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for op in Levenshtein.editops(ref, hyp):
        counts[op.tag] += 1
    return counts["replace"], counts["delete"], counts["insert"]


def cer(reference, hypothesis):
    s, d, i = char_errors(reference, hypothesis)
    return (s + d + i) / max(len(normalize(reference)), 1)


def disagreement(a, b):
    """How different two transcripts of the same audio are, 0 (same) to 1, relative to the longer one."""
    a, b = normalize(a), normalize(b)
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)
