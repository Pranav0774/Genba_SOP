"""Catch safety points and steps that say the opposite of the transcript (English).

The model sometimes flips an instruction: the speech says "be sure to attend training" and the list says
"Do not attend training". For each item, the transcript sentence with the most words in common is found;
the item is flagged when one says "do not" and the other clearly says "do" ("be sure to", "always",
"make sure"...), or the item gives a positive instruction where the sentence forbids it.

It only flags; it never changes an item. Sentences without a clear instruction ("there is a risk of...")
are never used as evidence, so a paraphrase of a warning is not flagged.
"""
import re

NEGATIVE = re.compile(r"\b(do not|don't|dont|never|must not|mustn't|should not|shouldn't|cannot|can't|"
                      r"may not|avoid|prohibited|forbidden|not allowed|no one)\b", re.IGNORECASE)
POSITIVE = re.compile(r"\b(be sure to|make sure|ensure|always|must|need to|have to|remember to|"
                      r"please|let's|be certain)\b", re.IGNORECASE)
# A sentence only counts as forbidding something when it is an instruction: "Do not ..." at the start of a
# clause, or "must not", "never", "prohibited"... Conversational "if you can't see", "I don't have" is not.
PROHIBITION = re.compile(r"(?:^|[.,;:!?]\s*|\b(?:please|and|so|but|then|also)\s+)(?:do not|don't|dont)\b|"
                         r"\b(?:never|must not|mustn't|should not|shouldn't|prohibited|forbidden|not allowed)\b",
                         re.IGNORECASE)
# Plain commands ("Use push rods when ...") also count as "do this": a sentence starting with one of these verbs.
COMMAND_VERBS = set("""use wear keep check turn cut carry perform inspect stop remove pull push select work release
replace unplug plug attend hold put lift lower raise place fasten tighten secure clean confirm make set install
attach connect disconnect switch press move store follow observe read wait stand sit look watch mark measure
maintain apply open close lock unlock adjust insert take bring return report call ask verify test""".split())
MIN_SHARED_WORDS = 2
MIN_OVERLAP = 0.6
STOPWORDS = set("""a an the and or but if when while then than to of in on at by for from with without into onto
over under as is are was were be been being it its this that these those there here you your yours one ones
he she his her they them their we our us i me my can could will would should shall may might do does did
done not no so such any all some each every very more most less other such type types""".split())
MARKER_WORDS = set("""sure make ensure always must need have remember please let lets certain never avoid
prohibited forbidden allowed dont don't cannot can't mustn't shouldn't""".split())


def _stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text):
    words = re.findall(r"[a-z][a-z']+", text.lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS and w not in MARKER_WORDS}


def transcript_sentences(raw_transcript):
    """Sentences of a '[s -> e] text' transcript; speech recognition splits sentences across segments."""
    text = re.sub(r"\[\d+(?:\.\d+)?s\s*->\s*\d+(?:\.\d+)?s\]", " ", raw_transcript)
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def best_sentence(item, sentences):
    words = _content_words(item)
    best, best_score = None, 0.0
    for sentence in sentences:
        shared = len(words & _content_words(sentence))
        score = shared / max(len(words), 1)
        if shared >= MIN_SHARED_WORDS and score > best_score:
            best, best_score = sentence, score
    return (best, best_score) if best_score >= MIN_OVERLAP else (None, best_score)


def conflict(item, sentences):
    """The transcript sentence an item contradicts, or None."""
    sentence, _ = best_sentence(item, sentences)
    if sentence is None:
        return None
    item_negative = bool(NEGATIVE.search(item))
    sentence_negative = bool(NEGATIVE.search(sentence))
    first_word = re.sub(r"^(please|always|then|and)\s+", "", sentence.strip().lower()).split(" ")[0].strip(",.")
    sentence_positive = (bool(POSITIVE.search(sentence)) or first_word in COMMAND_VERBS) and not sentence_negative
    if item_negative and sentence_positive:
        return sentence
    if not item_negative and PROHIBITION.search(sentence):
        return sentence
    return None


def conflicts(items, raw_transcript):
    """[(item, contradicted sentence)] for every flagged item."""
    sentences = transcript_sentences(raw_transcript)
    found = []
    for item in items:
        sentence = conflict(item, sentences)
        if sentence:
            found.append((item, sentence))
    return found
