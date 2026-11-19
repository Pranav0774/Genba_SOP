"""Translate reviewed SOP text between Japanese and English with the local Ollama model.

Machine translation from a 7B model is a draft: the app shows every translation for a reviewer to
correct before the second document is built. Checks here only reject output that is plainly wrong
(wrong script, Chinese characters, English words inside Japanese)."""
import os
import re
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 300
GLOSSARY_PATH = r"D:\genba-sop\reference\glossary.txt"

VERIFY_MARK_JA = "（要確認）"
VERIFY_MARK_EN = " (please verify)"

PROMPTS = {
    ("Japanese", "English", "sentence"): """Translate this Japanese workplace training instruction into one short, clear English sentence, in the imperative where it is an instruction. Use only English words. The text comes from speech recognition and may contain small errors: translate what it most plausibly means and do not add details that are not in it. Output only the translation.
{hints}
{text}
""",
    ("Japanese", "English", "term"): """Translate this Japanese name of a tool, machine or material into its usual English name: a short noun phrase, not a sentence. Output only the name.
{hints}
{text}
""",
    ("English", "Japanese", "sentence"): """Translate this English workplace training instruction into one short, natural Japanese sentence in the plain instruction form (for example "〜する" or "〜してください"). Write in Japanese only: kanji and kana as used in Japan, never Chinese characters that Japanese does not use. Write loanwords in katakana (for example ドリル). Do not leave any English words. Output only the translation.

Examples:
Mark screw locations -> ネジの位置に印を付ける
Apply glue to each joint -> 各接合部に接着剤を塗る
{hints}
{text}
""",
    ("English", "Japanese", "term"): """Translate this English name of a tool, machine or material into its usual Japanese name: a short noun phrase, not a sentence. Write loanwords in katakana. Output only the name.
{hints}
{text}
""",
}

# added to the prompt when an earlier answer was rejected (the model slips into Chinese or mixes scripts)
RETRY_NOTES = {
    "English": "IMPORTANT: answer in English letters only. No Chinese, no Japanese characters.\n",
    "Japanese": "IMPORTANT: answer in Japanese only (kana and kanji used in Japan). No English words, no Chinese characters.\n",
}

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_DIMENSIONS = re.compile(r"\d+\s*[x×]\s*\d+")
_LOWERCASE_LATIN = re.compile(r"[a-z]")
_UPPERCASE_WORD = re.compile(r"[A-Z]{2,}")
_SPACED_CJK = re.compile(r"[぀-ヿ㐀-鿿] (?=[぀-ヿ㐀-鿿])")

_glossary = None


def glossary():
    """[(Japanese, English)] from reference/glossary.txt."""
    global _glossary
    if _glossary is None:
        _glossary = []
        if os.path.exists(GLOSSARY_PATH):
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if "|" in line and not line.lstrip().startswith("#"):
                        ja, en = (part.strip() for part in line.split("|", 1))
                        if ja and en:
                            _glossary.append((ja, en))
    return _glossary


def glossary_hints(text, source):
    """The glossary pairs whose source-language term appears in the text, as prompt lines."""
    found = []
    for ja, en in glossary():
        term, other = (ja, en) if source == "Japanese" else (en, ja)
        if source == "Japanese":
            hit = term in text
        else:  # whole English words only, so "jack" does not match "Jackery"
            hit = re.search(r"\b" + re.escape(term) + r"s?\b", text, re.IGNORECASE) is not None
        if hit:
            found.append(f"- {term} = {other}")
    if not found:
        return ""
    return "Use these terms where they apply:\n" + "\n".join(found[:8]) + "\n"


def _valid(translation, target, kind, source_text=""):
    if not translation:
        return False
    if target == "English":
        return not _CJK.search(translation)
    from japanese_text import non_japanese_characters

    if non_japanese_characters(translation):
        return False
    if _LOWERCASE_LATIN.search(_DIMENSIONS.sub("", translation)):
        return False  # English words left inside the Japanese
    if any(word not in source_text for word in _UPPERCASE_WORD.findall(translation)):
        return False  # "PEOPLE 方向に..." - a capitalised English word the source does not contain
    if len(_SPACED_CJK.findall(translation)) >= 2:
        return False  # "方 向 に 插 電 源" - characters spaced out one by one, a broken answer
    return kind == "term" or bool(_KANA.search(translation))


def is_acceptable(translation, source_text, target, kind="sentence"):
    """Whether a saved translation would pass today's checks (the review mark is ignored)."""
    plain = (translation or "").replace(VERIFY_MARK_JA, "").replace(VERIFY_MARK_EN, "").strip()
    source = source_text.replace(VERIFY_MARK_JA, "").replace("要確認", "").strip()
    return _valid(plain, target, kind, source)


def _ask(prompt, seed):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": seed, "num_predict": 200, "num_ctx": 2048},
    }
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(10 * attempt)
    raise last_error


def translate_line(text, source, target, kind="sentence"):
    """Returns (translation, ok). ok is False when no acceptable translation came back; the
    translation is then empty and a reviewer should supply one. A 要確認 mark is carried over."""
    verify = "要確認" in text
    plain = text.replace(VERIFY_MARK_JA, "").replace("要確認", "").strip()
    if not plain:
        return "", True
    prompt = PROMPTS[(source, target, kind)].format(text=plain, hints=glossary_hints(plain, source))
    for attempt in range(5):
        retry_note = RETRY_NOTES[target] if attempt else ""
        raw = _ask(retry_note + prompt, seed=7 + attempt)
        translation = raw.split("\n")[0].strip().strip('"「」')
        if target == "Japanese":
            from japanese_text import to_japanese_kanji

            translation = to_japanese_kanji(translation)
        if _valid(translation, target, kind, plain):
            if verify:
                translation += VERIFY_MARK_JA if target == "Japanese" else VERIFY_MARK_EN
            return translation, True
    return "", False


def translate_many(items, source, target, cache, progress=None):
    """items: [(text, kind)]. Each distinct text is translated once. cache maps text -> translation and
    is updated, so edited or already-translated lines are not translated again. Returns the failed texts."""
    failed = []
    todo = [(t, k) for t, k in dict.fromkeys(items) if t and t not in cache]
    for i, (text, kind) in enumerate(todo, 1):
        translation, ok = translate_line(text, source, target, kind)
        cache[text] = translation
        if not ok:
            failed.append(text)
        if progress:
            progress(i, len(todo))
    return failed
