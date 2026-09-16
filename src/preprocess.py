from __future__ import annotations
import re
import unicodedata
import regex as regex_u

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """Conservative cleaning shared by IMDB and Vietnamese transfer data.

    Intentionally does not lowercase, remove punctuation, or perform Vietnamese
    word segmentation. Those choices belong to later preprocessing experiments.
    """
    if text is None:
        return ""
    t = str(text).strip()
    t = _HTML_TAG_RE.sub(" ", t)
    t = t.replace("\u00a0", " ")
    t = regex_u.sub(r"\p{C}+", " ", t)
    t = unicodedata.normalize("NFC", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t
