"""Deterministic text normalization and tokenization for retrieval features.

The retrieval pipeline deliberately keeps punctuation that is semantic inside
technical names.  In particular, generic ``\\w+`` tokenizers silently turn
``C++``, ``C#`` and ``.NET`` into misleading fragments and commonly drop the
single-letter language ``R``.  The helpers here are top-level callables so a
fitted scikit-learn vectorizer remains serializable.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from typing import Final

PRESERVED_TOKENS: Final[tuple[str, ...]] = ("C++", "C#", ".NET", "R", "Go")

_WHITESPACE = re.compile(r"\s+")
_WORD_TOKEN = re.compile(
    r"(?iu)"
    r"(?<![\w+#.])(?:c\+\+|c\#|\.net|r|go)(?![\w+#])"
    r"|[^\W_]+(?:['’-][^\W_]+)*"
    r"|\d+(?:[.,]\d+)*"
)


class TextNormalizationError(ValueError):
    """Raised when text cannot be normalized deterministically."""


def _missing(value: object) -> bool:
    if value is None:
        return True
    value_type = type(value)
    # Importing pandas here would make the serving vectorizer's smallest module
    # unnecessarily heavy.  Both sentinels have stable scalar types and must be
    # caught before any boolean coercion (``bool(pd.NA)`` is deliberately an
    # error).
    if value_type.__module__.startswith("pandas.") and value_type.__name__ in {
        "NAType",
        "NaTType",
    }:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    # Supports numpy floating scalars without importing numpy in this small,
    # frequently imported module.
    if value_type.__module__.startswith("numpy") and value_type.__name__.startswith("float"):
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            return False
    if value_type.__module__.startswith("numpy") and value_type.__name__ in {
        "datetime64",
        "timedelta64",
    }:
        return str(value) == "NaT"
    return False


def normalize_text(value: object, *, casefold: bool = True) -> str:
    """Return stable NFKC, entity and whitespace-normalized text.

    Punctuation is not stripped, so the configured technical terms survive
    intact.  NFKC is applied before case folding; full-width ``Ｃ＋＋`` therefore
    becomes ``c++`` in retrieval text.  Set ``casefold=False`` for display text.
    ``None`` and floating missing values normalize to the empty string.
    """

    if _missing(value):
        return ""
    try:
        text = html.unescape(str(value))
    except Exception as exc:  # pragma: no cover - exceptionally defensive
        raise TextNormalizationError(f"Could not convert {value!r} to text") from exc

    text = unicodedata.normalize("NFKC", text)
    # Format controls (BOM, zero-width joiners/spaces, bidi marks) are invisible
    # and can otherwise make byte-identical-looking inputs produce new features.
    text = "".join(character for character in text if unicodedata.category(character) != "Cf")
    text = _WHITESPACE.sub(" ", text).strip()
    return text.casefold() if casefold else text


def tokenize_words(value: object) -> list[str]:
    """Tokenize normalized text while retaining the five technical tokens.

    Returned tokens are case-folded.  The observable preserved forms are
    ``c++``, ``c#``, ``.net``, ``r`` and ``go``; punctuation outside terms is
    only a boundary and never becomes a token.
    """

    return [match.group(0) for match in _WORD_TOKEN.finditer(normalize_text(value))]


def normalized_join(*parts: object, separator: str = " ") -> str:
    """Normalize nonblank parts and join them without introducing empty fields."""

    normalized = [normalize_text(part) for part in parts]
    return separator.join(part for part in normalized if part)


__all__ = [
    "PRESERVED_TOKENS",
    "TextNormalizationError",
    "normalize_text",
    "normalized_join",
    "tokenize_words",
]
