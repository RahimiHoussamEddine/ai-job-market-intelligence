from __future__ import annotations

import math

import numpy as np
import pandas as pd

from job_market_intelligence.text import normalize_text, normalized_join, tokenize_words


def test_normalize_text_is_nfkc_casefolded_and_whitespace_stable() -> None:
    value = " \ufeffＣ＋＋\u00a0Developer\t&AMP;\nR Engineer "

    assert normalize_text(value) == "c++ developer & r engineer"
    assert normalize_text(value, casefold=False) == "C++ Developer & R Engineer"
    assert normalize_text(None) == ""
    assert normalize_text(math.nan) == ""
    assert normalize_text(pd.NA) == ""
    assert normalize_text(pd.NaT) == ""
    assert normalize_text(np.datetime64("NaT", "ns")) == ""


def test_tokenizer_preserves_protocol_technical_tokens() -> None:
    tokens = tokenize_words("C++ / C# / .NET / R / Go developers")

    assert tokens == ["c++", "c#", ".net", "r", "go", "developers"]


def test_normalized_join_skips_blank_parts() -> None:
    assert normalized_join("  Data  ", None, "ＳＣＩＥＮＣＥ") == "data science"
