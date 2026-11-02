"""Deterministic sparse TF-IDF representations used by both experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from .text import normalize_text, tokenize_words


class LexicalModelError(ValueError):
    """Raised when a lexical candidate has an invalid or unfitted contract."""


Analyzer = Literal["word", "char", "word_char"]


def _documents(values: Iterable[object]) -> list[str]:
    result = [normalize_text(value) for value in values]
    if not result:
        raise LexicalModelError("at least one document is required")
    if not any(result):
        raise LexicalModelError("all documents are blank after normalization")
    return result


class TfidfIndex(TransformerMixin, BaseEstimator):
    """Serializable fitted transformer with unit-normalized CSR output.

    ``transform([query])`` is intentionally the same serving interface consumed
    by the API artifact loader.  A combined word/character representation is
    normalized after concatenation so a sparse dot product is exact cosine.
    """

    def __init__(
        self,
        *,
        analyzer: Analyzer = "word_char",
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int = 2,
        sublinear_tf: bool = True,
        dtype: type[np.floating] = np.float64,
    ) -> None:
        self.analyzer = analyzer
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.dtype = dtype

    def _build_transformer(self) -> TfidfVectorizer | FeatureUnion:
        if self.analyzer not in {"word", "char", "word_char"}:
            raise LexicalModelError(f"unsupported analyzer: {self.analyzer!r}")
        if self.min_df < 1:
            raise LexicalModelError("min_df must be positive")
        if self.word_ngram_range[0] < 1 or self.word_ngram_range[0] > self.word_ngram_range[1]:
            raise LexicalModelError("word_ngram_range is invalid")
        if self.char_ngram_range[0] < 1 or self.char_ngram_range[0] > self.char_ngram_range[1]:
            raise LexicalModelError("char_ngram_range is invalid")

        word = TfidfVectorizer(
            analyzer="word",
            tokenizer=tokenize_words,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
            dtype=self.dtype,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            preprocessor=normalize_text,
            lowercase=False,
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
            dtype=self.dtype,
        )
        if self.analyzer == "word":
            return word
        if self.analyzer == "char":
            return char
        return FeatureUnion(
            [("word", word), ("char", char)],
            n_jobs=1,
            verbose_feature_names_out=True,
        )

    def fit(self, raw_documents: Sequence[object], y: object = None) -> TfidfIndex:
        documents = _documents(raw_documents)
        self.transformer_ = self._build_transformer()
        try:
            self.transformer_.fit(documents)
        except ValueError as exc:
            raise LexicalModelError(f"TF-IDF candidate could not be fit: {exc}") from exc
        self.n_features_out_ = int(len(self.transformer_.get_feature_names_out()))
        if self.n_features_out_ < 1:
            raise LexicalModelError("TF-IDF candidate produced no features")
        return self

    def fit_transform(
        self, raw_documents: Sequence[object], y: object = None, **fit_params: object
    ) -> sparse.csr_matrix:
        if fit_params:
            raise LexicalModelError("unexpected fit parameters")
        documents = _documents(raw_documents)
        self.transformer_ = self._build_transformer()
        try:
            matrix = self.transformer_.fit_transform(documents)
        except ValueError as exc:
            raise LexicalModelError(f"TF-IDF candidate could not be fit: {exc}") from exc
        self.n_features_out_ = int(matrix.shape[1])
        return _normalized_csr(matrix, dtype=self.dtype)

    def transform(self, raw_documents: Sequence[object]) -> sparse.csr_matrix:
        check_is_fitted(self, ("transformer_", "n_features_out_"))
        documents = [normalize_text(value) for value in raw_documents]
        if not documents:
            return sparse.csr_matrix((0, self.n_features_out_), dtype=self.dtype)
        matrix = self.transformer_.transform(documents)
        if matrix.shape[1] != self.n_features_out_:
            raise LexicalModelError("fitted TF-IDF feature count changed during transform")
        return _normalized_csr(matrix, dtype=self.dtype)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, ("transformer_", "n_features_out_"))
        if input_features is not None:
            raise LexicalModelError("raw text transformers do not accept input feature names")
        return np.asarray(self.transformer_.get_feature_names_out(), dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "transformer_") and hasattr(self, "n_features_out_")


class WeightedTextTransformer:
    """Concatenate fitted unit-normalized text components using sqrt weights.

    The dot product of the resulting unit vectors equals the declared weighted
    sum of component cosine similarities, so experiment and serving semantics
    remain identical for lexical/LSA title hybrids.
    """

    __slots__ = ("components", "n_features_out_", "weights")

    def __init__(
        self,
        components: Sequence[object],
        weights: Sequence[float],
    ) -> None:
        if not components or len(components) != len(weights):
            raise LexicalModelError("hybrid components and weights must be nonempty/equal length")
        numeric = tuple(float(weight) for weight in weights)
        if any(not np.isfinite(weight) or weight <= 0 for weight in numeric):
            raise LexicalModelError("hybrid transformer weights must be finite and positive")
        if not np.isclose(sum(numeric), 1.0, atol=1e-12):
            raise LexicalModelError("hybrid transformer weights must sum to one")
        counts: list[int] = []
        for component in components:
            count = getattr(component, "n_features_out_", None)
            if not isinstance(count, int) or count < 1:
                raise LexicalModelError("hybrid components must be fitted and expose feature count")
            if not callable(getattr(component, "transform", None)):
                raise LexicalModelError("hybrid component lacks transform")
            counts.append(count)
        self.components = tuple(components)
        self.weights = numeric
        self.n_features_out_ = sum(counts)

    def transform(self, raw_documents: Sequence[object]) -> sparse.csr_matrix:
        blocks: list[sparse.csr_matrix] = []
        for component, weight in zip(self.components, self.weights, strict=True):
            transformed = component.transform(raw_documents)
            block = sparse.csr_matrix(transformed, dtype=np.float64)
            block = _normalized_csr(block, dtype=np.float64)
            blocks.append(block * np.sqrt(weight))
        matrix = sparse.hstack(blocks, format="csr", dtype=np.float64)
        return _normalized_csr(matrix, dtype=np.float64)


def _normalized_csr(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    dtype: type[np.floating] = np.float32,
) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=dtype)
    result.sum_duplicates()
    result.sort_indices()
    result = normalize(result, norm="l2", axis=1, copy=False)
    result = sparse.csr_matrix(result, dtype=dtype)
    result.sum_duplicates()
    result.sort_indices()
    return result


def cosine_scores(
    query_matrix: sparse.spmatrix,
    catalog_matrix: sparse.spmatrix,
) -> np.ndarray:
    """Return exact cosine scores for two unit-normalized sparse matrices."""

    left = _normalized_csr(query_matrix, dtype=np.float64)
    right = _normalized_csr(catalog_matrix, dtype=np.float64)
    if left.shape[1] != right.shape[1]:
        raise LexicalModelError("query and catalog matrices have different feature counts")
    values = left @ right.T
    return np.asarray(values.toarray(), dtype=np.float64)


__all__ = [
    "Analyzer",
    "LexicalModelError",
    "TfidfIndex",
    "WeightedTextTransformer",
    "cosine_scores",
]
