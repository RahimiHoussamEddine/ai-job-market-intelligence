"""Frozen latent-semantic representation over leakage-safe lexical features."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_is_fitted

from .lexical import Analyzer, TfidfIndex


class EmbeddingError(ValueError):
    """Raised when an LSA candidate cannot satisfy its declared dimension."""


class LSAEmbedder(TransformerMixin, BaseEstimator):
    """Deterministic TruncatedSVD followed by row-wise L2 normalization."""

    def __init__(self, n_components: int, *, random_state: int = 20260814) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, matrix: sparse.spmatrix | np.ndarray, y: object = None) -> LSAEmbedder:
        validated = _validated_matrix(matrix)
        self._validate_dimension(validated.shape)
        self.svd_ = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=self.random_state,
        )
        self.svd_.fit(validated)
        self.n_features_in_ = int(validated.shape[1])
        self.explained_variance_ratio_sum_ = float(
            np.sum(self.svd_.explained_variance_ratio_, dtype=np.float64)
        )
        return self

    def fit_transform(
        self,
        matrix: sparse.spmatrix | np.ndarray,
        y: object = None,
        **fit_params: object,
    ) -> np.ndarray:
        if fit_params:
            raise EmbeddingError("unexpected fit parameters")
        validated = _validated_matrix(matrix)
        self._validate_dimension(validated.shape)
        self.svd_ = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=self.random_state,
        )
        transformed = self.svd_.fit_transform(validated)
        self.n_features_in_ = int(validated.shape[1])
        self.explained_variance_ratio_sum_ = float(
            np.sum(self.svd_.explained_variance_ratio_, dtype=np.float64)
        )
        return _normalize_dense(transformed)

    def transform(self, matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("svd_", "n_features_in_"))
        validated = _validated_matrix(matrix)
        if validated.shape[1] != self.n_features_in_:
            raise EmbeddingError(
                f"matrix has {validated.shape[1]} features; expected {self.n_features_in_}"
            )
        return _normalize_dense(self.svd_.transform(validated))

    def inverse_transform(self, embedded: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("svd_", "n_features_in_"))
        values = check_array(embedded, ensure_2d=True, dtype=np.float64)
        if values.shape[1] != self.n_components:
            raise EmbeddingError(
                f"embedded matrix has {values.shape[1]} columns; expected {self.n_components}"
            )
        return np.asarray(self.svd_.inverse_transform(values), dtype=np.float64)

    def _validate_dimension(self, shape: tuple[int, int]) -> None:
        if isinstance(self.n_components, bool) or self.n_components < 1:
            raise EmbeddingError("n_components must be a positive integer")
        if shape[1] <= self.n_components:
            raise EmbeddingError(
                f"candidate dimension {self.n_components} requires more than "
                f"{self.n_components} fitted lexical features; got {shape[1]}"
            )
        if shape[0] < 2:
            raise EmbeddingError("LSA requires at least two discovery documents")

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "svd_") and hasattr(self, "n_features_in_")


class TextLSAIndex:
    """Serializable raw-text → fitted TF-IDF → unit-normalized LSA pipeline."""

    __slots__ = (
        "analyzer",
        "embedder_",
        "min_df",
        "n_components",
        "n_features_out_",
        "random_state",
        "vectorizer_",
    )

    def __init__(
        self,
        n_components: int,
        *,
        analyzer: Analyzer = "word_char",
        min_df: int = 2,
        random_state: int = 20260814,
    ) -> None:
        self.n_components = n_components
        self.analyzer = analyzer
        self.min_df = min_df
        self.random_state = random_state

    def fit(self, raw_documents: list[object]) -> TextLSAIndex:
        self.vectorizer_ = TfidfIndex(
            analyzer=self.analyzer,
            min_df=self.min_df,
            dtype=np.float64,
        )
        lexical = self.vectorizer_.fit_transform(raw_documents)
        self.embedder_ = LSAEmbedder(
            self.n_components,
            random_state=self.random_state,
        )
        self.embedder_.fit(lexical)
        self.n_features_out_ = self.n_components
        return self

    def fit_transform(self, raw_documents: list[object]) -> np.ndarray:
        self.vectorizer_ = TfidfIndex(
            analyzer=self.analyzer,
            min_df=self.min_df,
            dtype=np.float64,
        )
        lexical = self.vectorizer_.fit_transform(raw_documents)
        self.embedder_ = LSAEmbedder(
            self.n_components,
            random_state=self.random_state,
        )
        result = self.embedder_.fit_transform(lexical)
        self.n_features_out_ = self.n_components
        return result

    def transform(self, raw_documents: list[object]) -> np.ndarray:
        if not hasattr(self, "vectorizer_") or not hasattr(self, "embedder_"):
            raise EmbeddingError("text LSA index is not fitted")
        return self.embedder_.transform(self.vectorizer_.transform(raw_documents))


def _validated_matrix(
    matrix: sparse.spmatrix | np.ndarray,
) -> sparse.csr_matrix | np.ndarray:
    if sparse.issparse(matrix):
        result = sparse.csr_matrix(matrix, dtype=np.float64)
        if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
            raise EmbeddingError("matrix must be nonempty and two-dimensional")
        if not np.isfinite(result.data).all():
            raise EmbeddingError("matrix contains non-finite values")
        return result
    try:
        return check_array(
            matrix,
            ensure_2d=True,
            ensure_min_samples=1,
            ensure_min_features=1,
            dtype=np.float64,
        )
    except ValueError as exc:
        raise EmbeddingError(f"invalid matrix: {exc}") from exc


def _normalize_dense(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32)
    result = np.asarray(normalize(result, norm="l2", axis=1), dtype=np.float32)
    if not np.isfinite(result).all():
        raise EmbeddingError("LSA produced non-finite values")
    return result


def dense_cosine_scores(query_embeddings: np.ndarray, catalog_embeddings: np.ndarray) -> np.ndarray:
    """Return cosine scores, normalizing defensively at the component boundary."""

    left = _normalize_dense(check_array(query_embeddings, ensure_2d=True, dtype=np.float64))
    right = _normalize_dense(check_array(catalog_embeddings, ensure_2d=True, dtype=np.float64))
    if left.shape[1] != right.shape[1]:
        raise EmbeddingError("query and catalog embeddings have different dimensions")
    return np.asarray(left @ right.T, dtype=np.float64)


__all__ = ["EmbeddingError", "LSAEmbedder", "TextLSAIndex", "dense_cosine_scores"]
