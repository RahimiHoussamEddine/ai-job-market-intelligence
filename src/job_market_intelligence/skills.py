"""Missing-aware structured similarities and directional development distance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .profiles import ProfileError, StructuredMatrix, build_structured_matrix

FORBIDDEN_STRUCTURED_COLUMNS: Final = frozenset(
    {
        "related_onet_soc_code",
        "query_onet_soc_code",
        "candidate_onet_soc_code",
        "relatedness_tier",
        "relatedness_index",
        "relevance_grade",
        "nem_code",
        "employment_2024",
        "employment_2034",
        "employment_change_2024_34",
        "employment_change_pct",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
    }
)


class StructuredFeatureError(ValueError):
    """Raised when structured retrieval inputs leak targets or lose missingness."""


@dataclass(frozen=True, slots=True)
class StructuredScores:
    occupation_codes: tuple[str, ...]
    similarity: np.ndarray
    overlap_count: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.occupation_codes), len(self.occupation_codes))
        if self.similarity.shape != expected or self.overlap_count.shape != expected:
            raise StructuredFeatureError("structured score matrices do not match catalog size")
        if self.overlap_count.dtype.kind not in {"i", "u"}:
            raise StructuredFeatureError("overlap counts must have integer dtype")


def reject_structured_leakage(frame: pd.DataFrame) -> None:
    """Reject qrel, SOC-graph, crosswalk, and BLS outcome fields."""

    normalized = {str(column).strip().casefold() for column in frame.columns}
    leaked = sorted(normalized & FORBIDDEN_STRUCTURED_COLUMNS)
    if leaked:
        raise StructuredFeatureError(
            "structured retrieval frame contains forbidden target/outcome fields: "
            + ", ".join(leaked)
        )


def _matrix_for_families(
    ratings: pd.DataFrame,
    occupation_codes: Sequence[str],
    families: Iterable[str],
) -> StructuredMatrix:
    reject_structured_leakage(ratings)
    requested = tuple(sorted(set(families)))
    if not requested:
        raise StructuredFeatureError("at least one structured feature family is required")
    required = {
        "onet_soc_code",
        "feature_family",
        "element_id",
        "scale_id",
        "normalized_value",
        "observed",
    }
    missing = sorted(required - set(ratings.columns))
    if missing:
        raise StructuredFeatureError(f"structured ratings are missing columns: {missing}")
    unknown_families = set(requested) - set(ratings["feature_family"].dropna().astype(str))
    if unknown_families:
        raise StructuredFeatureError(
            "requested structured families are absent: " + ", ".join(sorted(unknown_families))
        )
    codes = tuple(str(code) for code in occupation_codes)
    if not codes or len(codes) != len(set(codes)):
        raise StructuredFeatureError("occupation code order must be nonempty and unique")
    subset = ratings.loc[ratings["feature_family"].isin(requested)].copy()
    try:
        built = build_structured_matrix(subset, occupation_codes=codes)
    except ProfileError as exc:
        raise StructuredFeatureError(str(exc)) from exc
    if built.occupation_codes == codes:
        return built
    source_indexes = {code: index for index, code in enumerate(built.occupation_codes)}
    try:
        order = [source_indexes[code] for code in codes]
    except KeyError as exc:  # pragma: no cover - guarded by build_structured_matrix
        raise StructuredFeatureError(f"structured matrix omitted occupation {exc.args[0]}") from exc
    values = np.asarray(built.values[order], dtype=np.float32)
    observed = np.asarray(built.observed[order], dtype=np.bool_)
    values.setflags(write=False)
    observed.setflags(write=False)
    return StructuredMatrix(codes, built.feature_keys, values, observed)


def missing_aware_similarity(
    matrix: StructuredMatrix,
    *,
    minimum_overlap: int = 5,
) -> StructuredScores:
    """Compute pairwise cosine only on jointly observed dimensions.

    Cells with insufficient overlap or a zero vector remain ``NaN``.  That
    missingness is meaningful and must not be converted to a neutral score.
    """

    if isinstance(minimum_overlap, bool) or minimum_overlap < 1:
        raise StructuredFeatureError("minimum_overlap must be positive")
    count = len(matrix.occupation_codes)
    similarities = np.full((count, count), np.nan, dtype=np.float32)
    overlaps = np.zeros((count, count), dtype=np.int32)
    for left_index in range(count):
        for right_index in range(left_index, count):
            overlap = matrix.observed[left_index] & matrix.observed[right_index]
            overlap_count = int(np.count_nonzero(overlap))
            overlaps[left_index, right_index] = overlap_count
            overlaps[right_index, left_index] = overlap_count
            if overlap_count < minimum_overlap:
                continue
            left = matrix.values[left_index, overlap].astype(np.float64, copy=False)
            right = matrix.values[right_index, overlap].astype(np.float64, copy=False)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0.0:
                continue
            score = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
            similarities[left_index, right_index] = score
            similarities[right_index, left_index] = score
    similarities.setflags(write=False)
    overlaps.setflags(write=False)
    return StructuredScores(matrix.occupation_codes, similarities, overlaps)


def structured_similarity_from_ratings(
    ratings: pd.DataFrame,
    occupation_codes: Sequence[str],
    *,
    families: Iterable[str] = ("transferable_skill", "knowledge"),
    minimum_overlap: int = 5,
) -> StructuredScores:
    matrix = _matrix_for_families(ratings, occupation_codes, families)
    return missing_aware_similarity(matrix, minimum_overlap=minimum_overlap)


def positive_gap_distance_matrix(
    ratings: pd.DataFrame,
    occupation_codes: Sequence[str],
    *,
    family: str = "transferable_skill",
    minimum_overlap: int = 5,
) -> np.ndarray:
    """Directional RMS of positive target-minus-source normalized gaps.

    Row ``i``, column ``j`` answers how much higher destination ``j`` is on
    dimensions observed for both occupations.  Shared strengths and dimensions
    where the source is already higher add no development gap.  Missing or
    insufficient overlap remains ``NaN``.
    """

    if isinstance(minimum_overlap, bool) or minimum_overlap < 1:
        raise StructuredFeatureError("minimum_overlap must be positive")
    matrix = _matrix_for_families(ratings, occupation_codes, (family,))
    count = len(matrix.occupation_codes)
    distances = np.full((count, count), np.nan, dtype=np.float32)
    for source_index in range(count):
        for target_index in range(count):
            overlap = matrix.observed[source_index] & matrix.observed[target_index]
            if int(np.count_nonzero(overlap)) < minimum_overlap:
                continue
            source = matrix.values[source_index, overlap].astype(np.float64, copy=False)
            target = matrix.values[target_index, overlap].astype(np.float64, copy=False)
            positive_gap = np.maximum(target - source, 0.0)
            distances[source_index, target_index] = float(
                np.sqrt(np.mean(np.square(positive_gap), dtype=np.float64))
            )
    distances.setflags(write=False)
    return distances


class StructuredSimilarityIndex:
    """Immutable code-indexed structured component for experiment orchestration."""

    __slots__ = (
        "occupation_codes",
        "code_to_index",
        "scores",
        "development_distance",
        "minimum_overlap",
    )

    def __init__(
        self,
        ratings: pd.DataFrame,
        occupation_codes: Sequence[str],
        *,
        minimum_overlap: int = 5,
    ) -> None:
        codes = tuple(str(code) for code in occupation_codes)
        if not codes or len(codes) != len(set(codes)):
            raise StructuredFeatureError("occupation code order must be nonempty and unique")
        structured = structured_similarity_from_ratings(
            ratings,
            codes,
            minimum_overlap=minimum_overlap,
        )
        development = positive_gap_distance_matrix(
            ratings,
            codes,
            family="transferable_skill",
            minimum_overlap=minimum_overlap,
        )
        # Structured similarity and the directional skill-gap explanation are
        # one frozen evidence pair.  Apply this validity mask here so locked
        # evaluation, production refitting, and serving materialization all
        # rank with identical missingness semantics.
        masked_scores = np.asarray(structured.similarity, dtype=np.float32).copy()
        masked_development = np.asarray(development, dtype=np.float32).copy()
        pair_available = np.isfinite(masked_scores) & np.isfinite(masked_development)
        masked_scores[~pair_available] = np.nan
        masked_development[~pair_available] = np.nan
        masked_scores.setflags(write=False)
        masked_development.setflags(write=False)
        self.occupation_codes = codes
        self.code_to_index = {code: index for index, code in enumerate(codes)}
        self.scores = masked_scores
        self.development_distance = masked_development
        self.minimum_overlap = minimum_overlap

    def row(self, query_code: str) -> np.ndarray:
        try:
            return self.scores[self.code_to_index[query_code]]
        except KeyError as exc:
            raise StructuredFeatureError(f"unknown occupation code: {query_code}") from exc

    def distance_row(self, source_code: str) -> np.ndarray:
        try:
            return self.development_distance[self.code_to_index[source_code]]
        except KeyError as exc:
            raise StructuredFeatureError(f"unknown occupation code: {source_code}") from exc


__all__ = [
    "FORBIDDEN_STRUCTURED_COLUMNS",
    "StructuredFeatureError",
    "StructuredScores",
    "StructuredSimilarityIndex",
    "missing_aware_similarity",
    "positive_gap_distance_matrix",
    "reject_structured_leakage",
    "structured_similarity_from_ratings",
]
