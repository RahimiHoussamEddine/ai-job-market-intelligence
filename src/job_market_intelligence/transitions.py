"""Frozen adjacent-occupation scores and serving neighbor materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pandas as pd

from .retrieval import RetrievalError, combine_component_scores, rank_score_matrix

MAX_STORED_TRANSITIONS: Final = 250


class TransitionError(ValueError):
    """Raised when transition component matrices are misaligned or unsafe."""


def _validated_components(
    occupation_codes: Sequence[str],
    lexical_scores: np.ndarray,
    lsa_scores: np.ndarray,
    structured_scores: np.ndarray,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    codes = tuple(str(code) for code in occupation_codes)
    if not codes or len(codes) != len(set(codes)):
        raise TransitionError("occupation code order must be nonempty and unique")
    expected = (len(codes), len(codes))
    components = {
        "lexical": np.asarray(lexical_scores, dtype=np.float64),
        "lsa": np.asarray(lsa_scores, dtype=np.float64),
        "structured": np.asarray(structured_scores, dtype=np.float64),
    }
    for name, matrix in components.items():
        if matrix.shape != expected:
            raise TransitionError(f"{name} score shape {matrix.shape} differs from {expected}")
        if name != "structured" and not np.isfinite(matrix).all():
            raise TransitionError(f"{name} component contains missing/non-finite scores")
        if np.isinf(matrix).any():
            raise TransitionError(f"{name} component contains infinite scores")
    return codes, components


def combine_transition_scores(
    occupation_codes: Sequence[str],
    lexical_scores: np.ndarray,
    lsa_scores: np.ndarray,
    structured_scores: np.ndarray,
    *,
    weights: Mapping[str, float],
) -> np.ndarray:
    """Apply a frozen interpretable weight vector without missing-value imputation."""

    _, components = _validated_components(
        occupation_codes, lexical_scores, lsa_scores, structured_scores
    )
    try:
        return combine_component_scores(components, weights)
    except RetrievalError as exc:
        raise TransitionError(str(exc)) from exc


def _component_text(name: str, value: float | None, weight: float) -> str:
    if value is None or not np.isfinite(value):
        return f"{name}=unavailable (weight renormalized)"
    return f"{name}={value:.3f} at weight {weight:.2f}"


def build_transition_candidates(
    occupation_codes: Sequence[str],
    lexical_scores: np.ndarray,
    lsa_scores: np.ndarray,
    structured_scores: np.ndarray,
    development_distances: np.ndarray,
    *,
    weights: Mapping[str, float],
    limit: int = MAX_STORED_TRANSITIONS,
) -> pd.DataFrame:
    """Materialize up to 250 deterministic neighbors per source occupation."""

    if isinstance(limit, bool) or not 1 <= limit <= MAX_STORED_TRANSITIONS:
        raise TransitionError(f"limit must be between 1 and {MAX_STORED_TRANSITIONS}")
    codes, components = _validated_components(
        occupation_codes, lexical_scores, lsa_scores, structured_scores
    )
    distances = np.asarray(development_distances, dtype=np.float64)
    if distances.shape != (len(codes), len(codes)) or np.isinf(distances).any():
        raise TransitionError("development-distance matrix is misaligned or infinite")
    if np.any(distances[np.isfinite(distances)] < 0):
        raise TransitionError("development distances cannot be negative")
    # Serving exposes structured similarity and development distance as one
    # evidence pair.  If either side lacks enough jointly observed transferable
    # ratings, neither is used or displayed for that pair.
    structured = components["structured"].copy()
    pair_available = np.isfinite(structured) & np.isfinite(distances)
    structured[~pair_available] = np.nan
    distances = distances.copy()
    distances[~pair_available] = np.nan
    components["structured"] = structured
    try:
        total = combine_component_scores(components, weights)
        ranked = rank_score_matrix(codes, codes, total, k=limit, exclude_self=True)
    except RetrievalError as exc:
        raise TransitionError(str(exc)) from exc
    if ranked.empty:
        raise TransitionError("transition candidate build produced no finite neighbors")
    code_index = {code: index for index, code in enumerate(codes)}
    rows: list[dict[str, object]] = []
    normalized_weights = {name: float(weights[name]) for name in ("lexical", "lsa", "structured")}
    for row in ranked.itertuples(index=False):
        query_index = code_index[str(row.query_code)]
        destination_index = code_index[str(row.candidate_code)]
        lexical = float(components["lexical"][query_index, destination_index])
        lsa = float(components["lsa"][query_index, destination_index])
        structured_raw = float(components["structured"][query_index, destination_index])
        structured = structured_raw if np.isfinite(structured_raw) else None
        distance_raw = float(distances[query_index, destination_index])
        distance = distance_raw if np.isfinite(distance_raw) else None
        explanation = "; ".join(
            (
                _component_text("lexical", lexical, normalized_weights["lexical"]),
                _component_text("LSA", lsa, normalized_weights["lsa"]),
                _component_text("structured", structured, normalized_weights["structured"]),
            )
        )
        rows.append(
            {
                "query_code": str(row.query_code),
                "destination_code": str(row.candidate_code),
                "rank": int(row.rank),
                "total_score": float(row.score),
                "lexical_score": lexical,
                "lsa_score": lsa,
                "structured_score": structured,
                "development_distance": distance,
                "explanation": explanation,
            }
        )
    result = pd.DataFrame(rows)
    expected_queries = set(codes)
    if set(result["query_code"]) != expected_queries:
        missing = sorted(expected_queries - set(result["query_code"]))[:5]
        raise TransitionError(f"some occupations have no finite transition candidates: {missing}")
    expected_per_query = min(limit, len(codes) - 1)
    counts = result.groupby("query_code", sort=False).size()
    if bool(counts.ne(expected_per_query).any()):
        raise TransitionError(
            "selected transition representation cannot produce the exact stored "
            f"candidate count ({expected_per_query}) for every occupation"
        )
    return result.sort_values(["query_code", "rank"], kind="stable").reset_index(drop=True)


def same_major_soc_diagnostic(
    rankings: pd.DataFrame,
    occupation_metadata: pd.DataFrame,
    *,
    k: int = 10,
) -> float:
    """Descriptive sanity statistic; this function never returns model scores."""

    required_rankings = {"query_code", "candidate_code", "rank"}
    required_metadata = {"onet_soc_code", "soc_major_group"}
    if missing := sorted(required_rankings - set(rankings.columns)):
        raise TransitionError(f"rankings are missing columns: {missing}")
    if missing := sorted(required_metadata - set(occupation_metadata.columns)):
        raise TransitionError(f"occupation metadata are missing columns: {missing}")
    if k < 1:
        raise TransitionError("k must be positive")
    majors = occupation_metadata.set_index("onet_soc_code")["soc_major_group"].astype(str)
    top = rankings.loc[rankings["rank"].le(k)].copy()
    top["query_major"] = top["query_code"].map(majors)
    top["candidate_major"] = top["candidate_code"].map(majors)
    if top.empty or bool(top[["query_major", "candidate_major"]].isna().any().any()):
        raise TransitionError("diagnostic rankings reference missing occupation metadata")
    return float(top["query_major"].eq(top["candidate_major"]).mean())


__all__ = [
    "MAX_STORED_TRANSITIONS",
    "TransitionError",
    "build_transition_candidates",
    "combine_transition_scores",
    "same_major_soc_diagnostic",
]
