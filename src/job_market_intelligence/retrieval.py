"""Leakage-safe query construction, score composition, and deterministic ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import sparse

from .lexical import TfidfIndex, cosine_scores
from .ranking_metrics import deterministic_rank
from .text import normalize_text, normalized_join

ALIAS_QUERY_COLUMNS: Final = (
    "query_id",
    "query_code",
    "target_code",
    "query_text",
    "normalized_query",
    "soc_major_group",
    "fold",
    "role",
)


class RetrievalError(ValueError):
    """Raised when a retrieval input could leak a held-out query or target."""


@dataclass(frozen=True, slots=True)
class TitleDocumentSet:
    documents: pd.DataFrame
    included_alias_keys: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        expected = ["onet_soc_code", "canonical_title", "title_text", "indexed_alias_count"]
        if list(self.documents.columns) != expected:
            raise RetrievalError(f"title document columns must be {expected!r}")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RetrievalError(f"{context} is missing columns: {missing}")


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def prepare_alias_queries(
    aliases: pd.DataFrame,
    occupations: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    maximum_per_occupation: int = 25,
    seed: int = 20260814,
    allowed_query_codes: Iterable[str] | None = None,
    ambiguity_scope: str = "allowed_codes",
) -> pd.DataFrame:
    """Construct deterministic, unambiguous, held-out alternate-title queries.

    Development roles use ``allowed_codes`` so fold-zero aliases cannot
    causally remove a development query. The locked test uses ``full_catalog``:
    ambiguity is established across every occupation before the requested test
    queries are retained.
    """

    if isinstance(maximum_per_occupation, bool) or maximum_per_occupation < 1:
        raise RetrievalError("maximum_per_occupation must be positive")
    _require_columns(
        aliases,
        ("onet_soc_code", "job_title", "normalized_job_title"),
        "aliases",
    )
    _require_columns(
        occupations,
        ("onet_soc_code", "canonical_title", "soc_major_group"),
        "occupations",
    )
    _require_columns(
        fold_manifest,
        ("onet_soc_code", "fold", "role"),
        "fold manifest",
    )
    if occupations.empty or bool(occupations["onet_soc_code"].duplicated().any()):
        raise RetrievalError("occupations must be nonempty and unique by code")
    if bool(fold_manifest["onet_soc_code"].duplicated().any()):
        raise RetrievalError("fold manifest must be unique by occupation code")

    occupation_lookup = occupations.set_index("onet_soc_code")
    manifest_lookup = fold_manifest.set_index("onet_soc_code")
    unknown_alias_codes = set(aliases["onet_soc_code"].astype(str)) - set(occupation_lookup.index)
    if unknown_alias_codes:
        raise RetrievalError(
            "aliases reference unknown occupation codes: "
            + ", ".join(sorted(unknown_alias_codes)[:5])
        )
    if set(occupation_lookup.index) != set(manifest_lookup.index):
        raise RetrievalError("fold manifest and occupation catalog have different code coverage")
    allowed = (
        set(occupation_lookup.index)
        if allowed_query_codes is None
        else {str(code) for code in allowed_query_codes}
    )
    if unknown := allowed - set(occupation_lookup.index):
        raise RetrievalError(
            "allowed alias query codes are outside the catalog: " + ", ".join(sorted(unknown)[:5])
        )
    if not allowed:
        raise RetrievalError("allowed alias query codes cannot be empty")
    if ambiguity_scope not in {"allowed_codes", "full_catalog"}:
        raise RetrievalError("ambiguity_scope must be either 'allowed_codes' or 'full_catalog'")
    ambiguity_codes = set(occupation_lookup.index) if ambiguity_scope == "full_catalog" else allowed

    rows: list[dict[str, object]] = []
    for row in aliases.itertuples(index=False):
        code = str(row.onet_soc_code)
        # Development filters before ambiguity checks so a held-out alias
        # cannot causally remove a development query. Locked evaluation uses
        # the full catalog here and filters to test codes after ambiguity.
        if code not in ambiguity_codes:
            continue
        display = normalize_text(row.job_title, casefold=False)
        normalized = normalize_text(row.normalized_job_title)
        if not display or not normalized:
            raise RetrievalError(f"alias for {code} is blank after normalization")
        canonical = normalize_text(occupation_lookup.at[code, "canonical_title"])
        # A canonical-title echo is not a discovery query.
        if normalized == canonical:
            continue
        rows.append(
            {
                "query_code": code,
                "target_code": code,
                "query_text": display,
                "normalized_query": normalized,
            }
        )
    work = pd.DataFrame(rows)
    if work.empty:
        raise RetrievalError("no usable alternate-title queries remain")
    # One deterministic display row per normalized alias/occupation.
    work = work.sort_values(
        ["query_code", "normalized_query", "query_text"], kind="stable"
    ).drop_duplicates(["query_code", "normalized_query"], keep="first")
    ambiguity = work.groupby("normalized_query", sort=False)["query_code"].nunique()
    ambiguous = set(ambiguity[ambiguity.gt(1)].index)
    work = work.loc[~work["normalized_query"].isin(ambiguous)].copy()
    work = work.loc[work["query_code"].isin(allowed)].copy()
    if work.empty:
        raise RetrievalError("no requested alternate titles remain after ambiguity filtering")
    work["_selection_hash"] = [
        _stable_hash(seed, row.query_code, row.normalized_query)
        for row in work.itertuples(index=False)
    ]
    work = (
        work.sort_values(["query_code", "_selection_hash", "normalized_query"], kind="stable")
        .groupby("query_code", sort=False, group_keys=False)
        .head(maximum_per_occupation)
        .copy()
    )
    work["query_id"] = [
        _stable_hash("alias-query-v1", row.query_code, row.normalized_query)
        for row in work.itertuples(index=False)
    ]
    work["soc_major_group"] = work["query_code"].map(occupation_lookup["soc_major_group"])
    work["fold"] = work["query_code"].map(manifest_lookup["fold"])
    work["role"] = work["query_code"].map(manifest_lookup["role"])
    if bool(work[["soc_major_group", "fold", "role"]].isna().any().any()):
        raise RetrievalError("alias query could not be assigned occupation split metadata")
    result = work.loc[:, list(ALIAS_QUERY_COLUMNS)].sort_values(
        ["query_code", "query_id"], kind="stable"
    )
    if bool(result["query_id"].duplicated().any()):
        raise RetrievalError("deterministic alias query IDs are not unique")
    return result.reset_index(drop=True)


def build_title_documents(
    occupations: pd.DataFrame,
    aliases: pd.DataFrame,
    *,
    excluded_alias_codes: Iterable[str] = (),
    allowed_alias_codes: Iterable[str] | None = None,
) -> TitleDocumentSet:
    """Build catalog documents while explicitly excluding evaluation aliases.

    Canonical titles always remain searchable.  Only alternate titles are
    controlled by the fold boundary.
    """

    _require_columns(occupations, ("onet_soc_code", "canonical_title"), "occupations")
    _require_columns(
        aliases,
        ("onet_soc_code", "job_title", "normalized_job_title"),
        "aliases",
    )
    catalog = occupations.loc[:, ["onet_soc_code", "canonical_title"]].copy()
    catalog["onet_soc_code"] = catalog["onet_soc_code"].astype(str)
    if catalog.empty or bool(catalog["onet_soc_code"].duplicated().any()):
        raise RetrievalError("occupation catalog must be nonempty and unique")
    valid_codes = set(catalog["onet_soc_code"])
    excluded = {str(code) for code in excluded_alias_codes}
    allowed = (
        valid_codes if allowed_alias_codes is None else {str(code) for code in allowed_alias_codes}
    )
    if unknown := (excluded | allowed) - valid_codes:
        raise RetrievalError(
            "alias inclusion policy references unknown codes: " + ", ".join(sorted(unknown)[:5])
        )
    permitted = allowed - excluded
    alias_rows = aliases.loc[aliases["onet_soc_code"].astype(str).isin(permitted)].copy()
    alias_rows["onet_soc_code"] = alias_rows["onet_soc_code"].astype(str)
    unknown_aliases = set(alias_rows["onet_soc_code"]) - valid_codes
    if unknown_aliases:
        raise RetrievalError("aliases reference occupations outside the catalog")

    included_keys: set[tuple[str, str]] = set()
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in alias_rows.itertuples(index=False):
        normalized = normalize_text(row.normalized_job_title)
        display = normalize_text(row.job_title, casefold=False)
        if not normalized or not display:
            continue
        key = (str(row.onet_soc_code), normalized)
        if key in included_keys:
            continue
        included_keys.add(key)
        grouped.setdefault(str(row.onet_soc_code), []).append((normalized, display))

    records: list[dict[str, object]] = []
    for row in catalog.sort_values("onet_soc_code", kind="stable").itertuples(index=False):
        code = str(row.onet_soc_code)
        canonical = normalize_text(row.canonical_title, casefold=False)
        canonical_normalized = normalize_text(canonical)
        additions = [
            display
            for normalized, display in sorted(grouped.get(code, []), key=lambda item: item)
            if normalized != canonical_normalized
        ]
        records.append(
            {
                "onet_soc_code": code,
                "canonical_title": canonical,
                "title_text": normalized_join(canonical, *additions, separator="\n"),
                "indexed_alias_count": len(additions),
            }
        )
    result = pd.DataFrame(records)
    return TitleDocumentSet(result, frozenset(included_keys))


def assert_query_aliases_excluded(
    document_set: TitleDocumentSet,
    alias_queries: pd.DataFrame,
) -> None:
    """Prove that each evaluated alias key is absent from its target document."""

    _require_columns(alias_queries, ("query_code", "normalized_query"), "alias queries")
    leaking = [
        (str(row.query_code), normalize_text(row.normalized_query))
        for row in alias_queries.itertuples(index=False)
        if (str(row.query_code), normalize_text(row.normalized_query))
        in document_set.included_alias_keys
    ]
    if leaking:
        raise RetrievalError(
            "held-out aliases entered target title documents: "
            + ", ".join(f"{code}:{alias}" for code, alias in leaking[:5])
        )


def simplex_weight_grid(
    component_names: Sequence[str], *, step: float = 0.25
) -> tuple[dict[str, float], ...]:
    """Enumerate a lexicographically stable nonnegative simplex grid."""

    names = tuple(str(name).strip() for name in component_names)
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise RetrievalError("component names must be nonempty, nonblank, and unique")
    reciprocal = round(1.0 / step) if step > 0 else 0
    if step <= 0 or not np.isclose(reciprocal * step, 1.0, atol=1e-12):
        raise RetrievalError("step must divide one exactly")

    allocations: list[tuple[int, ...]] = []

    def enumerate_allocations(prefix: tuple[int, ...], remaining: int, slots: int) -> None:
        if slots == 1:
            allocations.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            enumerate_allocations((*prefix, value), remaining - value, slots - 1)

    enumerate_allocations((), reciprocal, len(names))
    return tuple(
        {name: value / reciprocal for name, value in zip(names, allocation, strict=True)}
        for allocation in allocations
    )


def combine_component_scores(
    components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine components and renormalize weights where a score is missing.

    Renormalizing over observed components is not mean/zero imputation: a cell
    with no positively weighted observed component remains missing.
    """

    if set(components) != set(weights) or not components:
        raise RetrievalError("component score names must exactly match weight names")
    numeric_weights = {name: float(value) for name, value in weights.items()}
    if any(not np.isfinite(value) or value < 0 for value in numeric_weights.values()):
        raise RetrievalError("component weights must be finite and nonnegative")
    if not np.isclose(sum(numeric_weights.values()), 1.0, atol=1e-12):
        raise RetrievalError("component weights must sum to one")
    shapes = {np.asarray(matrix).shape for matrix in components.values()}
    if len(shapes) != 1:
        raise RetrievalError("component score matrices must share one shape")
    shape = next(iter(shapes))
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    for name, matrix in components.items():
        weight = numeric_weights[name]
        if weight == 0.0:
            continue
        values = np.asarray(matrix, dtype=np.float64)
        observed = np.isfinite(values)
        numerator[observed] += weight * values[observed]
        denominator[observed] += weight
    result = np.full(shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def rank_score_matrix(
    query_codes: Sequence[str],
    candidate_codes: Sequence[str],
    scores: np.ndarray,
    *,
    k: int,
    exclude_self: bool = True,
    eligible_candidate_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert a component matrix to stable, contiguous, long-form rankings."""

    queries = tuple(str(code) for code in query_codes)
    candidates = tuple(str(code) for code in candidate_codes)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(queries), len(candidates)):
        raise RetrievalError(f"score matrix shape {values.shape} does not match queries/catalog")
    if len(queries) != len(set(queries)) or len(candidates) != len(set(candidates)):
        raise RetrievalError("query and candidate code orders must each be unique")
    eligible = (
        set(candidates) if eligible_candidate_codes is None else set(eligible_candidate_codes)
    )
    if unknown := eligible - set(candidates):
        raise RetrievalError(
            "eligible candidate set contains unknown codes: " + ", ".join(sorted(unknown)[:5])
        )
    candidate_mask = np.asarray([code in eligible for code in candidates], dtype=bool)
    rows: list[dict[str, object]] = []
    for query_index, query_code in enumerate(queries):
        finite_mask = np.isfinite(values[query_index]) & candidate_mask
        kept_codes = np.asarray(candidates, dtype=object)[finite_mask].tolist()
        kept_scores = values[query_index, finite_mask]
        if not kept_codes:
            # No finite score is a valid empty retrieval result. Metrics that
            # receive an expected query universe score this query as zero.
            continue
        excluded = (query_code,) if exclude_self else ()
        ranked = deterministic_rank(kept_codes, kept_scores, k=k, exclude_ids=excluded)
        for item in ranked:
            rows.append(
                {
                    "query_code": query_code,
                    "candidate_code": item.candidate_id,
                    "rank": item.rank,
                    "score": item.score,
                }
            )
    return pd.DataFrame(rows, columns=["query_code", "candidate_code", "rank", "score"])


def rank_alias_queries(
    transformer: TfidfIndex,
    catalog_matrix: sparse.spmatrix,
    catalog_codes: Sequence[str],
    alias_queries: pd.DataFrame,
    *,
    eligible_candidate_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the full-catalog rank of each alternate-title target."""

    _require_columns(
        alias_queries,
        ("query_id", "query_code", "target_code", "query_text"),
        "alias queries",
    )
    codes = tuple(str(code) for code in catalog_codes)
    if len(codes) != len(set(codes)) or catalog_matrix.shape[0] != len(codes):
        raise RetrievalError("catalog code order and matrix rows do not align")
    eligible = set(codes) if eligible_candidate_codes is None else set(eligible_candidate_codes)
    if unknown := eligible - set(codes):
        raise RetrievalError(f"eligible title candidates are unknown: {sorted(unknown)[:5]}")
    query_matrix = transformer.transform(alias_queries["query_text"].tolist())
    score_matrix = cosine_scores(query_matrix, catalog_matrix)
    rows: list[dict[str, object]] = []
    for query_index, query in enumerate(alias_queries.itertuples(index=False)):
        target = str(query.target_code)
        if target not in eligible:
            raise RetrievalError(f"alias target {target} is outside the eligible candidate pool")
        mask = np.asarray([code in eligible for code in codes], dtype=bool)
        ranked = deterministic_rank(
            np.asarray(codes, dtype=object)[mask].tolist(),
            score_matrix[query_index, mask],
        )
        rank_by_code = {item.candidate_id: item.rank for item in ranked}
        rows.append(
            {
                "query_id": str(query.query_id),
                "query_code": str(query.query_code),
                "target_code": target,
                "rank": rank_by_code.get(target),
            }
        )
    return pd.DataFrame(rows)


def candidate_codes_for_phase(
    fold_manifest: pd.DataFrame,
    phase: str,
) -> tuple[str, ...]:
    """Enforce the locked-test exclusion from all development candidate pools."""

    _require_columns(fold_manifest, ("onet_soc_code", "role"), "fold manifest")
    if phase in {"selection", "confirmation"}:
        mask = fold_manifest["role"].ne("test")
    elif phase in {"locked_test", "production"}:
        mask = pd.Series(True, index=fold_manifest.index)
    else:
        raise RetrievalError(f"unknown retrieval phase: {phase!r}")
    return tuple(fold_manifest.loc[mask, "onet_soc_code"].astype(str).sort_values(kind="stable"))


def fitting_codes_for_phase(fold_manifest: pd.DataFrame, phase: str) -> tuple[str, ...]:
    """Return codes allowed to fit vocabulary/IDF/SVD for a phase."""

    _require_columns(fold_manifest, ("onet_soc_code", "role"), "fold manifest")
    if phase == "selection":
        roles = {"discovery"}
    elif phase == "confirmation":
        roles = {"discovery", "selection"}
    elif phase in {"locked_test", "production"}:
        roles = {"discovery", "selection", "confirmation"}
    else:
        raise RetrievalError(f"unknown fitting phase: {phase!r}")
    return tuple(
        fold_manifest.loc[fold_manifest["role"].isin(roles), "onet_soc_code"]
        .astype(str)
        .sort_values(kind="stable")
    )


__all__ = [
    "ALIAS_QUERY_COLUMNS",
    "RetrievalError",
    "TitleDocumentSet",
    "assert_query_aliases_excluded",
    "build_title_documents",
    "candidate_codes_for_phase",
    "combine_component_scores",
    "fitting_codes_for_phase",
    "prepare_alias_queries",
    "rank_alias_queries",
    "rank_score_matrix",
    "simplex_weight_grid",
]
