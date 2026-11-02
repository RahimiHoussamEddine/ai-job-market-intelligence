"""Deterministic ranking metrics and preregistered query-level uncertainty."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

TITLE_RESULT_COLUMNS: Final = ("query_id", "query_code", "target_code", "rank")
RANKING_COLUMNS: Final = ("query_code", "candidate_code", "rank")
QREL_COLUMNS: Final = (
    "query_onet_soc_code",
    "candidate_onet_soc_code",
    "relatedness_tier",
    "relevance_grade",
)


class RankingMetricError(ValueError):
    """Raised when rankings are ambiguous, incomplete, or non-deterministic."""


@dataclass(frozen=True, slots=True)
class RankedItem:
    rank: int
    candidate_id: str
    score: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    replicates: int
    seed: int


def _require_columns(frame: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RankingMetricError(f"{context} is missing columns: {missing}")


def deterministic_rank(
    candidate_ids: Sequence[object],
    scores: Sequence[float] | np.ndarray,
    *,
    k: int | None = None,
    exclude_ids: Iterable[object] = (),
) -> tuple[RankedItem, ...]:
    """Rank by descending finite score, breaking ties by candidate ID."""

    identifiers = np.asarray([str(value).strip() for value in candidate_ids], dtype=object)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(identifiers) != len(values):
        raise RankingMetricError("candidate_ids and scores must have equal length")
    if len(identifiers) == 0:
        raise RankingMetricError("candidate list must be nonempty")
    if any(not value for value in identifiers):
        raise RankingMetricError("candidate IDs must be nonblank")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise RankingMetricError("candidate IDs must be unique")
    if not np.isfinite(values).all():
        raise RankingMetricError("ranking scores must be finite")
    if k is not None and (isinstance(k, bool) or k < 1):
        raise RankingMetricError("k must be a positive integer")
    excluded = {str(value).strip() for value in exclude_ids}
    keep = np.asarray([identifier not in excluded for identifier in identifiers], dtype=bool)
    identifiers = identifiers[keep]
    values = values[keep]
    # np.lexsort uses its final key as primary: -score first, code second.
    order = np.lexsort((identifiers.astype(str), -values))
    if k is not None:
        order = order[:k]
    return tuple(
        RankedItem(
            rank=index + 1, candidate_id=str(identifiers[position]), score=float(values[position])
        )
        for index, position in enumerate(order)
    )


def title_query_metrics(query_results: pd.DataFrame) -> pd.DataFrame:
    """Return one metric row per held-out alias query."""

    _require_columns(query_results, TITLE_RESULT_COLUMNS, "title query results")
    result = query_results.loc[:, list(TITLE_RESULT_COLUMNS)].copy()
    if result.empty or bool(result["query_id"].duplicated().any()):
        raise RankingMetricError("title query results must be nonempty and unique by query_id")
    for column in ("query_id", "query_code", "target_code"):
        result[column] = result[column].astype("string").str.strip()
        if bool(result[column].isna().any() | result[column].eq("").any()):
            raise RankingMetricError(f"title query results {column} contains blank values")
    if bool(result["query_code"].ne(result["target_code"]).any()):
        raise RankingMetricError("alternate-title query_code must equal its target occupation")
    numeric_rank = pd.to_numeric(result["rank"], errors="coerce")
    supplied = result["rank"].notna()
    invalid = supplied & (numeric_rank.isna() | numeric_rank.lt(1) | numeric_rank.mod(1).ne(0))
    if bool(invalid.any()):
        raise RankingMetricError("title result ranks must be positive integers or missing")
    result["rank"] = pd.array(numeric_rank, dtype="Int64")
    result["top1_accuracy"] = result["rank"].eq(1).fillna(False).astype(float)
    result["recall_at_5"] = result["rank"].le(5).fillna(False).astype(float)
    result["mrr"] = result["rank"].map(
        lambda value: 0.0 if pd.isna(value) else 1.0 / int(value), na_action=None
    )
    return result


def macro_title_metrics(query_results: pd.DataFrame) -> dict[str, float | int]:
    """Macro-average alias metrics by occupation, plus query-weighted diagnostics."""

    per_query = title_query_metrics(query_results)
    metric_columns = ["top1_accuracy", "recall_at_5", "mrr"]
    per_occupation = per_query.groupby("target_code", sort=True)[metric_columns].mean()
    output: dict[str, float | int] = {
        metric: float(per_occupation[metric].mean()) for metric in metric_columns
    }
    output.update(
        {f"query_weighted_{metric}": float(per_query[metric].mean()) for metric in metric_columns}
    )
    output["n_queries"] = int(len(per_query))
    output["n_occupations"] = int(len(per_occupation))
    return output


def _dcg(relevances: Sequence[int], k: int) -> float:
    values = np.asarray(relevances[:k], dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2, dtype=np.float64))
    gains = np.exp2(values) - 1.0
    return float(np.sum(gains / discounts, dtype=np.float64))


def ndcg_at_k(relevances: Sequence[int], ideal_relevances: Sequence[int], k: int) -> float:
    """Compute graded nDCG with exponential gains and base-two discounts."""

    if k < 1:
        raise RankingMetricError("k must be positive")
    if any(value < 0 for value in (*relevances, *ideal_relevances)):
        raise RankingMetricError("relevance grades cannot be negative")
    ideal = _dcg(sorted(ideal_relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return _dcg(relevances, k) / ideal


def transition_query_metrics(
    rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    expected_query_codes: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Compute preregistered adjacent-occupation metrics for each query."""

    _require_columns(rankings, RANKING_COLUMNS, "transition rankings")
    _require_columns(qrels, QREL_COLUMNS, "related-occupation qrels")
    if rankings.empty and expected_query_codes is None:
        raise RankingMetricError("transition rankings must be nonempty")
    ranking = rankings.loc[:, list(RANKING_COLUMNS)].copy()
    for column in ("query_code", "candidate_code"):
        ranking[column] = ranking[column].astype("string").str.strip()
        if bool(ranking[column].isna().any() | ranking[column].eq("").any()):
            raise RankingMetricError(f"transition rankings {column} contains blank values")
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
    if bool(
        ranking["rank"].isna().any()
        or ranking["rank"].lt(1).any()
        or ranking["rank"].mod(1).ne(0).any()
    ):
        raise RankingMetricError("transition ranks must be positive integers")
    ranking["rank"] = ranking["rank"].astype(int)
    if bool(ranking["query_code"].eq(ranking["candidate_code"]).any()):
        raise RankingMetricError("transition rankings contain a self-match")
    if bool(ranking.duplicated(["query_code", "candidate_code"]).any()):
        raise RankingMetricError("transition rankings contain duplicate candidates")
    if bool(ranking.duplicated(["query_code", "rank"]).any()):
        raise RankingMetricError("transition rankings contain duplicate ranks")
    for query_code, group in ranking.groupby("query_code", sort=False):
        observed_ranks = sorted(group["rank"].tolist())
        if observed_ranks != list(range(1, len(group) + 1)):
            raise RankingMetricError(f"ranks for {query_code} must be contiguous from one")

    judgments = qrels.loc[:, list(QREL_COLUMNS)].copy()
    judgments = judgments.rename(
        columns={
            "query_onet_soc_code": "query_code",
            "candidate_onet_soc_code": "candidate_code",
        }
    )
    for column in ("query_code", "candidate_code", "relatedness_tier"):
        judgments[column] = judgments[column].astype("string").str.strip()
    judgments["relevance_grade"] = pd.to_numeric(judgments["relevance_grade"], errors="coerce")
    valid_tiers = {"Primary-Short": 3, "Primary-Long": 2, "Supplemental": 1}
    if bool(judgments.duplicated(["query_code", "candidate_code"]).any()):
        raise RankingMetricError("qrels contain duplicate query/candidate pairs")
    if any(
        valid_tiers.get(str(row.relatedness_tier)) != row.relevance_grade
        for row in judgments.itertuples(index=False)
    ):
        raise RankingMetricError("qrel tier-to-grade mapping differs from the protocol")

    query_codes = (
        sorted({str(code) for code in expected_query_codes})
        if expected_query_codes is not None
        else sorted(set(ranking["query_code"]))
    )
    if not query_codes:
        raise RankingMetricError("expected transition query set must be nonempty")
    unexpected_rankings = set(ranking["query_code"]) - set(query_codes)
    if unexpected_rankings:
        raise RankingMetricError(
            "rankings contain queries outside the expected set: "
            + ", ".join(sorted(unexpected_rankings)[:5])
        )
    missing_qrels = set(query_codes) - set(judgments["query_code"])
    if missing_qrels:
        raise RankingMetricError(
            "ranked queries lack qrels: " + ", ".join(sorted(missing_qrels)[:5])
        )
    rows: list[dict[str, float | str]] = []
    qrel_groups = {code: group for code, group in judgments.groupby("query_code", sort=False)}
    rank_groups = {
        code: group.sort_values("rank", kind="stable")
        for code, group in ranking.groupby("query_code", sort=False)
    }
    for query_code in query_codes:
        query_qrels = qrel_groups[query_code]
        grade_by_candidate = dict(
            zip(
                query_qrels["candidate_code"].astype(str),
                query_qrels["relevance_grade"].astype(int),
                strict=True,
            )
        )
        ranked_group = rank_groups.get(query_code)
        ranked_codes = (
            [] if ranked_group is None else ranked_group["candidate_code"].astype(str).tolist()
        )
        observed_grades = [grade_by_candidate.get(code, 0) for code in ranked_codes]
        ideal_grades = query_qrels["relevance_grade"].astype(int).tolist()
        short = set(
            query_qrels.loc[
                query_qrels["relatedness_tier"].eq("Primary-Short"), "candidate_code"
            ].astype(str)
        )
        primary = set(
            query_qrels.loc[
                query_qrels["relatedness_tier"].isin(["Primary-Short", "Primary-Long"]),
                "candidate_code",
            ].astype(str)
        )
        if not short or not primary:
            raise RankingMetricError(f"qrels for {query_code} lack primary judgments")
        first_relevant = next(
            (rank for rank, grade in enumerate(observed_grades[:10], start=1) if grade > 0),
            None,
        )
        binary_observed = [1 if grade >= 2 else 0 for grade in observed_grades]
        binary_ideal = [1 if grade >= 2 else 0 for grade in ideal_grades]
        rows.append(
            {
                "query_code": query_code,
                "ndcg_at_10": ndcg_at_k(observed_grades, ideal_grades, 10),
                "primary_short_recall_at_5": len(short & set(ranked_codes[:5])) / len(short),
                "primary_recall_at_10": len(primary & set(ranked_codes[:10])) / len(primary),
                "ndcg_at_20": ndcg_at_k(observed_grades, ideal_grades, 20),
                "mrr_at_10": 0.0 if first_relevant is None else 1.0 / first_relevant,
                "binary_primary_ndcg_at_10": ndcg_at_k(binary_observed, binary_ideal, 10),
            }
        )
    return pd.DataFrame(rows).sort_values("query_code", kind="stable").reset_index(drop=True)


def transition_metrics(rankings: pd.DataFrame, qrels: pd.DataFrame) -> dict[str, float | int]:
    """Macro-average the query-level transition benchmark metrics."""

    per_query = transition_query_metrics(rankings, qrels)
    metrics = [column for column in per_query.columns if column != "query_code"]
    result: dict[str, float | int] = {metric: float(per_query[metric].mean()) for metric in metrics}
    result["n_queries"] = int(len(per_query))
    return result


def _aligned_numeric(
    values: Sequence[float] | pd.Series,
    strata: Sequence[object] | pd.Series,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        isinstance(values, pd.Series)
        and isinstance(strata, pd.Series)
        and not values.index.equals(strata.index)
    ):
        raise RankingMetricError(f"{context} values and strata indexes are not aligned")
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(strata, dtype=object).reshape(-1)
    if len(numeric) == 0 or len(numeric) != len(labels):
        raise RankingMetricError(f"{context} values and strata must be nonempty and equal length")
    if not np.isfinite(numeric).all():
        raise RankingMetricError(f"{context} values must be finite")
    if any(pd.isna(label) or not str(label).strip() for label in labels):
        raise RankingMetricError(f"{context} strata must be nonmissing and nonblank")
    return numeric, labels.astype(str)


def _bootstrap_means(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if isinstance(replicates, bool) or replicates < 2:
        raise RankingMetricError("bootstrap replicates must be at least two")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))]
    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indexes, size=len(indexes), replace=True) for indexes in groups]
        )
        result[replicate] = float(np.mean(values[sampled]))
    return result


def stratified_bootstrap_interval(
    values: Sequence[float] | pd.Series,
    strata: Sequence[object] | pd.Series,
    *,
    replicates: int = 2000,
    seed: int = 20260814,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Percentile interval and bootstrap SE, preserving SOC-stratum sizes."""

    numeric, labels = _aligned_numeric(values, strata, context="bootstrap")
    if not 0.0 < confidence < 1.0:
        raise RankingMetricError("confidence must lie strictly between zero and one")
    draws = _bootstrap_means(numeric, labels, replicates=replicates, seed=seed)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(np.mean(numeric)),
        standard_error=float(np.std(draws, ddof=1)),
        ci_lower=float(np.quantile(draws, alpha)),
        ci_upper=float(np.quantile(draws, 1.0 - alpha)),
        replicates=replicates,
        seed=seed,
    )


def paired_stratified_bootstrap(
    champion: Sequence[float] | pd.Series,
    baseline: Sequence[float] | pd.Series,
    strata: Sequence[object] | pd.Series,
    *,
    replicates: int = 2000,
    seed: int = 20260814,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Paired champion-minus-baseline interval over identical query units."""

    if (
        isinstance(champion, pd.Series)
        and isinstance(baseline, pd.Series)
        and not champion.index.equals(baseline.index)
    ):
        raise RankingMetricError("paired bootstrap inputs are not index-aligned")
    champion_values, labels = _aligned_numeric(champion, strata, context="paired bootstrap")
    baseline_values, baseline_labels = _aligned_numeric(
        baseline, strata, context="paired bootstrap"
    )
    if not np.array_equal(labels, baseline_labels):
        raise RankingMetricError("paired bootstrap strata changed between systems")
    return stratified_bootstrap_interval(
        champion_values - baseline_values,
        labels,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )


def query_stratified_standard_error(
    values: Sequence[float] | pd.Series,
    strata: Sequence[object] | pd.Series,
    *,
    replicates: int = 2000,
    seed: int = 20260814,
) -> float:
    """Return the bootstrap SE used by the frozen one-SE selection rule."""

    return stratified_bootstrap_interval(
        values, strata, replicates=replicates, seed=seed
    ).standard_error


__all__ = [
    "BootstrapInterval",
    "QREL_COLUMNS",
    "RANKING_COLUMNS",
    "RankedItem",
    "RankingMetricError",
    "deterministic_rank",
    "macro_title_metrics",
    "ndcg_at_k",
    "paired_stratified_bootstrap",
    "query_stratified_standard_error",
    "stratified_bootstrap_interval",
    "title_query_metrics",
    "transition_metrics",
    "transition_query_metrics",
]
