"""Transparent development-distance/BLS Pareto frontiers at NEM granularity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

SUPPORTED_OUTCOMES: Final = frozenset(
    {
        "median_annual_wage_2024",
        "employment_change_pct",
        "annual_openings_2024_34",
    }
)
BLS_CONTEXT_COLUMNS: Final = (
    "employment_2024",
    "employment_2034",
    "employment_change_2024_34",
    "employment_change_pct",
    "annual_openings_2024_34",
    "median_annual_wage_2024",
    "typical_education",
    "work_experience",
    "on_the_job_training",
)


class FrontierError(ValueError):
    """Raised when a frontier would duplicate or aggregate shared BLS figures."""


@dataclass(frozen=True, slots=True)
class FrontierResult:
    source_code: str
    outcome: str
    frontier: pd.DataFrame
    eligible_count: int
    excluded_missing_distance: int
    excluded_missing_outcome: int


def assert_unsummed_nem_market(market: pd.DataFrame) -> None:
    """Prove shared NEM rows repeat identical line-item values, never sums."""

    required = {"onet_soc_code", "nem_code", *BLS_CONTEXT_COLUMNS}
    if missing := sorted(required - set(market.columns)):
        raise FrontierError(f"occupation_market is missing columns: {missing}")
    if market.empty or bool(market.duplicated(["onet_soc_code", "nem_code"]).any()):
        raise FrontierError("occupation_market must be nonempty and unique by O*NET/NEM pair")
    for nem_code, group in market.groupby("nem_code", sort=False, dropna=False):
        if pd.isna(nem_code) or not str(nem_code).strip():
            raise FrontierError("occupation_market contains a missing NEM code")
        for column in BLS_CONTEXT_COLUMNS:
            normalized = group[column].astype("string").fillna("<MISSING>")
            if normalized.nunique(dropna=False) != 1:
                raise FrontierError(
                    f"shared NEM code {nem_code} has inconsistent unsummed {column} values"
                )


def pareto_frontier(
    transitions: pd.DataFrame,
    occupation_market: pd.DataFrame,
    *,
    source_code: str,
    outcome: str,
    candidate_limit: int = 250,
) -> FrontierResult:
    """Return nondominated NEM destinations without constructing an opaque score."""

    if outcome not in SUPPORTED_OUTCOMES:
        raise FrontierError(f"unsupported BLS outcome: {outcome!r}")
    if isinstance(candidate_limit, bool) or not 1 <= candidate_limit <= 250:
        raise FrontierError("candidate_limit must be between 1 and 250")
    required_transitions = {
        "query_code",
        "destination_code",
        "rank",
        "development_distance",
    }
    if missing := sorted(required_transitions - set(transitions.columns)):
        raise FrontierError(f"transitions are missing columns: {missing}")
    assert_unsummed_nem_market(occupation_market)
    selected = transitions.loc[
        transitions["query_code"].astype(str).eq(str(source_code))
        & pd.to_numeric(transitions["rank"], errors="coerce").le(candidate_limit)
    ].copy()
    if selected.empty:
        raise FrontierError(f"no stored transition candidates for source {source_code}")
    if bool(selected["destination_code"].astype(str).eq(str(source_code)).any()):
        raise FrontierError("frontier candidates contain a self-match")
    if bool(selected["destination_code"].duplicated().any()):
        raise FrontierError("frontier candidates repeat an O*NET destination")
    selected["development_distance"] = pd.to_numeric(
        selected["development_distance"], errors="coerce"
    )
    if bool(np.isinf(selected["development_distance"].dropna()).any()):
        raise FrontierError("development distances contain infinity")
    if bool(selected["development_distance"].dropna().lt(0).any()):
        raise FrontierError("development distances cannot be negative")
    missing_distance = int(selected["development_distance"].isna().sum())

    joined = selected.merge(
        occupation_market,
        left_on="destination_code",
        right_on="onet_soc_code",
        how="left",
        validate="one_to_many",
    )
    joined[outcome] = pd.to_numeric(joined[outcome], errors="coerce")
    missing_outcome = int(joined.loc[joined["development_distance"].notna(), outcome].isna().sum())
    eligible = joined.loc[joined["development_distance"].notna() & joined[outcome].notna()].copy()
    if bool(np.isinf(eligible[outcome]).any()):
        raise FrontierError(f"{outcome} contains infinity")
    if eligible.empty:
        return FrontierResult(
            str(source_code),
            outcome,
            pd.DataFrame(
                columns=[
                    "destination_code",
                    "nem_code",
                    "development_distance",
                    "outcome",
                    "outcome_value",
                ]
            ),
            0,
            missing_distance,
            missing_outcome,
        )

    # The best (lowest-distance) O*NET specialty is the representative route to
    # a shared NEM line item.  No BLS field is summed or averaged.
    eligible = eligible.sort_values(
        ["nem_code", "development_distance", "destination_code"], kind="stable"
    ).drop_duplicates("nem_code", keep="first")
    eligible = eligible.reset_index(drop=True)
    distance = eligible["development_distance"].to_numpy(dtype=np.float64)
    values = eligible[outcome].to_numpy(dtype=np.float64)
    efficient = np.ones(len(eligible), dtype=bool)
    for index in range(len(eligible)):
        no_worse = (distance <= distance[index]) & (values >= values[index])
        strict = (distance < distance[index]) | (values > values[index])
        if bool(np.any(no_worse & strict)):
            efficient[index] = False
    frontier = eligible.loc[efficient].copy()
    frontier["outcome"] = outcome
    frontier["outcome_value"] = frontier[outcome].astype(float)
    output_columns = [
        "destination_code",
        "nem_code",
        "development_distance",
        "outcome",
        "outcome_value",
        *BLS_CONTEXT_COLUMNS,
    ]
    frontier = frontier.loc[:, output_columns].sort_values(
        ["development_distance", "outcome_value", "nem_code"],
        ascending=[True, False, True],
        kind="stable",
    )
    return FrontierResult(
        source_code=str(source_code),
        outcome=outcome,
        frontier=frontier.reset_index(drop=True),
        eligible_count=int(len(eligible)),
        excluded_missing_distance=missing_distance,
        excluded_missing_outcome=missing_outcome,
    )


def materialize_pareto_frontiers(
    transitions: pd.DataFrame,
    occupation_market: pd.DataFrame,
    *,
    outcomes: tuple[str, ...] = tuple(sorted(SUPPORTED_OUTCOMES)),
    candidate_limit: int = 250,
) -> pd.DataFrame:
    """Materialize every source/outcome NEM-grain frontier without aggregation."""

    requested = tuple(str(value) for value in outcomes)
    if not requested or len(requested) != len(set(requested)):
        raise FrontierError("frontier outcomes must be nonempty and unique")
    unsupported = set(requested) - SUPPORTED_OUTCOMES
    if unsupported:
        raise FrontierError(f"unsupported BLS outcomes: {sorted(unsupported)}")
    if "query_code" not in transitions:
        raise FrontierError("transitions are missing query_code")
    sources = tuple(sorted(transitions["query_code"].dropna().astype(str).unique()))
    if not sources:
        raise FrontierError("transitions contain no source occupations")
    frames: list[pd.DataFrame] = []
    for source_code in sources:
        for outcome in requested:
            result = pareto_frontier(
                transitions,
                occupation_market,
                source_code=source_code,
                outcome=outcome,
                candidate_limit=candidate_limit,
            )
            frame = result.frontier.copy()
            if frame.empty:
                continue
            frame.insert(0, "source_code", result.source_code)
            frame["eligible_nem_count"] = result.eligible_count
            frame["excluded_missing_distance"] = result.excluded_missing_distance
            frame["excluded_missing_outcome"] = result.excluded_missing_outcome
            frames.append(frame)
    columns = [
        "source_code",
        "destination_code",
        "nem_code",
        "development_distance",
        "outcome",
        "outcome_value",
        *BLS_CONTEXT_COLUMNS,
        "eligible_nem_count",
        "excluded_missing_distance",
        "excluded_missing_outcome",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(frames, ignore_index=True)
        .loc[:, columns]
        .sort_values(
            ["source_code", "outcome", "development_distance", "outcome_value", "nem_code"],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


__all__ = [
    "BLS_CONTEXT_COLUMNS",
    "FrontierError",
    "FrontierResult",
    "SUPPORTED_OUTCOMES",
    "assert_unsummed_nem_market",
    "materialize_pareto_frontiers",
    "pareto_frontier",
]
