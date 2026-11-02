from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from job_market_intelligence.frontier import (
    FrontierError,
    materialize_pareto_frontiers,
    pareto_frontier,
)
from job_market_intelligence.transitions import build_transition_candidates


def _components() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    codes = ["11-1001.00", "15-1001.00", "15-1002.00", "29-1001.00"]
    lexical = np.array(
        [
            [1.0, 0.8, 0.7, 0.2],
            [0.8, 1.0, 0.6, 0.3],
            [0.7, 0.6, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ]
    )
    lsa = lexical - 0.05
    structured = np.array(
        [
            [1.0, 0.9, np.nan, 0.1],
            [0.9, 1.0, 0.5, 0.2],
            [np.nan, 0.5, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0],
        ]
    )
    distance = np.array(
        [
            [0.0, 0.2, 0.3, np.nan],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.3, 0.0, 0.6],
            [0.7, 0.6, 0.5, 0.0],
        ]
    )
    return codes, lexical, lsa, structured, distance


def test_transition_materialization_stores_top_250_contract_columns() -> None:
    codes, lexical, lsa, structured, distance = _components()
    neighbors = build_transition_candidates(
        codes,
        lexical,
        lsa,
        structured,
        distance,
        weights={"lexical": 0.5, "lsa": 0.25, "structured": 0.25},
        limit=3,
    )
    assert list(neighbors.columns) == [
        "query_code",
        "destination_code",
        "rank",
        "total_score",
        "lexical_score",
        "lsa_score",
        "structured_score",
        "development_distance",
        "explanation",
    ]
    assert neighbors.groupby("query_code").size().eq(3).all()
    assert not neighbors["query_code"].eq(neighbors["destination_code"]).any()
    unavailable = neighbors.loc[
        neighbors["query_code"].eq("11-1001.00") & neighbors["destination_code"].eq("15-1002.00")
    ].iloc[0]
    assert pd.isna(unavailable["structured_score"])
    assert pd.isna(unavailable["development_distance"])
    assert "weight renormalized" in unavailable["explanation"]


def _market() -> pd.DataFrame:
    common = {
        "employment_2024": 1000,
        "employment_2034": 1100,
        "employment_change_2024_34": 100,
        "typical_education": "Bachelor's degree",
        "work_experience": "None",
        "on_the_job_training": "None",
    }
    return pd.DataFrame(
        [
            {
                "onet_soc_code": "15-1001.00",
                "nem_code": "15-1000",
                **common,
                "employment_change_pct": 10.0,
                "annual_openings_2024_34": 100,
                "median_annual_wage_2024": 100000,
            },
            {
                "onet_soc_code": "15-1002.00",
                "nem_code": "15-1000",
                **common,
                "employment_change_pct": 10.0,
                "annual_openings_2024_34": 100,
                "median_annual_wage_2024": 100000,
            },
            {
                "onet_soc_code": "29-1001.00",
                "nem_code": "29-1001",
                **common,
                "employment_change_pct": 20.0,
                "annual_openings_2024_34": 80,
                "median_annual_wage_2024": 90000,
            },
        ]
    )


def test_frontier_is_unique_at_nem_grain_and_never_sums_shared_values() -> None:
    transitions = pd.DataFrame(
        [
            {
                "query_code": "11-1001.00",
                "destination_code": "15-1001.00",
                "rank": 1,
                "development_distance": 0.2,
            },
            {
                "query_code": "11-1001.00",
                "destination_code": "15-1002.00",
                "rank": 2,
                "development_distance": 0.3,
            },
            {
                "query_code": "11-1001.00",
                "destination_code": "29-1001.00",
                "rank": 3,
                "development_distance": 0.4,
            },
        ]
    )
    result = pareto_frontier(
        transitions,
        _market(),
        source_code="11-1001.00",
        outcome="employment_change_pct",
    )
    assert result.eligible_count == 2
    assert result.frontier["nem_code"].is_unique
    assert set(result.frontier["nem_code"]) == {"15-1000", "29-1001"}
    assert (
        result.frontier.loc[result.frontier["nem_code"].eq("15-1000"), "destination_code"].item()
        == "15-1001.00"
    )

    materialized = materialize_pareto_frontiers(
        transitions,
        _market(),
        outcomes=("employment_change_pct",),
    )
    assert set(materialized["source_code"]) == {"11-1001.00"}
    assert (
        materialized.groupby(["source_code", "outcome"])["nem_code"]
        .apply(lambda values: values.is_unique)
        .all()
    )

    inconsistent = _market()
    inconsistent.loc[1, "median_annual_wage_2024"] = 99999
    with pytest.raises(FrontierError, match="inconsistent unsummed"):
        pareto_frontier(
            transitions,
            inconsistent,
            source_code="11-1001.00",
            outcome="employment_change_pct",
        )
