from __future__ import annotations

import math

import pandas as pd
import pytest
from data_test_support import rating_row

from job_market_intelligence.crosswalk import (
    CrosswalkError,
    join_market_context,
    normalize_bls_market,
    normalize_crosswalk,
    parse_thousands_to_people,
)
from job_market_intelligence.profiles import (
    ProfileError,
    build_occupations,
    build_structured_matrix,
    build_structured_ratings,
    missing_aware_cosine,
)


def test_bls_thousands_are_converted_exactly_to_person_units() -> None:
    assert parse_thousands_to_people("1,562.0") == 1_562_000
    assert parse_thousands_to_people("0.001") == 1
    with pytest.raises(CrosswalkError, match="whole people"):
        parse_thousands_to_people("0.0001")


def test_bls_market_keeps_only_line_items_and_never_aggregates_specialties() -> None:
    source = pd.DataFrame(
        [
            {
                "nem_title": "Management occupations",
                "nem_code": "11-0000",
                "occupation_type": "Summary",
                "employment_2024_thousands": "10",
                "employment_2034_thousands": "11",
                "employment_change_2024_34_thousands": "1",
                "employment_change_pct": "10",
                "annual_openings_2024_34_thousands": "2",
                "median_annual_wage_2024": "$100,000",
                "typical_education": "Varies",
                "work_experience": "Varies",
                "on_the_job_training": "Varies",
            },
            {
                "nem_title": "Chief executives",
                "nem_code": "11-1011",
                "occupation_type": "Line item",
                "employment_2024_thousands": "1.2",
                "employment_2034_thousands": "1.3",
                "employment_change_2024_34_thousands": "0.1",
                "employment_change_pct": "8.3",
                "annual_openings_2024_34_thousands": "0.2",
                "median_annual_wage_2024": "$206,680",
                "typical_education": "Bachelor's degree",
                "work_experience": "5 years or more",
                "on_the_job_training": "None",
            },
        ]
    )

    result = normalize_bls_market(source)

    assert result["nem_code"].tolist() == ["11-1011"]
    assert result.loc[0, "employment_2024"] == 1_200
    assert result.loc[0, "annual_openings_2024_34"] == 200


def test_many_to_many_crosswalk_join_preserves_each_official_pair_unsummed() -> None:
    market = normalize_bls_market(
        pd.DataFrame(
            [
                {
                    "nem_title": "Software developers",
                    "nem_code": "15-1252",
                    "occupation_type": "Line item",
                    "employment_2024_thousands": "100.0",
                    "employment_2034_thousands": "120.0",
                    "employment_change_2024_34_thousands": "20.0",
                    "employment_change_pct": "20.0",
                    "annual_openings_2024_34_thousands": "10.0",
                    "median_annual_wage_2024": "130000",
                    "typical_education": "Bachelor's degree",
                    "work_experience": "None",
                    "on_the_job_training": "None",
                }
            ]
        )
    )
    crosswalk = normalize_crosswalk(
        pd.DataFrame(
            [
                {
                    "onet_soc_code": "15-1252.00",
                    "onet_soc_title": "Software Developers",
                    "nem_code": "15-1252",
                    "nem_title": "Software developers",
                },
                {
                    "onet_soc_code": "15-1252.01",
                    "onet_soc_title": "Video Game Designers",
                    "nem_code": "15-1252",
                    "nem_title": "Software developers",
                },
            ]
        )
    )

    result = join_market_context(
        crosswalk,
        market,
        valid_onet_codes=["15-1252.00", "15-1252.01"],
    )

    assert len(result) == 2
    assert result["employment_2024"].tolist() == [100_000, 100_000]
    assert result["employment_2024"].sum() == 200_000  # demonstrates why callers must not sum


def test_current_job_zone_contract_retains_date_and_domain_provenance() -> None:
    occupation_data = pd.DataFrame(
        [{"onet_soc_code": "11-1011.00", "title": "Chief Executives", "description": "Lead."}]
    )
    zones = pd.DataFrame(
        [
            {
                "onet_soc_code": "11-1011.00",
                "job_zone": "4",
                "date": "07/2026",
                "domain_source": "Analyst",
            }
        ]
    )

    result = build_occupations(occupation_data, zones)

    assert result.loc[0, "job_zone"] == 4
    assert result.loc[0, "job_zone_date"].isoformat() == "2026-07-01"
    assert result.loc[0, "job_zone_domain_source"] == "Analyst"
    zones.loc[0, "job_zone"] = "5"
    with pytest.raises(ProfileError, match="outside 1..4"):
        build_occupations(occupation_data, zones)


def test_structured_ratings_preserve_precision_and_mask_suppressed_values() -> None:
    observed = rating_row("11-1011.00", "Chief Executives", "2.A.1", "Leadership")
    suppressed = rating_row(
        "13-2011.00",
        "Accountants and Auditors",
        "2.A.1",
        "Leadership",
        recommend_suppress="Y",
    )
    knowledge = [
        rating_row("11-1011.00", "Chief Executives", "2.C.1", "Economics"),
        rating_row("13-2011.00", "Accountants and Auditors", "2.C.1", "Economics"),
    ]

    ratings = build_structured_ratings(
        pd.DataFrame([observed, suppressed]), pd.DataFrame(knowledge)
    )
    masked = ratings.loc[
        (ratings["onet_soc_code"] == "13-2011.00")
        & (ratings["feature_family"] == "transferable_skill")
    ].iloc[0]

    assert masked["raw_value"] == 3.0
    assert masked["sample_size"] == 120
    assert masked["standard_error"] == 0.1
    assert masked["lower_ci_bound"] == 2.8
    assert masked["upper_ci_bound"] == 3.2
    assert masked["rating_date"].isoformat() == "2026-07-01"
    assert masked["domain_source"] == "Analyst"
    assert not masked["observed"]
    assert pd.isna(masked["normalized_value"])


def test_missing_aware_cosine_returns_nan_below_overlap_threshold() -> None:
    ratings = build_structured_ratings(
        pd.DataFrame(
            [
                rating_row("11-1011.00", "Chief Executives", "2.A.1", "Leadership"),
                rating_row("13-2011.00", "Accountants and Auditors", "2.A.1", "Leadership"),
            ]
        ),
        pd.DataFrame(
            [
                rating_row("11-1011.00", "Chief Executives", "2.C.1", "Economics"),
                rating_row("13-2011.00", "Accountants and Auditors", "2.C.1", "Economics"),
            ]
        ),
    )
    matrix = build_structured_matrix(ratings)

    similarity = missing_aware_cosine(matrix, minimum_overlap=3)

    assert math.isnan(float(similarity[0, 1]))
