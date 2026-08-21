"""Small synthetic records for data-layer tests; never used as release evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from job_market_intelligence.crosswalk import (
    join_market_context,
    normalize_bls_market,
    normalize_crosswalk,
)
from job_market_intelligence.profiles import (
    build_aliases,
    build_occupation_profiles,
    build_occupations,
    build_structured_ratings,
    build_tasks,
)


def rating_row(
    code: str,
    title: str,
    element_id: str,
    element_name: str,
    **overrides: Any,
) -> dict[str, object]:
    row: dict[str, object] = {
        "onet_soc_code": code,
        "title": title,
        "element_id": element_id,
        "element_name": element_name,
        "scale_id": "IM",
        "scale_name": "Importance",
        "data_value": "3.0",
        "n": "120",
        "standard_error": "0.1",
        "lower_ci_bound": "2.8",
        "upper_ci_bound": "3.2",
        "recommend_suppress": "N",
        "not_relevant": "N",
        "date": "07/2026",
        "domain_source": "Analyst",
    }
    row.update(overrides)
    return row


def canonical_frames() -> dict[str, pd.DataFrame]:
    occupation_data = pd.DataFrame(
        [
            {
                "onet_soc_code": "11-1011.00",
                "title": "Chief Executives",
                "description": "Determine organizational direction.",
            },
            {
                "onet_soc_code": "13-2011.00",
                "title": "Accountants and Auditors",
                "description": "Examine financial records.",
            },
        ]
    )
    job_zones = pd.DataFrame(
        [
            {
                "onet_soc_code": "11-1011.00",
                "job_zone": "4",
                "date": "07/2026",
                "domain_source": "Analyst",
            },
            {
                "onet_soc_code": "13-2011.00",
                "job_zone": "3",
                "date": "07/2026",
                "domain_source": "Analyst",
            },
        ]
    )
    occupations = build_occupations(occupation_data, job_zones)
    aliases = build_aliases(
        pd.DataFrame(
            [
                {
                    "onet_soc_code": "11-1011.00",
                    "job_title": "CEO",
                    "short_title": "",
                    "sources": "Employer",
                },
                {
                    "onet_soc_code": "13-2011.00",
                    "job_title": "CPA",
                    "short_title": "",
                    "sources": "Employer",
                },
            ]
        )
    )
    tasks = build_tasks(
        pd.DataFrame(
            [
                {
                    "onet_soc_code": "11-1011.00",
                    "task_id": "1",
                    "task": "Direct organizational policy.",
                    "task_type": "Core",
                    "incumbents_responding": "10",
                    "date": "07/2026",
                    "domain_source": "Analyst",
                },
                {
                    "onet_soc_code": "13-2011.00",
                    "task_id": "2",
                    "task": "Audit financial records.",
                    "task_type": "Core",
                    "incumbents_responding": "12",
                    "date": "07/2026",
                    "domain_source": "Analyst",
                },
            ]
        )
    )
    skill_ratings = build_structured_ratings(
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
    profiles = build_occupation_profiles(occupations, aliases, tasks, skill_ratings)
    related_qrels = pd.DataFrame(
        [
            {
                "query_onet_soc_code": "11-1011.00",
                "candidate_onet_soc_code": "13-2011.00",
                "relatedness_tier": "Primary-Short",
                "relatedness_index": 1,
                "relevance_grade": 3,
            },
            {
                "query_onet_soc_code": "13-2011.00",
                "candidate_onet_soc_code": "11-1011.00",
                "relatedness_tier": "Primary-Short",
                "relatedness_index": 1,
                "relevance_grade": 3,
            },
        ]
    )
    market = normalize_bls_market(
        pd.DataFrame(
            [
                {
                    "nem_title": "Chief executives",
                    "nem_code": "11-1011",
                    "occupation_type": "Line item",
                    "employment_2024_thousands": "313.2",
                    "employment_2034_thousands": "329.1",
                    "employment_change_2024_34_thousands": "15.9",
                    "employment_change_pct": "5.1",
                    "annual_openings_2024_34_thousands": "19.8",
                    "median_annual_wage_2024": "$206,680",
                    "typical_education": "Bachelor's degree",
                    "work_experience": "5 years or more",
                    "on_the_job_training": "None",
                },
                {
                    "nem_title": "Accountants and auditors",
                    "nem_code": "13-2011",
                    "occupation_type": "Line item",
                    "employment_2024_thousands": "1562.0",
                    "employment_2034_thousands": "1633.0",
                    "employment_change_2024_34_thousands": "71.0",
                    "employment_change_pct": "4.5",
                    "annual_openings_2024_34_thousands": "124.2",
                    "median_annual_wage_2024": "$81,680",
                    "typical_education": "Bachelor's degree",
                    "work_experience": "None",
                    "on_the_job_training": "None",
                },
            ]
        )
    )
    crosswalk = normalize_crosswalk(
        pd.DataFrame(
            [
                {
                    "onet_soc_code": "11-1011.00",
                    "onet_soc_title": "Chief Executives",
                    "nem_code": "11-1011",
                    "nem_title": "Chief executives",
                },
                {
                    "onet_soc_code": "13-2011.00",
                    "onet_soc_title": "Accountants and Auditors",
                    "nem_code": "13-2011",
                    "nem_title": "Accountants and auditors",
                },
            ]
        )
    )
    occupation_market = join_market_context(
        crosswalk, market, valid_onet_codes=occupations["onet_soc_code"]
    )
    return {
        "occupations": occupations,
        "aliases": aliases,
        "tasks": tasks,
        "skill_ratings": skill_ratings,
        "related_qrels": related_qrels,
        "bls_market": market,
        "onet_bls_crosswalk": crosswalk,
        "occupation_market": occupation_market,
        "occupation_profiles": profiles,
    }
