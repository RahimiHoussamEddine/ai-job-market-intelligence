from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from data_test_support import rating_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stage(connection: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    temporary = f"_{name}"
    connection.register(temporary, frame)
    connection.execute(f'CREATE TABLE "stg_{name}" AS SELECT * FROM "{temporary}"')
    connection.unregister(temporary)


def test_sql_transformation_mirror_compiles_and_retains_lineage() -> None:
    connection = duckdb.connect()
    try:
        _stage(
            connection,
            "occupation_data",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "title": "Chief Executives",
                        "description": "Lead   organizations.",
                    },
                    {
                        "onet_soc_code": "13-2011.00",
                        "title": "Accountants and Auditors",
                        "description": "Audit records.",
                    },
                ]
            ),
        )
        _stage(
            connection,
            "job_zones",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "job_zone": "4",
                        "date": "07/2026",
                        "domain_source": "Analyst   survey",
                    }
                ]
            ),
        )
        _stage(
            connection,
            "job_titles",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "job_title": "Chief   Executive Officer",
                        "short_title": "CEO",
                        "source": "Employer",
                    }
                ]
            ),
        )
        _stage(
            connection,
            "task_statements",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "task_id": "1",
                        "task": "Direct   policy.",
                        "task_type": "Core",
                        "incumbents_responding": "10",
                        "date": "07/2026",
                        "domain_source": "Analyst",
                    }
                ]
            ),
        )
        transferable = pd.DataFrame(
            [rating_row("11-1011.00", "Chief Executives", "2.A.1", "Leadership")]
        )
        knowledge = pd.DataFrame(
            [rating_row("11-1011.00", "Chief Executives", "2.C.1", "Economics")]
        )
        _stage(connection, "transferable_skills", transferable)
        _stage(connection, "knowledge", knowledge)
        _stage(
            connection,
            "related_occupations",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "related_onet_soc_code": "13-2011.00",
                        "relatedness_tier": "Primary-Short",
                        "index": "1",
                    }
                ]
            ),
        )
        _stage(
            connection,
            "bls_occupation",
            pd.DataFrame(
                [
                    {
                        "nem_title": "Chief   executives",
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
                    }
                ]
            ),
        )
        _stage(
            connection,
            "bls_crosswalk",
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": "11-1011.00",
                        "onet_soc_title": "Chief Executives",
                        "nem_code": "11-1011",
                        "nem_title": "Chief executives",
                    }
                ]
            ),
        )

        for filename in (
            "skill_vectors.sql",
            "occupation_profiles.sql",
            "market_crosswalk.sql",
            "quality_audits.sql",
        ):
            connection.execute((PROJECT_ROOT / "sql" / filename).read_text(encoding="utf-8"))

        occupation = connection.execute(
            "SELECT description, job_zone, job_zone_domain_source FROM occupations "
            "WHERE onet_soc_code='11-1011.00'"
        ).fetchone()
        rating = connection.execute(
            "SELECT sample_size, standard_error, domain_source FROM skill_ratings "
            "WHERE feature_family='transferable_skill'"
        ).fetchone()
        market = connection.execute(
            "SELECT nem_title, employment_2024, annual_openings_2024_34 FROM bls_market"
        ).fetchone()

        assert occupation == ("Lead organizations.", 4, "Analyst survey")
        assert rating == (120, 0.1, "Analyst")
        assert market == ("Chief executives", 1_200, 200)
        assert connection.execute("SELECT count(*) FROM quality_audit_results").fetchone()[0] > 0
    finally:
        connection.close()
