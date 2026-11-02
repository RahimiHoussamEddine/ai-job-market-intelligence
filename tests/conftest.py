from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from job_market_intelligence.artifacts import (
    CORE_TABLE_SCHEMAS,
    parquet_file_manifest,
    sha256_file,
)
from job_market_intelligence.service import clear_service_cache


def _write_table(
    bundle: Path, logical_name: str, rows: list[dict[str, object]]
) -> dict[str, object]:
    path = bundle / f"{logical_name}.parquet"
    table = pa.Table.from_pylist(rows, schema=CORE_TABLE_SCHEMAS[logical_name])
    pq.write_table(table, path, compression="zstd")
    return parquet_file_manifest(logical_name, path)


@pytest.fixture
def synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project_root = tmp_path / "synthetic-project"
    bundle = project_root / "artifacts" / "v1"
    config_dir = project_root / "configs"
    bundle.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    config_path = config_dir / "project.toml"
    config_path.write_text(
        '[project]\nname = "synthetic-test-fixture"\n[profiles]\njob_zone_levels = 4\n',
        encoding="utf-8",
    )
    requirements_path = project_root / "requirements.lock"
    requirements_path.write_text("synthetic-test-lock==1\n", encoding="utf-8")
    monkeypatch.setenv("JOB_MARKET_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("JOB_INTELLIGENCE_ARTIFACT_DIR", str(bundle))
    clear_service_cache()

    occupation_rows = [
        {
            "onet_soc_code": "15-1252.00",
            "base_soc_code": "15-1252",
            "soc_major_group": "15",
            "canonical_title": "Software Developers",
            "description": "Develop and maintain software applications.",
            "job_zone": 4,
            "title_text": "software developer applications programmer",
            "work_text": "design develop test software applications",
            "core_task_count": 8,
            "alias_count": 5,
            "structured_observed_count": 2,
        },
        {
            "onet_soc_code": "15-2051.00",
            "base_soc_code": "15-2051",
            "soc_major_group": "15",
            "canonical_title": "Data Scientists",
            "description": "Use analytical techniques to extract insights from data.",
            "job_zone": 4,
            "title_text": "data scientist analytics",
            "work_text": "analyze data build statistical models",
            "core_task_count": 7,
            "alias_count": 4,
            "structured_observed_count": 1,
        },
        {
            "onet_soc_code": "29-1141.00",
            "base_soc_code": "29-1141",
            "soc_major_group": "29",
            "canonical_title": "Registered Nurses",
            "description": "Assess patient health and provide nursing care.",
            "job_zone": 3,
            "title_text": "registered nurse patient care",
            "work_text": "assess patients administer nursing care",
            "core_task_count": 9,
            "alias_count": 6,
            "structured_observed_count": 2,
        },
    ]
    market_rows = [
        {
            "onet_soc_code": "15-1252.00",
            "nem_code": "15-1252",
            "employment_2024": 1693900,
            "employment_2034": 1981200,
            "employment_change_2024_34": 287300,
            "employment_change_pct": 17.0,
            "annual_openings_2024_34": 140100,
            "median_annual_wage_2024": 130000,
            "typical_education": "Bachelor's degree",
            "work_experience": "None",
            "on_the_job_training": "None",
        },
        {
            "onet_soc_code": "15-2051.00",
            "nem_code": "15-2051",
            "employment_2024": 245900,
            "employment_2034": 319700,
            "employment_change_2024_34": 73800,
            "employment_change_pct": 30.0,
            "annual_openings_2024_34": 20400,
            "median_annual_wage_2024": 110000,
            "typical_education": "Bachelor's degree",
            "work_experience": "None",
            "on_the_job_training": "None",
        },
        {
            "onet_soc_code": "29-1141.00",
            "nem_code": "29-1141",
            "employment_2024": 3391900,
            "employment_2034": 3595400,
            "employment_change_2024_34": 203500,
            "employment_change_pct": 6.0,
            "annual_openings_2024_34": 197200,
            "median_annual_wage_2024": 86000,
            "typical_education": "Bachelor's degree",
            "work_experience": "None",
            "on_the_job_training": "None",
        },
    ]
    skill_rows = [
        {
            "onet_soc_code": "15-1252.00",
            "feature_family": "transferable_skill",
            "element_id": "2.A.1.a",
            "scale_id": "IM",
            "element_name": "Critical Thinking",
            "scale_name": "Importance",
            "raw_value": 4.5,
            "normalized_value": 0.90,
            "sample_size": 120,
            "standard_error": 0.10,
            "lower_ci_bound": 4.3,
            "upper_ci_bound": 4.7,
            "observed": True,
            "recommend_suppress": False,
            "not_relevant": False,
            "rating_date": date(2026, 5, 1),
            "domain_source": "Incumbent",
        },
        {
            "onet_soc_code": "15-1252.00",
            "feature_family": "transferable_skill",
            "element_id": "2.B.1.a",
            "scale_id": "IM",
            "element_name": "Programming",
            "scale_name": "Importance",
            "raw_value": 4.8,
            "normalized_value": 0.96,
            "sample_size": 120,
            "standard_error": 0.08,
            "lower_ci_bound": 4.6,
            "upper_ci_bound": 5.0,
            "observed": True,
            "recommend_suppress": False,
            "not_relevant": False,
            "rating_date": None,
            "domain_source": "Incumbent",
        },
        {
            "onet_soc_code": "15-2051.00",
            "feature_family": "transferable_skill",
            "element_id": "2.A.1.a",
            "scale_id": "IM",
            "element_name": "Critical Thinking",
            "scale_name": "Importance",
            "raw_value": 4.8,
            "normalized_value": 0.96,
            "sample_size": 80,
            "standard_error": 0.12,
            "lower_ci_bound": 4.5,
            "upper_ci_bound": 5.0,
            "observed": True,
            "recommend_suppress": False,
            "not_relevant": False,
            "rating_date": None,
            "domain_source": "Analyst",
        },
        {
            "onet_soc_code": "15-2051.00",
            "feature_family": "transferable_skill",
            "element_id": "2.B.1.a",
            "scale_id": "IM",
            "element_name": "Programming",
            "scale_name": "Importance",
            "raw_value": 1.0,
            "normalized_value": None,
            "sample_size": None,
            "standard_error": None,
            "lower_ci_bound": None,
            "upper_ci_bound": None,
            "observed": False,
            "recommend_suppress": True,
            "not_relevant": False,
            "rating_date": None,
            "domain_source": "Incumbent",
        },
        {
            "onet_soc_code": "29-1141.00",
            "feature_family": "transferable_skill",
            "element_id": "2.A.1.a",
            "scale_id": "IM",
            "element_name": "Critical Thinking",
            "scale_name": "Importance",
            "raw_value": 4.2,
            "normalized_value": 0.84,
            "sample_size": 200,
            "standard_error": 0.06,
            "lower_ci_bound": 4.1,
            "upper_ci_bound": 4.3,
            "observed": True,
            "recommend_suppress": False,
            "not_relevant": False,
            "rating_date": None,
            "domain_source": "Incumbent",
        },
    ]
    cluster_rows = [
        {
            "onet_soc_code": "15-1252.00",
            "cluster_id": 0,
            "cluster_label": "software and analytics",
            "pca_x": -0.8,
            "pca_y": 0.2,
        },
        {
            "onet_soc_code": "15-2051.00",
            "cluster_id": 0,
            "cluster_label": "software and analytics",
            "pca_x": -0.5,
            "pca_y": 0.5,
        },
        {
            "onet_soc_code": "29-1141.00",
            "cluster_id": 1,
            "cluster_label": "patient care",
            "pca_x": 1.2,
            "pca_y": -0.3,
        },
    ]
    neighbor_rows = [
        {
            "query_code": "15-1252.00",
            "destination_code": "15-2051.00",
            "rank": 1,
            "total_score": 0.90,
            "lexical_score": 0.82,
            "lsa_score": 0.91,
            "structured_score": 0.80,
            "development_distance": 0.20,
            "explanation": "Shared analytical and programming work.",
        },
        {
            "query_code": "15-1252.00",
            "destination_code": "29-1141.00",
            "rank": 2,
            "total_score": 0.30,
            "lexical_score": 0.20,
            "lsa_score": 0.35,
            "structured_score": 0.40,
            "development_distance": 0.60,
            "explanation": "Some shared problem-solving ratings.",
        },
        {
            "query_code": "15-2051.00",
            "destination_code": "15-1252.00",
            "rank": 1,
            "total_score": 0.90,
            "lexical_score": 0.82,
            "lsa_score": 0.91,
            "structured_score": 0.80,
            "development_distance": 0.20,
            "explanation": "Shared analytical and programming work.",
        },
        {
            "query_code": "15-2051.00",
            "destination_code": "29-1141.00",
            "rank": 2,
            "total_score": 0.40,
            "lexical_score": 0.30,
            "lsa_score": 0.45,
            "structured_score": None,
            "development_distance": None,
            "explanation": "Text evidence only; structured overlap is insufficient.",
        },
        {
            "query_code": "29-1141.00",
            "destination_code": "15-2051.00",
            "rank": 1,
            "total_score": 0.40,
            "lexical_score": 0.30,
            "lsa_score": 0.45,
            "structured_score": 0.50,
            "development_distance": 0.50,
            "explanation": "Shared analysis tasks.",
        },
        {
            "query_code": "29-1141.00",
            "destination_code": "15-1252.00",
            "rank": 2,
            "total_score": 0.30,
            "lexical_score": 0.20,
            "lsa_score": 0.35,
            "structured_score": 0.40,
            "development_distance": 0.60,
            "explanation": "Some shared problem-solving ratings.",
        },
    ]
    catalog_rows = [
        {"row_index": index, "onet_soc_code": row["onet_soc_code"]}
        for index, row in enumerate(occupation_rows)
    ]

    table_rows = {
        "occupations": occupation_rows,
        "occupation_market": market_rows,
        "skill_ratings": skill_rows,
        "clusters": cluster_rows,
        "neighbors": neighbor_rows,
        "search_catalog": catalog_rows,
    }
    files = {
        logical_name: _write_table(bundle, logical_name, rows)
        for logical_name, rows in table_rows.items()
    }

    documents = [row["title_text"] for row in occupation_rows]
    vectorizer = TfidfVectorizer(dtype=np.float64)
    title_matrix = vectorizer.fit_transform(documents).tocsr()
    title_matrix.sum_duplicates()
    title_matrix.sort_indices()
    vectorizer_path = bundle / "title_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    files["title_vectorizer"] = {
        "path": vectorizer_path.name,
        "format": "joblib",
        "bytes": vectorizer_path.stat().st_size,
        "sha256": sha256_file(vectorizer_path),
    }

    matrix_path = bundle / "title_matrix.npz"
    sparse.save_npz(matrix_path, title_matrix, compressed=True)
    files["title_matrix"] = {
        "path": matrix_path.name,
        "format": "scipy_npz",
        "bytes": matrix_path.stat().st_size,
        "sha256": sha256_file(matrix_path),
        "shape": list(title_matrix.shape),
        "dtype": str(title_matrix.dtype),
    }

    smoke_path = bundle / "smoke_queries.json"
    smoke_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queries": [{"id": "arbitrary-text", "query": "software applications"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files["smoke_queries"] = {
        "path": smoke_path.name,
        "format": "json",
        "bytes": smoke_path.stat().st_size,
        "sha256": sha256_file(smoke_path),
    }

    manifest = {
        "schema_version": 1,
        "artifact_version": "synthetic-v1",
        "snapshot_date": "2026-08-14",
        "models": {
            "title_candidate_id": "synthetic_title_tfidf",
            "transition_candidate_id": "synthetic_transition_hybrid",
            "title_feature_count": title_matrix.shape[1],
            "title_matrix_normalized": True,
            "stored_transition_limit": 250,
            "neighbor_display_max": 25,
            "transition_components": ["lexical", "lsa", "structured"],
        },
        "provenance": {
            "repository_commit": "a" * 40,
            "project_config_sha256": sha256_file(config_path),
            "source_manifest_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
            "locked_rankings_sha256": "d" * 64,
            "locked_metrics_sha256": "e" * 64,
            "environment_lock_sha256": sha256_file(requirements_path),
            "processed_manifest_sha256": "4" * 64,
            "feature_manifest_sha256": "5" * 64,
            "metric_contract_sha256": "6" * 64,
            "candidate_manifest_sha256": "9" * 64,
            "evaluation_universe_sha256": "0" * 64,
            "implementation_manifest_sha256": "a" * 64,
            "title_candidate_grid_sha256": "7" * 64,
            "transition_candidate_grid_sha256": "8" * 64,
            "source_hashes": {
                "onet_occupation_data": "1" * 64,
                "onet_related_occupations": "2" * 64,
                "bls_2024_34": "3" * 64,
            },
        },
        "evidence": {"status": "not_generated", "report_files": {}},
        "files": files,
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    yield bundle
    clear_service_cache()


@pytest.fixture
def resign_artifact_file(synthetic_bundle: Path):
    def resign(logical_name: str) -> None:
        manifest_path = synthetic_bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = synthetic_bundle / manifest["files"][logical_name]["path"]
        manifest["files"][logical_name]["bytes"] = path.stat().st_size
        manifest["files"][logical_name]["sha256"] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return resign


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
