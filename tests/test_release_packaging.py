from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from job_market_intelligence.artifacts import (
    ManifestValidationError,
    _validate_candidate_spec_payload,
    load_artifact_bundle,
    write_artifact_bundle,
)
from job_market_intelligence.clustering import fit_exploratory_clusters
from job_market_intelligence.config import load_config
from job_market_intelligence.frontier import materialize_pareto_frontiers
from job_market_intelligence.modeling import (
    CandidateSpec,
    fit_production_title_index,
    fit_production_transition_components,
)
from job_market_intelligence.transitions import build_transition_candidates


def test_candidate_spec_accepts_zero_weight_disabled_components() -> None:
    payload = {
        "candidate_id": "transition_hybrid",
        "task": "transition",
        "representation": "transition_hybrid",
        "analyzer": None,
        "lsa_dimensions": 64,
        "weights": [["lexical", 0.25], ["lsa", 0.0], ["structured", 0.75]],
    }

    assert _validate_candidate_spec_payload(payload, "test_candidate") == payload

    invalid = {**payload, "weights": [["lexical", -0.25], ["structured", 1.25]]}
    with pytest.raises(ManifestValidationError, match="finite and non-negative"):
        _validate_candidate_spec_payload(invalid, "test_candidate")


def _synthetic_release_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = [f"{('11', '15', '29')[index % 3]}-{1000 + index:04d}.00" for index in range(60)]
    profiles: list[dict[str, object]] = []
    aliases: list[dict[str, object]] = []
    ratings: list[dict[str, object]] = []
    market: list[dict[str, object]] = []
    for index, code in enumerate(codes):
        canonical = f"Synthetic Occupation {index} Specialist"
        work_text = (
            f"perform synthetic occupation token{index} systems analysis reporting "
            f"workflow{index} method{index} technology{index}"
        )
        profiles.append(
            {
                "onet_soc_code": code,
                "base_soc_code": code.split(".", 1)[0],
                "soc_major_group": code[:2],
                "canonical_title": canonical,
                "description": f"Synthetic description for occupation {index}.",
                "job_zone": (2, 3, 4, 5)[index % 4],
                "title_text": f"{canonical}\nSynthetic role {index}",
                "work_text": work_text,
                "core_task_count": 2,
                "alias_count": 1,
                "structured_observed_count": 4,
            }
        )
        aliases.append(
            {
                "onet_soc_code": code,
                "job_title": f"Synthetic Role {index}",
                "normalized_job_title": f"synthetic role {index}",
            }
        )
        for feature_index, family in enumerate(
            ("transferable_skill", "transferable_skill", "knowledge", "knowledge")
        ):
            normalized = ((index + feature_index) % 10) / 10
            raw_value = 1.0 + 4.0 * normalized
            ratings.append(
                {
                    "onet_soc_code": code,
                    "feature_family": family,
                    "element_id": f"E{feature_index}",
                    "scale_id": "IM",
                    "element_name": f"Synthetic feature {feature_index}",
                    "scale_name": "Importance",
                    "raw_value": raw_value,
                    "normalized_value": normalized,
                    "sample_size": 20,
                    "standard_error": 0.05,
                    "lower_ci_bound": raw_value - 0.1,
                    "upper_ci_bound": raw_value + 0.1,
                    "observed": True,
                    "recommend_suppress": False,
                    "not_relevant": False,
                    "rating_date": date(2026, 1, 1),
                    "domain_source": "synthetic-test",
                }
            )
        market.append(
            {
                "onet_soc_code": code,
                "nem_code": code.split(".", 1)[0],
                "employment_2024": 1000 + index,
                "employment_2034": 1100 + index,
                "employment_change_2024_34": 100,
                "employment_change_pct": 10.0,
                "annual_openings_2024_34": 50 + index,
                "median_annual_wage_2024": 60000 + index * 100,
                "typical_education": "Bachelor's degree",
                "work_experience": "None",
                "on_the_job_training": "None",
            }
        )
    return tuple(pd.DataFrame(rows) for rows in (profiles, aliases, ratings, market))


def test_synthetic_production_fit_round_trips_through_release_writer(tmp_path: Path) -> None:
    profiles, aliases, ratings, market = _synthetic_release_frames()
    title_spec = CandidateSpec(
        "title_word_char",
        "title",
        "canonical_title",
        "word_char",
        0,
        (("lexical", 1.0),),
    )
    transition_spec = CandidateSpec(
        "transition_title_word_char",
        "transition",
        "canonical_title",
        "word_char",
        0,
        (("lexical", 1.0),),
    )
    fit_codes = tuple(sorted(profiles["onet_soc_code"].astype(str))[:-12])
    title = fit_production_title_index(
        profiles,
        aliases,
        title_spec,
        fit_codes=fit_codes,
        min_df=1,
    )
    transition = fit_production_transition_components(
        profiles,
        ratings,
        transition_spec,
        fit_codes=fit_codes,
        minimum_structured_overlap=1,
        min_df=1,
        diagnostic_lsa_dimensions=8,
    )
    neighbors = build_transition_candidates(
        transition.occupation_codes,
        transition.lexical_scores,
        transition.lsa_scores,
        transition.structured_scores,
        transition.development_distances,
        weights=transition.weights,
    )
    clusters = fit_exploratory_clusters(
        transition.cluster_embedding,
        transition.occupation_codes,
        transition.term_matrix,
        transition.feature_names,
        soc_major_groups=profiles.sort_values("onet_soc_code")["soc_major_group"].tolist(),
    )
    search_catalog = pd.DataFrame({"row_index": range(60), "onet_soc_code": title.occupation_codes})
    project_config = Path(__file__).parents[1] / "configs" / "project.toml"
    config = load_config(project_config)
    profiles = profiles.sort_values("onet_soc_code", kind="stable").reset_index(drop=True)
    ratings = ratings.sort_values(
        ["onet_soc_code", "feature_family", "element_id", "scale_id"], kind="stable"
    ).reset_index(drop=True)
    market = market.sort_values(["onet_soc_code", "nem_code"], kind="stable").reset_index(drop=True)
    output = write_artifact_bundle(
        tmp_path / config.project.artifact_version,
        artifact_version=config.project.artifact_version,
        snapshot_date=config.project.snapshot_date.isoformat(),
        occupations=profiles,
        occupation_market=market,
        skill_ratings=ratings,
        clusters=clusters.assignments,
        neighbors=neighbors,
        search_catalog=search_catalog,
        title_vectorizer=title.transformer,
        title_matrix=title.matrix,
        smoke_queries=[
            {
                "id": "synthetic-title",
                "query": str(profiles.iloc[0]["canonical_title"]),
            }
        ],
        evidence_reports={},
        model_metadata={
            "title_candidate_id": title_spec.candidate_id,
            "transition_candidate_id": transition_spec.candidate_id,
        },
        provenance={
            "repository_commit": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "split_manifest_sha256": "c" * 64,
            "locked_rankings_sha256": "d" * 64,
            "locked_metrics_sha256": "e" * 64,
            "processed_manifest_sha256": "1" * 64,
            "feature_manifest_sha256": "2" * 64,
            "metric_contract_sha256": "3" * 64,
            "candidate_manifest_sha256": "4" * 64,
            "evaluation_universe_sha256": "5" * 64,
            "implementation_manifest_sha256": "6" * 64,
            "title_candidate_grid_sha256": "7" * 64,
            "transition_candidate_grid_sha256": "8" * 64,
            "source_hashes": {"synthetic": "f" * 64},
        },
        project_config_path=project_config,
    )
    loaded = load_artifact_bundle(output, project_config_path=project_config)
    assert loaded.search_matrix.shape == title.matrix.shape
    assert loaded.tables["neighbors"].num_rows == 60 * 59
    frontiers = materialize_pareto_frontiers(neighbors, market)
    assert set(frontiers["source_code"]) == set(title.occupation_codes)
