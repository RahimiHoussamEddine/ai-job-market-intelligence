from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from job_market_intelligence.config import load_config
from job_market_intelligence.evaluation import (
    EvaluationIntegrityError,
    build_evaluation_universe,
    build_title_benchmark_evidence,
    build_transition_benchmark_evidence,
    generate_locked_rankings,
    run_development_benchmarks,
    transition_evaluation_query_codes,
    verify_development_decision,
)
from job_market_intelligence.modeling import (
    CandidateSpec,
    multi_task_development_decision_payload,
)
from job_market_intelligence.splits import build_fold_manifest


def _experiment_frames() -> tuple[pd.DataFrame, ...]:
    codes = [f"{'11' if index < 10 else '15'}-{1000 + index:04d}.00" for index in range(20)]
    role_by_position = {
        0: (3, "discovery"),
        1: (4, "discovery"),
        2: (2, "selection"),
        3: (1, "confirmation"),
        4: (0, "test"),
    }
    profiles: list[dict[str, object]] = []
    aliases: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    ratings: list[dict[str, object]] = []
    for index, code in enumerate(codes):
        fold, role = role_by_position[index % 5]
        major = code[:2]
        profiles.append(
            {
                "onet_soc_code": code,
                "base_soc_code": code[:7],
                "soc_major_group": major,
                "canonical_title": f"Occupation {index} Analyst",
                "work_text": f"analyze occupation token{index} systems and reports",
            }
        )
        aliases.append(
            {
                "onet_soc_code": code,
                "job_title": f"Occupation {index} Specialist",
                "normalized_job_title": f"occupation {index} specialist",
            }
        )
        folds.append(
            {
                "onet_soc_code": code,
                "base_soc_code": code[:7],
                "soc_major_group": major,
                "fold": fold,
                "role": role,
            }
        )
        for feature_index in range(2):
            ratings.append(
                {
                    "onet_soc_code": code,
                    "feature_family": ("transferable_skill" if feature_index == 0 else "knowledge"),
                    "element_id": f"E{feature_index}",
                    "scale_id": "IM",
                    "normalized_value": ((index + feature_index) % 10) / 10,
                    "observed": True,
                }
            )
    qrels: list[dict[str, object]] = []
    tiers = (("Primary-Short", 3), ("Primary-Long", 2), ("Supplemental", 1))
    test_codes = {code for index, code in enumerate(codes) if index % 5 == 4}
    for index, query in enumerate(codes):
        pool = [
            codes[(index + offset) % len(codes)]
            for offset in range(1, len(codes))
            if codes[(index + offset) % len(codes)] != query
            and (query in test_codes or codes[(index + offset) % len(codes)] not in test_codes)
        ]
        for offset, (candidate, (tier, grade)) in enumerate(
            zip(pool, tiers, strict=False), start=1
        ):
            qrels.append(
                {
                    "query_onet_soc_code": query,
                    "candidate_onet_soc_code": candidate,
                    "relatedness_tier": tier,
                    "relatedness_index": offset,
                    "relevance_grade": grade,
                }
            )
    return tuple(pd.DataFrame(rows) for rows in (profiles, aliases, ratings, qrels, folds))


def _candidate_specs() -> tuple[tuple[CandidateSpec, ...], tuple[CandidateSpec, ...]]:
    title = (
        CandidateSpec(
            "title_word_char",
            "title",
            "canonical_title",
            "word_char",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "title_char",
            "title",
            "canonical_title",
            "char",
            0,
            (("lexical", 1.0),),
        ),
    )
    transition = (
        CandidateSpec(
            "transition_title_word_char",
            "transition",
            "canonical_title",
            "word_char",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_work_word",
            "transition",
            "work_profile",
            "word",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_structured",
            "transition",
            "structured",
            None,
            0,
            (("structured", 1.0),),
        ),
    )
    return title, transition


def test_reduced_end_to_end_development_and_locked_runner_never_uses_test_for_selection() -> None:
    profiles, aliases, ratings, qrels, folds = _experiment_frames()
    title_specs, transition_specs = _candidate_specs()
    development = run_development_benchmarks(
        profiles,
        aliases,
        ratings,
        qrels,
        folds,
        title_candidates=title_specs,
        transition_candidates=transition_specs,
        min_df=1,
        minimum_structured_overlap=1,
        bootstrap_replicates=100,
    )
    test_codes = set(folds.loc[folds["role"].eq("test"), "onet_soc_code"])
    assert not set(development.alias_queries["query_code"]) & test_codes
    assert not set(development.transition_rankings["query_code"]) & test_codes
    assert development.title_confirmation.release_candidate_id in {
        candidate.candidate_id for candidate in title_specs
    }
    assert development.transition_confirmation.release_candidate_id in {
        candidate.candidate_id for candidate in transition_specs
    }

    decision = multi_task_development_decision_payload(
        title_selection=development.title_selection,
        title_confirmation=development.title_confirmation,
        transition_selection=development.transition_selection,
        transition_confirmation=development.transition_confirmation,
        experiment_version="synthetic-v2",
        project_config_sha256="a" * 64,
        fold_manifest_sha256="b" * 64,
        feature_manifest_sha256="c" * 64,
        metric_contract_sha256="d" * 64,
        environment_lock_sha256="e" * 64,
        processed_manifest_sha256="f" * 64,
        candidate_manifest_sha256="3" * 64,
        evaluation_universe_sha256="4" * 64,
        implementation_manifest_sha256="5" * 64,
        title_candidate_grid_sha256="1" * 64,
        transition_candidate_grid_sha256="2" * 64,
    )
    assert verify_development_decision(decision) == decision["decision_sha256"]
    tampered_decision = dict(decision)
    tampered_confirmation = dict(decision["title_confirmation"])
    tampered_confirmation["fallback_applied"] = not tampered_confirmation["fallback_applied"]
    tampered_decision["title_confirmation"] = tampered_confirmation
    tampered_core = dict(tampered_decision)
    tampered_core.pop("decision_sha256")
    tampered_decision["decision_sha256"] = hashlib.sha256(
        json.dumps(tampered_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with pytest.raises(EvaluationIntegrityError, match="fallback/release semantics"):
        verify_development_decision(tampered_decision)
    locked = generate_locked_rankings(
        profiles,
        aliases,
        ratings,
        folds,
        decision,
        title_candidates=title_specs,
        transition_candidates=transition_specs,
        min_df=1,
        minimum_structured_overlap=1,
    )
    assert set(locked["query_code"]) <= test_codes
    assert set(locked["benchmark"]) == {"title", "transition"}

    slices = [
        {
            "dimension": dimension,
            "value": value,
            "estimate": 0.5,
            "ci_lower": 0.4,
            "ci_upper": 0.6,
            "n_units": 4,
            "descriptive_small_slice": True,
        }
        for dimension, value in (
            ("soc_major_group", "11"),
            ("job_zone", "2"),
            ("profile_completeness", "1-49"),
            ("rating_recency", "2024+"),
        )
    ]
    title_evidence = build_title_benchmark_evidence(
        locked,
        selected_candidate_id=decision["release_candidates"]["title"],
        baseline_candidate_id="title_word_char",
        slices=slices,
        bootstrap_replicates=100,
    )
    expected_test_queries = transition_evaluation_query_codes(qrels, folds, role="test")
    transition_evidence = build_transition_benchmark_evidence(
        locked,
        qrels,
        selected_candidate_id=decision["release_candidates"]["transition"],
        baseline_candidate_id="transition_title_word_char",
        expected_query_codes=expected_test_queries,
        slices=slices,
        bootstrap_replicates=100,
    )
    assert title_evidence["primary_metric"] == "top1_accuracy"
    assert transition_evidence["primary_metric"] == "ndcg_at_10"

    # Target-free generation preserves every fold-zero query, while evidence
    # evaluates only the preregistered judged subset and rejects silent drops.
    unjudged_code = sorted(test_codes)[0]
    subset_qrels = qrels.loc[qrels["query_onet_soc_code"].ne(unjudged_code)]
    subset_codes = transition_evaluation_query_codes(subset_qrels, folds, role="test")
    assert unjudged_code not in subset_codes
    assert unjudged_code in set(locked.loc[locked["benchmark"].eq("transition"), "query_code"])
    subset_evidence = build_transition_benchmark_evidence(
        locked,
        subset_qrels,
        selected_candidate_id=decision["release_candidates"]["transition"],
        baseline_candidate_id="transition_title_word_char",
        expected_query_codes=subset_codes,
        slices=slices,
        bootstrap_replicates=100,
    )
    assert all(metric["n_queries"] == len(subset_codes) for metric in subset_evidence["metrics"])
    if decision["release_candidates"]["title"] == "title_word_char":
        assert title_evidence["baseline_deltas"] == []
    if decision["release_candidates"]["transition"] == "transition_title_word_char":
        assert transition_evidence["baseline_deltas"] == []


def test_test_only_aliases_and_qrels_cannot_change_any_development_decision() -> None:
    profiles, aliases, ratings, qrels, folds = _experiment_frames()
    title_specs, transition_specs = _candidate_specs()
    arguments = {
        "title_candidates": title_specs,
        "transition_candidates": transition_specs,
        "min_df": 1,
        "minimum_structured_overlap": 1,
        "bootstrap_replicates": 100,
    }
    original = run_development_benchmarks(profiles, aliases, ratings, qrels, folds, **arguments)
    selection_alias = aliases.loc[
        aliases["onet_soc_code"].isin(folds.loc[folds["role"].eq("selection"), "onet_soc_code"])
    ].iloc[0]
    test_code = str(folds.loc[folds["role"].eq("test"), "onet_soc_code"].iloc[0])
    adversarial_aliases = pd.concat(
        [
            aliases,
            pd.DataFrame(
                [
                    {
                        "onet_soc_code": test_code,
                        "job_title": selection_alias["job_title"],
                        "normalized_job_title": selection_alias["normalized_job_title"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    test_codes = set(folds.loc[folds["role"].eq("test"), "onet_soc_code"].astype(str))
    adversarial_qrels = qrels.copy()
    hidden = adversarial_qrels["query_onet_soc_code"].astype(str).isin(
        test_codes
    ) | adversarial_qrels["candidate_onet_soc_code"].astype(str).isin(test_codes)
    adversarial_qrels.loc[hidden, "relevance_grade"] = 99
    changed = run_development_benchmarks(
        profiles,
        adversarial_aliases,
        ratings,
        adversarial_qrels,
        folds,
        **arguments,
    )
    assert_frame_equal(original.alias_queries, changed.alias_queries)
    for first, second in (
        (original.title_target_ranks, changed.title_target_ranks),
        (original.transition_rankings, changed.transition_rankings),
        (original.title_selection_results, changed.title_selection_results),
        (original.title_confirmation_results, changed.title_confirmation_results),
        (original.transition_selection_results, changed.transition_selection_results),
        (original.transition_confirmation_results, changed.transition_confirmation_results),
    ):
        assert_frame_equal(
            first.drop(columns="latency_ms"),
            second.drop(columns="latency_ms"),
        )
    assert original.title_selection == changed.title_selection
    assert original.title_confirmation == changed.title_confirmation
    assert original.transition_selection == changed.transition_selection
    assert original.transition_confirmation == changed.transition_confirmation
    assert set(original.title_confirmation_results["candidate_id"]) == {
        original.title_selection.selected_candidate_id,
        "title_word_char",
    }
    assert set(original.transition_confirmation_results["candidate_id"]) == {
        original.transition_selection.selected_candidate_id,
        "transition_title_word_char",
    }


def test_verified_snapshot_evaluation_universe_ids_are_frozen_without_metrics() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "project.toml")
    tables = root / "data" / "processed" / "snapshots" / config.project.artifact_version / "tables"
    if not tables.is_dir():
        pytest.skip(
            "requires the verified local O*NET/BLS snapshot; run "
            "`make data-download` and `make data-build`"
        )
    profiles = pd.read_parquet(tables / "occupation_profiles.parquet")
    aliases = pd.read_parquet(tables / "aliases.parquet")
    qrels = pd.read_parquet(tables / "related_qrels.parquet")
    folds = build_fold_manifest(
        profiles,
        n_splits=config.splits.n_splits,
        random_seed=config.project.random_seed,
        role_folds={
            "discovery": config.splits.discovery_folds,
            "selection": (config.splits.selection_fold,),
            "confirmation": (config.splits.confirmation_fold,),
            "test": (config.splits.test_fold,),
        },
    )
    universe = build_evaluation_universe(
        aliases,
        profiles,
        qrels,
        folds,
        maximum_aliases_per_occupation=25,
        seed=20260814,
        implementation_manifest_sha256="0" * 64,
    )
    expected = {
        "title": {
            "selection": (
                203,
                4506,
                "d93b7b16c18e17c161d1a997b9641c2e267c2c880a371deb5125ffdb8fc453a4",
            ),
            "confirmation": (
                204,
                4418,
                "bf661fa43d16eeb911dd51eb7e0068533658ea045313afa1441a919ac24e675f",
            ),
            "test": (203, 4087, "9c79b51dff661d0b9d022ca46cb64306c878880241a936624d7ce5e0f7964424"),
        },
        "transition": {
            "selection": (
                203,
                183,
                "5affa9ef88e5986961ba02ce21a05a7261c817b5f04b56728e7b4f3249e83786",
            ),
            "confirmation": (
                204,
                179,
                "ab72f85e191c2c4d7dba88f626555b833563f636ddb332ac1cae8455258bbf5d",
            ),
            "test": (203, 188, "cde3cf3b7eef5bae8252fcb0402ab0c501557304422d6ed4672a17e829523213"),
        },
    }
    for role, (assigned, eligible, digest) in expected["title"].items():
        entry = universe["title"]["roles"][role]
        assert (
            entry["assigned_occupation_count"],
            entry["eligible_query_count"],
            entry["query_ids_sha256"],
        ) == (assigned, eligible, digest)
    for role, (assigned, eligible, digest) in expected["transition"].items():
        entry = universe["transition"]["roles"][role]
        assert (
            entry["assigned_query_count"],
            entry["eligible_query_count"],
            entry["query_codes_sha256"],
        ) == (assigned, eligible, digest)
