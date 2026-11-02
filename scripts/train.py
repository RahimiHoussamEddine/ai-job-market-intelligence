#!/usr/bin/env python3
"""Refit frozen champions and publish the validated serving release bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from job_market_intelligence.artifacts import CORE_TABLE_SCHEMAS, write_artifact_bundle
from job_market_intelligence.clustering import (
    PREDECLARED_K_VALUES,
    PREDECLARED_SEEDS,
    fit_exploratory_clusters,
)
from job_market_intelligence.config import load_config
from job_market_intelligence.evaluation import (
    require_reporting_git_state,
    resume_reporting_only,
    sha256_file,
    validate_evaluation_universe,
    verify_development_decision,
    verify_implementation_manifest,
)
from job_market_intelligence.lexical import cosine_scores
from job_market_intelligence.modeling import (
    CandidateSpec,
    candidate_grid_digest,
    fit_production_title_index,
    fit_production_transition_components,
    title_candidate_grid,
    transition_candidate_grid,
)
from job_market_intelligence.retrieval import (
    fitting_codes_for_phase,
    prepare_alias_queries,
)
from job_market_intelligence.splits import fold_manifest_digest
from job_market_intelligence.transitions import build_transition_candidates
from job_market_intelligence.warehouse import load_processed_manifest, verify_processed_snapshot


class TrainCommandError(RuntimeError):
    """Raised when release inputs drift from the locked experiment."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainCommandError(f"could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise TrainCommandError(f"JSON object required: {path}")
    return value


def _entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["tables"] if entry["name"] == name]
    if len(matches) != 1:
        raise TrainCommandError(f"processed manifest lacks exactly one {name}")
    return matches[0]


def _table(snapshot: Path, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    entry = _entry(manifest, name)
    path = snapshot / entry["parquet_path"]
    if sha256_file(path) != entry["parquet_sha256"]:
        raise TrainCommandError(f"processed table checksum changed: {name}")
    return pd.read_parquet(path)


def _core_frame(name: str, frame: pd.DataFrame, *sort_by: str) -> pd.DataFrame:
    columns = CORE_TABLE_SCHEMAS[name].names
    if missing := sorted(set(columns) - set(frame.columns)):
        raise TrainCommandError(f"{name} is missing serving columns: {missing}")
    result = frame.loc[:, columns].sort_values(list(sort_by), kind="stable")
    return result.reset_index(drop=True)


def _candidate(candidates: tuple[CandidateSpec, ...], candidate_id: str) -> CandidateSpec:
    matches = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise TrainCommandError(f"frozen candidate is absent from production grid: {candidate_id}")
    return matches[0]


def _wrapper(
    report_type: str,
    payload: dict[str, Any],
    *,
    artifact_version: str,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": report_type,
        "evidence_status": "locked",
        "split": split,
        "artifact_version": artifact_version,
        "payload": payload,
    }


def _assert_bound_path(
    state: dict[str, Any],
    repository: Path,
    logical_name: str,
    expected: Path,
) -> None:
    entry = state["inputs"].get(logical_name)
    if not isinstance(entry, dict):
        raise TrainCommandError(f"locked state lacks input pin {logical_name}")
    actual = (repository / entry["path"]).resolve()
    if actual != expected.resolve() or entry["sha256"] != sha256_file(expected):
        raise TrainCommandError(f"release path differs from locked input {logical_name}")


def _smoke_queries(
    title_index: Any,
    occupations: pd.DataFrame,
    *,
    preferred_codes: tuple[str, ...],
) -> list[dict[str, str]]:
    """Choose three deterministic fit-code queries proven nonzero and positive."""

    preferred = set(preferred_codes)
    candidates = occupations.loc[
        occupations["onet_soc_code"].isin(preferred),
        ["onet_soc_code", "canonical_title"],
    ].sort_values("onet_soc_code", kind="stable")
    usable: list[str] = []
    seen: set[str] = set()
    for title in candidates["canonical_title"].astype(str):
        normalized_key = title.casefold().strip()
        if not normalized_key or normalized_key in seen:
            continue
        transformed = sparse.csr_matrix(title_index.transformer.transform([title]), dtype="float64")
        if transformed.nnz == 0:
            continue
        similarities = title_index.matrix @ transformed.T
        if float(similarities.max()) <= 0:
            continue
        seen.add(normalized_key)
        usable.append(title)
    if len(usable) < 3:
        raise TrainCommandError(
            "fitted title transformer cannot supply three nonzero target-free smoke queries"
        )
    selected = [usable[position] for position in (0, len(usable) // 2, len(usable) - 1)]
    return [
        {"id": f"catalog-title-{index + 1:02d}", "query": query}
        for index, query in enumerate(selected)
    ]


def _assert_production_replays_locked_rankings(
    *,
    locked_rankings: pd.DataFrame,
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    evaluation_universe: dict[str, Any],
    title_index: Any,
    transition: Any,
    serving_neighbors: pd.DataFrame,
    maximum_aliases_per_occupation: int,
    random_seed: int,
) -> None:
    """Prove the serving refit exactly replays selected locked ranking evidence."""

    catalog_codes = tuple(str(code) for code in title_index.occupation_codes)
    if catalog_codes != tuple(str(code) for code in transition.occupation_codes):
        raise TrainCommandError("production replay catalogs are misaligned")
    test_codes = tuple(
        fold_manifest.loc[fold_manifest["role"].eq("test"), "onet_soc_code"]
        .astype(str)
        .sort_values(kind="stable")
    )
    title_queries = prepare_alias_queries(
        aliases,
        profiles,
        fold_manifest,
        maximum_per_occupation=maximum_aliases_per_occupation,
        seed=random_seed,
        allowed_query_codes=test_codes,
        ambiguity_scope="full_catalog",
    ).sort_values("query_id", kind="stable")
    if tuple(title_queries["query_id"].astype(str)) != tuple(
        evaluation_universe["title"]["roles"]["test"]["query_ids"]
    ):
        raise TrainCommandError("production replay title queries differ from frozen universe")
    query_matrix = title_index.transformer.transform(
        title_queries["query_text"].astype(str).tolist()
    )
    title_scores = cosine_scores(
        sparse.csr_matrix(query_matrix, dtype=np.float64), title_index.matrix
    )
    code_array = np.asarray(catalog_codes, dtype=object)
    code_index = {code: index for index, code in enumerate(catalog_codes)}
    title_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(title_queries.itertuples(index=False)):
        target = str(query.target_code)
        target_index = code_index[target]
        values = title_scores[query_index]
        target_score = float(values[target_index])
        outranking = (values > target_score) | (
            (values == target_score) & (code_array.astype(str) < target)
        )
        title_rows.append(
            {
                "query_id": str(query.query_id),
                "query_code": str(query.query_code),
                "candidate_code": target,
                "rank": int(np.count_nonzero(outranking)) + 1,
                "score": target_score,
            }
        )
    observed_title = locked_rankings.loc[
        locked_rankings["benchmark"].eq("title")
        & locked_rankings["candidate_id"].eq(title_index.candidate_id),
        ["query_id", "query_code", "candidate_code", "rank", "score"],
    ].sort_values("query_id", kind="stable")
    replay_title = pd.DataFrame(title_rows).sort_values("query_id", kind="stable")
    for column in ("query_id", "query_code", "candidate_code", "rank"):
        if observed_title[column].tolist() != replay_title[column].tolist():
            raise TrainCommandError(f"production title replay differs in {column}")
    if not np.allclose(observed_title["score"], replay_title["score"], rtol=1e-10, atol=1e-12):
        raise TrainCommandError("production title replay scores differ from locked rankings")

    replay_transition = serving_neighbors.loc[
        serving_neighbors["query_code"].isin(test_codes)
        & serving_neighbors["rank"].le(min(20, len(catalog_codes) - 1)),
        ["query_code", "destination_code", "rank", "total_score"],
    ].rename(columns={"destination_code": "candidate_code", "total_score": "score"})
    observed_transition = locked_rankings.loc[
        locked_rankings["benchmark"].eq("transition")
        & locked_rankings["candidate_id"].eq(transition.candidate_id),
        ["query_code", "candidate_code", "rank", "score"],
    ]
    sort_columns = ["query_code", "rank"]
    replay_transition = replay_transition.sort_values(sort_columns, kind="stable")
    observed_transition = observed_transition.sort_values(sort_columns, kind="stable")
    for column in ("query_code", "candidate_code", "rank"):
        if observed_transition[column].tolist() != replay_transition[column].tolist():
            raise TrainCommandError(f"production transition replay differs in {column}")
    if not np.allclose(
        observed_transition["score"],
        replay_transition["score"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise TrainCommandError("production transition replay scores differ from locked rankings")


def package_release(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repository = config.root.resolve()
    snapshot = Path(args.snapshot).resolve()
    development = Path(args.development).resolve()
    locked = Path(args.locked).resolve()
    source_manifest = Path(args.source_manifest).resolve()
    output = Path(args.output).resolve()
    paths = (snapshot, development, locked, source_manifest, output.parent)
    if any(not path.is_relative_to(repository) for path in paths):
        raise TrainCommandError("all release paths must remain inside the repository")
    if output.name != config.project.artifact_version:
        raise TrainCommandError("artifact output directory must equal project.artifact_version")
    canonical = {
        "snapshot": repository
        / "data"
        / "processed"
        / "snapshots"
        / config.project.artifact_version,
        "development": repository / "reports" / "experiments" / config.project.artifact_version,
        "locked": repository / "reports" / "locked" / config.project.artifact_version,
        "source_manifest": repository / "reports" / "source_manifest.json",
        "output": repository / "artifacts" / config.project.artifact_version,
    }
    actual = {
        "snapshot": snapshot,
        "development": development,
        "locked": locked,
        "source_manifest": source_manifest,
        "output": output,
    }
    if mismatched := [name for name in canonical if actual[name] != canonical[name].resolve()]:
        raise TrainCommandError(
            "release paths differ from the canonical v1 contract: " + ", ".join(mismatched)
        )

    session = resume_reporting_only(locked, repository=repository)
    state = session.state()
    metrics = session.load_metrics_for_reporting()
    reporting_git = require_reporting_git_state(
        repository,
        state=state,
        output_dir=locked,
    )
    _assert_bound_path(state, repository, "processed_manifest", snapshot / "manifest.json")
    _assert_bound_path(state, repository, "source_manifest", source_manifest)
    _assert_bound_path(
        state,
        repository,
        "development_decision",
        development / "development_decision.json",
    )
    _assert_bound_path(
        state,
        repository,
        "fold_manifest",
        development / "fold_manifest.csv",
    )
    for logical_name, filename in (
        ("fold_digest", "fold_manifest.sha256.json"),
        ("candidate_manifest", "candidate_manifest.json"),
        ("feature_manifest", "feature_manifest.json"),
        ("metric_contract", "metric_contract.json"),
        ("evaluation_universe", "evaluation_universe.json"),
        ("implementation_manifest", "implementation_manifest.json"),
    ):
        _assert_bound_path(
            state,
            repository,
            logical_name,
            development / filename,
        )
    verify_processed_snapshot(
        snapshot,
        config=config,
        source_manifest_path=source_manifest,
    )
    processed_manifest = load_processed_manifest(snapshot)
    decision = _json_object(development / "development_decision.json")
    if verify_development_decision(decision) != state["decision_sha256"]:
        raise TrainCommandError("development decision differs from locked state")
    if decision.get("schema_version") != 2:
        raise TrainCommandError("release packaging requires a multi-task decision")
    fold_manifest = pd.read_csv(
        development / "fold_manifest.csv",
        dtype={
            "onet_soc_code": "string",
            "base_soc_code": "string",
            "soc_major_group": "string",
            "role": "string",
            "split_method": "string",
        },
    )
    if fold_manifest_digest(fold_manifest) != decision["fold_manifest_sha256"]:
        raise TrainCommandError("fold manifest differs from the frozen decision")
    exact_contract_digests = {
        "feature_manifest_sha256": sha256_file(development / "feature_manifest.json"),
        "metric_contract_sha256": sha256_file(development / "metric_contract.json"),
        "candidate_manifest_sha256": sha256_file(development / "candidate_manifest.json"),
        "evaluation_universe_sha256": sha256_file(development / "evaluation_universe.json"),
        "implementation_manifest_sha256": sha256_file(development / "implementation_manifest.json"),
        "environment_lock_sha256": sha256_file(repository / "requirements.lock"),
        "processed_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "project_config_sha256": sha256_file(config.path),
    }
    for field, digest in exact_contract_digests.items():
        if decision.get(field) != digest:
            raise TrainCommandError(f"release {field} differs from the frozen decision")
    implementation = _json_object(development / "implementation_manifest.json")
    implementation_digest = verify_implementation_manifest(implementation, repository)
    evaluation_universe = _json_object(development / "evaluation_universe.json")
    if evaluation_universe["implementation_manifest_sha256"] != implementation_digest:
        raise TrainCommandError("evaluation universe binds a different implementation")
    validate_evaluation_universe(evaluation_universe, fold_manifest)
    fit_codes = fitting_codes_for_phase(fold_manifest, "locked_test")

    title_candidates = title_candidate_grid()
    transition_candidates = transition_candidate_grid()
    if decision["title_candidate_grid_sha256"] != candidate_grid_digest(
        title_candidates
    ) or decision["transition_candidate_grid_sha256"] != candidate_grid_digest(
        transition_candidates
    ):
        raise TrainCommandError("frozen candidate grid differs from production code")
    candidate_manifest = _json_object(development / "candidate_manifest.json")
    if (
        candidate_manifest.get("title_sha256") != decision["title_candidate_grid_sha256"]
        or candidate_manifest.get("transition_sha256")
        != decision["transition_candidate_grid_sha256"]
    ):
        raise TrainCommandError("candidate manifest grid links differ from the decision")
    releases = decision["release_candidates"]
    title_spec = _candidate(title_candidates, releases["title"])
    transition_spec = _candidate(transition_candidates, releases["transition"])

    profiles = _table(snapshot, processed_manifest, "occupation_profiles")
    aliases = _table(snapshot, processed_manifest, "aliases")
    ratings = _table(snapshot, processed_manifest, "skill_ratings")
    market = _table(snapshot, processed_manifest, "occupation_market")
    title_index = fit_production_title_index(
        profiles,
        aliases,
        title_spec,
        fit_codes=fit_codes,
        min_df=int(config.raw["candidates"]["minimum_document_frequency"]),
        random_seed=config.project.random_seed,
    )
    transition = fit_production_transition_components(
        profiles,
        ratings,
        transition_spec,
        fit_codes=fit_codes,
        minimum_structured_overlap=config.profiles.minimum_structured_overlap,
        min_df=int(config.raw["candidates"]["minimum_document_frequency"]),
        random_seed=config.project.random_seed,
    )
    if transition.occupation_codes != title_index.occupation_codes:
        raise TrainCommandError("title and transition production catalogs are misaligned")
    neighbors = build_transition_candidates(
        transition.occupation_codes,
        transition.lexical_scores,
        transition.lsa_scores,
        transition.structured_scores,
        transition.development_distances,
        weights=transition.weights,
        limit=int(config.raw["serving"]["frontier_max_candidates"]),
    )
    locked_rankings = session.load_rankings_for_reporting()
    _assert_production_replays_locked_rankings(
        locked_rankings=locked_rankings,
        profiles=profiles,
        aliases=aliases,
        fold_manifest=fold_manifest,
        evaluation_universe=evaluation_universe,
        title_index=title_index,
        transition=transition,
        serving_neighbors=neighbors,
        maximum_aliases_per_occupation=int(
            config.raw["title_benchmark"]["maximum_queries_per_occupation"]
        ),
        random_seed=config.project.random_seed,
    )
    profile_by_code = profiles.set_index("onet_soc_code")
    aligned_profiles = profile_by_code.loc[list(transition.occupation_codes)]
    clusters = fit_exploratory_clusters(
        transition.cluster_embedding,
        transition.occupation_codes,
        transition.term_matrix,
        transition.feature_names,
        k_values=PREDECLARED_K_VALUES,
        seeds=PREDECLARED_SEEDS,
        bootstrap_fraction=float(config.raw["clustering"]["bootstrap_fraction"]),
        soc_major_groups=aligned_profiles["soc_major_group"].astype(str).tolist(),
    )
    selected_diagnostic = clusters.diagnostics.loc[
        clusters.diagnostics["k"].eq(clusters.selected_k)
    ].iloc[0]

    occupations = _core_frame("occupations", profiles, "onet_soc_code")
    occupation_market = _core_frame("occupation_market", market, "onet_soc_code", "nem_code")
    serving_ratings = _core_frame(
        "skill_ratings",
        ratings,
        "onet_soc_code",
        "feature_family",
        "element_id",
        "scale_id",
    )
    assignments = _core_frame("clusters", clusters.assignments, "onet_soc_code")
    serving_neighbors = _core_frame("neighbors", neighbors, "query_code", "rank")
    search_catalog = pd.DataFrame(
        {
            "row_index": range(len(title_index.occupation_codes)),
            "onet_soc_code": title_index.occupation_codes,
        }
    )
    search_catalog = _core_frame("search_catalog", search_catalog, "row_index")

    occupation_count = len(occupations)
    occupations_with_bls = occupation_market["onet_soc_code"].nunique()
    cluster_payload = {
        "selected_k": clusters.selected_k,
        "stability_metric": "pairwise_adjusted_rand_stability",
        "stability_estimate": float(selected_diagnostic["mean_pairwise_adjusted_rand"]),
        "silhouette": float(selected_diagnostic["silhouette"]),
        "seeds": list(PREDECLARED_SEEDS),
    }
    coverage_payload = {
        "counts": {
            "occupation_count": occupation_count,
            "occupations_with_bls": int(occupations_with_bls),
            "nem_occupation_count": int(occupation_market["nem_code"].nunique()),
            "crosswalk_rows": len(occupation_market),
        },
        "rates": {
            "bls_coverage": float(occupations_with_bls / occupation_count),
            "rated_occupation_coverage": float(
                occupations["structured_observed_count"].gt(0).mean()
            ),
            "job_zone_coverage": float(occupations["job_zone"].notna().mean()),
        },
        "notes": [
            "Coverage is measured at serving-table grain.",
            "Official O*NET 30.3 Job Zone labels are preserved as 2, 3, 4, and 5.",
            "Pair-level structured scores additionally require a defined transferable-skill development distance.",
        ],
    }
    limitations_payload = {
        "items": [
            "O*NET Related Occupations are evaluation judgments, not independent semantic truth.",
            "BLS national projections are contextual scenarios, not live vacancies or forecasts for individuals.",
            "Structured similarity is unavailable where occupations lack the minimum jointly observed ratings.",
            "Clusters are exploratory summaries and do not define occupational quality or career suitability.",
            "Pareto frontiers preserve shared NEM rows without summing duplicated national estimates.",
        ]
    }
    evidence_reports = {
        "title_benchmark": _wrapper(
            "title_benchmark",
            metrics["benchmarks"]["title_benchmark"],
            artifact_version=config.project.artifact_version,
            split="locked_test",
        ),
        "transition_benchmark": _wrapper(
            "transition_benchmark",
            metrics["benchmarks"]["transition_benchmark"],
            artifact_version=config.project.artifact_version,
            split="locked_test",
        ),
        "cluster_stability": _wrapper(
            "cluster_stability",
            cluster_payload,
            artifact_version=config.project.artifact_version,
            split="release_context",
        ),
        "data_coverage": _wrapper(
            "data_coverage",
            coverage_payload,
            artifact_version=config.project.artifact_version,
            split="release_context",
        ),
        "limitations": _wrapper(
            "limitations",
            limitations_payload,
            artifact_version=config.project.artifact_version,
            split="release_context",
        ),
    }
    smoke_queries = _smoke_queries(
        title_index,
        occupations,
        preferred_codes=fit_codes,
    )
    source_hashes = {
        str(entry["source_id"]): str(entry["sha256"]) for entry in processed_manifest["source_pins"]
    }
    provenance = {
        "repository_commit": reporting_git.commit,
        "project_config_sha256": decision["project_config_sha256"],
        "source_manifest_sha256": sha256_file(source_manifest),
        "split_manifest_sha256": decision["fold_manifest_sha256"],
        "locked_rankings_sha256": state["rankings"]["sha256"],
        "locked_metrics_sha256": state["metrics"]["sha256"],
        "environment_lock_sha256": decision["environment_lock_sha256"],
        "processed_manifest_sha256": decision["processed_manifest_sha256"],
        "feature_manifest_sha256": decision["feature_manifest_sha256"],
        "metric_contract_sha256": decision["metric_contract_sha256"],
        "candidate_manifest_sha256": decision["candidate_manifest_sha256"],
        "evaluation_universe_sha256": decision["evaluation_universe_sha256"],
        "implementation_manifest_sha256": decision["implementation_manifest_sha256"],
        "title_candidate_grid_sha256": decision["title_candidate_grid_sha256"],
        "transition_candidate_grid_sha256": decision["transition_candidate_grid_sha256"],
        "source_hashes": source_hashes,
    }
    published = write_artifact_bundle(
        output,
        artifact_version=config.project.artifact_version,
        snapshot_date=config.project.snapshot_date,
        occupations=occupations,
        occupation_market=occupation_market,
        skill_ratings=serving_ratings,
        clusters=assignments,
        neighbors=serving_neighbors,
        search_catalog=search_catalog,
        title_vectorizer=title_index.transformer,
        title_matrix=title_index.matrix,
        smoke_queries=smoke_queries,
        evidence_reports=evidence_reports,
        model_metadata={
            "title_candidate_id": title_spec.candidate_id,
            "transition_candidate_id": transition_spec.candidate_id,
        },
        provenance=provenance,
        locked_metrics_path=locked / "locked_metrics.json",
        development_decision_path=development / "development_decision.json",
        fold_manifest_path=development / "fold_manifest.csv",
        fold_digest_path=development / "fold_manifest.sha256.json",
        processed_manifest_path=snapshot / "manifest.json",
        feature_manifest_path=development / "feature_manifest.json",
        metric_contract_path=development / "metric_contract.json",
        candidate_manifest_path=development / "candidate_manifest.json",
        source_manifest_path=source_manifest,
        evaluation_universe_path=development / "evaluation_universe.json",
        implementation_manifest_path=development / "implementation_manifest.json",
        locked_rankings_path=locked / "locked_rankings.parquet",
        project_config_path=config.path,
    )
    print(
        json.dumps(
            {
                "artifact": str(published.relative_to(repository)),
                "title_candidate_id": title_spec.candidate_id,
                "transition_candidate_id": transition_spec.candidate_id,
                "occupation_count": occupation_count,
                "neighbor_rows": len(serving_neighbors),
                "selected_k": clusters.selected_k,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default="configs/project.toml")
    value.add_argument("--snapshot", required=True)
    value.add_argument("--development", required=True)
    value.add_argument("--locked", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--output", required=True)
    return value


if __name__ == "__main__":
    package_release(parser().parse_args())
