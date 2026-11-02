#!/usr/bin/env python3
"""Prepare and execute the preregistered development benchmark exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from job_market_intelligence.config import load_config
from job_market_intelligence.evaluation import (
    GitSnapshot,
    build_evaluation_universe,
    build_implementation_manifest,
    load_evaluation_qrels,
    require_clean_git_commit,
    run_development_confirmation,
    run_development_selection,
    sha256_file,
    validate_evaluation_universe,
    verify_implementation_manifest,
)
from job_market_intelligence.modeling import (
    OPERATIONAL_COST_FORMULA,
    OPERATIONAL_COST_POLICY,
    SelectionDecision,
    candidate_grid_digest,
    multi_task_development_decision_payload,
    title_candidate_grid,
    transition_candidate_grid,
)
from job_market_intelligence.retrieval import fitting_codes_for_phase
from job_market_intelligence.splits import (
    build_fold_manifest,
    fold_manifest_digest,
    write_fold_manifest,
)
from job_market_intelligence.warehouse import load_processed_manifest, verify_processed_snapshot


class BenchmarkCommandError(RuntimeError):
    pass


STAGE_STATE_FILENAMES = {
    "selection": "selection_run_state.json",
    "confirmation": "confirmation_run_state.json",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise BenchmarkCommandError(f"refusing to overwrite different evidence: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _publish_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            temporary,
            compression="zstd",
            use_dictionary=False,
        )
        payload = temporary.read_bytes()
        _publish_bytes(path, payload)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_bytes(path: Path, payload: bytes) -> None:
    """Durably replace a state file whose initial intent is already frozen."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _query_binding(universe: dict[str, Any], role: str) -> dict[str, Any]:
    title = universe["title"]["roles"][role]
    transition = universe["transition"]["roles"][role]
    return {
        "role": role,
        "title_query_count": int(title["eligible_query_count"]),
        "title_query_ids_sha256": str(title["query_ids_sha256"]),
        "transition_query_count": int(transition["eligible_query_count"]),
        "transition_query_codes_sha256": str(transition["query_codes_sha256"]),
    }


def _stage_state_path(output: Path, stage: str) -> Path:
    try:
        return output / STAGE_STATE_FILENAMES[stage]
    except KeyError as exc:
        raise BenchmarkCommandError(f"unknown development stage: {stage}") from exc


def _load_stage_state(output: Path, stage: str) -> dict[str, Any]:
    path = _stage_state_path(output, stage)
    if not path.is_file() or path.is_symlink():
        raise BenchmarkCommandError(f"{stage} run state is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkCommandError(f"{stage} run state is invalid JSON") from exc
    expected = {
        "schema_version",
        "stage",
        "experiment_version",
        "repository_commit",
        "preparation_sha256",
        "implementation_manifest_sha256",
        "input_bindings_sha256",
        "query_binding",
        "upstream",
        "intent_sha256",
        "status",
        "result",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise BenchmarkCommandError(f"{stage} run state fields differ from the contract")
    if (
        value["schema_version"] != 1
        or value["stage"] != stage
        or value["status"] not in {"started", "complete"}
        or not isinstance(value["query_binding"], dict)
        or not isinstance(value["upstream"], dict)
    ):
        raise BenchmarkCommandError(f"{stage} run state contract is invalid")
    intent = {
        key: value[key] for key in expected if key not in {"intent_sha256", "status", "result"}
    }
    if value["intent_sha256"] != _canonical_digest(intent):
        raise BenchmarkCommandError(f"{stage} run intent hash does not verify")
    if value["status"] == "started" and value["result"] is not None:
        raise BenchmarkCommandError(f"{stage} started state cannot contain a result")
    if value["status"] == "complete":
        result = value["result"]
        if (
            not isinstance(result, dict)
            or not result
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in result.values()
            )
        ):
            raise BenchmarkCommandError(f"{stage} completed state has invalid result hashes")
    return value


def _begin_stage(
    output: Path,
    *,
    stage: str,
    role: str,
    git: GitSnapshot,
    prepared: dict[str, Any],
    universe: dict[str, Any],
    upstream: dict[str, str] | None = None,
) -> str:
    path = _stage_state_path(output, stage)
    if os.path.lexists(path):
        raise BenchmarkCommandError(
            f"{stage} is one-shot and already has a run intent; never rerun it"
        )
    intent = {
        "schema_version": 1,
        "stage": stage,
        "experiment_version": prepared["experiment_version"],
        "repository_commit": git.commit,
        "preparation_sha256": prepared["preparation_sha256"],
        "implementation_manifest_sha256": prepared["contracts"]["implementation_manifest.json"],
        "input_bindings_sha256": _canonical_digest(prepared["inputs"]),
        "query_binding": _query_binding(universe, role),
        "upstream": dict(upstream or {}),
    }
    digest = _canonical_digest(intent)
    _publish_bytes(
        path,
        _json_bytes(
            {
                **intent,
                "intent_sha256": digest,
                "status": "started",
                "result": None,
            }
        ),
    )
    return digest


def _complete_stage(
    output: Path,
    *,
    stage: str,
    intent_sha256: str,
    result: dict[str, str],
) -> dict[str, Any]:
    state = _load_stage_state(output, stage)
    if state["status"] != "started" or state["intent_sha256"] != intent_sha256:
        raise BenchmarkCommandError(f"{stage} cannot transition to complete")
    completed = {**state, "status": "complete", "result": dict(result)}
    _replace_bytes(_stage_state_path(output, stage), _json_bytes(completed))
    return _load_stage_state(output, stage)


def _require_completed_stage(output: Path, stage: str) -> dict[str, Any]:
    state = _load_stage_state(output, stage)
    if state["status"] != "complete":
        raise BenchmarkCommandError(
            f"{stage} did not complete; create a new experiment version rather than rerunning"
        )
    return state


def _verify_completed_stage(
    output: Path,
    stage: str,
    *,
    config: Any,
    prepared: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    universe: dict[str, Any],
    role: str,
    expected_result_names: set[str],
) -> dict[str, Any]:
    state = _require_completed_stage(output, stage)
    expected_bindings = {
        "experiment_version": config.project.artifact_version,
        "preparation_sha256": prepared["preparation_sha256"],
        "implementation_manifest_sha256": prepared["contracts"]["implementation_manifest.json"],
        "input_bindings_sha256": _canonical_digest(inputs),
        "query_binding": _query_binding(universe, role),
    }
    for name, expected in expected_bindings.items():
        if state[name] != expected:
            raise BenchmarkCommandError(f"{stage} run state {name} differs from frozen inputs")
    if set(state["result"]) != expected_result_names:
        raise BenchmarkCommandError(f"{stage} completed output set differs from the contract")
    for name, digest in state["result"].items():
        path = output / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise BenchmarkCommandError(f"{stage} output changed after completion: {name}")
    return state


def _manifest_entry(processed_manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [entry for entry in processed_manifest["tables"] if entry["name"] == name]
    if len(matches) != 1:
        raise BenchmarkCommandError(f"processed manifest lacks exactly one {name} table")
    return matches[0]


def _table(snapshot: Path, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    entry = _manifest_entry(manifest, name)
    path = snapshot / entry["parquet_path"]
    if sha256_file(path) != entry["parquet_sha256"]:
        raise BenchmarkCommandError(f"processed table checksum changed: {name}")
    return pd.read_parquet(path)


def _input_bindings(
    config: Any,
    snapshot: Path,
    source_manifest: Path,
    processed_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    paths = {
        "project_config": config.path,
        "environment_lock": config.root / "requirements.lock",
        "processed_manifest": snapshot / "manifest.json",
        "source_manifest": source_manifest,
    }
    for name in ("occupation_profiles", "aliases", "skill_ratings", "related_qrels"):
        entry = _manifest_entry(processed_manifest, name)
        table_path = snapshot / entry["parquet_path"]
        if sha256_file(table_path) != entry["parquet_sha256"]:
            raise BenchmarkCommandError(f"processed table checksum changed: {name}")
        paths[f"table.{name}"] = table_path
    bindings: dict[str, dict[str, Any]] = {}
    for logical_name, raw_path in sorted(paths.items()):
        path = Path(raw_path).resolve()
        if not path.is_relative_to(config.root) or not path.is_file() or path.is_symlink():
            raise BenchmarkCommandError(
                f"development input is missing, unsafe, or outside the repository: {logical_name}"
            )
        bindings[logical_name] = {
            "path": path.relative_to(config.root).as_posix(),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
    return bindings


def _canonical_output(config: Any, value: str | Path) -> Path:
    output = Path(value).resolve()
    expected = (config.root / "reports" / "experiments" / config.project.artifact_version).resolve()
    if output != expected:
        raise BenchmarkCommandError(f"development output must be the canonical path {expected}")
    return output


def _context(
    args: argparse.Namespace,
) -> tuple[
    Any,
    Path,
    Path,
    Path,
    dict[str, Any],
    GitSnapshot,
    dict[str, dict[str, Any]],
]:
    config = load_config(args.config)
    git = require_clean_git_commit(config.root)
    snapshot = Path(args.snapshot).resolve()
    output = _canonical_output(config, args.output)
    source_manifest = Path(args.source_manifest).resolve()
    if not snapshot.is_relative_to(config.root) or not source_manifest.is_relative_to(config.root):
        raise BenchmarkCommandError("all development inputs must remain inside the repository")
    verify_processed_snapshot(
        snapshot,
        config=config,
        source_manifest_path=source_manifest,
    )
    processed_manifest = load_processed_manifest(snapshot)
    inputs = _input_bindings(config, snapshot, source_manifest, processed_manifest)
    return config, snapshot, output, source_manifest, processed_manifest, git, inputs


def _fold_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "onet_soc_code": "string",
            "base_soc_code": "string",
            "soc_major_group": "string",
            "role": "string",
            "split_method": "string",
        },
    )


def _compact_list_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_qrels(snapshot: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    entry = _manifest_entry(manifest, "related_qrels")
    return load_evaluation_qrels(
        snapshot / entry["parquet_path"], expected_sha256=entry["parquet_sha256"]
    )


def _prepared_contracts(
    output: Path,
    *,
    repository: Path,
    fold_manifest: pd.DataFrame,
    inputs: dict[str, dict[str, Any]],
    qrels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    preparation = json.loads((output / "preparation_manifest.json").read_text("utf-8"))
    digest = preparation.pop("preparation_sha256", None)
    if (
        hashlib.sha256(
            json.dumps(preparation, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        != digest
    ):
        raise BenchmarkCommandError("preparation manifest self-hash does not verify")
    if preparation.get("inputs") != inputs:
        raise BenchmarkCommandError("development inputs differ from prepared checksums")
    for filename, expected in preparation["contracts"].items():
        if sha256_file(output / filename) != expected:
            raise BenchmarkCommandError(f"prepared contract changed: {filename}")
    if fold_manifest_digest(fold_manifest) != preparation["fold_manifest_sha256"]:
        raise BenchmarkCommandError("prepared fold manifest semantic digest changed")
    universe = json.loads((output / "evaluation_universe.json").read_text("utf-8"))
    validate_evaluation_universe(universe, fold_manifest, qrels=qrels)
    implementation = json.loads((output / "implementation_manifest.json").read_text("utf-8"))
    implementation_digest = verify_implementation_manifest(implementation, repository)
    if universe["implementation_manifest_sha256"] != implementation_digest:
        raise BenchmarkCommandError("evaluation universe binds a different implementation")
    return {**preparation, "preparation_sha256": digest}


def prepare(args: argparse.Namespace) -> None:
    config, snapshot, output, _source_manifest, processed_manifest, _git, inputs = _context(args)
    forbidden_outputs = (
        "selection_decision.json",
        "development_decision.json",
        "development_summary.json",
    )
    if any((output / name).exists() for name in forbidden_outputs):
        raise BenchmarkCommandError("development experiment already advanced beyond prepare")
    profiles = _table(snapshot, processed_manifest, "occupation_profiles")
    aliases = _table(snapshot, processed_manifest, "aliases")
    ratings = _table(snapshot, processed_manifest, "skill_ratings")
    qrels = _load_qrels(snapshot, processed_manifest)
    fold_manifest = build_fold_manifest(
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
    fold_digest = write_fold_manifest(
        fold_manifest,
        output / "fold_manifest.csv",
        digest_path=output / "fold_manifest.sha256.json",
    )
    title_candidates = title_candidate_grid()
    transition_candidates = transition_candidate_grid()
    candidate_contract = {
        "schema_version": 1,
        "title_sha256": candidate_grid_digest(title_candidates),
        "transition_sha256": candidate_grid_digest(transition_candidates),
        "title": [asdict(candidate) for candidate in title_candidates],
        "transition": [asdict(candidate) for candidate in transition_candidates],
    }
    _publish_bytes(output / "candidate_manifest.json", _json_bytes(candidate_contract))
    fit_roles = {
        "selection": ["discovery"],
        "confirmation": ["discovery", "selection"],
        "locked_test": ["discovery", "selection", "confirmation"],
        "serving": ["discovery", "selection", "confirmation"],
    }
    fit_codes = {
        phase: fitting_codes_for_phase(
            fold_manifest, "locked_test" if phase == "serving" else phase
        )
        for phase in fit_roles
    }
    feature_contract = {
        "schema_version": 1,
        "fit_policy": {
            phase: {
                "roles": fit_roles[phase],
                "fit_code_count": len(fit_codes[phase]),
                "fit_codes_sha256": _compact_list_digest(fit_codes[phase]),
            }
            for phase in fit_roles
        },
        "candidate_universe": {
            "development": "all non-test occupations",
            "locked_test": "full catalog",
        },
        "title_fields": ["canonical_title", "phase-permitted alternate titles"],
        "work_fields": ["canonical_title", "description", "core tasks"],
        "structured_fields": [
            "feature_family",
            "element_id",
            "scale_id",
            "normalized_value",
            "observed",
        ],
        "structured_missingness": {
            "similarity": "observed skill plus knowledge cosine",
            "shared_pair_validity": (
                "similarity unavailable unless transferable-skill development distance is defined"
            ),
            "hybrid_policy": "renormalize finite positive-weight components; never impute",
        },
        "structured_feature_keys": sorted(
            {
                f"{row.feature_family}:{row.element_id}:{row.scale_id}"
                for row in ratings.itertuples(index=False)
            }
        ),
        "forbidden": [
            "related-occupation judgments",
            "SOC hierarchy",
            "BLS outcomes and crosswalk fields",
            "fold labels as model features",
        ],
    }
    _publish_bytes(output / "feature_manifest.json", _json_bytes(feature_contract))
    metric_contract = {
        "schema_version": 1,
        "title": {
            "primary": "macro_occupation_top1_accuracy",
            "supporting": ["recall_at_5", "mrr"],
        },
        "transition": {
            "primary": "macro_query_ndcg_at_10",
            "supporting": [
                "primary_short_recall_at_5",
                "primary_recall_at_10",
                "ndcg_at_20",
                "mrr_at_10",
                "binary_primary_ndcg_at_10",
            ],
        },
        "bootstrap_replicates": int(config.raw["selection"]["bootstrap_replicates"]),
        "bootstrap_stratum": "soc_major_group",
        "selection_rule": str(config.raw["selection"]["rule"]),
        "confirmation_gate": str(config.raw["selection"]["confirmation_gate"]),
        "tie_break": [
            "latent_dimensions",
            "serialized_artifact_bytes",
            "operational_cost_units",
            "candidate_id",
        ],
        "operational_cost": {
            "policy": OPERATIONAL_COST_POLICY,
            "formula": OPERATIONAL_COST_FORMULA,
            "definitions": {
                "score_cells": "query_count * candidate_count",
                "query_count": "fixed evaluation query batch row count",
            },
            "measured_latency_role": "diagnostic_only_not_a_selection_key",
        },
        "deployment_feasibility": {
            "release_requires_positive_always_finite_text_weight": True,
            "pure_structured": "evaluated_ablation_but_champion_ineligible",
        },
    }
    _publish_bytes(output / "metric_contract.json", _json_bytes(metric_contract))
    implementation_manifest = build_implementation_manifest(config.root)
    _publish_bytes(
        output / "implementation_manifest.json",
        _json_bytes(implementation_manifest),
    )
    evaluation_universe = build_evaluation_universe(
        aliases,
        profiles,
        qrels,
        fold_manifest,
        maximum_aliases_per_occupation=int(
            config.raw["title_benchmark"]["maximum_queries_per_occupation"]
        ),
        seed=config.project.random_seed,
        implementation_manifest_sha256=implementation_manifest["implementation_sha256"],
    )
    _publish_bytes(output / "evaluation_universe.json", _json_bytes(evaluation_universe))
    contract_names = (
        "fold_manifest.csv",
        "fold_manifest.sha256.json",
        "candidate_manifest.json",
        "feature_manifest.json",
        "metric_contract.json",
        "evaluation_universe.json",
        "implementation_manifest.json",
    )
    core = {
        "schema_version": 1,
        "experiment_version": config.project.artifact_version,
        "fold_manifest_sha256": fold_digest,
        "inputs": inputs,
        "contracts": {name: sha256_file(output / name) for name in contract_names},
    }
    preparation = {
        **core,
        "preparation_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _publish_bytes(output / "preparation_manifest.json", _json_bytes(preparation))
    print(json.dumps(preparation, sort_keys=True))


def select(args: argparse.Namespace) -> None:
    config, snapshot, output, _source_manifest, processed_manifest, git, inputs = _context(args)
    selection_paths = tuple(output.glob("selection_*.parquet")) + (
        output / "selection_decision.json",
    )
    if any(path.exists() for path in selection_paths):
        raise BenchmarkCommandError("selection is one-shot and has already started")
    profiles = _table(snapshot, processed_manifest, "occupation_profiles")
    aliases = _table(snapshot, processed_manifest, "aliases")
    ratings = _table(snapshot, processed_manifest, "skill_ratings")
    qrels = _load_qrels(snapshot, processed_manifest)
    folds = _fold_frame(output / "fold_manifest.csv")
    prepared = _prepared_contracts(
        output,
        repository=config.root,
        fold_manifest=folds,
        inputs=inputs,
        qrels=qrels,
    )
    universe = json.loads((output / "evaluation_universe.json").read_text("utf-8"))
    intent_sha256 = _begin_stage(
        output,
        stage="selection",
        role="selection",
        git=git,
        prepared=prepared,
        universe=universe,
    )
    result = run_development_selection(
        profiles,
        aliases,
        ratings,
        qrels,
        folds,
        maximum_aliases_per_occupation=int(
            config.raw["title_benchmark"]["maximum_queries_per_occupation"]
        ),
        minimum_structured_overlap=config.profiles.minimum_structured_overlap,
        min_df=int(config.raw["candidates"]["minimum_document_frequency"]),
        random_seed=config.project.random_seed,
        bootstrap_replicates=int(config.raw["selection"]["bootstrap_replicates"]),
    )
    frames = {
        "selection_alias_queries": result.alias_queries,
        "selection_title_target_ranks": result.title_target_ranks,
        "selection_transition_rankings": result.transition_rankings,
        "selection_title_results": result.title_results,
        "selection_transition_results": result.transition_results,
    }
    for name, frame in frames.items():
        _publish_parquet(output / f"{name}.parquet", frame)
    core = {
        "schema_version": 1,
        "experiment_version": config.project.artifact_version,
        "preparation_sha256": prepared["preparation_sha256"],
        "title_selection": asdict(result.title_selection),
        "transition_selection": asdict(result.transition_selection),
        "proposed_candidates": {
            "title": result.title_selection.selected_candidate_id,
            "transition": result.transition_selection.selected_candidate_id,
        },
        "outputs": {
            name: {
                "path": f"{name}.parquet",
                "sha256": sha256_file(output / f"{name}.parquet"),
                "row_count": len(frame),
            }
            for name, frame in frames.items()
        },
    }
    proposal = {
        **core,
        "selection_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _publish_bytes(output / "selection_decision.json", _json_bytes(proposal))
    _complete_stage(
        output,
        stage="selection",
        intent_sha256=intent_sha256,
        result={
            **{f"{name}.parquet": sha256_file(output / f"{name}.parquet") for name in frames},
            "selection_decision.json": sha256_file(output / "selection_decision.json"),
        },
    )
    print(json.dumps(proposal, sort_keys=True))


def _selection_decision(value: Any) -> SelectionDecision:
    if not isinstance(value, dict):
        raise BenchmarkCommandError("selection decision payload is invalid")
    return SelectionDecision(**value)


def confirm(args: argparse.Namespace) -> None:
    config, snapshot, output, _source_manifest, processed_manifest, git, inputs = _context(args)
    confirmation_paths = tuple(output.glob("confirmation_*.parquet")) + (
        output / "development_decision.json",
        output / "development_summary.json",
    )
    if any(path.exists() for path in confirmation_paths):
        raise BenchmarkCommandError("confirmation is one-shot and has already started")
    profiles = _table(snapshot, processed_manifest, "occupation_profiles")
    aliases = _table(snapshot, processed_manifest, "aliases")
    ratings = _table(snapshot, processed_manifest, "skill_ratings")
    qrels = _load_qrels(snapshot, processed_manifest)
    folds = _fold_frame(output / "fold_manifest.csv")
    prepared = _prepared_contracts(
        output,
        repository=config.root,
        fold_manifest=folds,
        inputs=inputs,
        qrels=qrels,
    )
    universe = json.loads((output / "evaluation_universe.json").read_text("utf-8"))
    selection_state = _verify_completed_stage(
        output,
        "selection",
        config=config,
        prepared=prepared,
        inputs=inputs,
        universe=universe,
        role="selection",
        expected_result_names={
            "selection_alias_queries.parquet",
            "selection_title_target_ranks.parquet",
            "selection_transition_rankings.parquet",
            "selection_title_results.parquet",
            "selection_transition_results.parquet",
            "selection_decision.json",
        },
    )
    proposal = json.loads((output / "selection_decision.json").read_text("utf-8"))
    selection_digest = proposal.pop("selection_sha256", None)
    if (
        hashlib.sha256(
            json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        != selection_digest
    ):
        raise BenchmarkCommandError("selection decision self-hash does not verify")
    if proposal["preparation_sha256"] != prepared["preparation_sha256"]:
        raise BenchmarkCommandError("selection decision binds a different preparation")
    for entry in proposal["outputs"].values():
        path = output / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise BenchmarkCommandError("selection evidence changed before confirmation")
    if selection_state["result"]["selection_decision.json"] != sha256_file(
        output / "selection_decision.json"
    ):
        raise BenchmarkCommandError("selection run state does not bind its decision")
    intent_sha256 = _begin_stage(
        output,
        stage="confirmation",
        role="confirmation",
        git=git,
        prepared=prepared,
        universe=universe,
        upstream={
            "selection_state_sha256": sha256_file(_stage_state_path(output, "selection")),
            "selection_decision_sha256": selection_state["result"]["selection_decision.json"],
        },
    )
    result = run_development_confirmation(
        profiles,
        aliases,
        ratings,
        qrels,
        folds,
        title_proposed_candidate_id=proposal["proposed_candidates"]["title"],
        transition_proposed_candidate_id=proposal["proposed_candidates"]["transition"],
        maximum_aliases_per_occupation=int(
            config.raw["title_benchmark"]["maximum_queries_per_occupation"]
        ),
        minimum_structured_overlap=config.profiles.minimum_structured_overlap,
        min_df=int(config.raw["candidates"]["minimum_document_frequency"]),
        random_seed=config.project.random_seed,
    )
    frames = {
        "confirmation_alias_queries": result.alias_queries,
        "confirmation_title_target_ranks": result.title_target_ranks,
        "confirmation_transition_rankings": result.transition_rankings,
        "confirmation_title_results": result.title_results,
        "confirmation_transition_results": result.transition_results,
    }
    for name, frame in frames.items():
        _publish_parquet(output / f"{name}.parquet", frame)
    decision = multi_task_development_decision_payload(
        title_selection=_selection_decision(proposal["title_selection"]),
        title_confirmation=result.title_confirmation,
        transition_selection=_selection_decision(proposal["transition_selection"]),
        transition_confirmation=result.transition_confirmation,
        experiment_version=config.project.artifact_version,
        project_config_sha256=config.fingerprint,
        fold_manifest_sha256=prepared["fold_manifest_sha256"],
        feature_manifest_sha256=prepared["contracts"]["feature_manifest.json"],
        metric_contract_sha256=prepared["contracts"]["metric_contract.json"],
        environment_lock_sha256=sha256_file(config.root / "requirements.lock"),
        processed_manifest_sha256=sha256_file(snapshot / "manifest.json"),
        candidate_manifest_sha256=prepared["contracts"]["candidate_manifest.json"],
        evaluation_universe_sha256=prepared["contracts"]["evaluation_universe.json"],
        implementation_manifest_sha256=prepared["contracts"]["implementation_manifest.json"],
        title_candidate_grid_sha256=candidate_grid_digest(title_candidate_grid()),
        transition_candidate_grid_sha256=candidate_grid_digest(transition_candidate_grid()),
    )
    _publish_bytes(output / "development_decision.json", _json_bytes(decision))
    summary = {
        "schema_version": 1,
        "experiment_version": config.project.artifact_version,
        "selection_sha256": selection_digest,
        "decision_sha256": decision["decision_sha256"],
        "release_candidates": decision["release_candidates"],
        "confirmation_outputs": {
            name: {
                "path": f"{name}.parquet",
                "sha256": sha256_file(output / f"{name}.parquet"),
                "row_count": len(frame),
            }
            for name, frame in frames.items()
        },
    }
    _publish_bytes(output / "development_summary.json", _json_bytes(summary))
    _complete_stage(
        output,
        stage="confirmation",
        intent_sha256=intent_sha256,
        result={
            **{f"{name}.parquet": sha256_file(output / f"{name}.parquet") for name in frames},
            "development_decision.json": sha256_file(output / "development_decision.json"),
            "development_summary.json": sha256_file(output / "development_summary.json"),
        },
    )
    print(json.dumps(summary, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default="configs/project.toml")
    commands = value.add_subparsers(dest="command", required=True)
    for command, function in (
        ("prepare", prepare),
        ("select", select),
        ("confirm", confirm),
    ):
        stage = commands.add_parser(command)
        stage.add_argument("--snapshot", required=True)
        stage.add_argument("--output", required=True)
        stage.add_argument("--source-manifest", required=True)
        stage.set_defaults(function=function)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
