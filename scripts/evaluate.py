#!/usr/bin/env python3
"""One-shot locked ranking generation and reporting-only aggregation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from job_market_intelligence.config import load_config
from job_market_intelligence.evaluation import (
    EvaluationIntegrityError,
    begin_locked_run,
    build_locked_descriptive_slices,
    build_title_benchmark_evidence,
    build_transition_benchmark_evidence,
    generate_locked_rankings,
    load_evaluation_qrels,
    resume_reporting_only,
    sha256_file,
    validate_evaluation_universe,
    verify_development_decision,
    verify_implementation_manifest,
)
from job_market_intelligence.modeling import (
    candidate_grid_digest,
    title_candidate_grid,
    transition_candidate_grid,
)
from job_market_intelligence.splits import fold_manifest_digest
from job_market_intelligence.warehouse import load_processed_manifest, verify_processed_snapshot


class EvaluateCommandError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluateCommandError(f"JSON object required: {path}")
    return value


def _publish_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise EvaluateCommandError(f"refusing to overwrite different report: {path}")
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
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["tables"] if entry["name"] == name]
    if len(matches) != 1:
        raise EvaluateCommandError(f"processed manifest lacks exactly one {name}")
    return matches[0]


def _table(snapshot: Path, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    entry = _entry(manifest, name)
    path = snapshot / entry["parquet_path"]
    if sha256_file(path) != entry["parquet_sha256"]:
        raise EvaluateCommandError(f"processed table checksum changed: {name}")
    return pd.read_parquet(path)


def _input_pins(
    *,
    config_path: Path,
    snapshot: Path,
    processed_manifest: dict[str, Any],
    development: Path,
    source_manifest: Path,
) -> dict[str, tuple[Path, str]]:
    paths: dict[str, Path] = {
        "project_config": config_path,
        "environment_lock": config_path.parent.parent / "requirements.lock",
        "processed_manifest": snapshot / "manifest.json",
        "source_manifest": source_manifest,
        "fold_manifest": development / "fold_manifest.csv",
        "fold_digest": development / "fold_manifest.sha256.json",
        "candidate_manifest": development / "candidate_manifest.json",
        "feature_manifest": development / "feature_manifest.json",
        "metric_contract": development / "metric_contract.json",
        "evaluation_universe": development / "evaluation_universe.json",
        "implementation_manifest": development / "implementation_manifest.json",
        "development_decision": development / "development_decision.json",
    }
    for name in ("occupation_profiles", "aliases", "skill_ratings", "related_qrels"):
        entry = _entry(processed_manifest, name)
        path = snapshot / entry["parquet_path"]
        if sha256_file(path) != entry["parquet_sha256"]:
            raise EvaluateCommandError(f"processed table checksum changed: {name}")
        paths[f"table.{name}"] = path
    return {name: (path, sha256_file(path)) for name, path in paths.items()}


def _canonical_locked_output(repository: Path, artifact_version: str, value: str | Path) -> Path:
    output = Path(value).resolve()
    expected = (repository / "reports" / "locked" / artifact_version).resolve()
    if output != expected:
        raise EvaluateCommandError(f"locked output must be the canonical path {expected}")
    return output


def _verify_frozen_contracts(
    *,
    config: Any,
    snapshot: Path,
    development: Path,
    decision: dict[str, Any],
    fold_manifest: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    exact_digests = {
        "project_config_sha256": sha256_file(config.path),
        "environment_lock_sha256": sha256_file(config.root / "requirements.lock"),
        "processed_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "feature_manifest_sha256": sha256_file(development / "feature_manifest.json"),
        "metric_contract_sha256": sha256_file(development / "metric_contract.json"),
        "candidate_manifest_sha256": sha256_file(development / "candidate_manifest.json"),
        "evaluation_universe_sha256": sha256_file(development / "evaluation_universe.json"),
        "implementation_manifest_sha256": sha256_file(development / "implementation_manifest.json"),
    }
    for field, actual in exact_digests.items():
        if decision.get(field) != actual:
            raise EvaluateCommandError(f"{field} differs from the frozen decision")
    semantic_fold_digest = fold_manifest_digest(fold_manifest)
    if semantic_fold_digest != decision["fold_manifest_sha256"]:
        raise EvaluateCommandError("fold manifest differs from the frozen decision")
    fold_sidecar = _json(development / "fold_manifest.sha256.json")
    if (
        fold_sidecar.get("semantic_sha256") != semantic_fold_digest
        or fold_sidecar.get("path") != "fold_manifest.csv"
    ):
        raise EvaluateCommandError("fold digest sidecar differs from fold manifest")
    title_candidates = title_candidate_grid()
    transition_candidates = transition_candidate_grid()
    expected_candidate_manifest = {
        "schema_version": 1,
        "title_sha256": candidate_grid_digest(title_candidates),
        "transition_sha256": candidate_grid_digest(transition_candidates),
        "title": [json.loads(json.dumps(asdict(candidate))) for candidate in title_candidates],
        "transition": [
            json.loads(json.dumps(asdict(candidate))) for candidate in transition_candidates
        ],
    }
    candidates = _json(development / "candidate_manifest.json")
    if candidates != expected_candidate_manifest:
        raise EvaluateCommandError("candidate manifest differs from the production grid")
    if (
        decision["title_candidate_grid_sha256"] != candidates["title_sha256"]
        or decision["transition_candidate_grid_sha256"] != candidates["transition_sha256"]
    ):
        raise EvaluateCommandError("candidate grid digest differs from the frozen decision")
    universe = _json(development / "evaluation_universe.json")
    implementation = _json(development / "implementation_manifest.json")
    implementation_digest = verify_implementation_manifest(implementation, config.root)
    if universe["implementation_manifest_sha256"] != implementation_digest:
        raise EvaluateCommandError("evaluation universe binds a different implementation")
    validated = validate_evaluation_universe(universe, fold_manifest)
    return dict(validated["transition"])


def begin_and_rank(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repository = config.root
    snapshot = Path(args.snapshot).resolve()
    development = Path(args.development).resolve()
    output = _canonical_locked_output(repository, config.project.artifact_version, args.output)
    source_manifest = Path(args.source_manifest).resolve()
    for path in (snapshot, development, output.parent, source_manifest.parent):
        if not path.is_relative_to(repository):
            raise EvaluateCommandError("all locked-run paths must remain inside the repository")
    verify_processed_snapshot(
        snapshot,
        config=config,
        source_manifest_path=source_manifest,
    )
    processed_manifest = load_processed_manifest(snapshot)
    decision = _json(development / "development_decision.json")
    verify_development_decision(decision)
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
    _verify_frozen_contracts(
        config=config,
        snapshot=snapshot,
        development=development,
        decision=decision,
        fold_manifest=fold_manifest,
    )
    pins = _input_pins(
        config_path=config.path,
        snapshot=snapshot,
        processed_manifest=processed_manifest,
        development=development,
        source_manifest=source_manifest,
    )
    session = begin_locked_run(
        output,
        repository=repository,
        experiment_version=config.project.artifact_version,
        decision=decision,
        expected_inputs=pins,
    )
    profiles = _table(snapshot, processed_manifest, "occupation_profiles")
    aliases = _table(snapshot, processed_manifest, "aliases")
    ratings = _table(snapshot, processed_manifest, "skill_ratings")
    rankings = generate_locked_rankings(
        profiles,
        aliases,
        ratings,
        fold_manifest,
        decision,
        maximum_aliases_per_occupation=int(
            config.raw["title_benchmark"]["maximum_queries_per_occupation"]
        ),
        minimum_structured_overlap=config.profiles.minimum_structured_overlap,
        min_df=int(config.raw["candidates"]["minimum_document_frequency"]),
        random_seed=config.project.random_seed,
    )
    digest = session.publish_rankings(rankings)
    print(json.dumps({"rankings_sha256": digest, "row_count": len(rankings)}, sort_keys=True))


def _evidence_wrapper(
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


def report_only(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    repository = config.root
    output = _canonical_locked_output(repository, config.project.artifact_version, args.output)
    session = resume_reporting_only(output, repository=repository)
    state = session.state()
    rankings = session.load_rankings_for_reporting()
    inputs = state["inputs"]
    qrel_entry = inputs["table.related_qrels"]
    qrel_path = repository / qrel_entry["path"]
    qrels = load_evaluation_qrels(qrel_path, expected_sha256=qrel_entry["sha256"])
    releases = state["release_candidates"]
    if set(releases) != {"title", "transition"}:
        raise EvaluateCommandError("reporting requires a multi-task release decision")
    replicates = int(config.raw["selection"]["bootstrap_replicates"])
    if state["status"] == "rankings_written":
        fold_manifest = pd.read_csv(
            repository / inputs["fold_manifest"]["path"],
            dtype={
                "onet_soc_code": "string",
                "base_soc_code": "string",
                "soc_major_group": "string",
                "role": "string",
                "split_method": "string",
            },
        )
        universe = _json(repository / inputs["evaluation_universe"]["path"])
        aliases_path = repository / inputs["table.aliases"]["path"]
        query_universe = validate_evaluation_universe(
            universe,
            fold_manifest,
            aliases=pd.read_parquet(aliases_path),
            profiles=pd.read_parquet(repository / inputs["table.occupation_profiles"]["path"]),
            qrels=qrels,
            seed=config.project.random_seed,
        )
        for candidate_id in {releases["title"], "title_word_char"}:
            observed_title_ids = tuple(
                sorted(
                    rankings.loc[
                        rankings["benchmark"].eq("title")
                        & rankings["candidate_id"].eq(candidate_id),
                        "query_id",
                    ].astype(str)
                )
            )
            if observed_title_ids != query_universe["title"]["test"]:
                raise EvaluateCommandError(
                    f"locked title rankings differ from frozen query IDs for {candidate_id}"
                )
        profiles_path = repository / inputs["table.occupation_profiles"]["path"]
        ratings_path = repository / inputs["table.skill_ratings"]["path"]
        slices = build_locked_descriptive_slices(
            rankings,
            qrels,
            pd.read_parquet(profiles_path),
            pd.read_parquet(ratings_path),
            title_candidate_id=releases["title"],
            transition_candidate_id=releases["transition"],
            expected_transition_query_codes=query_universe["transition"]["test"],
            bootstrap_replicates=replicates,
            bootstrap_seed=config.project.random_seed,
        )
        title_payload = build_title_benchmark_evidence(
            rankings,
            selected_candidate_id=releases["title"],
            baseline_candidate_id="title_word_char",
            slices=slices["title"],
            bootstrap_replicates=replicates,
            bootstrap_seed=config.project.random_seed,
        )
        transition_payload = build_transition_benchmark_evidence(
            rankings,
            qrels,
            selected_candidate_id=releases["transition"],
            baseline_candidate_id="transition_title_word_char",
            expected_query_codes=query_universe["transition"]["test"],
            slices=slices["transition"],
            bootstrap_replicates=replicates,
            bootstrap_seed=config.project.random_seed,
        )
        report = {
            "schema_version": 1,
            "experiment_version": config.project.artifact_version,
            "split": "locked_test",
            "rankings_sha256": state["rankings"]["sha256"],
            "decision_sha256": state["decision_sha256"],
            "benchmarks": {
                "title_benchmark": title_payload,
                "transition_benchmark": transition_payload,
            },
        }
        session.publish_metrics(report, repository=repository)
        state = session.state()
    elif state["status"] == "complete":
        report = session.load_metrics_for_reporting()
        title_payload = report["benchmarks"]["title_benchmark"]
        transition_payload = report["benchmarks"]["transition_benchmark"]
    else:
        raise EvaluationIntegrityError("locked run is not ready for reporting")

    title_wrapper = _evidence_wrapper(
        "title_benchmark",
        title_payload,
        artifact_version=config.project.artifact_version,
        split="locked_test",
    )
    transition_wrapper = _evidence_wrapper(
        "transition_benchmark",
        transition_payload,
        artifact_version=config.project.artifact_version,
        split="locked_test",
    )
    _publish_json(output / "title_benchmark.json", title_wrapper)
    _publish_json(output / "transition_benchmark.json", transition_wrapper)
    print(
        json.dumps(
            {
                "metrics_sha256": state["metrics"]["sha256"],
                "title_report": "title_benchmark.json",
                "transition_report": "transition_benchmark.json",
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default="configs/project.toml")
    commands = value.add_subparsers(dest="command", required=True)
    begin = commands.add_parser("begin-and-rank")
    begin.add_argument("--snapshot", required=True)
    begin.add_argument("--development", required=True)
    begin.add_argument("--output", required=True)
    begin.add_argument("--source-manifest", required=True)
    begin.set_defaults(function=begin_and_rank)
    report = commands.add_parser("report-only")
    report.add_argument("--output", required=True)
    report.set_defaults(function=report_only)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
