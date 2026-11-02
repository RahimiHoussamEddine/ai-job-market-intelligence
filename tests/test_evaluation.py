from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from job_market_intelligence.evaluation import (
    EvaluationIntegrityError,
    begin_locked_run,
    evaluate_title_rankings,
    require_clean_git_commit,
    resume_reporting_only,
    sha256_file,
    validate_locked_rankings,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=synthetic@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _decision() -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": 1,
        "experiment_version": "synthetic-v1",
        "release_candidate_id": "candidate-a",
        "selection": {"selected_candidate_id": "candidate-a"},
        "confirmation": {"release_candidate_id": "candidate-a"},
        "project_config_sha256": "a" * 64,
        "fold_manifest_sha256": "b" * 64,
        "candidate_grid_sha256": "c" * 64,
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**core, "decision_sha256": digest}


def _rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": "title",
                "candidate_id": "candidate-a",
                "query_id": "query-1",
                "query_code": "15-1001.00",
                "candidate_code": "15-1001.00",
                "rank": 1,
                "score": 0.9,
                "split": "locked_test",
            },
        ]
    )


def test_locked_ranking_schema_rejects_noncanonical_order() -> None:
    valid = _rankings()
    assert len(validate_locked_rankings(valid)) == 1
    assert evaluate_title_rankings(valid, candidate_id="candidate-a")["top1_accuracy"] == 1.0
    invalid = pd.DataFrame(
        [
            {
                "benchmark": "transition",
                "candidate_id": "candidate-a",
                "query_id": "query-1",
                "query_code": "15-1001.00",
                "candidate_code": "15-1002.00",
                "rank": 2,
                "score": 0.8,
                "split": "locked_test",
            }
        ]
    )
    with pytest.raises(EvaluationIntegrityError, match="contiguous/sorted"):
        validate_locked_rankings(invalid)


def test_one_shot_locked_run_and_reporting_only_resume(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    pinned = repository / "input.txt"
    pinned.write_text("immutable input\n", encoding="utf-8")
    _git(repository, "add", "input.txt")
    _git(repository, "commit", "-qm", "initial")
    original = require_clean_git_commit(repository).commit

    output = repository / "reports" / "locked-v1"
    session = begin_locked_run(
        output,
        repository=repository,
        experiment_version="synthetic-v1",
        decision=_decision(),
        expected_inputs={"input": (pinned, sha256_file(pinned))},
    )
    rankings_digest = session.publish_rankings(_rankings())
    with pytest.raises(EvaluationIntegrityError, match="already started"):
        begin_locked_run(
            output,
            repository=repository,
            experiment_version="synthetic-v1",
            decision=_decision(),
            expected_inputs={"input": (pinned, sha256_file(pinned))},
        )

    report = {
        "schema_version": 1,
        "experiment_version": "synthetic-v1",
        "split": "locked_test",
        "rankings_sha256": rankings_digest,
        "decision_sha256": _decision()["decision_sha256"],
        "benchmarks": {"title": {"top1_accuracy": 1.0}},
    }
    unrelated = repository / "unexpected.txt"
    unrelated.write_text("not allowed during reporting\n", encoding="utf-8")
    with pytest.raises(EvaluationIntegrityError, match="outside output files"):
        session.publish_metrics(report, repository=repository)
    unrelated.unlink()

    pinned.write_text("changed input\n", encoding="utf-8")
    with pytest.raises(EvaluationIntegrityError, match="outside output files|input changed"):
        session.publish_metrics(report, repository=repository)
    pinned.write_text("immutable input\n", encoding="utf-8")

    # Initial reporting works at the original clean-code HEAD while the exact
    # state/ranking outputs are intentionally untracked.
    session.publish_metrics(report, repository=repository)
    metrics = json.loads(session.metrics_path.read_text(encoding="utf-8"))
    assert metrics["original_repository_commit"] == original
    assert metrics["reporting_repository_commit"] == original

    resumed = resume_reporting_only(output, repository=repository)
    assert len(resumed.load_rankings_for_reporting()) == 1

    pinned.write_text("corrupted after ranking\n", encoding="utf-8")
    with pytest.raises(EvaluationIntegrityError, match="outside output files|input changed"):
        resume_reporting_only(output, repository=repository)
    pinned.write_text("immutable input\n", encoding="utf-8")

    original_rankings = session.rankings_path.read_bytes()
    session.rankings_path.write_bytes(b"tampered")
    with pytest.raises(EvaluationIntegrityError, match="differ"):
        session.load_rankings_for_reporting()
    session.rankings_path.write_bytes(original_rankings)

    _git(repository, "add", "reports")
    _git(repository, "commit", "-qm", "locked outputs")
    with pytest.raises(EvaluationIntegrityError, match="HEAD to equal"):
        resume_reporting_only(output, repository=repository)
