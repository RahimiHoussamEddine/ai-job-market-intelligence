from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from job_market_intelligence.evaluation import GitSnapshot
from scripts.benchmark import (
    BenchmarkCommandError,
    _begin_stage,
    _complete_stage,
    _load_stage_state,
    _verify_completed_stage,
)


def _universe() -> dict[str, object]:
    return {
        "title": {
            "roles": {
                "selection": {
                    "eligible_query_count": 2,
                    "query_ids_sha256": "a" * 64,
                }
            }
        },
        "transition": {
            "roles": {
                "selection": {
                    "eligible_query_count": 3,
                    "query_codes_sha256": "b" * 64,
                }
            }
        },
    }


def test_started_selection_intent_persists_and_blocks_retry(tmp_path) -> None:
    prepared = {
        "experiment_version": "v1",
        "preparation_sha256": "1" * 64,
        "contracts": {"implementation_manifest.json": "2" * 64},
        "inputs": {"snapshot": {"sha256": "3" * 64}},
    }
    git = GitSnapshot(tmp_path, "f" * 40)
    universe = _universe()

    intent = _begin_stage(
        tmp_path,
        stage="selection",
        role="selection",
        git=git,
        prepared=prepared,
        universe=universe,
    )

    state = _load_stage_state(tmp_path, "selection")
    assert state["status"] == "started"
    assert state["intent_sha256"] == intent
    assert (
        state["input_bindings_sha256"]
        == hashlib.sha256(
            b'{"snapshot":{"sha256":"3333333333333333333333333333333333333333333333333333333333333333"}}'
        ).hexdigest()
    )
    with pytest.raises(BenchmarkCommandError, match="never rerun"):
        _begin_stage(
            tmp_path,
            stage="selection",
            role="selection",
            git=git,
            prepared=prepared,
            universe=universe,
        )

    assert state["status"] == "started"
    assert state["result"] is None


def test_completed_stage_binds_every_published_output_and_universe(tmp_path) -> None:
    prepared = {
        "experiment_version": "v1",
        "preparation_sha256": "1" * 64,
        "contracts": {"implementation_manifest.json": "2" * 64},
        "inputs": {"snapshot": {"sha256": "3" * 64}},
    }
    inputs = {"snapshot": {"sha256": "3" * 64}}
    config = SimpleNamespace(project=SimpleNamespace(artifact_version="v1"))
    universe = _universe()
    intent = _begin_stage(
        tmp_path,
        stage="selection",
        role="selection",
        git=GitSnapshot(tmp_path, "f" * 40),
        prepared=prepared,
        universe=universe,
    )
    output = tmp_path / "selection_decision.json"
    output.write_text('{"selected":true}\n', encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    _complete_stage(
        tmp_path,
        stage="selection",
        intent_sha256=intent,
        result={output.name: digest},
    )

    state = _verify_completed_stage(
        tmp_path,
        "selection",
        config=config,
        prepared=prepared,
        inputs=inputs,
        universe=universe,
        role="selection",
        expected_result_names={output.name},
    )
    assert state["status"] == "complete"

    with pytest.raises(BenchmarkCommandError, match="changed after completion"):
        output.write_text('{"selected":false}\n', encoding="utf-8")
        _verify_completed_stage(
            tmp_path,
            "selection",
            config=config,
            prepared=prepared,
            inputs=inputs,
            universe=universe,
            role="selection",
            expected_result_names={output.name},
        )
