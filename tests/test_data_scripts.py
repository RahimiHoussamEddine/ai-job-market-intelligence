from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.audit_data import main as audit_main
from scripts.build_warehouse import main as build_main
from scripts.download_data import main as download_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_download_plan_is_read_only_and_exposes_all_locked_requests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = download_main(
        ["--config", str(PROJECT_ROOT / "configs" / "project.toml"), "--plan"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["mode"] == "plan"
    assert payload["snapshot_date"] == "2026-08-21"
    assert len(payload["sources"]) == 9
    assert all(item["url"].startswith("https://") for item in payload["sources"])


@pytest.mark.parametrize("entrypoint", [build_main, audit_main])
def test_build_and_audit_scripts_fail_closed_without_source_snapshot(
    entrypoint: Callable[[list[str] | None], int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "--config",
        str(PROJECT_ROOT / "configs" / "project.toml"),
        "--source-snapshot",
        str(tmp_path / "missing-source"),
    ]
    if entrypoint is audit_main:
        arguments.append("--source-only")

    exit_code = entrypoint(arguments)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error:")
