from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.app_test import TMP_DIR

from job_market_intelligence.cli import main as cli_main
from job_market_intelligence.service import clear_service_cache


@pytest.fixture(scope="module", autouse=True)
def cleanup_streamlit_testing_directory():
    """Close Streamlit's module-level TemporaryDirectory before interpreter exit."""

    yield
    TMP_DIR.cleanup()


def test_cli_exercises_every_serving_view(
    synthetic_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commands = (
        (["verify-artifact"], "smoke_checks"),
        (["search", "software applications", "-k", "2"], "results"),
        (["occupation", "15-1252.00"], "occupation"),
        (["neighbors", "15-1252.00", "-k", "2"], "neighbors"),
        (
            ["compare-skills", "15-1252.00", "15-2051.00", "--limit", "10"],
            "differences",
        ),
        (["frontier", "15-1252.00", "--maximum-job-zone", "4"], "frontier"),
        (["clusters"], "summaries"),
        (["evidence"], "reports"),
    )

    for command, expected_key in commands:
        exit_code = cli_main(["--artifact-dir", str(synthetic_bundle), *command])
        captured = capsys.readouterr()
        assert exit_code == 0, captured.err
        assert captured.err == ""
        payload = json.loads(captured.out)
        assert expected_key in payload


def test_cli_distinguishes_invalid_request_missing_and_invalid_artifact(
    synthetic_bundle: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_main(["--artifact-dir", str(synthetic_bundle), "search", "   ", "-k", "1"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert json.loads(captured.err)["status"] == "invalid_request"

    missing = synthetic_bundle.parent / "missing-v1"
    clear_service_cache()
    exit_code = cli_main(["--artifact-dir", str(missing), "verify-artifact"])
    captured = capsys.readouterr()
    assert exit_code == 3
    assert json.loads(captured.err)["status"] == "artifact_unavailable"

    invalid = synthetic_bundle.parent / "invalid-v1"
    shutil.copytree(synthetic_bundle, invalid)
    occupation_path = invalid / "occupations.parquet"
    occupation_path.write_bytes(occupation_path.read_bytes() + b"tampered")
    clear_service_cache()
    exit_code = cli_main(["--artifact-dir", str(invalid), "verify-artifact"])
    captured = capsys.readouterr()
    assert exit_code == 3
    assert json.loads(captured.err)["status"] == "artifact_unavailable"


def test_streamlit_app_exercises_all_five_tabs_with_valid_artifact(
    synthetic_bundle: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app" / "main.py"), default_timeout=20).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Discover",
        "Transitions",
        "Skills",
        "Clusters",
        "Evidence & lineage",
    ]
    assert len(app.get("plotly_chart")) >= 3
    assert len(app.dataframe) >= 6
    assert any("Locked evidence has not been generated" in item.value for item in app.info)

    app.text_input[0].set_value("software applications")
    next(button for button in app.button if button.label == "Search").click()
    app.run()

    assert not app.exception
    assert len(app.dataframe) >= 7


def test_streamlit_app_has_a_safe_missing_artifact_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "private-release" / "missing-v1"
    monkeypatch.setenv("JOB_INTELLIGENCE_ARTIFACT_DIR", str(missing))
    clear_service_cache()
    root = Path(__file__).resolve().parents[1]

    app = AppTest.from_file(str(root / "app" / "main.py"), default_timeout=20).run()

    assert not app.exception
    assert not app.tabs
    assert any("not available" in item.value for item in app.error)
    assert any("artifact_missing" in item.value for item in app.info)
    assert all(str(missing) not in item.value for item in (*app.error, *app.info))
