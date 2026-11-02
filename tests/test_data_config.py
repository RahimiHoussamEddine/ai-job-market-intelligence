from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from job_market_intelligence.config import (
    ONET_30_3_STORED_JOB_ZONE_VALUES,
    ONET_TABLES,
    ConfigError,
    load_config,
)
from job_market_intelligence.sources import build_source_specs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"


def _write_config(tmp_path: Path, transform: Callable[[str], str]) -> Path:
    path = tmp_path / "configs" / "project.toml"
    path.parent.mkdir(parents=True)
    path.write_text(transform(CONFIG_PATH.read_text(encoding="utf-8")), encoding="utf-8")
    return path


def test_locked_config_uses_actual_snapshot_date_and_four_job_zones() -> None:
    config = load_config(CONFIG_PATH)

    assert config.project.snapshot_date.isoformat() == "2026-08-21"
    assert config.project.artifact_version == "v3"
    assert config.onet.release == "30.3"
    assert config.onet.tables == ONET_TABLES
    assert config.profiles.job_zone_levels == 4
    assert ONET_30_3_STORED_JOB_ZONE_VALUES == (2, 3, 4, 5)
    assert config.fingerprint == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert config.fingerprint != config.semantic_fingerprint


def test_source_plan_is_nine_official_lowercase_urls() -> None:
    specs = build_source_specs(load_config(CONFIG_PATH))

    assert len(specs) == 9
    assert [spec.source_id for spec in specs[:7]] == [f"onet.{table}" for table in ONET_TABLES]
    assert all(spec.filename == spec.filename.casefold() for spec in specs)
    assert all(" " not in spec.requested_url for spec in specs)
    assert all(spec.requested_url.startswith("https://") for spec in specs)
    assert specs[-2].requested_url == ("https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx")
    assert specs[-1].requested_url.endswith("nem-onet-to-soc-crosswalk.xlsx")


def test_config_rejects_legacy_five_level_job_zone_contract(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path, lambda text: text.replace("job_zone_levels = 4", "job_zone_levels = 5")
    )

    with pytest.raises(ConfigError, match="four Job Zone categories"):
        load_config(path)


def test_config_rejects_nonofficial_source_host(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda text: text.replace("https://www.bls.gov/", "https://mirror.example/", 1),
    )

    with pytest.raises(ConfigError, match="official www.bls.gov"):
        load_config(path)
