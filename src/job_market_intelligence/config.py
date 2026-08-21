"""Typed, validated access to the locked project configuration.

The configuration file is part of the experiment contract.  This module keeps
configuration loading side-effect free: importing it never discovers the
network, creates directories, or mutates a snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

ONET_TABLES: tuple[str, ...] = (
    "occupation_data",
    "job_titles",
    "task_statements",
    "transferable_skills",
    "knowledge",
    "job_zones",
    "related_occupations",
)

ONET_FILENAMES: Mapping[str, str] = MappingProxyType(
    {table: f"{table}.csv" for table in ONET_TABLES}
)


class ConfigError(ValueError):
    """Raised when the locked project configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    name: str
    artifact_version: str
    snapshot_date: date
    random_seed: int


@dataclass(frozen=True, slots=True)
class OnetSourceConfig:
    publisher: str
    release: str
    release_date: str
    license: str
    license_url: str
    base_url: str
    tables: tuple[str, ...]

    def filename(self, table: str) -> str:
        try:
            return ONET_FILENAMES[table]
        except KeyError as exc:
            raise ConfigError(f"Unknown O*NET table: {table!r}") from exc


@dataclass(frozen=True, slots=True)
class BlsSourceConfig:
    publisher: str
    projection_vintage: str
    projection_release_date: date
    occupation_url: str
    crosswalk_url: str
    reuse: str


@dataclass(frozen=True, slots=True)
class SplitConfig:
    method: str
    n_splits: int
    group: str
    stratum: str
    discovery_folds: tuple[int, ...]
    selection_fold: int
    confirmation_fold: int
    test_fold: int


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    core_task_type: str
    preserve_tokens: tuple[str, ...]
    minimum_structured_overlap: int
    job_zone_levels: int


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """The portions of ``project.toml`` used by the data pipeline.

    ``raw`` preserves the complete TOML mapping so downstream experiment code
    can read its own locked sections without a second parser or a lossy rewrite.
    """

    project: ProjectMetadata
    onet: OnetSourceConfig
    bls: BlsSourceConfig
    splits: SplitConfig
    profiles: ProfileConfig
    raw: Mapping[str, Any]
    path: Path
    # ``fingerprint`` is the authoritative file-byte digest used by release
    # provenance.  The semantic digest is also retained to diagnose harmless
    # TOML formatting-only changes without treating them as the same contract.
    fingerprint: str
    semantic_fingerprint: str

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    @property
    def snapshot_id(self) -> str:
        return self.project.artifact_version


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ConfigError(f"Missing required setting [{section}].{key}") from exc


def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{field} must be an ISO date, got {value!r}") from exc


def _validate_https(url: str, field: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ConfigError(f"{field} must be an absolute HTTPS URL without credentials")
    return url.rstrip("/")


def _validate_official_host(url: str, field: str, expected_host: str) -> str:
    validated = _validate_https(url, field)
    if (urlparse(validated).hostname or "").casefold() != expected_host:
        raise ConfigError(f"{field} must use the official {expected_host} host")
    return validated


def _immutable(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, Path)):
        return str(value)
    return value


def config_fingerprint(raw: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 over the semantic TOML values."""

    payload = json.dumps(
        _jsonable(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``configs/project.toml``."""

    configured_root = os.environ.get("JOB_MARKET_PROJECT_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "configs" / "project.toml").is_file():
            raise ConfigError(
                f"JOB_MARKET_PROJECT_ROOT does not contain configs/project.toml: {root}"
            )
        return root
    candidate = Path(start or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "configs" / "project.toml").is_file():
            return directory
    raise ConfigError(f"Could not find configs/project.toml from {candidate}")


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load and strictly validate the data-pipeline configuration."""

    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else find_project_root() / "configs" / "project.toml"
    )
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("rb") as handle:
        raw_mutable = tomllib.load(handle)

    project_raw = _required(raw_mutable, "project", "root")
    sources_raw = _required(raw_mutable, "sources", "root")
    onet_raw = _required(sources_raw, "onet", "sources")
    bls_raw = _required(sources_raw, "bls", "sources")
    splits_raw = _required(raw_mutable, "splits", "root")
    profiles_raw = _required(raw_mutable, "profiles", "root")

    artifact_version = str(_required(project_raw, "artifact_version", "project"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", artifact_version):
        raise ConfigError("project.artifact_version is not a safe snapshot identifier")

    configured_tables = tuple(str(item) for item in _required(onet_raw, "tables", "sources.onet"))
    if configured_tables != ONET_TABLES:
        raise ConfigError(
            "sources.onet.tables must contain the seven locked tables in protocol order; "
            f"expected {ONET_TABLES!r}, got {configured_tables!r}"
        )

    release = str(_required(onet_raw, "release", "sources.onet"))
    if not re.fullmatch(r"\d+\.\d+", release):
        raise ConfigError(f"Invalid O*NET release: {release!r}")

    project = ProjectMetadata(
        name=str(_required(project_raw, "name", "project")),
        artifact_version=artifact_version,
        snapshot_date=_parse_date(
            _required(project_raw, "snapshot_date", "project"), "snapshot_date"
        ),
        random_seed=int(_required(project_raw, "random_seed", "project")),
    )
    onet = OnetSourceConfig(
        publisher=str(_required(onet_raw, "publisher", "sources.onet")),
        release=release,
        release_date=str(_required(onet_raw, "release_date", "sources.onet")),
        license=str(_required(onet_raw, "license", "sources.onet")),
        license_url=_validate_official_host(
            str(_required(onet_raw, "license_url", "sources.onet")),
            "sources.onet.license_url",
            "creativecommons.org",
        ),
        base_url=_validate_official_host(
            str(_required(onet_raw, "base_url", "sources.onet")),
            "sources.onet.base_url",
            "www.onetcenter.org",
        ),
        tables=configured_tables,
    )
    bls = BlsSourceConfig(
        publisher=str(_required(bls_raw, "publisher", "sources.bls")),
        projection_vintage=str(_required(bls_raw, "projection_vintage", "sources.bls")),
        projection_release_date=_parse_date(
            _required(bls_raw, "projection_release_date", "sources.bls"),
            "sources.bls.projection_release_date",
        ),
        occupation_url=_validate_official_host(
            str(_required(bls_raw, "occupation_url", "sources.bls")),
            "sources.bls.occupation_url",
            "www.bls.gov",
        ),
        crosswalk_url=_validate_official_host(
            str(_required(bls_raw, "crosswalk_url", "sources.bls")),
            "sources.bls.crosswalk_url",
            "www.bls.gov",
        ),
        reuse=str(_required(bls_raw, "reuse", "sources.bls")),
    )
    splits = SplitConfig(
        method=str(_required(splits_raw, "method", "splits")),
        n_splits=int(_required(splits_raw, "n_splits", "splits")),
        group=str(_required(splits_raw, "group", "splits")),
        stratum=str(_required(splits_raw, "stratum", "splits")),
        discovery_folds=tuple(
            int(item) for item in _required(splits_raw, "discovery_folds", "splits")
        ),
        selection_fold=int(_required(splits_raw, "selection_fold", "splits")),
        confirmation_fold=int(_required(splits_raw, "confirmation_fold", "splits")),
        test_fold=int(_required(splits_raw, "test_fold", "splits")),
    )
    all_folds = (
        *splits.discovery_folds,
        splits.selection_fold,
        splits.confirmation_fold,
        splits.test_fold,
    )
    if sorted(all_folds) != list(range(splits.n_splits)):
        raise ConfigError("Split fold assignments must cover each configured fold exactly once")
    profiles = ProfileConfig(
        core_task_type=str(_required(profiles_raw, "core_task_type", "profiles")),
        preserve_tokens=tuple(
            str(item) for item in _required(profiles_raw, "preserve_tokens", "profiles")
        ),
        minimum_structured_overlap=int(
            _required(profiles_raw, "minimum_structured_overlap", "profiles")
        ),
        job_zone_levels=int(_required(profiles_raw, "job_zone_levels", "profiles")),
    )
    if profiles.minimum_structured_overlap < 1:
        raise ConfigError("profiles.minimum_structured_overlap must be positive")
    if release == "30.3" and profiles.job_zone_levels != 4:
        raise ConfigError("O*NET 30.3 requires the current four-level Job Zone framework")

    immutable_raw = _immutable(raw_mutable)
    return ProjectConfig(
        project=project,
        onet=onet,
        bls=bls,
        splits=splits,
        profiles=profiles,
        raw=immutable_raw,
        path=config_path,
        fingerprint=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        semantic_fingerprint=config_fingerprint(immutable_raw),
    )


__all__ = [
    "BlsSourceConfig",
    "ConfigError",
    "ONET_FILENAMES",
    "ONET_TABLES",
    "OnetSourceConfig",
    "ProfileConfig",
    "ProjectConfig",
    "ProjectMetadata",
    "SplitConfig",
    "config_fingerprint",
    "find_project_root",
    "load_config",
]
