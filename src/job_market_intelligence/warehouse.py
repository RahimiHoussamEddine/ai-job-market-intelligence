"""Atomic DuckDB/Parquet snapshot construction and strict quality audits."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ProjectConfig
from .crosswalk import (
    CrosswalkError,
    base_soc_code,
    canonical_onet_soc_code,
    join_market_context,
    normalize_bls_market,
    normalize_crosswalk,
    soc_major_group,
)
from .profiles import (
    ProfileError,
    build_aliases,
    build_occupation_profiles,
    build_occupations,
    build_structured_ratings,
    build_tasks,
    normalize_text,
)
from .schemas import (
    ONET_CONTRACTS,
    SchemaError,
    read_bls_workbook,
    read_onet_csv,
    schema_fingerprint,
)
from .sources import (
    EXPECTED_SOURCE_LAYOUT,
    SourceManifest,
    load_manifest,
    sha256_file,
    source_path,
    verify_snapshot,
)

PROCESSED_MANIFEST_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TABLE_KEYS: Mapping[str, tuple[str, ...]] = {
    "occupations": ("onet_soc_code",),
    "aliases": ("alias_id",),
    "tasks": ("onet_soc_code", "task_id"),
    "skill_ratings": ("onet_soc_code", "feature_family", "element_id", "scale_id"),
    "related_qrels": ("query_onet_soc_code", "candidate_onet_soc_code"),
    "bls_market": ("nem_code",),
    "onet_bls_crosswalk": ("onet_soc_code", "nem_code"),
    "occupation_market": ("onet_soc_code", "nem_code"),
    "occupation_profiles": ("onet_soc_code",),
}

TABLE_SORT_KEYS: Mapping[str, tuple[str, ...]] = {
    "occupations": ("onet_soc_code",),
    "aliases": ("onet_soc_code", "normalized_job_title", "job_title", "alias_id"),
    "tasks": ("onet_soc_code", "task_id"),
    "skill_ratings": ("onet_soc_code", "feature_family", "element_id", "scale_id"),
    "related_qrels": (
        "query_onet_soc_code",
        "relatedness_index",
        "candidate_onet_soc_code",
    ),
    "bls_market": ("nem_code",),
    "onet_bls_crosswalk": ("onet_soc_code", "nem_code"),
    "occupation_market": ("onet_soc_code", "nem_code"),
    "occupation_profiles": ("onet_soc_code",),
}

TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "occupations": (
        "onet_soc_code",
        "base_soc_code",
        "soc_major_group",
        "canonical_title",
        "description",
        "job_zone",
        "job_zone_date",
        "job_zone_domain_source",
    ),
    "aliases": (
        "alias_id",
        "onet_soc_code",
        "job_title",
        "normalized_job_title",
        "short_title",
        "sources",
    ),
    "tasks": (
        "onet_soc_code",
        "task_id",
        "task",
        "normalized_task",
        "task_type",
        "incumbents_responding",
        "task_date",
        "domain_source",
    ),
    "skill_ratings": (
        "onet_soc_code",
        "feature_family",
        "element_id",
        "element_name",
        "scale_id",
        "scale_name",
        "raw_value",
        "sample_size",
        "standard_error",
        "lower_ci_bound",
        "upper_ci_bound",
        "normalized_value",
        "observed",
        "recommend_suppress",
        "not_relevant",
        "rating_date",
        "domain_source",
    ),
    "related_qrels": (
        "query_onet_soc_code",
        "candidate_onet_soc_code",
        "relatedness_tier",
        "relatedness_index",
        "relevance_grade",
    ),
    "bls_market": (
        "nem_code",
        "nem_title",
        "occupation_type",
        "employment_2024",
        "employment_2034",
        "employment_change_2024_34",
        "employment_change_pct",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
        "typical_education",
        "work_experience",
        "on_the_job_training",
    ),
    "onet_bls_crosswalk": (
        "onet_soc_code",
        "onet_soc_title",
        "nem_code",
        "nem_title",
    ),
    "occupation_market": (
        "onet_soc_code",
        "onet_soc_title",
        "nem_code",
        "nem_title",
        "occupation_type",
        "employment_2024",
        "employment_2034",
        "employment_change_2024_34",
        "employment_change_pct",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
        "typical_education",
        "work_experience",
        "on_the_job_training",
    ),
    "occupation_profiles": (
        "onet_soc_code",
        "base_soc_code",
        "soc_major_group",
        "canonical_title",
        "description",
        "job_zone",
        "job_zone_date",
        "job_zone_domain_source",
        "title_text",
        "work_text",
        "core_task_count",
        "alias_count",
        "structured_observed_count",
    ),
}

FEATURE_TABLES = ("occupations", "aliases", "tasks", "skill_ratings", "occupation_profiles")
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "related_onet_soc_code",
        "relatedness_tier",
        "relatedness_index",
        "relevance_grade",
        "nem_code",
        "employment_2024",
        "employment_2034",
        "employment_change_2024_34",
        "employment_change_pct",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
    }
)

ONET_30_3_RELEASE_COUNTS: Mapping[str, int] = {
    "occupation_data": 1_016,
    "job_titles": 57_543,
    "task_statements": 18_796,
    "transferable_skills": 44_700,
    "knowledge": 59_004,
    "job_zones": 923,
    "related_occupations": 18_460,
}


class WarehouseError(RuntimeError):
    """Raised for non-atomic builds, invalid normalized data, or audit failure."""


@dataclass(frozen=True, slots=True)
class AuditIssue:
    check: str
    severity: str
    count: int
    detail: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    issues: tuple[AuditIssue, ...]
    table_rows: Mapping[str, int]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" and issue.count > 0 for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error" and issue.count]
        if errors:
            summary = "; ".join(f"{issue.check}={issue.count} ({issue.detail})" for issue in errors)
            raise WarehouseError(f"Data quality audit failed: {summary}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "table_rows": dict(self.table_rows),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    snapshot_id: str
    snapshot_dir: Path
    warehouse_path: Path
    manifest_path: Path
    source_manifest_sha256: str
    warehouse_sha256: str
    table_rows: Mapping[str, int]


def _manifest_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "manifest.json" if value.is_dir() else value


def _source_manifest_sha256(path: str | Path) -> str:
    return sha256_file(_manifest_path(path))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_source_manifest(
    source_manifest_path: str | Path,
    report_path: str | Path,
) -> str:
    """Publish byte-identical source pins outside ignored raw storage.

    The raw manifest contains only relative object paths and public URLs.  An
    existing public manifest may be reused only when its bytes are identical.
    """

    verified = verify_snapshot(source_manifest_path)
    del verified  # verification is the purpose; bytes remain authoritative
    source_path_value = _manifest_path(source_manifest_path)
    payload = source_path_value.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = Path(report_path)
    if destination.exists() and destination.read_bytes() != payload:
        raise WarehouseError(
            f"Public source manifest already exists with different bytes: {destination}"
        )
    if not destination.exists():
        _atomic_bytes(destination, payload)
    checksum_payload = f"{digest}  {destination.name}\n".encode("ascii")
    checksum_path = destination.with_suffix(".sha256")
    if checksum_path.exists() and checksum_path.read_bytes() != checksum_payload:
        raise WarehouseError(f"Public source checksum sidecar conflicts: {checksum_path}")
    if not checksum_path.exists():
        _atomic_bytes(checksum_path, checksum_payload)
    return digest


def _source_map(manifest_path: str | Path, manifest: SourceManifest) -> dict[str, Path]:
    return {source.source_id: source_path(manifest_path, source) for source in manifest.sources}


def _normalize_related_qrels(frame: pd.DataFrame, occupation_codes: Iterable[str]) -> pd.DataFrame:
    required = {
        "onet_soc_code",
        "related_onet_soc_code",
        "relatedness_tier",
        "index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise WarehouseError(f"related_occupations missing columns: {missing}")
    grade_map = {"Primary-Short": 3, "Primary-Long": 2, "Supplemental": 1}
    valid_codes = set(occupation_codes)
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        try:
            query = canonical_onet_soc_code(row["onet_soc_code"])
            candidate = canonical_onet_soc_code(row["related_onet_soc_code"])
        except CrosswalkError as exc:
            raise WarehouseError(f"related_occupations: {exc}") from exc
        tier = normalize_text(row["relatedness_tier"])
        if tier not in grade_map:
            raise WarehouseError(f"Unknown Related Occupations tier {tier!r}")
        index_text = normalize_text(row["index"])
        try:
            index = int(index_text)
        except ValueError as exc:
            raise WarehouseError(f"Invalid Related Occupations index {index_text!r}") from exc
        if query == candidate:
            raise WarehouseError(f"Related Occupations contains forbidden self-match for {query}")
        if query not in valid_codes or candidate not in valid_codes:
            raise WarehouseError(
                f"Related Occupations references unknown code pair ({query}, {candidate})"
            )
        rows.append(
            {
                "query_onet_soc_code": query,
                "candidate_onet_soc_code": candidate,
                "relatedness_tier": tier,
                "relatedness_index": index,
                "relevance_grade": grade_map[tier],
            }
        )
    result = pd.DataFrame(rows)
    pair = ["query_onet_soc_code", "candidate_onet_soc_code"]
    if result.empty or result.duplicated(pair).any():
        raise WarehouseError("related_qrels must be nonempty and unique by query/candidate")
    duplicate_index = result.duplicated(["query_onet_soc_code", "relatedness_index"])
    if bool(duplicate_index.any()) or bool((result["relatedness_index"] < 1).any()):
        raise WarehouseError("related_qrels indexes must be positive and unique within query")
    return result.sort_values(
        ["query_onet_soc_code", "relatedness_index", "candidate_onet_soc_code"],
        kind="stable",
    ).reset_index(drop=True)


def audit_onet_release(
    raw: Mapping[str, pd.DataFrame],
    config: ProjectConfig,
) -> None:
    """Assert the publisher-documented O*NET 30.3 release cardinalities."""

    if config.onet.release != "30.3":
        raise WarehouseError(
            "No release cardinality contract is registered for O*NET "
            f"{config.onet.release}; create a new versioned contract before building"
        )
    failures: list[str] = []
    for table, expected in ONET_30_3_RELEASE_COUNTS.items():
        actual = len(raw.get(table, ()))
        if actual != expected:
            failures.append(f"{table}: expected {expected:,}, got {actual:,}")
        frame = raw.get(table)
        if frame is None:
            failures.append(f"{table}: source table is missing")
        else:
            missing_columns = sorted(set(ONET_CONTRACTS[table].columns) - set(frame.columns))
            if missing_columns:
                failures.append(f"{table}: missing columns {missing_columns}")
    if failures:
        raise WarehouseError("O*NET 30.3 release audit failed: " + "; ".join(failures))

    occupations = raw["occupation_data"]
    transferable = raw["transferable_skills"]
    knowledge = raw["knowledge"]
    job_zones = raw["job_zones"]
    related = raw["related_occupations"]
    try:
        canonical_titles = {
            canonical_onet_soc_code(row.onet_soc_code): normalize_text(row.title)
            for row in occupations.itertuples(index=False)
        }
    except CrosswalkError as exc:
        failures.append(f"occupation_data code format: {exc}")
        canonical_titles = {}
    if len(canonical_titles) != ONET_30_3_RELEASE_COUNTS["occupation_data"]:
        failures.append("occupation_data: canonical code coverage is not one-to-one")

    for table_name, frame in raw.items():
        if table_name == "occupation_data" or not {"onet_soc_code", "title"}.issubset(
            frame.columns
        ):
            continue
        mismatches = 0
        unknown = 0
        for row in frame[["onet_soc_code", "title"]].itertuples(index=False):
            try:
                code = canonical_onet_soc_code(row.onet_soc_code)
            except CrosswalkError:
                unknown += 1
                continue
            expected_title = canonical_titles.get(code)
            if expected_title is None:
                unknown += 1
            elif normalize_text(row.title) != expected_title:
                mismatches += 1
        if unknown:
            failures.append(f"{table_name}: {unknown} unknown/malformed occupation codes")
        if mismatches:
            failures.append(f"{table_name}: {mismatches} canonical-title mismatches")

    related_title_mismatches = 0
    related_unknown = 0
    for row in related[["related_onet_soc_code", "related_title"]].itertuples(index=False):
        try:
            code = canonical_onet_soc_code(row.related_onet_soc_code)
        except CrosswalkError:
            related_unknown += 1
            continue
        expected_title = canonical_titles.get(code)
        if expected_title is None:
            related_unknown += 1
        elif normalize_text(row.related_title) != expected_title:
            related_title_mismatches += 1
    if related_unknown:
        failures.append(
            f"related_occupations: {related_unknown} unknown/malformed related occupation codes"
        )
    if related_title_mismatches:
        failures.append(f"related_occupations: {related_title_mismatches} related-title mismatches")

    expected_dimensions = (
        ("transferable_skills", transferable, 894, 25, 2),
        ("knowledge", knowledge, 894, 33, 2),
    )
    for (
        name,
        frame,
        expected_occupations,
        expected_elements,
        expected_scales,
    ) in expected_dimensions:
        observed = (
            int(frame["onet_soc_code"].nunique()),
            int(frame["element_id"].nunique()),
            int(frame["scale_id"].nunique()),
        )
        expected = (expected_occupations, expected_elements, expected_scales)
        if observed != expected:
            failures.append(f"{name} dimensions: expected {expected}, got {observed}")
        expected_rows_per_occupation = expected_elements * expected_scales
        rows_per_occupation = frame.groupby("onet_soc_code", dropna=False).size()
        if not bool(rows_per_occupation.eq(expected_rows_per_occupation).all()):
            failures.append(
                f"{name}: every rated occupation must have {expected_rows_per_occupation} rows"
            )
        if set(frame["scale_id"].dropna()) != {"IM", "LV"}:
            failures.append(f"{name}: expected exactly IM and LV scales")

    try:
        zones = pd.to_numeric(job_zones["job_zone"], errors="raise").astype(int)
    except (TypeError, ValueError):
        failures.append("job_zones: values must be integers in the current 1-4 framework")
    else:
        if set(zones.unique()) != {1, 2, 3, 4} or not bool(zones.between(1, 4).all()):
            failures.append("job_zones: expected only and all current levels 1, 2, 3, and 4")

    query_sizes = related.groupby("onet_soc_code", dropna=False).size()
    if len(query_sizes) != 923 or not bool(query_sizes.eq(20).all()):
        failures.append("related_occupations: expected 923 queries with exactly 20 judgments each")
    tiers = related["relatedness_tier"].value_counts().to_dict()
    expected_tiers = {"Primary-Short": 4_615, "Primary-Long": 4_615, "Supplemental": 9_230}
    if tiers != expected_tiers:
        failures.append(f"related_occupations tier counts: expected {expected_tiers}, got {tiers}")
    per_query_tiers = (
        related.groupby(["onet_soc_code", "relatedness_tier"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    expected_per_query_tiers = {
        "Primary-Short": 5,
        "Primary-Long": 5,
        "Supplemental": 10,
    }
    if set(per_query_tiers.columns) != set(expected_per_query_tiers) or any(
        not bool(per_query_tiers[tier].eq(expected).all())
        for tier, expected in expected_per_query_tiers.items()
        if tier in per_query_tiers
    ):
        failures.append("related_occupations: each query must have 5/5/10 tier counts")
    try:
        related_indexes = pd.to_numeric(related["index"], errors="raise").astype(int)
    except (TypeError, ValueError):
        failures.append("related_occupations: rank indexes must be integers")
    else:
        indexed = related.assign(_release_index=related_indexes)
        index_sets = indexed.groupby("onet_soc_code", dropna=False)["_release_index"].agg(set)
        if not bool(index_sets.map(lambda values: values == set(range(1, 21))).all()):
            failures.append("related_occupations: each query must contain ranks 1 through 20")
    if failures:
        raise WarehouseError("O*NET 30.3 release audit failed: " + "; ".join(failures))


def load_source_frames(
    source_manifest_path: str | Path,
    config: ProjectConfig,
    *,
    enforce_release_contract: bool = True,
) -> dict[str, pd.DataFrame]:
    """Parse, normalize, and assemble every canonical processed table."""

    manifest = verify_snapshot(source_manifest_path, config=config)
    paths = _source_map(source_manifest_path, manifest)
    raw = {table: read_onet_csv(paths[f"onet.{table}"], table) for table in config.onet.tables}
    if enforce_release_contract:
        audit_onet_release(raw, config)
    occupations = build_occupations(
        raw["occupation_data"],
        raw["job_zones"],
        job_zone_levels=config.profiles.job_zone_levels,
    )
    aliases = build_aliases(raw["job_titles"])
    tasks = build_tasks(raw["task_statements"])
    skill_ratings = build_structured_ratings(raw["transferable_skills"], raw["knowledge"])
    profiles = build_occupation_profiles(
        occupations,
        aliases,
        tasks,
        skill_ratings,
        core_task_type=config.profiles.core_task_type,
    )
    related_qrels = _normalize_related_qrels(
        raw["related_occupations"], occupations["onet_soc_code"]
    )
    bls_raw = read_bls_workbook(paths["bls.occupation"], "occupation")
    crosswalk_raw = read_bls_workbook(paths["bls.crosswalk"], "crosswalk")
    try:
        bls_market = normalize_bls_market(bls_raw, line_items_only=True)
        onet_bls_crosswalk = normalize_crosswalk(crosswalk_raw)
        occupation_market = join_market_context(
            onet_bls_crosswalk,
            bls_market,
            valid_onet_codes=occupations["onet_soc_code"],
        )
    except CrosswalkError as exc:
        raise WarehouseError(str(exc)) from exc
    return {
        "occupations": occupations,
        "aliases": aliases,
        "tasks": tasks,
        "skill_ratings": skill_ratings,
        "related_qrels": related_qrels,
        "bls_market": bls_market,
        "onet_bls_crosswalk": onet_bls_crosswalk,
        "occupation_market": occupation_market,
        "occupation_profiles": profiles,
    }


def _issue(
    issues: list[AuditIssue], check: str, count: int, detail: str, severity: str = "error"
) -> None:
    if count:
        issues.append(AuditIssue(check, severity, int(count), detail))


def audit_frames(frames: Mapping[str, pd.DataFrame]) -> AuditReport:
    """Run leakage, key, unit, mask, and crosswalk cardinality gates."""

    missing_tables = sorted(set(TABLE_KEYS) - set(frames))
    if missing_tables:
        raise WarehouseError(f"Audit input is missing tables: {missing_tables}")
    issues: list[AuditIssue] = []
    rows = {table: len(frames[table]) for table in TABLE_KEYS}
    for table, key in TABLE_KEYS.items():
        frame = frames[table]
        actual_columns = tuple(frame.columns)
        if actual_columns != TABLE_COLUMNS[table]:
            raise WarehouseError(
                f"{table} columns differ from the processed schema; "
                f"expected {TABLE_COLUMNS[table]}, got {actual_columns}"
            )
        _issue(issues, f"{table}.empty", int(frame.empty), "table must be nonempty")
        absent = [column for column in key if column not in frame.columns]
        if absent:
            _issue(issues, f"{table}.key_columns", len(absent), f"missing {absent}")
            continue
        _issue(
            issues,
            f"{table}.duplicate_key",
            int(frame.duplicated(list(key)).sum()),
            f"key={key}",
        )
        _issue(
            issues,
            f"{table}.null_key",
            int(frame.loc[:, list(key)].isna().any(axis=1).sum()),
            f"key={key}",
        )

    occupations = frames["occupations"]
    occupation_codes = set(occupations["onet_soc_code"])
    invalid_base = sum(
        str(row.base_soc_code) != base_soc_code(row.onet_soc_code)
        for row in occupations.itertuples(index=False)
    )
    invalid_major = sum(
        str(row.soc_major_group) != soc_major_group(row.onet_soc_code)
        for row in occupations.itertuples(index=False)
    )
    _issue(issues, "occupations.base_soc_code", invalid_base, "must derive exactly from O*NET code")
    _issue(
        issues, "occupations.soc_major_group", invalid_major, "must derive exactly from O*NET code"
    )
    if "job_zone" in occupations:
        invalid_zone = int((~occupations["job_zone"].dropna().isin(range(1, 5))).sum())
        _issue(
            issues, "occupations.job_zone", invalid_zone, "O*NET 30.3 job zone must be 1..4 or null"
        )

    for table in ("aliases", "tasks", "skill_ratings", "occupation_profiles"):
        orphan = set(frames[table]["onet_soc_code"]) - occupation_codes
        _issue(issues, f"{table}.occupation_fk", len(orphan), f"unknown codes={sorted(orphan)[:5]}")

    ratings = frames["skill_ratings"]
    invalid_normalized = ratings["observed"] & (
        ratings["normalized_value"].isna()
        | (ratings["normalized_value"] < 0)
        | (ratings["normalized_value"] > 1)
    )
    unobserved_value = ~ratings["observed"] & ratings["normalized_value"].notna()
    masked_observed = ratings["observed"] & (
        ratings["recommend_suppress"] | ratings["not_relevant"]
    )
    _issue(
        issues,
        "skill_ratings.observed_value",
        int(invalid_normalized.sum()),
        "observed normalized values must be in [0,1]",
    )
    _issue(
        issues,
        "skill_ratings.unobserved_null",
        int(unobserved_value.sum()),
        "unobserved normalized values must be null",
    )
    _issue(
        issues,
        "skill_ratings.suppression_mask",
        int(masked_observed.sum()),
        "suppressed/not-relevant ratings cannot be observed",
    )
    _issue(
        issues,
        "skill_ratings.sample_size",
        int((ratings["sample_size"].dropna() < 0).sum()),
        "sample size cannot be negative",
    )
    _issue(
        issues,
        "skill_ratings.standard_error",
        int((ratings["standard_error"].dropna() < 0).sum()),
        "standard error cannot be negative",
    )
    partial_ci = ratings["lower_ci_bound"].isna() ^ ratings["upper_ci_bound"].isna()
    reversed_ci = (
        ratings["lower_ci_bound"].notna()
        & ratings["upper_ci_bound"].notna()
        & (ratings["lower_ci_bound"] > ratings["upper_ci_bound"])
    )
    excludes_value = (
        ratings["raw_value"].notna()
        & ratings["lower_ci_bound"].notna()
        & ratings["upper_ci_bound"].notna()
        & (
            (ratings["raw_value"] < ratings["lower_ci_bound"])
            | (ratings["raw_value"] > ratings["upper_ci_bound"])
        )
    )
    _issue(
        issues,
        "skill_ratings.confidence_interval",
        int((partial_ci | reversed_ci | excludes_value).sum()),
        "confidence bounds must be jointly present, ordered, and contain the raw value",
    )

    qrels = frames["related_qrels"]
    _issue(
        issues,
        "related_qrels.self_match",
        int((qrels["query_onet_soc_code"] == qrels["candidate_onet_soc_code"]).sum()),
        "self matches are prohibited",
    )
    qrel_orphans = (
        set(qrels["query_onet_soc_code"]) | set(qrels["candidate_onet_soc_code"])
    ) - occupation_codes
    _issue(
        issues,
        "related_qrels.occupation_fk",
        len(qrel_orphans),
        f"unknown codes={sorted(qrel_orphans)[:5]}",
    )
    _issue(
        issues,
        "related_qrels.relevance_grade",
        int((~qrels["relevance_grade"].isin([1, 2, 3])).sum()),
        "grade must be 1, 2, or 3",
    )
    expected_grades = qrels["relatedness_tier"].map(
        {"Primary-Short": 3, "Primary-Long": 2, "Supplemental": 1}
    )
    _issue(
        issues,
        "related_qrels.tier_grade_mapping",
        int((expected_grades.isna() | expected_grades.ne(qrels["relevance_grade"])).sum()),
        "tier/grade must follow the preregistered 3/2/1 convention",
    )

    for table in FEATURE_TABLES:
        leaked = FORBIDDEN_FEATURE_COLUMNS & set(frames[table].columns)
        _issue(
            issues,
            f"{table}.forbidden_feature_columns",
            len(leaked),
            f"leaked={sorted(leaked)}",
        )

    market = frames["bls_market"]
    _issue(
        issues,
        "bls_market.line_item_only",
        int((market["occupation_type"].str.casefold() != "line item").sum()),
        "summary rows must not enter market context",
    )
    for column in (
        "employment_2024",
        "employment_2034",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
    ):
        _issue(
            issues,
            f"bls_market.{column}.nonnegative",
            int((market[column].dropna() < 0).sum()),
            "must be a nonnegative integer in persons/dollars",
        )

    crosswalk = frames["onet_bls_crosswalk"]
    crosswalk_onet_orphans = set(crosswalk["onet_soc_code"]) - occupation_codes
    crosswalk_nem_orphans = set(crosswalk["nem_code"]) - set(market["nem_code"])
    _issue(
        issues,
        "onet_bls_crosswalk.occupation_fk",
        len(crosswalk_onet_orphans),
        f"unknown codes={sorted(crosswalk_onet_orphans)[:5]}",
    )
    _issue(
        issues,
        "onet_bls_crosswalk.market_fk",
        len(crosswalk_nem_orphans),
        f"unknown NEM codes={sorted(crosswalk_nem_orphans)[:5]}",
    )
    occupation_market = frames["occupation_market"]
    _issue(
        issues,
        "occupation_market.cardinality",
        abs(len(occupation_market) - len(crosswalk)),
        "market join must preserve one row per official code pair",
    )
    expected_pairs = set(
        map(tuple, crosswalk[["onet_soc_code", "nem_code"]].itertuples(index=False, name=None))
    )
    actual_pairs = set(
        map(
            tuple,
            occupation_market[["onet_soc_code", "nem_code"]].itertuples(index=False, name=None),
        )
    )
    _issue(
        issues,
        "occupation_market.mapping_pairs",
        len(expected_pairs ^ actual_pairs),
        "joined mapping pairs must equal official crosswalk pairs",
    )
    try:
        expected_market = join_market_context(
            crosswalk,
            market,
            valid_onet_codes=occupations["onet_soc_code"],
        )
    except CrosswalkError as exc:
        raise WarehouseError(f"Could not reconstruct occupation_market audit: {exc}") from exc
    actual_market = occupation_market.sort_values(
        list(TABLE_SORT_KEYS["occupation_market"]), kind="stable"
    ).reset_index(drop=True)
    if not expected_market.equals(actual_market):
        _issue(
            issues,
            "occupation_market.exact_join",
            1,
            "all market values must equal the unsummed official many-to-one join",
        )

    profiles = frames["occupation_profiles"]
    profile_identity = profiles.loc[:, list(TABLE_COLUMNS["occupations"])]
    occupation_identity = occupations.loc[:, list(TABLE_COLUMNS["occupations"])]
    if not profile_identity.equals(occupation_identity):
        _issue(
            issues,
            "occupation_profiles.identity",
            1,
            "profile identity and Job Zone provenance must match occupations exactly",
        )
    return AuditReport(tuple(issues), rows)


def _register_table(connection: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    temporary = f"_{name}_frame"
    connection.register(temporary, frame)
    try:
        connection.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "{temporary}"')
    finally:
        connection.unregister(temporary)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if not np.isfinite(value):
            raise WarehouseError("Non-finite value cannot enter deterministic table hash")
        return format(value, ".17g")
    return value


def semantic_table_sha256(frame: pd.DataFrame, key: Sequence[str]) -> str:
    ordered = frame.sort_values(list(key), kind="stable", na_position="last")
    digest = hashlib.sha256()
    digest.update((json.dumps(list(ordered.columns), separators=(",", ":")) + "\n").encode("utf-8"))
    for row in ordered.itertuples(index=False, name=None):
        payload = [_json_scalar(value) for value in row]
        digest.update(
            (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _duckdb_schema(connection: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    rows = connection.execute(f'DESCRIBE "{table}"').fetchall()
    return [
        {
            "name": str(row[0]),
            "type": str(row[1]),
            "nullable": str(row[2]).upper() == "YES",
        }
        for row in rows
    ]


def _arrow_schema(path: Path) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in pq.read_schema(path)
    ]


def _build_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
    }


def _build_stage(
    frames: Mapping[str, pd.DataFrame],
    stage: Path,
    source_manifest_path: str | Path,
    source_manifest: SourceManifest,
    config: ProjectConfig,
) -> dict[str, Any]:
    report = audit_frames(frames)
    report.raise_for_errors()
    source_manifest_file_sha = _source_manifest_sha256(source_manifest_path)
    warehouse_path = stage / "warehouse.duckdb"
    connection = duckdb.connect(str(warehouse_path))
    try:
        connection.execute("PRAGMA threads=1")
        connection.execute("PRAGMA preserve_insertion_order=true")
        for table in TABLE_KEYS:
            ordered = frames[table].sort_values(
                list(TABLE_SORT_KEYS[table]), kind="stable", na_position="last"
            )
            _register_table(connection, table, ordered)
        # This compatibility view makes the missing-aware representation clear
        # while retaining ``skill_ratings`` as the stable serving name.
        connection.execute("CREATE VIEW structured_ratings AS SELECT * FROM skill_ratings")
        connection.execute(
            """
            CREATE TABLE build_audit AS
            SELECT
                ?::VARCHAR AS check_name,
                ?::BOOLEAN AS passed,
                ?::VARCHAR AS source_manifest_sha256,
                ?::VARCHAR AS config_file_sha256,
                ?::INTEGER AS processed_manifest_version
            """,
            [
                "strict_python_audit",
                True,
                source_manifest_file_sha,
                config.fingerprint,
                PROCESSED_MANIFEST_VERSION,
            ],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()

    tables_dir = stage / "tables"
    tables_dir.mkdir()
    table_entries: list[dict[str, Any]] = []
    export_connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        for table in TABLE_KEYS:
            sort_key = TABLE_SORT_KEYS[table]
            key_sql = ", ".join(f'"{column}"' for column in sort_key)
            parquet_path = tables_dir / f"{table}.parquet"
            escaped_path = str(parquet_path).replace("'", "''")
            export_connection.execute(
                f'COPY (SELECT * FROM "{table}" ORDER BY {key_sql}) '
                f"TO '{escaped_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            frame = export_connection.execute(
                f'SELECT * FROM "{table}" ORDER BY {key_sql}'
            ).fetchdf()
            table_entries.append(
                {
                    "name": table,
                    "row_count": len(frame),
                    "columns": list(frame.columns),
                    "pandas_schema_sha256": schema_fingerprint(
                        frame.columns.tolist(), frame.dtypes.tolist()
                    ),
                    "duckdb_schema": _duckdb_schema(export_connection, table),
                    "parquet_schema": _arrow_schema(parquet_path),
                    "semantic_sha256": semantic_table_sha256(frame, sort_key),
                    "primary_key": list(TABLE_KEYS[table]),
                    "sort_order": list(sort_key),
                    "parquet_path": f"tables/{table}.parquet",
                    "parquet_byte_count": parquet_path.stat().st_size,
                    "parquet_sha256": sha256_file(parquet_path),
                }
            )
    finally:
        export_connection.close()
    warehouse_sha = sha256_file(warehouse_path)
    manifest = {
        "manifest_version": PROCESSED_MANIFEST_VERSION,
        "snapshot_id": source_manifest.snapshot_id,
        "project_name": config.project.name,
        "artifact_version": config.project.artifact_version,
        "snapshot_date": config.project.snapshot_date.isoformat(),
        "config_file_sha256": config.fingerprint,
        "config_semantic_sha256": config.semantic_fingerprint,
        "source_manifest_sha256": source_manifest_file_sha,
        "source_pins": [
            {
                "source_id": source.source_id,
                "sha256": source.sha256,
                "byte_count": source.byte_count,
                "row_count": source.row_count,
                "schema_fingerprint": source.schema_fingerprint,
                "parser_name": source.parser_name,
                "parser_version": source.parser_version,
            }
            for source in source_manifest.sources
        ],
        "build_environment": _build_environment(),
        "warehouse": {
            "path": "warehouse.duckdb",
            "byte_count": warehouse_path.stat().st_size,
            "sha256": warehouse_sha,
        },
        "tables": table_entries,
        "audit": report.to_dict(),
    }
    _write_json(stage / "manifest.json", manifest)
    return manifest


def load_processed_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _manifest_path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WarehouseError(f"Could not read processed manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WarehouseError("Processed manifest root must be an object")
    expected_fields = {
        "manifest_version",
        "snapshot_id",
        "project_name",
        "artifact_version",
        "snapshot_date",
        "config_file_sha256",
        "config_semantic_sha256",
        "source_manifest_sha256",
        "source_pins",
        "build_environment",
        "warehouse",
        "tables",
        "audit",
    }
    if set(value) != expected_fields:
        raise WarehouseError(
            "Processed manifest fields differ from schema v1: "
            f"missing={sorted(expected_fields - set(value))}, "
            f"unknown={sorted(set(value) - expected_fields)}"
        )
    version_value = value["manifest_version"]
    if (
        isinstance(version_value, bool)
        or not isinstance(version_value, int)
        or version_value != PROCESSED_MANIFEST_VERSION
    ):
        raise WarehouseError("Unsupported processed manifest version")
    return value


def _validated_relative_file(root: Path, value: Any, expected: str) -> Path:
    if not isinstance(value, str) or value != expected:
        raise WarehouseError(f"Processed path must be exactly {expected!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise WarehouseError(f"Unsafe processed path: {value!r}")
    unresolved = root
    for part in relative.parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise WarehouseError(f"Processed path contains a symbolic link: {value!r}")
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WarehouseError(f"Processed path escapes snapshot: {value!r}") from exc
    return candidate


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise WarehouseError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: Any, field: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WarehouseError(f"{field} must be an integer >= {minimum}")
    return value


def _source_pin_records(manifest: SourceManifest) -> list[dict[str, Any]]:
    return [
        {
            "source_id": source.source_id,
            "sha256": source.sha256,
            "byte_count": source.byte_count,
            "row_count": source.row_count,
            "schema_fingerprint": source.schema_fingerprint,
            "parser_name": source.parser_name,
            "parser_version": source.parser_version,
        }
        for source in manifest.sources
    ]


def verify_processed_snapshot(
    path: str | Path,
    *,
    config: ProjectConfig | None = None,
    source_manifest_path: str | Path | None = None,
) -> BuildResult:
    supplied_path = Path(path)
    if supplied_path.is_symlink():
        raise WarehouseError(f"Processed snapshot path cannot be a symbolic link: {path}")
    root = supplied_path.resolve()
    if root.is_file():
        root = root.parent
    if root.is_symlink() or not root.is_dir():
        raise WarehouseError(f"Processed snapshot directory is missing or linked: {root}")
    manifest = load_processed_manifest(root)
    for field in (
        "snapshot_id",
        "project_name",
        "artifact_version",
        "snapshot_date",
    ):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise WarehouseError(f"Processed manifest {field} must be a nonempty string")
    for field in (
        "config_file_sha256",
        "config_semantic_sha256",
        "source_manifest_sha256",
    ):
        _require_sha256(manifest[field], field)
    if manifest["snapshot_id"] != manifest["artifact_version"]:
        raise WarehouseError("Processed snapshot ID and artifact version differ")
    try:
        parsed_snapshot_date = date.fromisoformat(manifest["snapshot_date"])
    except ValueError as exc:
        raise WarehouseError("Processed snapshot date is not ISO YYYY-MM-DD") from exc
    if parsed_snapshot_date.isoformat() != manifest["snapshot_date"]:
        raise WarehouseError("Processed snapshot date is not canonical")
    if config is not None:
        expected_config_values = (
            config.snapshot_id,
            config.project.name,
            config.project.artifact_version,
            config.project.snapshot_date.isoformat(),
            config.fingerprint,
            config.semantic_fingerprint,
        )
        actual_config_values = (
            manifest["snapshot_id"],
            manifest["project_name"],
            manifest["artifact_version"],
            manifest["snapshot_date"],
            manifest["config_file_sha256"],
            manifest["config_semantic_sha256"],
        )
        if actual_config_values != expected_config_values:
            raise WarehouseError("Processed provenance differs from the locked configuration")

    if source_manifest_path is not None:
        source_manifest = verify_snapshot(source_manifest_path, config=config)
        source_manifest_sha = _source_manifest_sha256(source_manifest_path)
        if manifest["source_manifest_sha256"] != source_manifest_sha:
            raise WarehouseError("Processed source-manifest checksum mismatch")
        if manifest["source_pins"] != _source_pin_records(source_manifest):
            raise WarehouseError("Processed source pins differ from the verified source manifest")
    else:
        source_pins = manifest["source_pins"]
        expected_pin_fields = {
            "source_id",
            "sha256",
            "byte_count",
            "row_count",
            "schema_fingerprint",
            "parser_name",
            "parser_version",
        }
        if (
            not isinstance(source_pins, list)
            or len(source_pins) != 9
            or not all(isinstance(record, dict) for record in source_pins)
            or [record.get("source_id") for record in source_pins] != list(EXPECTED_SOURCE_LAYOUT)
        ):
            raise WarehouseError("Processed manifest must carry nine ordered source-pin records")
        for index, record in enumerate(source_pins):
            if set(record) != expected_pin_fields:
                raise WarehouseError(f"Source-pin record {index} differs from schema v1")
            _require_sha256(record["sha256"], f"source_pins[{index}].sha256")
            _require_sha256(
                record["schema_fingerprint"],
                f"source_pins[{index}].schema_fingerprint",
            )
            _require_nonnegative_int(
                record["byte_count"], f"source_pins[{index}].byte_count", positive=True
            )
            _require_nonnegative_int(
                record["row_count"], f"source_pins[{index}].row_count", positive=True
            )
            for field in ("source_id", "parser_name", "parser_version"):
                if not isinstance(record[field], str) or not record[field]:
                    raise WarehouseError(f"source_pins[{index}].{field} must be nonempty")

    if manifest["build_environment"] != _build_environment():
        raise WarehouseError("Processed snapshot was built with a different locked environment")
    warehouse_record = manifest["warehouse"]
    if not isinstance(warehouse_record, dict) or set(warehouse_record) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise WarehouseError("Processed warehouse record differs from schema v1")
    warehouse_path = _validated_relative_file(root, warehouse_record["path"], "warehouse.duckdb")
    if not warehouse_path.is_file():
        raise WarehouseError("Processed warehouse file is missing")
    if warehouse_path.stat().st_size != _require_nonnegative_int(
        warehouse_record["byte_count"], "warehouse.byte_count", positive=True
    ):
        raise WarehouseError("Processed warehouse byte-count mismatch")
    warehouse_sha = sha256_file(warehouse_path)
    if warehouse_sha != _require_sha256(warehouse_record["sha256"], "warehouse.sha256"):
        raise WarehouseError("Processed warehouse checksum mismatch")
    table_rows: dict[str, int] = {}
    raw_entries = manifest["tables"]
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, dict) for entry in raw_entries
    ):
        raise WarehouseError("Processed table manifest must be an ordered array of objects")
    entry_names = [entry.get("name") for entry in raw_entries]
    if entry_names != list(TABLE_KEYS):
        raise WarehouseError("Processed table manifest does not match the table contract")
    entries = {entry["name"]: entry for entry in raw_entries}

    expected_files = {root / "manifest.json", warehouse_path}
    for table, entry in entries.items():
        expected_files.add(
            _validated_relative_file(root, entry.get("parquet_path"), f"tables/{table}.parquet")
        )
    actual_files: set[Path] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise WarehouseError(f"Processed snapshot contains a symbolic link: {candidate}")
        if candidate.is_file():
            actual_files.add(candidate.resolve())
    expected_files = {candidate.resolve() for candidate in expected_files}
    if actual_files != expected_files:
        unexpected = sorted(str(path.relative_to(root)) for path in actual_files - expected_files)
        missing = sorted(str(path.relative_to(root)) for path in expected_files - actual_files)
        raise WarehouseError(
            f"Processed file set differs from manifest; unexpected={unexpected}, missing={missing}"
        )

    database_frames: dict[str, pd.DataFrame] = {}
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE'"
            ).fetchall()
        }
        if table_names != {*TABLE_KEYS, "build_audit"}:
            raise WarehouseError(f"Warehouse table set differs from contract: {table_names}")
        view_names = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema='main'"
            ).fetchall()
        }
        if "structured_ratings" not in view_names:
            raise WarehouseError("Warehouse structured_ratings compatibility view is missing")
        build_audit = connection.execute("SELECT * FROM build_audit").fetchone()
        expected_build_audit = (
            "strict_python_audit",
            True,
            manifest["source_manifest_sha256"],
            manifest["config_file_sha256"],
            PROCESSED_MANIFEST_VERSION,
        )
        if build_audit != expected_build_audit:
            raise WarehouseError("Warehouse build_audit provenance differs from manifest")

        expected_entry_fields = {
            "name",
            "row_count",
            "columns",
            "pandas_schema_sha256",
            "duckdb_schema",
            "parquet_schema",
            "semantic_sha256",
            "primary_key",
            "sort_order",
            "parquet_path",
            "parquet_byte_count",
            "parquet_sha256",
        }
        for table in TABLE_KEYS:
            entry = entries[table]
            if set(entry) != expected_entry_fields:
                raise WarehouseError(f"Processed table record fields differ for {table}")
            if entry["columns"] != list(TABLE_COLUMNS[table]):
                raise WarehouseError(f"Processed column contract mismatch for {table}")
            if entry["primary_key"] != list(TABLE_KEYS[table]):
                raise WarehouseError(f"Processed primary-key contract mismatch for {table}")
            if entry["sort_order"] != list(TABLE_SORT_KEYS[table]):
                raise WarehouseError(f"Processed sort-order contract mismatch for {table}")
            parquet_path = _validated_relative_file(
                root, entry["parquet_path"], f"tables/{table}.parquet"
            )
            if not parquet_path.is_file():
                raise WarehouseError(f"Processed Parquet file is missing for {table}")
            if parquet_path.stat().st_size != _require_nonnegative_int(
                entry["parquet_byte_count"], f"{table}.parquet_byte_count", positive=True
            ):
                raise WarehouseError(f"Processed Parquet byte-count mismatch for {table}")
            if sha256_file(parquet_path) != _require_sha256(
                entry["parquet_sha256"], f"{table}.parquet_sha256"
            ):
                raise WarehouseError(f"Processed Parquet checksum mismatch for {table}")
            row_count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            if row_count != _require_nonnegative_int(
                entry["row_count"], f"{table}.row_count", positive=True
            ):
                raise WarehouseError(f"Processed row-count mismatch for {table}")
            sort_sql = ", ".join(f'"{column}"' for column in TABLE_SORT_KEYS[table])
            frame = connection.execute(f'SELECT * FROM "{table}" ORDER BY {sort_sql}').fetchdf()
            database_frames[table] = frame
            if list(frame.columns) != entry["columns"]:
                raise WarehouseError(f"Warehouse column order mismatch for {table}")
            if schema_fingerprint(frame.columns.tolist(), frame.dtypes.tolist()) != _require_sha256(
                entry["pandas_schema_sha256"], f"{table}.pandas_schema_sha256"
            ):
                raise WarehouseError(f"Warehouse pandas schema mismatch for {table}")
            if _duckdb_schema(connection, table) != entry["duckdb_schema"]:
                raise WarehouseError(f"Warehouse DuckDB schema mismatch for {table}")
            if _arrow_schema(parquet_path) != entry["parquet_schema"]:
                raise WarehouseError(f"Warehouse Parquet schema mismatch for {table}")
            semantic_sha = _require_sha256(entry["semantic_sha256"], f"{table}.semantic_sha256")
            if semantic_table_sha256(frame, TABLE_SORT_KEYS[table]) != semantic_sha:
                raise WarehouseError(f"Processed semantic table checksum mismatch for {table}")
            parquet_frame = connection.execute(
                f"SELECT * FROM read_parquet(?) ORDER BY {sort_sql}", [str(parquet_path)]
            ).fetchdf()
            if semantic_table_sha256(parquet_frame, TABLE_SORT_KEYS[table]) != semantic_sha:
                raise WarehouseError(f"Parquet/database semantic mismatch for {table}")
            table_rows[table] = row_count
    finally:
        connection.close()

    audit = audit_frames(database_frames)
    audit.raise_for_errors()
    if manifest["audit"] != audit.to_dict():
        raise WarehouseError("Processed audit evidence differs from a fresh strict audit")
    return BuildResult(
        snapshot_id=str(manifest["snapshot_id"]),
        snapshot_dir=root,
        warehouse_path=warehouse_path,
        manifest_path=root / "manifest.json",
        source_manifest_sha256=str(manifest["source_manifest_sha256"]),
        warehouse_sha256=warehouse_sha,
        table_rows=table_rows,
    )


def build_warehouse(
    source_manifest_path: str | Path,
    output_dir: str | Path,
    config: ProjectConfig,
    *,
    publish_manifest_path: str | Path | None = None,
) -> BuildResult:
    """Build a new immutable processed snapshot and publish it atomically."""

    source_manifest = verify_snapshot(source_manifest_path, config=config)
    supplied_target = Path(output_dir).expanduser()
    if supplied_target.is_symlink():
        raise WarehouseError(f"Processed output cannot be a symbolic link: {supplied_target}")
    target = supplied_target.resolve()
    if target.exists():
        result = verify_processed_snapshot(
            target,
            config=config,
            source_manifest_path=source_manifest_path,
        )
        if result.source_manifest_sha256 != _source_manifest_sha256(source_manifest_path):
            raise WarehouseError("Existing processed snapshot was built from different source pins")
        if publish_manifest_path is not None:
            published_sha = publish_source_manifest(source_manifest_path, publish_manifest_path)
            if published_sha != result.source_manifest_sha256:
                raise WarehouseError(
                    "Published source manifest digest differs from processed provenance"
                )
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        frames = load_source_frames(source_manifest_path, config)
        _build_stage(frames, stage, source_manifest_path, source_manifest, config)
        verify_processed_snapshot(
            stage,
            config=config,
            source_manifest_path=source_manifest_path,
        )
        os.replace(stage, target)
        stage = target
        result = verify_processed_snapshot(
            target,
            config=config,
            source_manifest_path=source_manifest_path,
        )
        if publish_manifest_path is not None:
            published_sha = publish_source_manifest(source_manifest_path, publish_manifest_path)
            if published_sha != result.source_manifest_sha256:
                raise WarehouseError(
                    "Published source manifest digest differs from processed provenance"
                )
        return result
    except (ProfileError, CrosswalkError, SchemaError) as exc:
        raise WarehouseError(str(exc)) from exc
    finally:
        if stage.exists() and stage != target:
            shutil.rmtree(stage)


def build_snapshot(
    config: ProjectConfig,
    source_snapshot: str | Path,
    processed_root: str | Path | None = None,
    *,
    snapshot_id: str | None = None,
    publish_manifest_path: str | Path | None = None,
) -> BuildResult:
    source_manifest = load_manifest(source_snapshot)
    chosen_id = snapshot_id or source_manifest.snapshot_id
    if chosen_id != source_manifest.snapshot_id:
        raise WarehouseError("Processed snapshot ID must equal its immutable source snapshot ID")
    root = Path(processed_root or config.root / "data" / "processed").resolve()
    target = root / "snapshots" / chosen_id
    public_path = publish_manifest_path or config.root / "reports" / "source_manifest.json"
    return build_warehouse(
        source_snapshot,
        target,
        config,
        publish_manifest_path=public_path,
    )


def read_warehouse_frames(path: str | Path) -> dict[str, pd.DataFrame]:
    value = Path(path)
    warehouse_path = value / "warehouse.duckdb" if value.is_dir() else value
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        return {
            table: connection.execute(f'SELECT * FROM "{table}"').fetchdf() for table in TABLE_KEYS
        }
    finally:
        connection.close()


def audit_warehouse(path: str | Path) -> AuditReport:
    frames = read_warehouse_frames(path)
    report = audit_frames(frames)
    report.raise_for_errors()
    return report


__all__ = [
    "AuditIssue",
    "AuditReport",
    "BuildResult",
    "FEATURE_TABLES",
    "FORBIDDEN_FEATURE_COLUMNS",
    "PROCESSED_MANIFEST_VERSION",
    "ONET_30_3_RELEASE_COUNTS",
    "TABLE_KEYS",
    "WarehouseError",
    "audit_frames",
    "audit_onet_release",
    "audit_warehouse",
    "build_snapshot",
    "build_warehouse",
    "load_processed_manifest",
    "load_source_frames",
    "publish_source_manifest",
    "read_warehouse_frames",
    "semantic_table_sha256",
    "verify_processed_snapshot",
]
