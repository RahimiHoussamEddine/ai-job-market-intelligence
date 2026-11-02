"""Validated, checksum-addressed serving artifacts.

The serving boundary is intentionally narrower than the experiment boundary.  A
bundle contains only target-free occupation features and frozen model outputs;
Related Occupations judgments are never a serving dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import warnings
import zipfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .config import ConfigError, ProjectConfig, load_config
from .embeddings import TextLSAIndex
from .evaluation import (
    TITLE_QUERY_POLICY,
    EvaluationIntegrityError,
    verify_development_decision,
)
from .lexical import TfidfIndex, WeightedTextTransformer
from .modeling import (
    OPERATIONAL_COST_FORMULA,
    OPERATIONAL_COST_POLICY,
    CandidateSpec,
    candidate_grid_digest,
    title_candidate_grid,
    transition_candidate_grid,
)
from .retrieval import fitting_codes_for_phase
from .sources import ManifestError, SourceManifest, build_source_specs
from .splits import (
    SPLIT_MANIFEST_VERSION,
    SplitManifestError,
    assert_fold_manifest,
    fold_manifest_digest,
)

MANIFEST_FILENAME: Final = "manifest.json"
MANIFEST_SCHEMA_VERSION: Final = 1
STORED_TRANSITION_LIMIT: Final = 250
NEIGHBOR_DISPLAY_MAX: Final = 25
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40,64}$")
ONET_SOC_RE: Final = re.compile(r"^\d{2}-\d{4}\.\d{2}$")
BASE_SOC_RE: Final = re.compile(r"^\d{2}-\d{4}$")
SOC_MAJOR_RE: Final = re.compile(r"^\d{2}$")
PROCESSED_TABLE_NAMES: Final = (
    "occupations",
    "aliases",
    "tasks",
    "skill_ratings",
    "related_qrels",
    "bls_market",
    "onet_bls_crosswalk",
    "occupation_market",
    "occupation_profiles",
)

# These names identify evaluation targets, not permissible serving features.
# Matching is performed after punctuation/case normalization and intentionally
# does not ban ordinary prose words such as "related" in a title vocabulary.
FORBIDDEN_FIELD_PATTERNS: Final = (
    re.compile(r"(?:^|_)qrels?(?:_|$)"),
    re.compile(r"(?:^|_)related_occupations?(?:_|$)"),
    re.compile(r"(?:^|_)relevance(?:_tier|_grade|_label)?(?:_|$)"),
    re.compile(r"(?:^|_)(?:human_)?judg(?:e)?ments?(?:_|$)"),
    re.compile(r"(?:^|_)primary_(?:short|long)(?:_|$)"),
    re.compile(r"(?:^|_)supplemental_tier(?:_|$)"),
    re.compile(r"(?:^|_)evaluation_targets?(?:_|$)"),
)


def _field(name: str, data_type: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, data_type, nullable=nullable)


SAFE_LOCKED_RANKING_SCHEMA: Final = pa.schema(
    [
        _field("benchmark", pa.string()),
        _field("candidate_id", pa.string()),
        _field("query_id", pa.string()),
        _field("query_code", pa.string()),
        _field("candidate_code", pa.string()),
        _field("rank", pa.int64()),
        _field("score", pa.float64()),
        _field("split", pa.string()),
    ]
)


# Arrow schemas are the canonical dtype contract.  Pandas' dtype strings vary
# with backend configuration; Arrow types are stable in Parquet metadata.
CORE_TABLE_SCHEMAS: Final[Mapping[str, pa.Schema]] = MappingProxyType(
    {
        "occupations": pa.schema(
            [
                _field("onet_soc_code", pa.string()),
                _field("base_soc_code", pa.string()),
                _field("soc_major_group", pa.string()),
                _field("canonical_title", pa.string()),
                _field("description", pa.string()),
                _field("job_zone", pa.int64(), nullable=True),
                _field("title_text", pa.string()),
                _field("work_text", pa.string()),
                _field("core_task_count", pa.int64()),
                _field("alias_count", pa.int64()),
                _field("structured_observed_count", pa.int64()),
            ]
        ),
        "occupation_market": pa.schema(
            [
                _field("onet_soc_code", pa.string()),
                _field("nem_code", pa.string()),
                _field("employment_2024", pa.int64(), nullable=True),
                _field("employment_2034", pa.int64(), nullable=True),
                _field("employment_change_2024_34", pa.int64(), nullable=True),
                _field("employment_change_pct", pa.float64(), nullable=True),
                _field("annual_openings_2024_34", pa.int64(), nullable=True),
                _field("median_annual_wage_2024", pa.int64(), nullable=True),
                _field("typical_education", pa.string(), nullable=True),
                _field("work_experience", pa.string(), nullable=True),
                _field("on_the_job_training", pa.string(), nullable=True),
            ]
        ),
        "skill_ratings": pa.schema(
            [
                _field("onet_soc_code", pa.string()),
                _field("feature_family", pa.string()),
                _field("element_id", pa.string()),
                _field("scale_id", pa.string()),
                _field("element_name", pa.string()),
                _field("scale_name", pa.string()),
                _field("raw_value", pa.float64(), nullable=True),
                _field("normalized_value", pa.float64(), nullable=True),
                _field("sample_size", pa.int64(), nullable=True),
                _field("standard_error", pa.float64(), nullable=True),
                _field("lower_ci_bound", pa.float64(), nullable=True),
                _field("upper_ci_bound", pa.float64(), nullable=True),
                _field("observed", pa.bool_()),
                _field("recommend_suppress", pa.bool_()),
                _field("not_relevant", pa.bool_()),
                _field("rating_date", pa.date32(), nullable=True),
                _field("domain_source", pa.string(), nullable=True),
            ]
        ),
        "clusters": pa.schema(
            [
                _field("onet_soc_code", pa.string()),
                _field("cluster_id", pa.int64()),
                _field("cluster_label", pa.string()),
                _field("pca_x", pa.float64()),
                _field("pca_y", pa.float64()),
            ]
        ),
        "neighbors": pa.schema(
            [
                _field("query_code", pa.string()),
                _field("destination_code", pa.string()),
                _field("rank", pa.int64()),
                _field("total_score", pa.float64()),
                _field("lexical_score", pa.float64()),
                _field("lsa_score", pa.float64()),
                _field("structured_score", pa.float64(), nullable=True),
                _field("development_distance", pa.float64(), nullable=True),
                _field("explanation", pa.string()),
            ]
        ),
        "search_catalog": pa.schema(
            [
                _field("row_index", pa.int64()),
                _field("onet_soc_code", pa.string()),
            ]
        ),
    }
)

CORE_FILE_FORMATS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "occupations": "parquet",
        "occupation_market": "parquet",
        "skill_ratings": "parquet",
        "clusters": "parquet",
        "neighbors": "parquet",
        "search_catalog": "parquet",
        "title_vectorizer": "joblib",
        "title_matrix": "scipy_npz",
        "smoke_queries": "json",
    }
)

LOCKED_LINEAGE_FILE_FORMATS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "locked_metrics": "json",
        "locked_rankings": "parquet",
        "development_decision": "json",
        "fold_manifest": "csv",
        "fold_digest": "json",
        "processed_manifest": "json",
        "feature_manifest": "json",
        "metric_contract": "json",
        "candidate_manifest": "json",
        "source_manifest": "json",
        "evaluation_universe": "json",
        "implementation_manifest": "json",
    }
)

EVIDENCE_REPORT_TYPES: Final = frozenset(
    {
        "title_benchmark",
        "transition_benchmark",
        "cluster_stability",
        "data_coverage",
        "limitations",
    }
)

BENCHMARK_CONTRACTS: Final[Mapping[str, tuple[str, frozenset[str], str]]] = MappingProxyType(
    {
        "title_benchmark": (
            "top1_accuracy",
            frozenset({"top1_accuracy", "recall_at_5", "mrr"}),
            "title_candidate_id",
        ),
        "transition_benchmark": (
            "ndcg_at_10",
            frozenset(
                {
                    "ndcg_at_10",
                    "primary_short_recall_at_5",
                    "primary_recall_at_10",
                    "ndcg_at_20",
                    "mrr_at_10",
                    "binary_primary_ndcg_at_10",
                }
            ),
            "transition_candidate_id",
        ),
    }
)

BENCHMARK_BASELINE_CANDIDATES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "title_benchmark": "title_word_char",
        "transition_benchmark": "transition_title_word_char",
    }
)

PRIMARY_KEYS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "occupations": ("onet_soc_code",),
        "occupation_market": ("onet_soc_code", "nem_code"),
        "skill_ratings": ("onet_soc_code", "feature_family", "element_id", "scale_id"),
        "clusters": ("onet_soc_code",),
        "neighbors": ("query_code", "rank"),
        "search_catalog": ("row_index",),
    }
)

SORT_KEYS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "occupations": ("onet_soc_code",),
        "occupation_market": ("onet_soc_code", "nem_code"),
        "skill_ratings": ("onet_soc_code", "feature_family", "element_id", "scale_id"),
        "clusters": ("onet_soc_code",),
        "neighbors": ("query_code", "rank"),
        "search_catalog": ("row_index",),
    }
)


class ArtifactError(RuntimeError):
    """Base class for a serving-artifact failure."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when no serving bundle is present."""


class ManifestValidationError(ArtifactError):
    """Raised when the manifest itself violates the release contract."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when bytes do not match the checksum-addressed manifest."""


class ArtifactSchemaError(ArtifactError):
    """Raised when a table, matrix, or fitted model violates its schema."""


class ArtifactSafetyError(ArtifactError):
    """Raised when a serving bundle contains unsafe paths or evaluation targets."""


@dataclass(frozen=True, slots=True)
class LoadedArtifactBundle:
    """A fully verified serving bundle.

    Arrow tables are immutable.  The sparse matrix has write-protected backing
    arrays, and the mappings are read-only proxies.
    """

    root: Path
    manifest: Mapping[str, Any]
    tables: Mapping[str, pa.Table]
    search_transformer: Any
    search_matrix: sparse.csr_matrix
    smoke_queries: tuple[tuple[str, str], ...]
    evidence_reports: Mapping[str, Mapping[str, Any]]
    locked_lineage: Mapping[str, Any]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrow_schema_manifest(schema: pa.Schema) -> list[dict[str, Any]]:
    """Serialize the exact ordered Arrow schema used in a manifest."""

    return [
        {"name": field.name, "dtype": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def parquet_file_manifest(logical_name: str, path: Path) -> dict[str, Any]:
    """Build the deterministic manifest entry for a canonical Parquet table."""

    if logical_name not in CORE_TABLE_SCHEMAS:
        raise ValueError(f"unknown core table: {logical_name}")
    parquet_file = pq.ParquetFile(path)
    return {
        "path": path.name,
        "format": "parquet",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": parquet_file.metadata.num_rows,
        "columns": arrow_schema_manifest(CORE_TABLE_SCHEMAS[logical_name]),
        "primary_key": list(PRIMARY_KEYS[logical_name]),
        "sort_by": list(SORT_KEYS[logical_name]),
    }


def _normal_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _is_forbidden_field(value: str) -> bool:
    normalized = _normal_field_name(value)
    return any(pattern.search(normalized) for pattern in FORBIDDEN_FIELD_PATTERNS)


def _reject_forbidden_fields(values: Iterable[str], *, context: str) -> None:
    offending = sorted({value for value in values if _is_forbidden_field(value)})
    if offending:
        raise ArtifactSafetyError(
            f"{context} contains evaluation-only field(s): {', '.join(offending)}"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ManifestValidationError(f"non-finite JSON number is prohibited: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(f"{path.name} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"invalid JSON in {path.name}: {exc.msg}") from exc
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read declared JSON artifact: {path.name}") from exc


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{context} must be an object")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ManifestValidationError(f"{context} missing key(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ManifestValidationError(f"{context} has unknown key(s): {', '.join(sorted(unknown))}")


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestValidationError(f"{context} must be a trimmed, non-empty string")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ManifestValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, context: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ManifestValidationError(f"{context} must be a {qualifier} integer")
    return value


def _candidate_spec_payload(candidate: CandidateSpec) -> dict[str, Any]:
    """Return the exact JSON-native candidate definition used by the grid manifest."""

    payload = asdict(candidate)
    payload["weights"] = [[name, float(weight)] for name, weight in candidate.weights]
    return payload


def fit_codes_digest(codes: Iterable[str]) -> str:
    """Hash one canonical, sorted, unique production-fit occupation code list."""

    normalized = tuple(str(code) for code in codes)
    if not normalized or normalized != tuple(sorted(set(normalized))):
        raise ArtifactSchemaError("fit codes must be a non-empty sorted unique sequence")
    encoded = json.dumps(list(normalized), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_candidate_spec_payload(value: Any, context: str) -> Mapping[str, Any]:
    spec = _expect_mapping(value, context)
    _expect_exact_keys(
        spec,
        required={
            "candidate_id",
            "task",
            "representation",
            "analyzer",
            "lsa_dimensions",
            "weights",
        },
        context=context,
    )
    _require_nonempty_string(spec["candidate_id"], f"{context}.candidate_id")
    if spec["task"] not in {"title", "transition"}:
        raise ManifestValidationError(f"{context}.task must be title or transition")
    _require_nonempty_string(spec["representation"], f"{context}.representation")
    if spec["analyzer"] is not None and spec["analyzer"] not in {"word", "char", "word_char"}:
        raise ManifestValidationError(f"{context}.analyzer is invalid")
    _require_positive_int(spec["lsa_dimensions"], f"{context}.lsa_dimensions", allow_zero=True)
    weights = spec["weights"]
    if not isinstance(weights, list) or not weights:
        raise ManifestValidationError(f"{context}.weights must be a non-empty list")
    seen: set[str] = set()
    total = 0.0
    for index, item in enumerate(weights):
        if not isinstance(item, list) or len(item) != 2:
            raise ManifestValidationError(f"{context}.weights[{index}] must be [name, weight]")
        name = _require_nonempty_string(item[0], f"{context}.weights[{index}][0]")
        if name not in {"lexical", "lsa", "structured"} or name in seen:
            raise ManifestValidationError(f"{context}.weights has an invalid/repeated component")
        seen.add(name)
        weight = item[1]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ManifestValidationError(f"{context}.weights[{index}][1] must be numeric")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ManifestValidationError(f"{context}.weights must be finite and positive")
        total += numeric
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ManifestValidationError(f"{context}.weights must sum to one")
    return spec


def _validate_manifest(manifest: Any) -> Mapping[str, Any]:
    manifest = _expect_mapping(manifest, "manifest")
    _expect_exact_keys(
        manifest,
        required={
            "schema_version",
            "artifact_version",
            "snapshot_date",
            "models",
            "provenance",
            "evidence",
            "files",
        },
        optional={"created_at_utc"},
        context="manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"unsupported manifest schema_version: {manifest['schema_version']!r}"
        )
    _require_nonempty_string(manifest["artifact_version"], "artifact_version")
    snapshot_date = _require_nonempty_string(manifest["snapshot_date"], "snapshot_date")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot_date) is None:
        raise ManifestValidationError("snapshot_date must use YYYY-MM-DD")
    try:
        date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise ManifestValidationError("snapshot_date is not a real calendar date") from exc
    if "created_at_utc" in manifest:
        created = _require_nonempty_string(manifest["created_at_utc"], "created_at_utc")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created) is None:
            raise ManifestValidationError("created_at_utc must be a second-precision UTC timestamp")

    models = _expect_mapping(manifest["models"], "models")
    semantic_model_fields = {
        "title_candidate_spec",
        "transition_candidate_spec",
        "fit_phase",
        "fit_codes_sha256",
        "fit_code_count",
        "fit_fold_manifest_sha256",
    }
    _expect_exact_keys(
        models,
        required={
            "title_candidate_id",
            "transition_candidate_id",
            "title_feature_count",
            "title_matrix_normalized",
            "stored_transition_limit",
            "neighbor_display_max",
            "transition_components",
        },
        optional=semantic_model_fields,
        context="models",
    )
    _require_nonempty_string(models["title_candidate_id"], "models.title_candidate_id")
    _require_nonempty_string(models["transition_candidate_id"], "models.transition_candidate_id")
    _require_positive_int(models["title_feature_count"], "models.title_feature_count")
    if models["title_matrix_normalized"] is not True:
        raise ManifestValidationError("models.title_matrix_normalized must be true")
    stored_transition_limit = _require_positive_int(
        models["stored_transition_limit"], "models.stored_transition_limit"
    )
    if stored_transition_limit != STORED_TRANSITION_LIMIT:
        raise ManifestValidationError(
            f"models.stored_transition_limit must be the preregistered top-{STORED_TRANSITION_LIMIT}"
        )
    neighbor_display_max = _require_positive_int(
        models["neighbor_display_max"], "models.neighbor_display_max"
    )
    if neighbor_display_max != NEIGHBOR_DISPLAY_MAX:
        raise ManifestValidationError(f"models.neighbor_display_max must be {NEIGHBOR_DISPLAY_MAX}")
    components = models["transition_components"]
    if components != ["lexical", "lsa", "structured"]:
        raise ManifestValidationError(
            "models.transition_components must be ['lexical', 'lsa', 'structured']"
        )
    present_semantic_fields = semantic_model_fields & models.keys()
    if present_semantic_fields and present_semantic_fields != semantic_model_fields:
        missing = ", ".join(sorted(semantic_model_fields - present_semantic_fields))
        raise ManifestValidationError(f"models semantic binding is incomplete: {missing}")
    if present_semantic_fields:
        title_spec = _validate_candidate_spec_payload(
            models["title_candidate_spec"], "models.title_candidate_spec"
        )
        transition_spec = _validate_candidate_spec_payload(
            models["transition_candidate_spec"], "models.transition_candidate_spec"
        )
        if title_spec["task"] != "title" or transition_spec["task"] != "transition":
            raise ManifestValidationError("models candidate specs are assigned to the wrong tasks")
        if (
            title_spec["candidate_id"] != models["title_candidate_id"]
            or transition_spec["candidate_id"] != models["transition_candidate_id"]
        ):
            raise ManifestValidationError("models candidate specs differ from their candidate IDs")
        if models["fit_phase"] != "locked_test":
            raise ManifestValidationError("models.fit_phase must be 'locked_test'")
        _require_sha256(models["fit_codes_sha256"], "models.fit_codes_sha256")
        _require_positive_int(models["fit_code_count"], "models.fit_code_count")
        _require_sha256(models["fit_fold_manifest_sha256"], "models.fit_fold_manifest_sha256")

    provenance = _expect_mapping(manifest["provenance"], "provenance")
    _expect_exact_keys(
        provenance,
        required={
            "repository_commit",
            "project_config_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "locked_rankings_sha256",
            "locked_metrics_sha256",
            "environment_lock_sha256",
            "processed_manifest_sha256",
            "feature_manifest_sha256",
            "metric_contract_sha256",
            "candidate_manifest_sha256",
            "evaluation_universe_sha256",
            "implementation_manifest_sha256",
            "title_candidate_grid_sha256",
            "transition_candidate_grid_sha256",
            "source_hashes",
        },
        context="provenance",
    )
    commit = provenance["repository_commit"]
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ManifestValidationError("provenance.repository_commit must be a full Git object ID")
    for field_name in (
        "project_config_sha256",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "locked_rankings_sha256",
        "locked_metrics_sha256",
        "environment_lock_sha256",
        "processed_manifest_sha256",
        "feature_manifest_sha256",
        "metric_contract_sha256",
        "candidate_manifest_sha256",
        "evaluation_universe_sha256",
        "implementation_manifest_sha256",
        "title_candidate_grid_sha256",
        "transition_candidate_grid_sha256",
    ):
        _require_sha256(provenance[field_name], f"provenance.{field_name}")
    source_hashes = _expect_mapping(provenance["source_hashes"], "provenance.source_hashes")
    if not source_hashes:
        raise ManifestValidationError("provenance.source_hashes cannot be empty")
    for source_name, digest in source_hashes.items():
        _require_nonempty_string(source_name, "source hash name")
        _require_sha256(digest, f"provenance.source_hashes.{source_name}")

    evidence = _expect_mapping(manifest["evidence"], "evidence")
    _expect_exact_keys(evidence, required={"status", "report_files"}, context="evidence")
    if evidence["status"] not in {"not_generated", "locked"}:
        raise ManifestValidationError("evidence.status must be 'not_generated' or 'locked'")
    report_files = _expect_mapping(evidence["report_files"], "evidence.report_files")
    unknown_report_types = report_files.keys() - EVIDENCE_REPORT_TYPES
    if unknown_report_types:
        raise ManifestValidationError(
            "evidence.report_files has unknown report type(s): "
            + ", ".join(sorted(unknown_report_types))
        )
    if not all(isinstance(item, str) for item in report_files.values()):
        raise ManifestValidationError("evidence.report_files values must be logical file names")
    if len(report_files.values()) != len(set(report_files.values())):
        raise ManifestValidationError("evidence.report_files contains duplicate logical names")
    if evidence["status"] == "not_generated" and report_files:
        raise ManifestValidationError("not_generated evidence cannot bind report files")
    if evidence["status"] == "locked" and set(report_files) != EVIDENCE_REPORT_TYPES:
        missing = EVIDENCE_REPORT_TYPES - report_files.keys()
        raise ManifestValidationError(
            "locked evidence is missing report type(s): " + ", ".join(sorted(missing))
        )
    if evidence["status"] == "locked" and present_semantic_fields != semantic_model_fields:
        raise ManifestValidationError("locked evidence requires complete model semantic bindings")

    files = _expect_mapping(manifest["files"], "files")
    missing_files = CORE_FILE_FORMATS.keys() - files.keys()
    if missing_files:
        raise ManifestValidationError(
            f"files missing required artifact(s): {', '.join(sorted(missing_files))}"
        )
    _reject_forbidden_fields(files.keys(), context="logical artifact names")
    lineage_names = set(LOCKED_LINEAGE_FILE_FORMATS)
    present_lineage = lineage_names & files.keys()
    if evidence["status"] == "locked" and present_lineage != lineage_names:
        missing = ", ".join(sorted(lineage_names - present_lineage))
        raise ManifestValidationError(f"locked evidence is missing lineage artifact(s): {missing}")
    if evidence["status"] == "not_generated" and present_lineage:
        raise ManifestValidationError("not_generated evidence cannot bind locked release lineage")
    for report_name in report_files.values():
        if report_name not in files:
            raise ManifestValidationError(
                f"evidence report {report_name!r} is not checksum-bound in files"
            )
        if report_name in CORE_FILE_FORMATS or report_name in LOCKED_LINEAGE_FILE_FORMATS:
            raise ManifestValidationError(
                f"reserved artifact {report_name!r} cannot be an evidence report"
            )
    extra_names = files.keys() - CORE_FILE_FORMATS.keys() - lineage_names
    undeclared_extras = extra_names - set(report_files.values())
    if undeclared_extras:
        raise ManifestValidationError(
            "non-core files must be evidence reports: " + ", ".join(sorted(undeclared_extras))
        )

    seen_paths: set[str] = set()
    for logical_name, raw_entry in files.items():
        entry = _expect_mapping(raw_entry, f"files.{logical_name}")
        expected_format = CORE_FILE_FORMATS.get(logical_name)
        if expected_format is None:
            expected_format = LOCKED_LINEAGE_FILE_FORMATS.get(logical_name)
        if expected_format == "parquet" and logical_name in CORE_TABLE_SCHEMAS:
            required = {
                "path",
                "format",
                "bytes",
                "sha256",
                "rows",
                "columns",
                "primary_key",
                "sort_by",
            }
            optional: set[str] = set()
        elif expected_format == "scipy_npz":
            required = {"path", "format", "bytes", "sha256", "shape", "dtype"}
            optional = set()
        else:
            required = {"path", "format", "bytes", "sha256"}
            optional = set()
        if expected_format is None:
            required = {"path", "format", "bytes", "sha256"}
            optional = {"schema_version"}
        _expect_exact_keys(
            entry,
            required=required,
            optional=optional,
            context=f"files.{logical_name}",
        )
        path_text = _require_nonempty_string(entry["path"], f"files.{logical_name}.path")
        posix_path = PurePosixPath(path_text)
        if path_text != posix_path.as_posix() or path_text == MANIFEST_FILENAME:
            raise ManifestValidationError(
                f"files.{logical_name}.path must be canonical and cannot name manifest.json"
            )
        if path_text in seen_paths:
            raise ManifestValidationError(f"multiple logical files reference {path_text!r}")
        seen_paths.add(path_text)
        if expected_format is not None and entry["format"] != expected_format:
            raise ManifestValidationError(
                f"files.{logical_name}.format must be {expected_format!r}"
            )
        if expected_format is None and entry["format"] != "json":
            raise ManifestValidationError("evidence reports must be JSON")
        if expected_format is None and entry.get("schema_version", 1) != 1:
            raise ManifestValidationError(
                f"files.{logical_name}.schema_version must be 1 when declared"
            )
        required_suffix = {
            "parquet": ".parquet",
            "joblib": ".joblib",
            "scipy_npz": ".npz",
            "json": ".json",
            "csv": ".csv",
        }[entry["format"]]
        if not path_text.endswith(required_suffix):
            raise ManifestValidationError(
                f"files.{logical_name}.path must end with {required_suffix!r}"
            )
        _require_positive_int(entry["bytes"], f"files.{logical_name}.bytes")
        _require_sha256(entry["sha256"], f"files.{logical_name}.sha256")
        if expected_format == "parquet" and logical_name in CORE_TABLE_SCHEMAS:
            _require_positive_int(entry["rows"], f"files.{logical_name}.rows", allow_zero=True)
            expected_schema = CORE_TABLE_SCHEMAS[logical_name]
            if entry["columns"] != arrow_schema_manifest(expected_schema):
                raise ManifestValidationError(
                    f"files.{logical_name}.columns does not match the canonical ordered schema"
                )
            if entry["primary_key"] != list(PRIMARY_KEYS[logical_name]):
                raise ManifestValidationError(
                    f"files.{logical_name}.primary_key does not match the contract"
                )
            if entry["sort_by"] != list(SORT_KEYS[logical_name]):
                raise ManifestValidationError(
                    f"files.{logical_name}.sort_by does not match the contract"
                )
        elif expected_format == "scipy_npz":
            shape = entry["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 1
                    for item in shape
                )
            ):
                raise ManifestValidationError(
                    "files.title_matrix.shape must contain two positive integers"
                )
            if entry["dtype"] != "float64":
                raise ManifestValidationError("files.title_matrix.dtype must be 'float64'")
    return manifest


def _safe_artifact_path(root: Path, path_text: str) -> Path:
    if "\\" in path_text:
        raise ArtifactSafetyError(f"artifact path uses a backslash: {path_text!r}")
    posix_path = PurePosixPath(path_text)
    if (
        posix_path.is_absolute()
        or not posix_path.parts
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ArtifactSafetyError(f"artifact path is not a safe relative path: {path_text!r}")
    candidate = root.joinpath(*posix_path.parts)
    current = root
    for part in posix_path.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactSafetyError(f"symlinks are prohibited in artifact paths: {path_text!r}")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ArtifactSafetyError(f"artifact path escapes its bundle: {path_text!r}") from exc
    if not candidate.is_file():
        raise ArtifactIntegrityError(
            f"declared artifact is missing or not a regular file: {path_text}"
        )
    return candidate


def _verify_files(root: Path, manifest: Mapping[str, Any]) -> Mapping[str, Path]:
    if root.is_symlink():
        raise ArtifactSafetyError("the serving bundle root cannot be a symlink")
    paths: dict[str, Path] = {}
    declared_relative_paths = {MANIFEST_FILENAME}
    for logical_name, entry in manifest["files"].items():
        path = _safe_artifact_path(root, entry["path"])
        declared_relative_paths.add(path.relative_to(root).as_posix())
        actual_size = path.stat().st_size
        if actual_size != entry["bytes"]:
            raise ArtifactIntegrityError(
                f"size mismatch for {logical_name}: expected {entry['bytes']}, got {actual_size}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != entry["sha256"]:
            raise ArtifactIntegrityError(f"SHA-256 mismatch for {logical_name}")
        paths[logical_name] = path

    actual_relative_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise ArtifactSafetyError(
                    f"undeclared symlink in artifact bundle: {child.relative_to(root)}"
                )
        for name in file_names:
            actual_relative_paths.add((directory_path / name).relative_to(root).as_posix())
    undeclared = actual_relative_paths - declared_relative_paths
    if undeclared:
        raise ArtifactIntegrityError(
            "bundle contains file(s) not declared in the manifest: " + ", ".join(sorted(undeclared))
        )
    return MappingProxyType(paths)


def _resolve_project_config_path(root: Path, project_config_path: str | Path | None = None) -> Path:
    """Resolve the literal checked-in configuration used for provenance."""

    if project_config_path is not None:
        return Path(project_config_path)

    configured_root = os.environ.get("JOB_MARKET_PROJECT_ROOT")
    if configured_root:
        project_root = Path(configured_root)
    elif root.parent.name == "artifacts":
        project_root = root.parent.parent
    else:
        project_root = root.parent
    return project_root / "configs" / "project.toml"


def _verify_project_config(
    root: Path,
    manifest: Mapping[str, Any],
    project_config_path: str | Path | None = None,
) -> Path:
    """Bind provenance to the literal checked-in configuration and environment lock."""

    config_path = _resolve_project_config_path(root, project_config_path)
    if not config_path.is_file() or config_path.is_symlink():
        raise ArtifactIntegrityError(
            f"project configuration required for provenance is missing: {config_path}"
        )
    expected = manifest["provenance"]["project_config_sha256"]
    if sha256_file(config_path) != expected:
        raise ArtifactIntegrityError(
            "provenance.project_config_sha256 does not match configs/project.toml bytes"
        )
    requirements_path = config_path.parent.parent / "requirements.lock"
    if not requirements_path.is_file() or requirements_path.is_symlink():
        raise ArtifactIntegrityError(
            "requirements.lock required for provenance is missing or unsafe"
        )
    if sha256_file(requirements_path) != manifest["provenance"]["environment_lock_sha256"]:
        raise ArtifactIntegrityError(
            "provenance.environment_lock_sha256 does not match requirements.lock bytes"
        )
    return config_path


def _read_core_table(logical_name: str, path: Path, entry: Mapping[str, Any]) -> pa.Table:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise ArtifactSchemaError(f"{logical_name} is not a readable Parquet table") from exc
    schema = parquet_file.schema_arrow
    expected_schema = CORE_TABLE_SCHEMAS[logical_name]
    if not schema.equals(expected_schema, check_metadata=True):
        raise ArtifactSchemaError(f"{logical_name} has schema {schema}, expected {expected_schema}")
    if parquet_file.metadata.num_rows != entry["rows"]:
        raise ArtifactSchemaError(f"{logical_name} row count does not match its manifest")
    try:
        table = parquet_file.read().combine_chunks()
    except Exception as exc:
        raise ArtifactSchemaError(f"{logical_name} cannot be decoded as Parquet") from exc
    if table.num_rows != entry["rows"]:
        raise ArtifactSchemaError(f"{logical_name} decoded row count does not match its manifest")
    _validate_table_contract(logical_name, table)
    return table


def _column_values(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _validate_primary_key(logical_name: str, table: pa.Table) -> None:
    key_names = PRIMARY_KEYS[logical_name]
    keys = list(zip(*(_column_values(table, name) for name in key_names), strict=True))
    if any(any(value is None for value in key) for key in keys):
        raise ArtifactSchemaError(f"{logical_name} primary key contains null values")
    if len(keys) != len(set(keys)):
        raise ArtifactSchemaError(f"{logical_name} primary key is not unique")


def _validate_sort_order(logical_name: str, table: pa.Table) -> None:
    sort_names = SORT_KEYS[logical_name]
    rows = list(zip(*(_column_values(table, name) for name in sort_names), strict=True))
    if any(any(value is None for value in row) for row in rows):
        raise ArtifactSchemaError(f"{logical_name} sort key contains null values")
    if rows != sorted(rows):
        raise ArtifactSchemaError(f"{logical_name} is not sorted by {', '.join(sort_names)}")


def _require_trimmed_nonempty_column(table: pa.Table, name: str, logical_name: str) -> None:
    values = _column_values(table, name)
    if any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in values
    ):
        raise ArtifactSchemaError(f"{logical_name}.{name} contains blank or untrimmed values")


def _require_nullable_trimmed_column(table: pa.Table, name: str, logical_name: str) -> None:
    values = _column_values(table, name)
    if any(
        value is not None
        and (not isinstance(value, str) or not value.strip() or value != value.strip())
        for value in values
    ):
        raise ArtifactSchemaError(
            f"{logical_name}.{name} contains blank or untrimmed non-null values"
        )


def _require_finite_column(
    table: pa.Table, name: str, logical_name: str, *, nonnegative: bool = False
) -> None:
    for value in _column_values(table, name):
        if value is None:
            continue
        if not math.isfinite(float(value)) or (nonnegative and value < 0):
            qualifier = "finite non-negative" if nonnegative else "finite"
            raise ArtifactSchemaError(f"{logical_name}.{name} must contain only {qualifier} values")


def _validate_table_contract(logical_name: str, table: pa.Table) -> None:
    _reject_forbidden_fields(table.column_names, context=f"{logical_name} columns")
    for field in CORE_TABLE_SCHEMAS[logical_name]:
        if not field.nullable and table.column(field.name).null_count:
            raise ArtifactSchemaError(
                f"{logical_name}.{field.name} contains nulls but is non-nullable"
            )
    _validate_primary_key(logical_name, table)
    _validate_sort_order(logical_name, table)

    string_fields = [field.name for field in table.schema if pa.types.is_string(field.type)]
    required_strings = {
        "occupations": {
            "onet_soc_code",
            "base_soc_code",
            "soc_major_group",
            "canonical_title",
            "description",
            "title_text",
            "work_text",
        },
        "occupation_market": {"onet_soc_code", "nem_code"},
        "skill_ratings": {
            "onet_soc_code",
            "feature_family",
            "element_id",
            "scale_id",
            "element_name",
            "scale_name",
        },
        "clusters": {"onet_soc_code", "cluster_label"},
        "neighbors": {"query_code", "destination_code", "explanation"},
        "search_catalog": {"onet_soc_code"},
    }[logical_name]
    for field_name in string_fields:
        if field_name in required_strings:
            _require_trimmed_nonempty_column(table, field_name, logical_name)

    if logical_name == "occupations":
        codes = _column_values(table, "onet_soc_code")
        base_codes = _column_values(table, "base_soc_code")
        major_groups = _column_values(table, "soc_major_group")
        for code, base_code, major_group in zip(codes, base_codes, major_groups, strict=True):
            if ONET_SOC_RE.fullmatch(code) is None:
                raise ArtifactSchemaError("occupations.onet_soc_code has an invalid format")
            if BASE_SOC_RE.fullmatch(base_code) is None or base_code != code.split(".", 1)[0]:
                raise ArtifactSchemaError(
                    "occupations.base_soc_code must match its detailed O*NET-SOC code"
                )
            if SOC_MAJOR_RE.fullmatch(major_group) is None or major_group != code[:2]:
                raise ArtifactSchemaError(
                    "occupations.soc_major_group must match its O*NET-SOC code"
                )
        for name in ("core_task_count", "alias_count", "structured_observed_count"):
            if any(value < 0 for value in _column_values(table, name)):
                raise ArtifactSchemaError(f"occupations.{name} cannot be negative")
        zones = _column_values(table, "job_zone")
        if any(value is not None and value not in {2, 3, 4, 5} for value in zones):
            raise ArtifactSchemaError(
                "occupations.job_zone must be null or an official O*NET 30.3 stored label: "
                "2 (combined Job Zone 1-2), 3, 4, or 5"
            )
    elif logical_name == "occupation_market":
        if any(BASE_SOC_RE.fullmatch(value) is None for value in _column_values(table, "nem_code")):
            raise ArtifactSchemaError("occupation_market.nem_code has an invalid format")
        for name in ("typical_education", "work_experience", "on_the_job_training"):
            _require_nullable_trimmed_column(table, name, logical_name)
        for name in (
            "employment_2024",
            "employment_2034",
            "annual_openings_2024_34",
            "median_annual_wage_2024",
        ):
            values = _column_values(table, name)
            if any(value is not None and value < 0 for value in values):
                raise ArtifactSchemaError(f"occupation_market.{name} cannot be negative")
        _require_finite_column(table, "employment_change_pct", logical_name)
    elif logical_name == "skill_ratings":
        _require_nullable_trimmed_column(table, "domain_source", logical_name)
        for name in (
            "raw_value",
            "normalized_value",
            "standard_error",
            "lower_ci_bound",
            "upper_ci_bound",
        ):
            _require_finite_column(table, name, logical_name)
        if any(value is not None and value < 0 for value in _column_values(table, "sample_size")):
            raise ArtifactSchemaError("skill_ratings.sample_size cannot be negative")
        if any(
            value is not None and value < 0 for value in _column_values(table, "standard_error")
        ):
            raise ArtifactSchemaError("skill_ratings.standard_error cannot be negative")
        lower_bounds = _column_values(table, "lower_ci_bound")
        upper_bounds = _column_values(table, "upper_ci_bound")
        if any(
            (lower is None) != (upper is None)
            for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
        ):
            raise ArtifactSchemaError(
                "skill_ratings confidence interval bounds must be present or missing together"
            )
        if any(
            lower is not None and upper is not None and lower > upper
            for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
        ):
            raise ArtifactSchemaError("skill_ratings confidence interval bounds are reversed")
        observed = _column_values(table, "observed")
        suppressed = _column_values(table, "recommend_suppress")
        not_relevant = _column_values(table, "not_relevant")
        normalized = _column_values(table, "normalized_value")
        if any(value is not None and not 0 <= value <= 1 for value in normalized):
            raise ArtifactSchemaError("skill_ratings.normalized_value must lie in [0, 1]")
        raw_values = _column_values(table, "raw_value")
        for index, values in enumerate(
            zip(
                observed,
                suppressed,
                not_relevant,
                raw_values,
                normalized,
                lower_bounds,
                upper_bounds,
                strict=True,
            )
        ):
            (
                is_observed,
                is_suppressed,
                is_not_relevant,
                raw_value,
                normalized_value,
                lower,
                upper,
            ) = values
            if (is_suppressed or is_not_relevant) and is_observed:
                raise ArtifactSchemaError(
                    f"skill_ratings row {index} marks a suppressed/not-relevant value observed"
                )
            if (not is_observed) != (normalized_value is None):
                raise ArtifactSchemaError(
                    f"skill_ratings row {index} must have normalized_value iff observed is true"
                )
            if is_observed and raw_value is None:
                raise ArtifactSchemaError(
                    f"skill_ratings row {index} must retain raw_value when observed"
                )
            if (
                raw_value is not None
                and lower is not None
                and upper is not None
                and not lower - 1e-12 <= raw_value <= upper + 1e-12
            ):
                raise ArtifactSchemaError(
                    f"skill_ratings row {index} raw_value lies outside its confidence interval"
                )
    elif logical_name == "clusters":
        for name in ("pca_x", "pca_y"):
            _require_finite_column(table, name, logical_name)
        cluster_ids = _column_values(table, "cluster_id")
        if any(value < 0 for value in cluster_ids):
            raise ArtifactSchemaError("clusters.cluster_id cannot be negative")
        if set(cluster_ids) != set(range(len(set(cluster_ids)))):
            raise ArtifactSchemaError("clusters.cluster_id must be contiguous and zero-based")
    elif logical_name == "neighbors":
        for name in (
            "total_score",
            "lexical_score",
            "lsa_score",
            "structured_score",
        ):
            _require_finite_column(table, name, logical_name)
            if any(
                value is not None and not -1 - 1e-6 <= value <= 1 + 1e-6
                for value in _column_values(table, name)
            ):
                raise ArtifactSchemaError(f"neighbors.{name} must lie in [-1, 1]")
        _require_finite_column(table, "development_distance", logical_name, nonnegative=True)
        if any(
            value is not None and value > 1 + 1e-6
            for value in _column_values(table, "development_distance")
        ):
            raise ArtifactSchemaError("neighbors.development_distance must lie in [0, 1]")
        queries = _column_values(table, "query_code")
        destinations = _column_values(table, "destination_code")
        if any(
            query == destination for query, destination in zip(queries, destinations, strict=True)
        ):
            raise ArtifactSchemaError("neighbors contains a self-match")
        structured = _column_values(table, "structured_score")
        distances = _column_values(table, "development_distance")
        if any(
            (score is None) != (distance is None)
            for score, distance in zip(structured, distances, strict=True)
        ):
            raise ArtifactSchemaError(
                "neighbors structured_score and development_distance must be missing together"
            )


def _validate_foreign_keys(tables: Mapping[str, pa.Table], neighbor_limit: int) -> None:
    occupation_codes = set(_column_values(tables["occupations"], "onet_soc_code"))
    for logical_name, code_columns in {
        "occupation_market": ("onet_soc_code",),
        "skill_ratings": ("onet_soc_code",),
        "clusters": ("onet_soc_code",),
        "neighbors": ("query_code", "destination_code"),
        "search_catalog": ("onet_soc_code",),
    }.items():
        for column_name in code_columns:
            unknown = set(_column_values(tables[logical_name], column_name)) - occupation_codes
            if unknown:
                sample = ", ".join(sorted(unknown)[:3])
                raise ArtifactSchemaError(
                    f"{logical_name}.{column_name} references unknown occupation code(s): {sample}"
                )

    cluster_codes = _column_values(tables["clusters"], "onet_soc_code")
    if set(cluster_codes) != occupation_codes:
        raise ArtifactSchemaError("clusters must contain exactly one row for every occupation")
    cluster_labels: dict[int, str] = {}
    for cluster_id, label in zip(
        _column_values(tables["clusters"], "cluster_id"),
        _column_values(tables["clusters"], "cluster_label"),
        strict=True,
    ):
        previous = cluster_labels.setdefault(cluster_id, label)
        if previous != label:
            raise ArtifactSchemaError("each cluster_id must have one deterministic label")

    market_fields = [
        field.name
        for field in CORE_TABLE_SCHEMAS["occupation_market"]
        if field.name not in {"onet_soc_code", "nem_code"}
    ]
    by_nem: dict[str, tuple[Any, ...]] = {}
    market_rows = tables["occupation_market"].to_pylist()
    for row in market_rows:
        values = tuple(row[name] for name in market_fields)
        previous = by_nem.setdefault(row["nem_code"], values)
        if previous != values:
            raise ArtifactSchemaError(
                "shared NEM codes must retain identical unsummed BLS values and requirements"
            )

    catalog_codes = _column_values(tables["search_catalog"], "onet_soc_code")
    row_indices = _column_values(tables["search_catalog"], "row_index")
    if row_indices != list(range(len(row_indices))):
        raise ArtifactSchemaError("search_catalog.row_index must be contiguous and zero-based")
    if catalog_codes != sorted(occupation_codes):
        raise ArtifactSchemaError(
            "search_catalog must contain every occupation exactly once in code order"
        )

    query_codes = _column_values(tables["neighbors"], "query_code")
    ranks = _column_values(tables["neighbors"], "rank")
    destinations = _column_values(tables["neighbors"], "destination_code")
    totals = _column_values(tables["neighbors"], "total_score")
    by_query: dict[str, list[tuple[int, str, float]]] = {}
    for query, rank, destination, total in zip(
        query_codes, ranks, destinations, totals, strict=True
    ):
        by_query.setdefault(query, []).append((rank, destination, total))
    if set(by_query) != occupation_codes:
        raise ArtifactSchemaError("neighbors must contain a ranked list for every occupation")
    for query, ranked in by_query.items():
        expected_count = min(neighbor_limit, len(occupation_codes) - 1)
        if len(ranked) != expected_count:
            raise ArtifactSchemaError(
                f"neighbors for {query} must contain the complete stored top-{expected_count}"
            )
        if [rank for rank, _, _ in ranked] != list(range(1, len(ranked) + 1)):
            raise ArtifactSchemaError(f"neighbors ranks for {query} must be contiguous from one")
        destination_codes = [destination for _, destination, _ in ranked]
        if len(destination_codes) != len(set(destination_codes)):
            raise ArtifactSchemaError(f"neighbors for {query} contain duplicate destinations")
        expected_order = sorted(ranked, key=lambda item: (-item[2], item[1]))
        if ranked != expected_order:
            raise ArtifactSchemaError(
                f"neighbors for {query} must be score-descending with code-ascending ties"
            )


def _validate_transition_score_semantics(
    neighbors: pa.Table,
    candidate_spec: Mapping[str, Any],
) -> None:
    """Recompute every stored score and rank under the selected missing-aware weights."""

    weights = {name: float(weight) for name, weight in candidate_spec["weights"]}
    expanded = {
        component: weights.get(component, 0.0) for component in ("lexical", "lsa", "structured")
    }
    if expanded["lexical"] <= 0 and expanded["lsa"] <= 0:
        raise ArtifactSchemaError("transition serving candidate lacks an always-finite text weight")
    by_query: dict[str, list[tuple[int, str, float]]] = {}
    for row in neighbors.to_pylist():
        numerator = 0.0
        denominator = 0.0
        for component in ("lexical", "lsa", "structured"):
            weight = expanded[component]
            value = row[f"{component}_score"]
            if weight > 0 and value is not None:
                numerator += weight * float(value)
                denominator += weight
        if denominator <= 0:
            raise ArtifactSchemaError("transition score has no positively weighted observation")
        expected_score = numerator / denominator
        if not math.isclose(
            float(row["total_score"]),
            expected_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ArtifactIntegrityError(
                "neighbors.total_score differs from the selected missing-aware candidate weights"
            )
        by_query.setdefault(str(row["query_code"]), []).append(
            (int(row["rank"]), str(row["destination_code"]), expected_score)
        )
    for query_code, ranked in by_query.items():
        recomputed = sorted(ranked, key=lambda item: (-item[2], item[1]))
        expected = [
            (rank, destination) for rank, (_, destination, _) in enumerate(recomputed, start=1)
        ]
        observed = [(rank, destination) for rank, destination, _ in ranked]
        if observed != expected:
            raise ArtifactIntegrityError(
                f"neighbors ranks for {query_code} differ from recomputed candidate scores"
            )


def _validate_sparse_matrix(
    path: Path,
    entry: Mapping[str, Any],
    *,
    row_count: int,
    feature_count: int,
    normalized: bool,
) -> sparse.csr_matrix:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item.filename for item in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactSchemaError("title_matrix is not a readable NPZ archive") from exc
    expected_members = {"data.npy", "format.npy", "indices.npy", "indptr.npy", "shape.npy"}
    if len(members) != len(set(members)) or set(members) != expected_members:
        raise ArtifactSafetyError(
            "title_matrix NPZ must contain only the canonical CSR array members"
        )
    try:
        matrix = sparse.load_npz(path)
    except Exception as exc:
        raise ArtifactSchemaError("title_matrix is not a readable SciPy sparse matrix") from exc
    if not sparse.isspmatrix_csr(matrix):
        raise ArtifactSchemaError("title_matrix must be CSR")
    if list(matrix.shape) != entry["shape"]:
        raise ArtifactSchemaError("title_matrix shape does not match its manifest")
    if matrix.shape != (row_count, feature_count):
        raise ArtifactSchemaError(
            "title_matrix shape does not match search_catalog rows/title_feature_count"
        )
    if matrix.dtype != np.dtype(entry["dtype"]):
        raise ArtifactSchemaError("title_matrix dtype does not match its manifest")
    if not matrix.has_canonical_format:
        raise ArtifactSchemaError("title_matrix must use canonical sorted CSR indices")
    if not np.isfinite(matrix.data).all():
        raise ArtifactSchemaError("title_matrix contains non-finite values")
    if normalized:
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        if not np.allclose(norms, 1.0, rtol=1e-8, atol=1e-10):
            raise ArtifactSchemaError("every title_matrix row must have unit L2 norm")
    matrix.data.flags.writeable = False
    matrix.indices.flags.writeable = False
    matrix.indptr.flags.writeable = False
    return matrix


def _model_field_names(estimator: Any) -> set[str]:
    """Collect estimator metadata names without treating vocabulary tokens as fields."""

    result: set[str] = set()
    visited: set[int] = set()

    def visit(value: Any, *, depth: int) -> None:
        if depth > 8 or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Mapping):
            result.update(str(name) for name in value)
            for item in value.values():
                visit(item, depth=depth + 1)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item, depth=depth + 1)
            return
        feature_names = getattr(value, "feature_names_in_", None)
        if feature_names is not None:
            result.update(str(item) for item in np.asarray(feature_names).ravel())
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            result.update(str(name) for name in attributes)
            for name, item in attributes.items():
                # Learned vocabularies contain ordinary searchable text rather
                # than metadata field names; scanning their tokens would make
                # benign phrases look like evaluation schema leakage.
                if _normal_field_name(str(name)) not in {"vocabulary", "stop_words"}:
                    visit(item, depth=depth + 1)
        for attribute in ("steps", "transformer_list", "transformers_"):
            children = getattr(value, attribute, None)
            if children is None:
                continue
            for child in children:
                if isinstance(child, tuple) and child:
                    result.add(str(child[0]))
                    for item in child[1:]:
                        if not (isinstance(item, str) and item in {"drop", "passthrough"}):
                            visit(item, depth=depth + 1)
                else:
                    visit(child, depth=depth + 1)

    visit(estimator, depth=0)
    return result


def _validate_transformer(
    transformer: Any,
    *,
    feature_count: int,
    smoke_queries: tuple[tuple[str, str], ...],
    search_matrix: sparse.csr_matrix,
) -> None:
    if not callable(getattr(transformer, "transform", None)):
        raise ArtifactSchemaError("title_vectorizer must expose a fitted transform method")
    _reject_forbidden_fields(
        _model_field_names(transformer), context="title model feature metadata"
    )
    query_texts = [query for _, query in smoke_queries]
    try:
        output = transformer.transform(query_texts)
    except Exception as exc:
        raise ArtifactSchemaError(
            "title_vectorizer is not fitted or cannot transform text"
        ) from exc
    if sparse.issparse(output):
        output_shape = output.shape
        values = output.data
    else:
        output_array = np.asarray(output)
        output_shape = output_array.shape
        values = output_array
    if output_shape != (len(query_texts), feature_count):
        raise ArtifactSchemaError(
            "title_vectorizer output does not match smoke-query rows/title_feature_count"
        )
    try:
        finite = bool(np.isfinite(values).all())
    except (TypeError, ValueError) as exc:
        raise ArtifactSchemaError("title_vectorizer must produce numeric values") from exc
    if not finite:
        raise ArtifactSchemaError("title_vectorizer produced non-finite values")
    query_matrix = sparse.csr_matrix(output, dtype=np.float64)
    norms = np.sqrt(np.asarray(query_matrix.multiply(query_matrix).sum(axis=1)).ravel())
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ArtifactSchemaError("every target-free smoke query must produce a non-zero vector")
    query_matrix = sparse.diags(1.0 / norms) @ query_matrix
    raw_scores = search_matrix @ query_matrix.T
    scores = raw_scores.toarray() if sparse.issparse(raw_scores) else np.asarray(raw_scores)
    if not np.isfinite(scores).all() or np.any(np.max(scores, axis=0) <= 0):
        raise ArtifactSchemaError(
            "every target-free smoke query must produce a positive catalog similarity"
        )


def _validate_tfidf_semantics(
    transformer: Any,
    *,
    analyzer: str,
    config: ProjectConfig,
    context: str,
) -> None:
    if type(transformer) is not TfidfIndex:
        raise ArtifactSchemaError(f"{context} must be an exact TfidfIndex")
    expected_word = tuple(int(value) for value in config.raw["candidates"]["title_word_ngrams"])
    expected_char = tuple(
        int(value) for value in config.raw["candidates"]["title_character_ngrams"]
    )
    expected_min_df = int(config.raw["candidates"]["minimum_document_frequency"])
    if (
        transformer.analyzer != analyzer
        or tuple(transformer.word_ngram_range) != expected_word
        or tuple(transformer.char_ngram_range) != expected_char
        or transformer.min_df != expected_min_df
        or transformer.sublinear_tf is not True
        or np.dtype(transformer.dtype) != np.dtype(np.float64)
    ):
        raise ArtifactIntegrityError(f"{context} configuration differs from candidate/config")
    fitted = getattr(transformer, "transformer_", None)

    def validate_vectorizer(value: Any, *, kind: str) -> None:
        if type(value) is not TfidfVectorizer:
            raise ArtifactSchemaError(f"{context} fitted {kind} component has the wrong class")
        expected_analyzer = "word" if kind == "word" else "char_wb"
        expected_ngram = expected_word if kind == "word" else expected_char
        if (
            value.analyzer != expected_analyzer
            or tuple(value.ngram_range) != expected_ngram
            or value.min_df != expected_min_df
            or value.sublinear_tf is not True
            or value.norm != "l2"
            or np.dtype(value.dtype) != np.dtype(np.float64)
        ):
            raise ArtifactIntegrityError(f"{context} fitted {kind} configuration differs")

    if analyzer == "word":
        validate_vectorizer(fitted, kind="word")
    elif analyzer == "char":
        validate_vectorizer(fitted, kind="char")
    else:
        if type(fitted) is not FeatureUnion or [name for name, _ in fitted.transformer_list] != [
            "word",
            "char",
        ]:
            raise ArtifactSchemaError(f"{context} must contain exact word/char components")
        validate_vectorizer(fitted.transformer_list[0][1], kind="word")
        validate_vectorizer(fitted.transformer_list[1][1], kind="char")


def _validate_lsa_semantics(
    transformer: Any,
    *,
    dimensions: int,
    analyzer: str,
    config: ProjectConfig,
    context: str,
) -> None:
    if type(transformer) is not TextLSAIndex:
        raise ArtifactSchemaError(f"{context} must be an exact TextLSAIndex")
    if (
        transformer.n_components != dimensions
        or transformer.analyzer != analyzer
        or transformer.min_df != int(config.raw["candidates"]["minimum_document_frequency"])
        or transformer.random_state != config.project.random_seed
    ):
        raise ArtifactIntegrityError(f"{context} configuration differs from candidate/config")
    _validate_tfidf_semantics(
        getattr(transformer, "vectorizer_", None),
        analyzer=analyzer,
        config=config,
        context=f"{context}.vectorizer",
    )
    embedder = getattr(transformer, "embedder_", None)
    if (
        embedder is None
        or getattr(embedder, "n_components", None) != dimensions
        or getattr(embedder, "random_state", None) != config.project.random_seed
        or getattr(getattr(embedder, "svd_", None), "n_components", None) != dimensions
        or getattr(getattr(embedder, "svd_", None), "random_state", None)
        != config.project.random_seed
    ):
        raise ArtifactIntegrityError(f"{context} fitted SVD configuration differs")


def _validate_title_transformer_semantics(
    transformer: Any,
    *,
    candidate_spec: Mapping[str, Any],
    config: ProjectConfig,
) -> None:
    representation = str(candidate_spec["representation"])
    analyzer = candidate_spec["analyzer"]
    dimensions = int(candidate_spec["lsa_dimensions"])
    weights = {name: float(weight) for name, weight in candidate_spec["weights"]}
    if representation in {"canonical_title", "title_profile"}:
        _validate_tfidf_semantics(
            transformer,
            analyzer=str(analyzer),
            config=config,
            context="title_vectorizer",
        )
        if dimensions != 0 or weights != {"lexical": 1.0}:
            raise ArtifactIntegrityError("title lexical candidate semantics differ")
    elif representation == "title_lsa":
        _validate_lsa_semantics(
            transformer,
            dimensions=dimensions,
            analyzer="word_char",
            config=config,
            context="title_vectorizer",
        )
        if weights != {"lsa": 1.0}:
            raise ArtifactIntegrityError("title LSA candidate weights differ")
    elif representation == "title_hybrid":
        if type(transformer) is not WeightedTextTransformer:
            raise ArtifactSchemaError("title hybrid candidate requires WeightedTextTransformer")
        if len(transformer.components) != 2 or tuple(transformer.weights) != (
            weights.get("lexical"),
            weights.get("lsa"),
        ):
            raise ArtifactIntegrityError("title hybrid transformer weights differ from candidate")
        _validate_tfidf_semantics(
            transformer.components[0],
            analyzer="word_char",
            config=config,
            context="title_vectorizer.lexical",
        )
        _validate_lsa_semantics(
            transformer.components[1],
            dimensions=dimensions,
            analyzer="word_char",
            config=config,
            context="title_vectorizer.lsa",
        )
    else:
        raise ArtifactSchemaError("selected title candidate representation is unsupported")


def _load_smoke_queries(path: Path) -> tuple[tuple[str, str], ...]:
    payload = _load_json(path)
    payload = _expect_mapping(payload, "smoke_queries")
    _expect_exact_keys(payload, required={"schema_version", "queries"}, context="smoke_queries")
    if payload["schema_version"] != 1:
        raise ArtifactSchemaError("unsupported smoke query schema_version")
    queries = payload["queries"]
    if not isinstance(queries, list) or not queries:
        raise ArtifactSchemaError("smoke_queries.queries must be a non-empty list")
    result: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_query in enumerate(queries):
        query = _expect_mapping(raw_query, f"smoke_queries.queries[{index}]")
        _expect_exact_keys(
            query, required={"id", "query"}, context=f"smoke_queries.queries[{index}]"
        )
        _reject_forbidden_fields(query.keys(), context="smoke query fields")
        query_id = _require_nonempty_string(query["id"], f"smoke query {index} id")
        query_text = _require_nonempty_string(query["query"], f"smoke query {index} text")
        if len(query_text) > 512:
            raise ArtifactSchemaError("smoke query text cannot exceed 512 characters")
        if query_id in seen_ids:
            raise ArtifactSchemaError(f"duplicate smoke query id: {query_id}")
        seen_ids.add(query_id)
        result.append((query_id, query_text))
    return tuple(result)


def _evidence_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactSchemaError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactSchemaError(f"{context} must be a finite number")
    return number


def _validate_benchmark_slices(value: Any, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ArtifactSchemaError(f"{context} must be a non-empty list")
    dimensions = {
        "soc_major_group",
        "job_zone",
        "profile_completeness",
        "rating_recency",
    }
    seen: set[tuple[str, str]] = set()
    totals: dict[str, int] = {dimension: 0 for dimension in dimensions}
    for index, raw_slice in enumerate(value):
        item_context = f"{context}[{index}]"
        item = _expect_mapping(raw_slice, item_context)
        _expect_exact_keys(
            item,
            required={
                "dimension",
                "value",
                "estimate",
                "ci_lower",
                "ci_upper",
                "n_units",
                "descriptive_small_slice",
            },
            context=item_context,
        )
        dimension = _require_nonempty_string(item["dimension"], f"{item_context}.dimension")
        if dimension not in dimensions:
            raise ArtifactSchemaError(f"{item_context}.dimension is not preregistered")
        slice_value = _require_nonempty_string(item["value"], f"{item_context}.value")
        allowed_value = {
            "soc_major_group": SOC_MAJOR_RE.fullmatch(slice_value) is not None,
            "job_zone": slice_value in {"2", "3", "4", "5", "missing"},
            "profile_completeness": slice_value in {"none", "1-49", "50+"},
            "rating_recency": slice_value in {"missing", "through-2020", "2021-2023", "2024+"},
        }[dimension]
        if not allowed_value:
            raise ArtifactSchemaError(
                f"{item_context}.value is invalid for slice dimension {dimension!r}"
            )
        key = (dimension, slice_value)
        if key in seen:
            raise ArtifactSchemaError(f"{context} repeats slice {key!r}")
        seen.add(key)
        estimate = _evidence_number(item["estimate"], f"{item_context}.estimate")
        lower = _evidence_number(item["ci_lower"], f"{item_context}.ci_lower")
        upper = _evidence_number(item["ci_upper"], f"{item_context}.ci_upper")
        if not 0 <= lower <= estimate <= upper <= 1:
            raise ArtifactSchemaError(
                f"{item_context} estimate and confidence interval must lie in [0, 1]"
            )
        n_units = _require_positive_int(item["n_units"], f"{item_context}.n_units")
        small = item["descriptive_small_slice"]
        if not isinstance(small, bool) or small != (n_units < 20):
            raise ArtifactSchemaError(
                f"{item_context}.descriptive_small_slice must equal (n_units < 20)"
            )
        totals[dimension] += n_units
    present_dimensions = {dimension for dimension, _ in seen}
    if present_dimensions != dimensions:
        missing = ", ".join(sorted(dimensions - present_dimensions))
        raise ArtifactSchemaError(f"{context} is missing preregistered dimension(s): {missing}")
    if len(set(totals.values())) != 1:
        raise ArtifactSchemaError(f"{context} dimensions must cover the same analysis units")


def _validate_benchmark_payload(
    report_type: str,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    context: str,
) -> None:
    _expect_exact_keys(
        payload,
        required={
            "selected_candidate_id",
            "primary_metric",
            "metrics",
            "baseline_deltas",
            "slices",
        },
        context=context,
    )
    candidate_id = _require_nonempty_string(
        payload["selected_candidate_id"], f"{context}.selected_candidate_id"
    )
    primary_metric = _require_nonempty_string(
        payload["primary_metric"], f"{context}.primary_metric"
    )
    expected_primary, required_metrics, candidate_field = BENCHMARK_CONTRACTS[report_type]
    if candidate_id != manifest["models"][candidate_field]:
        raise ArtifactSchemaError(
            f"{context}.selected_candidate_id does not match the serving model"
        )
    if primary_metric != expected_primary:
        raise ArtifactSchemaError(
            f"{context}.primary_metric must be the preregistered {expected_primary!r}"
        )
    metrics = payload["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ArtifactSchemaError(f"{context}.metrics must be a non-empty list")
    seen_metrics: set[str] = set()
    for index, raw_metric in enumerate(metrics):
        metric = _expect_mapping(raw_metric, f"{context}.metrics[{index}]")
        _expect_exact_keys(
            metric,
            required={"name", "estimate", "ci_lower", "ci_upper", "n_queries"},
            context=f"{context}.metrics[{index}]",
        )
        name = _require_nonempty_string(metric["name"], f"{context}.metrics[{index}].name")
        if name in seen_metrics:
            raise ArtifactSchemaError(f"{context}.metrics repeats {name!r}")
        seen_metrics.add(name)
        estimate = _evidence_number(metric["estimate"], f"{context}.{name}.estimate")
        lower = _evidence_number(metric["ci_lower"], f"{context}.{name}.ci_lower")
        upper = _evidence_number(metric["ci_upper"], f"{context}.{name}.ci_upper")
        if lower > upper:
            raise ArtifactSchemaError(f"{context}.{name} confidence bounds are reversed")
        _require_positive_int(metric["n_queries"], f"{context}.{name}.n_queries")
        if not lower - 1e-12 <= estimate <= upper + 1e-12:
            raise ArtifactSchemaError(f"{context}.{name} estimate must lie within its interval")
        if not 0 <= lower <= estimate <= upper <= 1:
            raise ArtifactSchemaError(f"{context}.{name} retrieval metric must lie in [0, 1]")
    missing_metrics = required_metrics - seen_metrics
    unknown_metrics = seen_metrics - required_metrics
    if missing_metrics or unknown_metrics:
        details = []
        if missing_metrics:
            details.append("missing " + ", ".join(sorted(missing_metrics)))
        if unknown_metrics:
            details.append("unknown " + ", ".join(sorted(unknown_metrics)))
        raise ArtifactSchemaError(
            f"{context}.metrics must contain exactly the preregistered metrics ("
            + "; ".join(details)
            + ")"
        )
    _validate_benchmark_slices(payload["slices"], f"{context}.slices")
    deltas = payload["baseline_deltas"]
    if not isinstance(deltas, list):
        raise ArtifactSchemaError(f"{context}.baseline_deltas must be a list")
    expected_baseline_id = BENCHMARK_BASELINE_CANDIDATES[report_type]
    if candidate_id == expected_baseline_id:
        if deltas:
            raise ArtifactSchemaError(
                f"{context}.baseline_deltas must be empty when the selected candidate is "
                "the preregistered baseline"
            )
        return
    if not deltas:
        raise ArtifactSchemaError(
            f"{context}.baseline_deltas must be non-empty unless confirmation selected "
            "the preregistered baseline"
        )
    seen_deltas: set[tuple[str, str]] = set()
    for index, raw_delta in enumerate(deltas):
        delta = _expect_mapping(raw_delta, f"{context}.baseline_deltas[{index}]")
        _expect_exact_keys(
            delta,
            required={"baseline_candidate_id", "metric", "delta", "ci_lower", "ci_upper"},
            context=f"{context}.baseline_deltas[{index}]",
        )
        baseline_id = _require_nonempty_string(
            delta["baseline_candidate_id"],
            f"{context}.baseline_deltas[{index}].baseline_candidate_id",
        )
        if baseline_id != expected_baseline_id:
            raise ArtifactSchemaError(
                f"{context}.baseline_deltas[{index}].baseline_candidate_id must be the "
                f"preregistered {expected_baseline_id!r}"
            )
        if baseline_id == candidate_id:
            raise ArtifactSchemaError(
                f"{context}.baseline_deltas[{index}] cannot compare the selected candidate to itself"
            )
        metric_name = _require_nonempty_string(
            delta["metric"], f"{context}.baseline_deltas[{index}].metric"
        )
        delta_key = (baseline_id, metric_name)
        if delta_key in seen_deltas:
            raise ArtifactSchemaError(f"{context}.baseline_deltas repeats {delta_key!r}")
        seen_deltas.add(delta_key)
        if metric_name not in seen_metrics:
            raise ArtifactSchemaError(
                f"{context}.baseline_deltas[{index}].metric is absent from metrics"
            )
        estimate = _evidence_number(delta["delta"], f"{context}.baseline_deltas[{index}].delta")
        lower = _evidence_number(delta["ci_lower"], f"{context}.baseline_deltas[{index}].ci_lower")
        upper = _evidence_number(delta["ci_upper"], f"{context}.baseline_deltas[{index}].ci_upper")
        if lower > upper or not lower - 1e-12 <= estimate <= upper + 1e-12:
            raise ArtifactSchemaError(
                f"{context}.baseline_deltas[{index}] has an invalid confidence interval"
            )
        if not -1 <= lower <= estimate <= upper <= 1:
            raise ArtifactSchemaError(f"{context}.baseline_deltas[{index}] must lie in [-1, 1]")
    delta_metrics = {metric_name for _, metric_name in seen_deltas}
    if delta_metrics != seen_metrics:
        raise ArtifactSchemaError(
            f"{context}.baseline_deltas must cover every preregistered metric"
        )


def _validate_locked_benchmark_universes(
    benchmarks: Mapping[str, Any],
    universe: Mapping[str, Any],
) -> None:
    """Bind aggregate counts and descriptive slices to the frozen eligible units.

    Locked transition rankings deliberately retain every assigned test query so
    the production refit can be replayed without consulting qrels. Aggregate
    transition evidence, however, is defined only on the preregistered judged
    subset in ``evaluation_universe``. Title metric counts are alias-query
    counts while title slices are occupation-macro units, so those two title
    cardinalities are intentionally distinct as well.
    """

    expected = {
        "title_benchmark": (
            int(universe["title"]["roles"]["test"]["eligible_query_count"]),
            int(universe["title"]["roles"]["test"]["eligible_occupation_count"]),
        ),
        "transition_benchmark": (
            int(universe["transition"]["roles"]["test"]["eligible_query_count"]),
            int(universe["transition"]["roles"]["test"]["eligible_query_count"]),
        ),
    }
    for report_type, (expected_metric_queries, expected_slice_units) in expected.items():
        payload = _expect_mapping(
            benchmarks[report_type], f"locked_metrics.benchmarks.{report_type}"
        )
        for index, raw_metric in enumerate(payload["metrics"]):
            metric = _expect_mapping(
                raw_metric,
                f"locked_metrics.benchmarks.{report_type}.metrics[{index}]",
            )
            if metric["n_queries"] != expected_metric_queries:
                raise ArtifactIntegrityError(
                    f"locked {report_type} metric query counts differ from evaluation_universe"
                )
        slice_totals: dict[str, int] = {}
        for raw_slice in payload["slices"]:
            item = _expect_mapping(raw_slice, f"locked_metrics.benchmarks.{report_type}.slices")
            dimension = str(item["dimension"])
            slice_totals[dimension] = slice_totals.get(dimension, 0) + int(item["n_units"])
        if set(slice_totals.values()) != {expected_slice_units}:
            raise ArtifactIntegrityError(
                f"locked {report_type} slice units differ from evaluation_universe"
            )


def _validate_evidence_payload(
    report_type: str,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tables: Mapping[str, pa.Table],
) -> None:
    context = f"evidence.{report_type}.payload"
    if report_type in {"title_benchmark", "transition_benchmark"}:
        _validate_benchmark_payload(report_type, payload, manifest, context)
        return
    if report_type == "cluster_stability":
        _expect_exact_keys(
            payload,
            required={
                "selected_k",
                "stability_metric",
                "stability_estimate",
                "silhouette",
                "seeds",
            },
            context=context,
        )
        selected_k = _require_positive_int(payload["selected_k"], f"{context}.selected_k")
        if selected_k not in {12, 18, 24}:
            raise ArtifactSchemaError(f"{context}.selected_k is outside the preregistered grid")
        actual_k = len(set(_column_values(tables["clusters"], "cluster_id")))
        if selected_k != actual_k:
            raise ArtifactSchemaError(f"{context}.selected_k does not match clusters.parquet")
        metric = _require_nonempty_string(
            payload["stability_metric"], f"{context}.stability_metric"
        )
        if metric != "pairwise_adjusted_rand_stability":
            raise ArtifactSchemaError(f"{context}.stability_metric is not preregistered")
        stability = _evidence_number(payload["stability_estimate"], f"{context}.stability_estimate")
        silhouette = _evidence_number(payload["silhouette"], f"{context}.silhouette")
        if not -1 <= stability <= 1 or not -1 <= silhouette <= 1:
            raise ArtifactSchemaError(f"{context} stability and silhouette must lie in [-1, 1]")
        seeds = payload["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ArtifactSchemaError(f"{context}.seeds must be a non-empty integer list")
        for index, seed in enumerate(seeds):
            _require_positive_int(seed, f"{context}.seeds[{index}]", allow_zero=True)
        if seeds != [20260814, 20260815, 20260816, 20260817, 20260818]:
            raise ArtifactSchemaError(f"{context}.seeds do not match the preregistered seeds")
        return
    if report_type == "data_coverage":
        _expect_exact_keys(
            payload,
            required={"counts", "rates"},
            optional={"notes"},
            context=context,
        )
        counts = _expect_mapping(payload["counts"], f"{context}.counts")
        required_counts = {
            "occupation_count",
            "occupations_with_bls",
            "nem_occupation_count",
            "crosswalk_rows",
        }
        if set(counts) != required_counts:
            raise ArtifactSchemaError(
                f"{context}.counts must contain exactly the serving cardinalities"
            )
        _reject_forbidden_fields(counts.keys(), context=f"{context}.counts")
        for name, value in counts.items():
            _require_positive_int(value, f"{context}.counts.{name}", allow_zero=True)
        expected_counts = {
            "occupation_count": tables["occupations"].num_rows,
            "occupations_with_bls": len(
                set(_column_values(tables["occupation_market"], "onet_soc_code"))
            ),
            "nem_occupation_count": len(
                set(_column_values(tables["occupation_market"], "nem_code"))
            ),
            "crosswalk_rows": tables["occupation_market"].num_rows,
        }
        for name, expected in expected_counts.items():
            if counts[name] != expected:
                raise ArtifactSchemaError(
                    f"{context}.counts.{name} does not match the serving tables"
                )
        rates = _expect_mapping(payload["rates"], f"{context}.rates")
        expected_rates = {
            "bls_coverage": (
                expected_counts["occupations_with_bls"] / expected_counts["occupation_count"]
            ),
            "rated_occupation_coverage": (
                sum(
                    int(value) > 0
                    for value in _column_values(tables["occupations"], "structured_observed_count")
                )
                / expected_counts["occupation_count"]
            ),
            "job_zone_coverage": (
                sum(
                    value is not None for value in _column_values(tables["occupations"], "job_zone")
                )
                / expected_counts["occupation_count"]
            ),
        }
        if set(rates) != set(expected_rates):
            raise ArtifactSchemaError(
                f"{context}.rates must contain exactly the serving coverage rates"
            )
        _reject_forbidden_fields(rates.keys(), context=f"{context}.rates")
        for name, value in rates.items():
            rate = _evidence_number(value, f"{context}.rates.{name}")
            if not 0 <= rate <= 1:
                raise ArtifactSchemaError(f"{context}.rates.{name} must lie in [0, 1]")
            if not math.isclose(
                rate,
                expected_rates[name],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ArtifactIntegrityError(
                    f"{context}.rates.{name} differs from the serving tables"
                )
        if "notes" in payload and (
            not isinstance(payload["notes"], list)
            or not all(isinstance(item, str) and item.strip() for item in payload["notes"])
        ):
            raise ArtifactSchemaError(f"{context}.notes must be a list of non-empty strings")
        return
    if report_type == "limitations":
        _expect_exact_keys(payload, required={"items"}, context=context)
        items = payload["items"]
        if (
            not isinstance(items, list)
            or not items
            or not all(
                isinstance(item, str) and item.strip() and item == item.strip() for item in items
            )
        ):
            raise ArtifactSchemaError(f"{context}.items must be non-empty trimmed strings")
        if len(items) != len(set(items)):
            raise ArtifactSchemaError(f"{context}.items cannot contain duplicates")
        return
    raise ArtifactSchemaError(f"unsupported evidence report type: {report_type}")


def _validate_evidence_reports(
    manifest: Mapping[str, Any], paths: Mapping[str, Path], tables: Mapping[str, pa.Table]
) -> Mapping[str, Mapping[str, Any]]:
    if manifest["evidence"]["status"] == "not_generated":
        return MappingProxyType({})
    reports: dict[str, Mapping[str, Any]] = {}
    for report_type, logical_name in manifest["evidence"]["report_files"].items():
        payload = _expect_mapping(
            _load_json(paths[logical_name]), f"evidence report {logical_name}"
        )
        _expect_exact_keys(
            payload,
            required={
                "schema_version",
                "report_type",
                "evidence_status",
                "split",
                "artifact_version",
                "payload",
            },
            context=f"evidence report {logical_name}",
        )
        if payload["schema_version"] != 1:
            raise ArtifactSchemaError(f"unsupported evidence schema in {logical_name}")
        if payload["report_type"] != report_type:
            raise ArtifactSchemaError(f"evidence report type mismatch for {logical_name}")
        if payload["evidence_status"] != "locked":
            raise ArtifactSchemaError(f"evidence report {logical_name} is not locked")
        expected_split = (
            "locked_test"
            if report_type in {"title_benchmark", "transition_benchmark"}
            else "release_context"
        )
        if payload["split"] != expected_split:
            raise ArtifactSchemaError(
                f"evidence report {logical_name} must use split {expected_split!r}"
            )
        if payload["artifact_version"] != manifest["artifact_version"]:
            raise ArtifactSchemaError(f"evidence report {logical_name} artifact version mismatch")
        pending: list[Any] = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                _reject_forbidden_fields(value.keys(), context=f"evidence report {logical_name}")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        report_payload = _expect_mapping(payload["payload"], f"{logical_name}.payload")
        _validate_evidence_payload(report_type, report_payload, manifest, tables)
        reports[report_type] = payload
    return MappingProxyType(reports)


def _reject_nested_forbidden_fields(value: Any, *, context: str) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            _reject_forbidden_fields(item.keys(), context=context)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _load_locked_project_config(config_path: Path) -> ProjectConfig:
    try:
        return load_config(config_path)
    except (ConfigError, OSError, ValueError) as exc:
        raise ArtifactSchemaError("locked lineage requires a valid project configuration") from exc


def _read_fold_manifest(
    path: Path,
    *,
    config: ProjectConfig,
    occupations: pa.Table,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    string_columns = {
        "onet_soc_code": "string",
        "base_soc_code": "string",
        "soc_major_group": "string",
        "role": "string",
        "split_method": "string",
    }
    try:
        frame = pd.read_csv(path, dtype=string_columns, keep_default_na=False)
    except Exception as exc:
        raise ArtifactSchemaError("fold_manifest is not a readable canonical CSV") from exc
    role_folds = {
        "discovery": config.splits.discovery_folds,
        "selection": (config.splits.selection_fold,),
        "confirmation": (config.splits.confirmation_fold,),
        "test": (config.splits.test_fold,),
    }
    occupation_rows = occupations.select(
        ["onet_soc_code", "base_soc_code", "soc_major_group"]
    ).to_pylist()
    expected_codes = [str(row["onet_soc_code"]) for row in occupation_rows]
    try:
        assert_fold_manifest(
            frame,
            expected_codes=expected_codes,
            role_folds=role_folds,
            n_splits=config.splits.n_splits,
            random_seed=config.project.random_seed,
        )
        digest = fold_manifest_digest(frame)
    except (SplitManifestError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactSchemaError("fold_manifest violates the frozen split contract") from exc
    serving_identity = {
        str(row["onet_soc_code"]): (
            str(row["base_soc_code"]),
            str(row["soc_major_group"]),
        )
        for row in occupation_rows
    }
    fold_identity = {
        str(row.onet_soc_code): (str(row.base_soc_code), str(row.soc_major_group))
        for row in frame.itertuples(index=False)
    }
    if fold_identity != serving_identity:
        raise ArtifactIntegrityError(
            "fold_manifest occupation identity differs from serving occupations"
        )
    fit_codes = fitting_codes_for_phase(frame, "locked_test")
    if set(fit_codes) == set(expected_codes):
        raise ArtifactSchemaError("locked-test fit codes must exclude the test fold")
    return frame, digest, fit_codes


def _validate_fold_digest_payload(value: Any, *, semantic_digest: str) -> Mapping[str, Any]:
    payload = _expect_mapping(value, "fold_digest")
    _expect_exact_keys(
        payload,
        required={"manifest_version", "path", "semantic_sha256"},
        context="fold_digest",
    )
    if payload != {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "path": "fold_manifest.csv",
        "semantic_sha256": semantic_digest,
    }:
        raise ArtifactIntegrityError("fold_digest does not bind the bundled fold manifest")
    return payload


def _expected_candidate_manifest() -> dict[str, Any]:
    title = title_candidate_grid()
    transition = transition_candidate_grid()
    return {
        "schema_version": 1,
        "title_sha256": candidate_grid_digest(title),
        "transition_sha256": candidate_grid_digest(transition),
        "title": [_candidate_spec_payload(candidate) for candidate in title],
        "transition": [_candidate_spec_payload(candidate) for candidate in transition],
    }


def _validate_candidate_manifest(value: Any) -> Mapping[str, Any]:
    payload = _expect_mapping(value, "candidate_manifest")
    expected = _expected_candidate_manifest()
    if payload != expected:
        raise ArtifactIntegrityError(
            "candidate_manifest differs from the complete frozen candidate grids in current code"
        )
    return payload


def _selected_candidate_specs(
    candidate_manifest: Mapping[str, Any],
    *,
    title_candidate_id: str,
    transition_candidate_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    indexes = {
        task: {str(item["candidate_id"]): item for item in candidate_manifest[task]}
        for task in ("title", "transition")
    }
    try:
        return indexes["title"][title_candidate_id], indexes["transition"][transition_candidate_id]
    except KeyError as exc:
        raise ArtifactSchemaError(
            "selected serving candidate is absent from candidate_manifest"
        ) from exc


def _validate_implementation_manifest(value: Any) -> Mapping[str, Any]:
    payload = _expect_mapping(value, "implementation_manifest")
    _expect_exact_keys(
        payload,
        required={"schema_version", "scope", "files", "implementation_sha256"},
        context="implementation_manifest",
    )
    if payload["schema_version"] != 1:
        raise ArtifactSchemaError("implementation_manifest must use schema v1")
    expected_scope = {
        "prefixes": ["configs/", "docs/", "src/job_market_intelligence/"],
        "exact_paths": [
            "Makefile",
            "pyproject.toml",
            "requirements.lock",
            "scripts/analyze.py",
            "scripts/benchmark.py",
            "scripts/evaluate.py",
            "scripts/train.py",
        ],
    }
    if payload["scope"] != expected_scope:
        raise ArtifactSchemaError("implementation_manifest scope differs from protocol")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ArtifactSchemaError("implementation_manifest.files must be nonempty")
    paths: list[str] = []
    for index, raw_entry in enumerate(files):
        context = f"implementation_manifest.files[{index}]"
        entry = _expect_mapping(raw_entry, context)
        _expect_exact_keys(entry, required={"path", "sha256", "byte_count"}, context=context)
        path_text = _require_nonempty_string(entry["path"], f"{context}.path")
        relative = PurePosixPath(path_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in path_text
            or path_text != relative.as_posix()
        ):
            raise ArtifactSafetyError(f"{context}.path is unsafe")
        in_scope = path_text in expected_scope["exact_paths"] or any(
            path_text.startswith(prefix) for prefix in expected_scope["prefixes"]
        )
        if not in_scope:
            raise ArtifactSafetyError(f"{context}.path lies outside implementation scope")
        _require_sha256(entry["sha256"], f"{context}.sha256")
        _require_positive_int(entry["byte_count"], f"{context}.byte_count", allow_zero=True)
        paths.append(path_text)
    if paths != sorted(set(paths)):
        raise ArtifactSchemaError("implementation_manifest files are not sorted unique paths")
    implementation_digest = _require_sha256(
        payload["implementation_sha256"],
        "implementation_manifest.implementation_sha256",
    )
    core = dict(payload)
    core.pop("implementation_sha256")
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != implementation_digest:
        raise ArtifactIntegrityError("implementation_manifest self-hash does not verify")
    return payload


def _validate_evaluation_universe(
    value: Any,
    *,
    fold_manifest: pd.DataFrame,
    config: ProjectConfig,
) -> Mapping[str, Any]:
    universe = _expect_mapping(value, "evaluation_universe")
    _expect_exact_keys(
        universe,
        required={
            "schema_version",
            "implementation_manifest_sha256",
            "title",
            "transition",
        },
        context="evaluation_universe",
    )
    if universe["schema_version"] != 1:
        raise ArtifactSchemaError("evaluation_universe must use schema v1")
    _require_sha256(
        universe["implementation_manifest_sha256"],
        "evaluation_universe.implementation_manifest_sha256",
    )
    title = _expect_mapping(universe["title"], "evaluation_universe.title")
    _expect_exact_keys(
        title,
        required={"query_policy", "maximum_queries_per_occupation", "roles"},
        context="evaluation_universe.title",
    )
    if title["query_policy"] != TITLE_QUERY_POLICY or title[
        "maximum_queries_per_occupation"
    ] != int(config.raw["title_benchmark"]["maximum_queries_per_occupation"]):
        raise ArtifactIntegrityError("title evaluation universe differs from config")
    title_roles = _expect_mapping(title["roles"], "evaluation_universe.title.roles")
    transition = _expect_mapping(universe["transition"], "evaluation_universe.transition")
    _expect_exact_keys(
        transition,
        required={"query_policy", "roles"},
        context="evaluation_universe.transition",
    )
    if transition["query_policy"] != "post_boundary_primary_short_required":
        raise ArtifactSchemaError("transition evaluation universe protocol differs")
    transition_roles = _expect_mapping(transition["roles"], "evaluation_universe.transition.roles")
    expected_roles = {"selection", "confirmation", "test"}
    if set(title_roles) != expected_roles or set(transition_roles) != expected_roles:
        raise ArtifactSchemaError("evaluation universe roles differ")
    for role in ("selection", "confirmation", "test"):
        assigned = set(
            fold_manifest.loc[fold_manifest["role"].astype(str).eq(role), "onet_soc_code"].astype(
                str
            )
        )
        title_context = f"evaluation_universe.title.roles.{role}"
        title_entry = _expect_mapping(title_roles[role], title_context)
        _expect_exact_keys(
            title_entry,
            required={
                "assigned_occupation_count",
                "eligible_occupation_count",
                "eligible_query_count",
                "query_ids",
                "query_ids_sha256",
            },
            context=title_context,
        )
        query_ids = title_entry["query_ids"]
        if (
            not isinstance(query_ids, list)
            or not query_ids
            or any(not isinstance(query_id, str) or not query_id for query_id in query_ids)
            or query_ids != sorted(set(query_ids))
        ):
            raise ArtifactSchemaError(f"{title_context}.query_ids are not canonical")
        query_id_digest = hashlib.sha256(
            json.dumps(query_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            isinstance(title_entry["assigned_occupation_count"], bool)
            or title_entry["assigned_occupation_count"] != len(assigned)
            or isinstance(title_entry["eligible_occupation_count"], bool)
            or not isinstance(title_entry["eligible_occupation_count"], int)
            or not 1 <= title_entry["eligible_occupation_count"] <= len(assigned)
            or isinstance(title_entry["eligible_query_count"], bool)
            or title_entry["eligible_query_count"] != len(query_ids)
            or title_entry["query_ids_sha256"] != query_id_digest
        ):
            raise ArtifactIntegrityError(f"{title_context} is inconsistent with the fold manifest")

        context = f"evaluation_universe.transition.roles.{role}"
        entry = _expect_mapping(transition_roles[role], context)
        _expect_exact_keys(
            entry,
            required={
                "assigned_query_count",
                "eligible_query_count",
                "query_codes",
                "query_codes_sha256",
                "candidate_universe",
            },
            context=context,
        )
        codes = entry["query_codes"]
        if (
            not isinstance(codes, list)
            or not codes
            or any(
                not isinstance(code, str) or ONET_SOC_RE.fullmatch(code) is None for code in codes
            )
            or codes != sorted(set(codes))
        ):
            raise ArtifactSchemaError(f"{context}.query_codes must be sorted unique SOC codes")
        expected_code_digest = hashlib.sha256(
            json.dumps(codes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            entry["assigned_query_count"] != len(assigned)
            or isinstance(entry["assigned_query_count"], bool)
            or entry["eligible_query_count"] != len(codes)
            or isinstance(entry["eligible_query_count"], bool)
            or not set(codes) <= assigned
            or entry["query_codes_sha256"] != expected_code_digest
            or entry["candidate_universe"] != ("full_catalog" if role == "test" else "non_test")
        ):
            raise ArtifactIntegrityError(f"{context} is inconsistent with the fold manifest")
    return universe


def _validate_locked_rankings(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    fold_manifest: pd.DataFrame,
    universe: Mapping[str, Any],
    tables: Mapping[str, pa.Table],
) -> pd.DataFrame:
    try:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(SAFE_LOCKED_RANKING_SCHEMA, check_metadata=True):
            raise ArtifactSchemaError("locked_rankings has a noncanonical Arrow schema")
        frame = parquet.read().to_pandas()
    except ArtifactSchemaError:
        raise
    except Exception as exc:
        raise ArtifactSchemaError("locked_rankings is not readable Parquet") from exc
    if frame.empty or list(frame.columns) != SAFE_LOCKED_RANKING_SCHEMA.names:
        raise ArtifactSchemaError("locked_rankings must be nonempty with exact columns")
    text_columns = (
        "benchmark",
        "candidate_id",
        "query_id",
        "query_code",
        "candidate_code",
        "split",
    )
    for column in text_columns:
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in frame[column].tolist()
        ):
            raise ArtifactSchemaError(f"locked_rankings.{column} contains invalid text")
    if set(frame["benchmark"]) != {"title", "transition"} or set(frame["split"]) != {"locked_test"}:
        raise ArtifactSchemaError("locked_rankings benchmark/split contract differs")
    ranks = frame["rank"].to_numpy(dtype=np.int64)
    scores = frame["score"].to_numpy(dtype=np.float64)
    if np.any(ranks < 1) or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1 + 1e-12):
        raise ArtifactSchemaError("locked_rankings ranks/scores are invalid")
    group_keys = ["benchmark", "candidate_id", "query_id"]
    if bool(frame.duplicated([*group_keys, "rank"]).any()) or bool(
        frame.duplicated([*group_keys, "candidate_code"]).any()
    ):
        raise ArtifactSchemaError("locked_rankings repeats a rank or candidate")
    catalog_codes = set(_column_values(tables["occupations"], "onet_soc_code"))
    if set(frame["query_code"]) - catalog_codes or set(frame["candidate_code"]) - catalog_codes:
        raise ArtifactSchemaError("locked_rankings references a code outside the serving catalog")
    test_codes = set(
        fold_manifest.loc[fold_manifest["role"].astype(str).eq("test"), "onet_soc_code"].astype(str)
    )
    if set(frame["query_code"]) - test_codes:
        raise ArtifactIntegrityError("locked_rankings contains a non-test query")
    expected_candidates = {
        "title": {
            manifest["models"]["title_candidate_id"],
            BENCHMARK_BASELINE_CANDIDATES["title_benchmark"],
        },
        "transition": {
            manifest["models"]["transition_candidate_id"],
            BENCHMARK_BASELINE_CANDIDATES["transition_benchmark"],
        },
    }
    for benchmark, expected in expected_candidates.items():
        benchmark_frame = frame.loc[frame["benchmark"].eq(benchmark)]
        if set(benchmark_frame["candidate_id"]) != expected:
            raise ArtifactIntegrityError(
                f"locked_rankings {benchmark} candidates differ from release/baseline"
            )
        candidate_query_sets = {
            candidate: tuple(group["query_id"])
            for candidate, group in benchmark_frame.groupby("candidate_id", sort=True)
        }
        if len(set(candidate_query_sets.values())) != 1:
            raise ArtifactIntegrityError(
                f"locked_rankings {benchmark} candidates use different query universes"
            )
        if benchmark == "transition":
            assigned_test_codes = sorted(test_codes)
            for candidate_id, candidate_frame in benchmark_frame.groupby("candidate_id", sort=True):
                ranked_query_codes = sorted(set(candidate_frame["query_code"].astype(str)))
                if ranked_query_codes != assigned_test_codes:
                    raise ArtifactIntegrityError(
                        "locked_rankings transition candidate "
                        f"{candidate_id!r} does not cover every assigned test query"
                    )
    for key, group in frame.groupby(group_keys, sort=False):
        benchmark = str(group["benchmark"].iloc[0])
        if group["query_code"].nunique() != 1:
            raise ArtifactSchemaError(f"locked_rankings query {key!r} maps to multiple codes")
        if benchmark == "title":
            if (
                len(group) != 1
                or group["candidate_code"].iloc[0] != group["query_code"].iloc[0]
                or int(group["rank"].iloc[0]) > len(catalog_codes)
            ):
                raise ArtifactSchemaError("locked title evidence must be one valid target-rank row")
        else:
            if len(group) != min(20, len(catalog_codes) - 1):
                raise ArtifactSchemaError("locked transition evidence must store complete top-20")
            if group["rank"].tolist() != list(range(1, len(group) + 1)):
                raise ArtifactSchemaError("locked transition ranks are not contiguous")
            if bool(group["query_code"].eq(group["candidate_code"]).any()):
                raise ArtifactSchemaError("locked transition rankings contain a self-match")
            expected = group.sort_values(
                ["score", "candidate_code"], ascending=[False, True], kind="stable"
            )
            if group.index.tolist() != expected.index.tolist():
                raise ArtifactSchemaError("locked transition rankings violate score/code order")
    expected_frame = frame.sort_values(
        ["benchmark", "candidate_id", "query_id", "rank"], kind="stable"
    )
    if frame.index.tolist() != expected_frame.index.tolist():
        raise ArtifactSchemaError("locked_rankings rows are not canonically sorted")

    title_rows = frame.loc[
        frame["benchmark"].eq("title")
        & frame["candidate_id"].eq(manifest["models"]["title_candidate_id"])
    ]
    title_test = universe["title"]["roles"]["test"]
    if (
        title_rows["query_id"].tolist() != title_test["query_ids"]
        or title_rows["query_code"].nunique() != title_test["eligible_occupation_count"]
    ):
        raise ArtifactIntegrityError("locked title rankings differ from evaluation_universe")
    transition_rows = frame.loc[
        frame["benchmark"].eq("transition")
        & frame["candidate_id"].eq(manifest["models"]["transition_candidate_id"])
    ]
    transition_test_codes = set(transition_rows["query_code"].astype(str))
    eligible_transition_codes = set(universe["transition"]["roles"]["test"]["query_codes"])
    if not eligible_transition_codes <= transition_test_codes:
        raise ArtifactIntegrityError(
            "locked transition rankings omit an eligible evaluation_universe query"
        )
    locked_selected = transition_rows.loc[
        :, ["query_code", "candidate_code", "rank", "score"]
    ].sort_values(["query_code", "rank"], kind="stable")
    serving_selected = (
        tables["neighbors"]
        .to_pandas()
        .loc[
            lambda value: (
                value["query_code"].astype(str).isin(test_codes)
                & value["rank"].le(min(20, len(catalog_codes) - 1))
            ),
            ["query_code", "destination_code", "rank", "total_score"],
        ]
        .rename(columns={"destination_code": "candidate_code", "total_score": "score"})
        .sort_values(["query_code", "rank"], kind="stable")
    )
    for column in ("query_code", "candidate_code", "rank"):
        if locked_selected[column].tolist() != serving_selected[column].tolist():
            raise ArtifactIntegrityError(
                f"serving transition neighbors differ from locked rankings in {column}"
            )
    if not np.allclose(
        locked_selected["score"].to_numpy(dtype=np.float64),
        serving_selected["score"].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ArtifactIntegrityError(
            "serving transition neighbor scores differ from locked rankings"
        )
    return frame


def _validate_feature_manifest(value: Any, *, fold_manifest: pd.DataFrame) -> Mapping[str, Any]:
    payload = _expect_mapping(value, "feature_manifest")
    _expect_exact_keys(
        payload,
        required={
            "schema_version",
            "fit_policy",
            "candidate_universe",
            "title_fields",
            "work_fields",
            "structured_fields",
            "structured_missingness",
            "structured_feature_keys",
            "forbidden",
        },
        context="feature_manifest",
    )
    fit_roles = {
        "selection": ["discovery"],
        "confirmation": ["discovery", "selection"],
        "locked_test": ["discovery", "selection", "confirmation"],
        "serving": ["discovery", "selection", "confirmation"],
    }
    expected_fit_policy = {
        phase: {
            "roles": roles,
            "fit_code_count": len(
                fitting_codes_for_phase(
                    fold_manifest, "locked_test" if phase == "serving" else phase
                )
            ),
            "fit_codes_sha256": fit_codes_digest(
                fitting_codes_for_phase(
                    fold_manifest, "locked_test" if phase == "serving" else phase
                )
            ),
        }
        for phase, roles in fit_roles.items()
    }
    if (
        payload["schema_version"] != 1
        or payload["fit_policy"] != expected_fit_policy
        or payload["candidate_universe"]
        != {
            "development": "all non-test occupations",
            "locked_test": "full catalog",
        }
        or payload["title_fields"] != ["canonical_title", "phase-permitted alternate titles"]
        or payload["work_fields"] != ["canonical_title", "description", "core tasks"]
        or payload["structured_fields"]
        != [
            "feature_family",
            "element_id",
            "scale_id",
            "normalized_value",
            "observed",
        ]
        or payload["structured_missingness"]
        != {
            "similarity": "observed skill plus knowledge cosine",
            "shared_pair_validity": (
                "similarity unavailable unless transferable-skill development distance is defined"
            ),
            "hybrid_policy": "renormalize finite positive-weight components; never impute",
        }
    ):
        raise ArtifactSchemaError("feature_manifest differs from the frozen feature policy")
    keys = payload["structured_feature_keys"]
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(item, str) or item.count(":") != 2 for item in keys)
        or keys != sorted(set(keys))
    ):
        raise ArtifactSchemaError("feature_manifest structured feature keys are not canonical")
    forbidden = payload["forbidden"]
    if (
        not isinstance(forbidden, list)
        or len(forbidden) != 4
        or any(not isinstance(item, str) or not item for item in forbidden)
    ):
        raise ArtifactSchemaError("feature_manifest forbidden-feature policy differs")
    return payload


def _validate_metric_contract(value: Any, *, config: ProjectConfig) -> Mapping[str, Any]:
    payload = _expect_mapping(value, "metric_contract")
    expected = {
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
    if payload != expected:
        raise ArtifactIntegrityError("metric_contract differs from the preregistered config")
    return payload


def _validate_source_and_processed_manifests(
    source_value: Any,
    processed_value: Any,
    *,
    source_path: Path,
    config: ProjectConfig,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source_payload = _expect_mapping(source_value, "source_manifest")
    try:
        source = SourceManifest.from_dict(source_payload)
    except ManifestError as exc:
        raise ArtifactSchemaError("source_manifest violates public schema v1") from exc
    expected_specs = {item.source_id: item for item in build_source_specs(config)}
    if (
        source.snapshot_id != manifest["artifact_version"]
        or source.project_name != config.project.name
        or source.project_artifact_version != manifest["artifact_version"]
        or source.project_snapshot_date != manifest["snapshot_date"]
        or source.config_file_sha256 != config.fingerprint
        or source.config_semantic_sha256 != config.semantic_fingerprint
    ):
        raise ArtifactIntegrityError("source_manifest differs from artifact/config identity")
    for item in source.sources:
        expected = expected_specs.get(item.source_id)
        if expected is None or (
            item.family,
            item.table,
            item.publisher,
            item.vintage,
            item.filename,
            item.media_type,
            item.requested_url,
        ) != (
            expected.family,
            expected.table,
            expected.publisher,
            expected.vintage,
            expected.filename,
            expected.media_type,
            expected.requested_url,
        ):
            raise ArtifactIntegrityError(f"source_manifest identity differs for {item.source_id}")
    source_hashes = {item.source_id: item.sha256 for item in source.sources}
    if source_hashes != dict(manifest["provenance"]["source_hashes"]):
        raise ArtifactIntegrityError("source_manifest object hashes differ from provenance")
    if sha256_file(source_path) != manifest["provenance"]["source_manifest_sha256"]:
        raise ArtifactIntegrityError("source_manifest bytes differ from provenance")

    processed = _expect_mapping(processed_value, "processed_manifest")
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
    _expect_exact_keys(processed, required=expected_fields, context="processed_manifest")
    if (
        processed["manifest_version"] != 1
        or processed["snapshot_id"] != manifest["artifact_version"]
        or processed["project_name"] != config.project.name
        or processed["artifact_version"] != manifest["artifact_version"]
        or processed["snapshot_date"] != manifest["snapshot_date"]
        or processed["config_file_sha256"] != config.fingerprint
        or processed["config_semantic_sha256"] != config.semantic_fingerprint
        or processed["source_manifest_sha256"] != sha256_file(source_path)
    ):
        raise ArtifactIntegrityError("processed_manifest identity/provenance differs")
    expected_pins = [
        {
            "source_id": item.source_id,
            "sha256": item.sha256,
            "byte_count": item.byte_count,
            "row_count": item.row_count,
            "schema_fingerprint": item.schema_fingerprint,
            "parser_name": item.parser_name,
            "parser_version": item.parser_version,
        }
        for item in source.sources
    ]
    if processed["source_pins"] != expected_pins:
        raise ArtifactIntegrityError("processed_manifest source pins differ from source_manifest")
    table_entries = processed["tables"]
    if (
        not isinstance(table_entries, list)
        or [entry.get("name") for entry in table_entries if isinstance(entry, dict)]
        != list(PROCESSED_TABLE_NAMES)
        or len(table_entries) != len(PROCESSED_TABLE_NAMES)
    ):
        raise ArtifactSchemaError("processed_manifest table inventory differs from schema v1")
    for index, entry in enumerate(table_entries):
        if not isinstance(entry, dict):
            raise ArtifactSchemaError(f"processed_manifest.tables[{index}] must be an object")
        for digest_name in ("pandas_schema_sha256", "semantic_sha256", "parquet_sha256"):
            _require_sha256(
                entry.get(digest_name), f"processed_manifest.tables[{index}].{digest_name}"
            )
        _require_positive_int(
            entry.get("parquet_byte_count"),
            f"processed_manifest.tables[{index}].parquet_byte_count",
        )
        _require_positive_int(
            entry.get("row_count"), f"processed_manifest.tables[{index}].row_count"
        )
        if entry.get("parquet_path") != f"tables/{entry['name']}.parquet":
            raise ArtifactSchemaError("processed_manifest contains a noncanonical table path")
    return source_payload, processed


def _validate_confirmation_binding(
    decision: Mapping[str, Any],
    *,
    task: str,
    expected_baseline: str,
) -> str:
    selection = _expect_mapping(decision[f"{task}_selection"], f"decision.{task}_selection")
    confirmation = _expect_mapping(
        decision[f"{task}_confirmation"], f"decision.{task}_confirmation"
    )
    selected = _require_nonempty_string(
        selection.get("selected_candidate_id"),
        f"decision.{task}_selection.selected_candidate_id",
    )
    proposed = _require_nonempty_string(
        confirmation.get("proposed_candidate_id"),
        f"decision.{task}_confirmation.proposed_candidate_id",
    )
    baseline = _require_nonempty_string(
        confirmation.get("baseline_candidate_id"),
        f"decision.{task}_confirmation.baseline_candidate_id",
    )
    released = _require_nonempty_string(
        confirmation.get("release_candidate_id"),
        f"decision.{task}_confirmation.release_candidate_id",
    )
    fallback = confirmation.get("fallback_applied")
    if selected != proposed:
        raise ArtifactSchemaError(
            f"development decision {task} confirmation is not bound to its selected candidate"
        )
    if baseline != expected_baseline:
        raise ArtifactSchemaError(f"development decision {task} baseline is not preregistered")
    if not isinstance(fallback, bool):
        raise ArtifactSchemaError(
            f"development decision {task} confirmation fallback_applied must be boolean"
        )
    expected_release = baseline if fallback else proposed
    if released != expected_release:
        raise ArtifactSchemaError(
            f"development decision {task} release does not match its confirmation gate"
        )
    return released


def _validate_locked_lineage(
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    config_path: Path,
    tables: Mapping[str, pa.Table],
) -> Mapping[str, Any]:
    if manifest["evidence"]["status"] != "locked":
        return MappingProxyType({})

    config = _load_locked_project_config(config_path)
    fold_frame, fold_digest_value, fit_codes = _read_fold_manifest(
        paths["fold_manifest"], config=config, occupations=tables["occupations"]
    )
    fold_digest_payload = _validate_fold_digest_payload(
        _load_json(paths["fold_digest"]), semantic_digest=fold_digest_value
    )
    candidate_manifest = _validate_candidate_manifest(_load_json(paths["candidate_manifest"]))
    feature_manifest = _validate_feature_manifest(
        _load_json(paths["feature_manifest"]), fold_manifest=fold_frame
    )
    metric_contract = _validate_metric_contract(_load_json(paths["metric_contract"]), config=config)
    implementation_manifest = _validate_implementation_manifest(
        _load_json(paths["implementation_manifest"])
    )
    universe = _validate_evaluation_universe(
        _load_json(paths["evaluation_universe"]),
        fold_manifest=fold_frame,
        config=config,
    )
    if (
        universe["implementation_manifest_sha256"]
        != implementation_manifest["implementation_sha256"]
    ):
        raise ArtifactIntegrityError(
            "evaluation_universe differs from implementation_manifest semantic hash"
        )
    source_manifest, processed_manifest = _validate_source_and_processed_manifests(
        _load_json(paths["source_manifest"]),
        _load_json(paths["processed_manifest"]),
        source_path=paths["source_manifest"],
        config=config,
        manifest=manifest,
    )
    locked_rankings = _validate_locked_rankings(
        paths["locked_rankings"],
        manifest=manifest,
        fold_manifest=fold_frame,
        universe=universe,
        tables=tables,
    )

    metrics = _expect_mapping(_load_json(paths["locked_metrics"]), "locked_metrics")
    _expect_exact_keys(
        metrics,
        required={
            "schema_version",
            "experiment_version",
            "split",
            "rankings_sha256",
            "decision_sha256",
            "benchmarks",
            "original_repository_commit",
            "reporting_repository_commit",
        },
        context="locked_metrics",
    )
    if metrics["schema_version"] != 1 or metrics["split"] != "locked_test":
        raise ArtifactSchemaError("locked_metrics must be schema v1 locked_test evidence")
    if metrics["experiment_version"] != manifest["artifact_version"]:
        raise ArtifactSchemaError("locked_metrics experiment version differs from the artifact")
    benchmarks = _expect_mapping(metrics["benchmarks"], "locked_metrics.benchmarks")
    if set(benchmarks) != {"title_benchmark", "transition_benchmark"}:
        raise ArtifactSchemaError("locked_metrics must contain exactly both benchmark payloads")
    for report_type, payload in benchmarks.items():
        benchmark_payload = _expect_mapping(payload, f"locked_metrics.benchmarks.{report_type}")
        _validate_benchmark_payload(
            report_type,
            benchmark_payload,
            manifest,
            f"locked_metrics.benchmarks.{report_type}",
        )
    _validate_locked_benchmark_universes(benchmarks, universe)
    provenance = manifest["provenance"]
    if sha256_file(paths["locked_metrics"]) != provenance["locked_metrics_sha256"]:
        raise ArtifactIntegrityError(
            "lineage locked_metrics bytes differ from provenance.locked_metrics_sha256"
        )
    if metrics["rankings_sha256"] != provenance["locked_rankings_sha256"]:
        raise ArtifactIntegrityError(
            "locked_metrics rankings hash differs from provenance.locked_rankings_sha256"
        )
    if sha256_file(paths["locked_rankings"]) != provenance["locked_rankings_sha256"]:
        raise ArtifactIntegrityError(
            "lineage locked_rankings bytes differ from provenance.locked_rankings_sha256"
        )
    for name in ("original_repository_commit", "reporting_repository_commit"):
        value = metrics[name]
        if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
            raise ArtifactSchemaError(f"locked_metrics.{name} must be a full Git object ID")
    if (
        metrics["original_repository_commit"] != metrics["reporting_repository_commit"]
        or metrics["reporting_repository_commit"] != provenance["repository_commit"]
    ):
        raise ArtifactIntegrityError(
            "locked_metrics commits differ from the exact-HEAD release provenance"
        )

    decision = _expect_mapping(_load_json(paths["development_decision"]), "development_decision")
    decision_fields = {
        "schema_version",
        "experiment_version",
        "release_candidates",
        "title_selection",
        "title_confirmation",
        "transition_selection",
        "transition_confirmation",
        "project_config_sha256",
        "fold_manifest_sha256",
        "feature_manifest_sha256",
        "metric_contract_sha256",
        "environment_lock_sha256",
        "processed_manifest_sha256",
        "candidate_manifest_sha256",
        "evaluation_universe_sha256",
        "implementation_manifest_sha256",
        "title_candidate_grid_sha256",
        "transition_candidate_grid_sha256",
        "decision_sha256",
    }
    _expect_exact_keys(decision, required=decision_fields, context="development_decision")
    if decision["schema_version"] != 2:
        raise ArtifactSchemaError("development_decision must use multi-task schema v2")
    if decision["experiment_version"] != manifest["artifact_version"]:
        raise ArtifactSchemaError("development_decision experiment version differs from artifact")
    decision_core = dict(decision)
    decision_digest = decision_core.pop("decision_sha256")
    _require_sha256(decision_digest, "development_decision.decision_sha256")
    encoded = json.dumps(decision_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != decision_digest:
        raise ArtifactIntegrityError("development_decision self-hash does not verify")
    try:
        verify_development_decision(decision)
    except EvaluationIntegrityError as exc:
        raise ArtifactSchemaError(
            f"development_decision selection/gate semantics are invalid: {exc}"
        ) from exc
    if metrics["decision_sha256"] != decision_digest:
        raise ArtifactIntegrityError(
            "locked_metrics decision hash differs from development_decision"
        )

    releases = _expect_mapping(decision["release_candidates"], "decision.release_candidates")
    if set(releases) != {"title", "transition"}:
        raise ArtifactSchemaError("development_decision must freeze title and transition releases")
    title_release = _validate_confirmation_binding(
        decision,
        task="title",
        expected_baseline=BENCHMARK_BASELINE_CANDIDATES["title_benchmark"],
    )
    transition_release = _validate_confirmation_binding(
        decision,
        task="transition",
        expected_baseline=BENCHMARK_BASELINE_CANDIDATES["transition_benchmark"],
    )
    if releases != {"title": title_release, "transition": transition_release}:
        raise ArtifactSchemaError(
            "development_decision release candidates differ from confirmation gates"
        )
    if (
        manifest["models"]["title_candidate_id"] != title_release
        or manifest["models"]["transition_candidate_id"] != transition_release
    ):
        raise ArtifactSchemaError("serving model IDs differ from development_decision releases")

    decision_provenance_fields = {
        "project_config_sha256": "project_config_sha256",
        "fold_manifest_sha256": "split_manifest_sha256",
        "feature_manifest_sha256": "feature_manifest_sha256",
        "metric_contract_sha256": "metric_contract_sha256",
        "environment_lock_sha256": "environment_lock_sha256",
        "processed_manifest_sha256": "processed_manifest_sha256",
        "candidate_manifest_sha256": "candidate_manifest_sha256",
        "evaluation_universe_sha256": "evaluation_universe_sha256",
        "implementation_manifest_sha256": "implementation_manifest_sha256",
        "title_candidate_grid_sha256": "title_candidate_grid_sha256",
        "transition_candidate_grid_sha256": "transition_candidate_grid_sha256",
    }
    for decision_name, provenance_name in decision_provenance_fields.items():
        digest = decision[decision_name]
        _require_sha256(digest, f"development_decision.{decision_name}")
        if digest != provenance[provenance_name]:
            raise ArtifactIntegrityError(
                f"development_decision.{decision_name} differs from provenance.{provenance_name}"
            )
    exact_file_digest_fields = {
        "processed_manifest": "processed_manifest_sha256",
        "feature_manifest": "feature_manifest_sha256",
        "metric_contract": "metric_contract_sha256",
        "candidate_manifest": "candidate_manifest_sha256",
        "evaluation_universe": "evaluation_universe_sha256",
        "implementation_manifest": "implementation_manifest_sha256",
    }
    for logical_name, decision_name in exact_file_digest_fields.items():
        if sha256_file(paths[logical_name]) != decision[decision_name]:
            raise ArtifactIntegrityError(
                f"bundled {logical_name} bytes differ from development_decision.{decision_name}"
            )
    if decision["fold_manifest_sha256"] != fold_digest_value:
        raise ArtifactIntegrityError(
            "bundled fold_manifest semantic digest differs from development_decision"
        )
    if sha256_file(config_path) != decision["project_config_sha256"]:
        raise ArtifactIntegrityError(
            "development_decision project config hash differs from literal config bytes"
        )
    requirements_path = config_path.parent.parent / "requirements.lock"
    if sha256_file(requirements_path) != decision["environment_lock_sha256"]:
        raise ArtifactIntegrityError(
            "development_decision environment hash differs from requirements.lock bytes"
        )

    expected_title_grid = candidate_grid_digest(title_candidate_grid())
    expected_transition_grid = candidate_grid_digest(transition_candidate_grid())
    if decision["title_candidate_grid_sha256"] != expected_title_grid:
        raise ArtifactIntegrityError(
            "development_decision title candidate grid differs from current code"
        )
    if decision["transition_candidate_grid_sha256"] != expected_transition_grid:
        raise ArtifactIntegrityError(
            "development_decision transition candidate grid differs from current code"
        )
    title_specs = {candidate.candidate_id: candidate for candidate in title_candidate_grid()}
    transition_specs = {
        candidate.candidate_id: candidate for candidate in transition_candidate_grid()
    }
    if title_release not in title_specs or transition_release not in transition_specs:
        raise ArtifactSchemaError("development_decision release candidate is absent from code")
    transition_weights = transition_specs[transition_release].weight_map
    if transition_weights.get("lexical", 0.0) <= 0 and transition_weights.get("lsa", 0.0) <= 0:
        raise ArtifactSchemaError(
            "development_decision transition release lacks an always-finite text component"
        )
    selected_title_payload, selected_transition_payload = _selected_candidate_specs(
        candidate_manifest,
        title_candidate_id=title_release,
        transition_candidate_id=transition_release,
    )
    models = manifest["models"]
    if (
        models["title_candidate_spec"] != selected_title_payload
        or models["transition_candidate_spec"] != selected_transition_payload
    ):
        raise ArtifactIntegrityError(
            "serving candidate specs differ from the exact selected candidate manifest entries"
        )
    if (
        models["fit_phase"] != "locked_test"
        or models["fit_codes_sha256"] != fit_codes_digest(fit_codes)
        or models["fit_code_count"] != len(fit_codes)
        or models["fit_fold_manifest_sha256"] != fold_digest_value
    ):
        raise ArtifactIntegrityError(
            "serving fit-code/fold binding differs from the locked-test split"
        )
    serving_feature_keys = sorted(
        {
            f"{row['feature_family']}:{row['element_id']}:{row['scale_id']}"
            for row in tables["skill_ratings"].to_pylist()
        }
    )
    if feature_manifest["structured_feature_keys"] != serving_feature_keys:
        raise ArtifactIntegrityError(
            "feature_manifest structured keys differ from serving skill_ratings"
        )
    processed_rows = {
        str(entry["name"]): int(entry["row_count"]) for entry in processed_manifest["tables"]
    }
    if (
        processed_rows["occupation_profiles"] != tables["occupations"].num_rows
        or processed_rows["skill_ratings"] != tables["skill_ratings"].num_rows
        or processed_rows["occupation_market"] != tables["occupation_market"].num_rows
    ):
        raise ArtifactIntegrityError("serving table row counts differ from processed_manifest")
    _reject_nested_forbidden_fields(metrics, context="locked_metrics")
    _reject_nested_forbidden_fields(decision, context="development_decision")
    return MappingProxyType(
        {
            "locked_metrics": metrics,
            "locked_rankings": {
                "row_count": len(locked_rankings),
                "sha256": sha256_file(paths["locked_rankings"]),
            },
            "development_decision": decision,
            "fold_manifest": {"rows": fold_frame.to_dict("records")},
            "fold_digest": fold_digest_payload,
            "processed_manifest": processed_manifest,
            "feature_manifest": feature_manifest,
            "metric_contract": metric_contract,
            "candidate_manifest": candidate_manifest,
            "source_manifest": source_manifest,
            "evaluation_universe": universe,
            "implementation_manifest": implementation_manifest,
        }
    )


def _validate_lineage_report_bindings(
    lineage: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    if not lineage:
        return
    locked_benchmarks = lineage["locked_metrics"]["benchmarks"]
    for report_type in ("title_benchmark", "transition_benchmark"):
        if reports[report_type]["payload"] != locked_benchmarks[report_type]:
            raise ArtifactIntegrityError(
                f"{report_type} wrapper payload differs from locked_metrics"
            )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_artifact_bundle(
    bundle_path: str | Path, *, project_config_path: str | Path | None = None
) -> LoadedArtifactBundle:
    """Load a serving bundle only after all bytes and contracts are verified.

    Joblib is a pickle-based format.  It is deserialized only after the bundle's
    complete file set, byte sizes, and SHA-256 digests have been checked.  Bundle
    manifests must therefore come from the trusted release process.
    """

    root = Path(bundle_path)
    if not root.exists():
        raise ArtifactNotFoundError(f"serving artifact directory does not exist: {root}")
    if not root.is_dir():
        raise ArtifactNotFoundError(f"serving artifact path is not a directory: {root}")
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactNotFoundError(f"serving artifact manifest is missing: {manifest_path}")
    if manifest_path.is_symlink():
        raise ArtifactSafetyError("manifest.json cannot be a symlink")

    manifest = _validate_manifest(_load_json(manifest_path))
    paths = _verify_files(root, manifest)
    config_path = _verify_project_config(root, manifest, project_config_path)

    tables = {
        logical_name: _read_core_table(
            logical_name, paths[logical_name], manifest["files"][logical_name]
        )
        for logical_name in CORE_TABLE_SCHEMAS
    }
    _validate_foreign_keys(tables, manifest["models"]["stored_transition_limit"])

    smoke_queries = _load_smoke_queries(paths["smoke_queries"])
    matrix = _validate_sparse_matrix(
        paths["title_matrix"],
        manifest["files"]["title_matrix"],
        row_count=tables["search_catalog"].num_rows,
        feature_count=manifest["models"]["title_feature_count"],
        normalized=manifest["models"]["title_matrix_normalized"],
    )

    # Locked metrics/decision bytes and JSON reports are parsed, schema-checked,
    # and cross-linked before the pickle-based model is ever deserialized.
    locked_lineage = _validate_locked_lineage(
        manifest, paths, config_path=config_path, tables=tables
    )
    if locked_lineage:
        _validate_transition_score_semantics(
            tables["neighbors"], manifest["models"]["transition_candidate_spec"]
        )
    evidence_reports = _validate_evidence_reports(manifest, paths, tables)
    _validate_lineage_report_bindings(locked_lineage, evidence_reports)
    try:
        with warnings.catch_warnings():
            # joblib 1.5.3 assigns ``array.shape`` while decoding its own
            # trusted format, which NumPy 2.5 deprecates. Keep this one
            # dependency warning from turning a valid locked artifact into an
            # application failure; all other warnings remain visible.
            warnings.filterwarnings(
                "ignore",
                message=r"Setting the shape on a NumPy array has been deprecated.*",
                category=DeprecationWarning,
                module=r"joblib\.numpy_pickle",
            )
            transformer = joblib.load(paths["title_vectorizer"])
    except Exception as exc:
        raise ArtifactSchemaError(
            "title_vectorizer is not a readable fitted joblib artifact"
        ) from exc
    _validate_transformer(
        transformer,
        feature_count=manifest["models"]["title_feature_count"],
        smoke_queries=smoke_queries,
        search_matrix=matrix,
    )
    if locked_lineage:
        _validate_title_transformer_semantics(
            transformer,
            candidate_spec=manifest["models"]["title_candidate_spec"],
            config=_load_locked_project_config(config_path),
        )

    return LoadedArtifactBundle(
        root=root.resolve(),
        manifest=_deep_freeze(manifest),
        tables=MappingProxyType(tables),
        search_transformer=transformer,
        search_matrix=matrix,
        smoke_queries=smoke_queries,
        evidence_reports=_deep_freeze(dict(evidence_reports)),
        locked_lineage=_deep_freeze(dict(locked_lineage)),
    )


def _canonical_json_bytes(value: Any, *, context: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactSchemaError(f"{context} is not strict JSON data") from exc


def _generic_file_manifest(
    path: Path, artifact_format: str, *, relative_to: Path
) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "format": artifact_format,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _coerce_core_table(logical_name: str, value: Any) -> pa.Table:
    expected_schema = CORE_TABLE_SCHEMAS[logical_name]
    if isinstance(value, pa.Table):
        table = value.combine_chunks().replace_schema_metadata(None)
    else:
        raw_columns = getattr(value, "columns", None)
        if raw_columns is None or list(raw_columns) != expected_schema.names:
            raise ArtifactSchemaError(
                f"{logical_name} must expose exactly the canonical ordered columns"
            )
        try:
            table = (
                pa.Table.from_pandas(
                    value,
                    schema=expected_schema,
                    preserve_index=False,
                    safe=True,
                )
                .combine_chunks()
                .replace_schema_metadata(None)
            )
        except Exception as exc:
            raise ArtifactSchemaError(
                f"{logical_name} cannot be converted safely to its canonical Arrow schema"
            ) from exc
    if table.column_names != expected_schema.names or not table.schema.equals(
        expected_schema, check_metadata=True
    ):
        raise ArtifactSchemaError(
            f"{logical_name} must match its canonical ordered Arrow schema exactly"
        )
    _validate_table_contract(logical_name, table)
    return table


def _smoke_query_payload(smoke_queries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    for index, raw_query in enumerate(smoke_queries):
        if not isinstance(raw_query, Mapping):
            raise ArtifactSchemaError(f"smoke_queries[{index}] must be an object")
        queries.append(dict(raw_query))
    return {"schema_version": 1, "queries": queries}


def _evidence_payload(value: str | Path | Mapping[str, Any], *, report_type: str) -> Any:
    if isinstance(value, (str, Path)):
        source = Path(value)
        if not source.is_file() or source.is_symlink():
            raise ArtifactIntegrityError(
                f"evidence source for {report_type} must be a regular non-symlink file"
            )
        return _load_json(source)
    if not isinstance(value, Mapping):
        raise ArtifactSchemaError(
            f"evidence source for {report_type} must be a JSON path or complete report object"
        )
    return dict(value)


def _exact_lineage_source_bytes(
    value: str | Path,
    *,
    context: str,
    artifact_format: str,
) -> bytes:
    source = Path(value)
    if not source.is_file() or source.is_symlink():
        raise ArtifactIntegrityError(f"{context} must be a regular non-symlink file")
    if artifact_format == "json":
        _load_json(source)
    elif artifact_format == "csv":
        try:
            source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ArtifactIntegrityError(f"{context} must be UTF-8 CSV") from exc
    elif artifact_format == "parquet":
        try:
            pq.ParquetFile(source)
        except Exception as exc:
            raise ArtifactIntegrityError(f"{context} must be readable Parquet") from exc
    else:  # pragma: no cover - immutable internal format table
        raise ArtifactIntegrityError(f"unsupported lineage format for {context}")
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ArtifactIntegrityError(f"could not read exact bytes for {context}") from exc


def write_artifact_bundle(
    output_dir: str | Path,
    *,
    artifact_version: str,
    snapshot_date: str | date,
    occupations: Any,
    occupation_market: Any,
    skill_ratings: Any,
    clusters: Any,
    neighbors: Any,
    search_catalog: Any,
    title_vectorizer: Any,
    title_matrix: sparse.spmatrix,
    smoke_queries: Iterable[Mapping[str, Any]],
    evidence_reports: Mapping[str, str | Path | Mapping[str, Any]],
    model_metadata: Mapping[str, Any],
    provenance: Mapping[str, Any],
    locked_metrics_path: str | Path | None = None,
    locked_rankings_path: str | Path | None = None,
    development_decision_path: str | Path | None = None,
    fold_manifest_path: str | Path | None = None,
    fold_digest_path: str | Path | None = None,
    processed_manifest_path: str | Path | None = None,
    feature_manifest_path: str | Path | None = None,
    metric_contract_path: str | Path | None = None,
    candidate_manifest_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    evaluation_universe_path: str | Path | None = None,
    implementation_manifest_path: str | Path | None = None,
    project_config_path: str | Path | None = None,
    created_at_utc: str | None = None,
) -> Path:
    """Validate, stage, and publish one immutable serving bundle.

    The function never derives evidence, rankings, metrics, or model choices. It
    accepts already-frozen release outputs, writes only the serving contract,
    validates the complete staged directory through :func:`load_artifact_bundle`,
    and then renames it into place while holding an exclusive publication lock.
    Existing output directories are never replaced.
    """

    target = Path(output_dir)
    if target.name in {"", ".", ".."} or ".." in target.parts:
        raise ArtifactSafetyError("artifact output must name a version directory")
    parent = target.parent
    if any(ancestor.is_symlink() for ancestor in (parent, *parent.parents)):
        raise ArtifactSafetyError("artifact output ancestors cannot be symlinks")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ArtifactSafetyError("artifact output parent cannot be a symlink")
    if os.path.lexists(target):
        raise ArtifactIntegrityError(f"artifact output already exists: {target}")

    config_path = _resolve_project_config_path(target, project_config_path)
    if not config_path.is_file() or config_path.is_symlink():
        raise ArtifactIntegrityError(
            f"project configuration required for provenance is missing: {config_path}"
        )

    table_inputs = {
        "occupations": occupations,
        "occupation_market": occupation_market,
        "skill_ratings": skill_ratings,
        "clusters": clusters,
        "neighbors": neighbors,
        "search_catalog": search_catalog,
    }
    tables = {
        logical_name: _coerce_core_table(logical_name, value)
        for logical_name, value in table_inputs.items()
    }
    _validate_foreign_keys(tables, STORED_TRANSITION_LIMIT)

    models = dict(model_metadata)
    derived_model_metadata = {
        "title_feature_count": int(title_matrix.shape[1]),
        "title_matrix_normalized": True,
        "stored_transition_limit": STORED_TRANSITION_LIMIT,
        "neighbor_display_max": NEIGHBOR_DISPLAY_MAX,
        "transition_components": ["lexical", "lsa", "structured"],
    }
    for name, expected in derived_model_metadata.items():
        if name in models and models[name] != expected:
            raise ManifestValidationError(
                f"model_metadata.{name} conflicts with the serving contract"
            )
        models[name] = expected

    provenance_payload = dict(provenance)
    config_digest = sha256_file(config_path)
    configured_digest = provenance_payload.get("project_config_sha256")
    if configured_digest is not None and configured_digest != config_digest:
        raise ArtifactIntegrityError(
            "provenance.project_config_sha256 does not match project_config_path bytes"
        )
    provenance_payload["project_config_sha256"] = config_digest
    requirements_path = config_path.parent.parent / "requirements.lock"
    if not requirements_path.is_file() or requirements_path.is_symlink():
        raise ArtifactIntegrityError(
            "requirements.lock required for provenance is missing or unsafe"
        )
    environment_digest = sha256_file(requirements_path)
    configured_environment = provenance_payload.get("environment_lock_sha256")
    if configured_environment is not None and configured_environment != environment_digest:
        raise ArtifactIntegrityError(
            "provenance.environment_lock_sha256 does not match requirements.lock bytes"
        )
    provenance_payload["environment_lock_sha256"] = environment_digest

    report_sources = dict(evidence_reports)
    evidence_status = "locked" if report_sources else "not_generated"
    if report_sources and set(report_sources) != EVIDENCE_REPORT_TYPES:
        missing = EVIDENCE_REPORT_TYPES - report_sources.keys()
        unknown = report_sources.keys() - EVIDENCE_REPORT_TYPES
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ManifestValidationError(
            "locked evidence must provide exactly five reports (" + "; ".join(details) + ")"
        )
    lineage_arguments = {
        "locked_metrics": locked_metrics_path,
        "locked_rankings": locked_rankings_path,
        "development_decision": development_decision_path,
        "fold_manifest": fold_manifest_path,
        "fold_digest": fold_digest_path,
        "processed_manifest": processed_manifest_path,
        "feature_manifest": feature_manifest_path,
        "metric_contract": metric_contract_path,
        "candidate_manifest": candidate_manifest_path,
        "source_manifest": source_manifest_path,
        "evaluation_universe": evaluation_universe_path,
        "implementation_manifest": implementation_manifest_path,
    }
    supplied_lineage = {name for name, value in lineage_arguments.items() if value is not None}
    if report_sources and supplied_lineage != set(LOCKED_LINEAGE_FILE_FORMATS):
        missing = ", ".join(sorted(set(LOCKED_LINEAGE_FILE_FORMATS) - supplied_lineage))
        raise ManifestValidationError(
            f"locked evidence requires every exact safe lineage file: {missing}"
        )
    if not report_sources and supplied_lineage:
        raise ManifestValidationError(
            "locked release lineage cannot accompany not_generated evidence"
        )
    lineage_bytes = {
        name: _exact_lineage_source_bytes(
            value,
            context=name,
            artifact_format=LOCKED_LINEAGE_FILE_FORMATS[name],
        )
        for name, value in lineage_arguments.items()
        if value is not None
    }
    if report_sources:
        lineage_paths = {
            name: Path(value) for name, value in lineage_arguments.items() if value is not None
        }
        config = _load_locked_project_config(config_path)
        _fold_frame, split_digest, fit_codes = _read_fold_manifest(
            lineage_paths["fold_manifest"],
            config=config,
            occupations=tables["occupations"],
        )
        _validate_fold_digest_payload(
            _load_json(lineage_paths["fold_digest"]), semantic_digest=split_digest
        )
        candidate_payload = _validate_candidate_manifest(
            _load_json(lineage_paths["candidate_manifest"])
        )
        title_id = _require_nonempty_string(
            models.get("title_candidate_id"), "model_metadata.title_candidate_id"
        )
        transition_id = _require_nonempty_string(
            models.get("transition_candidate_id"), "model_metadata.transition_candidate_id"
        )
        title_spec, transition_spec = _selected_candidate_specs(
            candidate_payload,
            title_candidate_id=title_id,
            transition_candidate_id=transition_id,
        )
        locked_model_metadata = {
            "title_candidate_spec": dict(title_spec),
            "transition_candidate_spec": dict(transition_spec),
            "fit_phase": "locked_test",
            "fit_codes_sha256": fit_codes_digest(fit_codes),
            "fit_code_count": len(fit_codes),
            "fit_fold_manifest_sha256": split_digest,
        }
        for name, expected in locked_model_metadata.items():
            if name in models and models[name] != expected:
                raise ArtifactIntegrityError(
                    f"model_metadata.{name} differs from derived locked lineage"
                )
            models[name] = expected
        _validate_transition_score_semantics(tables["neighbors"], transition_spec)

        source_payload = _expect_mapping(
            _load_json(lineage_paths["source_manifest"]), "source_manifest"
        )
        try:
            source = SourceManifest.from_dict(source_payload)
        except ManifestError as exc:
            raise ArtifactSchemaError("source_manifest violates public schema v1") from exc
        derived_provenance = {
            "source_manifest_sha256": sha256_file(lineage_paths["source_manifest"]),
            "split_manifest_sha256": split_digest,
            "locked_rankings_sha256": sha256_file(lineage_paths["locked_rankings"]),
            "locked_metrics_sha256": sha256_file(lineage_paths["locked_metrics"]),
            "processed_manifest_sha256": sha256_file(lineage_paths["processed_manifest"]),
            "feature_manifest_sha256": sha256_file(lineage_paths["feature_manifest"]),
            "metric_contract_sha256": sha256_file(lineage_paths["metric_contract"]),
            "candidate_manifest_sha256": sha256_file(lineage_paths["candidate_manifest"]),
            "evaluation_universe_sha256": sha256_file(lineage_paths["evaluation_universe"]),
            "implementation_manifest_sha256": sha256_file(lineage_paths["implementation_manifest"]),
            "title_candidate_grid_sha256": candidate_grid_digest(title_candidate_grid()),
            "transition_candidate_grid_sha256": candidate_grid_digest(transition_candidate_grid()),
        }
        for name, expected in derived_provenance.items():
            if name in provenance_payload and provenance_payload[name] != expected:
                raise ArtifactIntegrityError(
                    f"provenance.{name} differs from exact locked lineage bytes"
                )
            provenance_payload[name] = expected
        derived_source_hashes = {item.source_id: item.sha256 for item in source.sources}
        if (
            "source_hashes" in provenance_payload
            and provenance_payload["source_hashes"] != derived_source_hashes
        ):
            raise ArtifactIntegrityError(
                "provenance.source_hashes differs from exact source_manifest"
            )
        provenance_payload["source_hashes"] = derived_source_hashes

    lock_path = parent / f".{target.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArtifactIntegrityError(f"artifact publication is already reserved: {target}") from exc
    staging: Path | None = None
    published = False
    try:
        os.close(lock_fd)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
        staging.chmod(0o755)
        files: dict[str, dict[str, Any]] = {}
        for logical_name, table in tables.items():
            artifact_path = staging / f"{logical_name}.parquet"
            pq.write_table(
                table,
                artifact_path,
                compression="zstd",
                version="2.6",
                use_dictionary=False,
                write_statistics=True,
            )
            files[logical_name] = parquet_file_manifest(logical_name, artifact_path)

        vectorizer_path = staging / "title_vectorizer.joblib"
        joblib.dump(title_vectorizer, vectorizer_path, compress=3)
        files["title_vectorizer"] = _generic_file_manifest(
            vectorizer_path, "joblib", relative_to=staging
        )

        matrix_path = staging / "title_matrix.npz"
        sparse.save_npz(matrix_path, title_matrix, compressed=True)
        files["title_matrix"] = {
            **_generic_file_manifest(matrix_path, "scipy_npz", relative_to=staging),
            "shape": [int(title_matrix.shape[0]), int(title_matrix.shape[1])],
            "dtype": str(title_matrix.dtype),
        }

        smoke_path = staging / "smoke_queries.json"
        smoke_path.write_bytes(
            _canonical_json_bytes(_smoke_query_payload(smoke_queries), context="smoke_queries")
        )
        files["smoke_queries"] = {
            **_generic_file_manifest(smoke_path, "json", relative_to=staging),
        }

        report_files: dict[str, str] = {}
        if report_sources:
            reports_dir = staging / "reports"
            reports_dir.mkdir()
            for report_type in sorted(EVIDENCE_REPORT_TYPES):
                logical_name = f"report_{report_type}"
                report_path = reports_dir / f"{report_type}.json"
                report_path.write_bytes(
                    _canonical_json_bytes(
                        _evidence_payload(report_sources[report_type], report_type=report_type),
                        context=f"evidence report {report_type}",
                    )
                )
                entry = _generic_file_manifest(report_path, "json", relative_to=staging)
                entry["schema_version"] = 1
                files[logical_name] = entry
                report_files[report_type] = logical_name

            lineage_dir = staging / "lineage"
            lineage_dir.mkdir()
            for logical_name in sorted(LOCKED_LINEAGE_FILE_FORMATS):
                suffix = {
                    "json": ".json",
                    "csv": ".csv",
                    "parquet": ".parquet",
                }[LOCKED_LINEAGE_FILE_FORMATS[logical_name]]
                lineage_path = lineage_dir / f"{logical_name}{suffix}"
                lineage_path.write_bytes(lineage_bytes[logical_name])
                files[logical_name] = _generic_file_manifest(
                    lineage_path,
                    LOCKED_LINEAGE_FILE_FORMATS[logical_name],
                    relative_to=staging,
                )

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_version": artifact_version,
            "snapshot_date": (
                snapshot_date.isoformat() if isinstance(snapshot_date, date) else snapshot_date
            ),
            "models": models,
            "provenance": provenance_payload,
            "evidence": {
                "status": evidence_status,
                "report_files": report_files,
            },
            "files": files,
        }
        if created_at_utc is not None:
            manifest["created_at_utc"] = created_at_utc
        _validate_manifest(manifest)
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json_bytes(manifest, context="manifest"))
        for candidate in staging.rglob("*"):
            candidate.chmod(0o755 if candidate.is_dir() else 0o644)
        staging.chmod(0o755)

        # This is deliberately the same load path used by the CLI, API, and UI.
        # It validates every staged byte/table/report/matrix before deserializing
        # the just-written model and before the directory becomes visible.
        load_artifact_bundle(staging, project_config_path=config_path)

        if os.path.lexists(target):
            raise ArtifactIntegrityError(f"artifact output appeared during publication: {target}")
        os.rename(staging, target)
        published = True
        return target.resolve()
    finally:
        if not published and staging is not None and staging.exists():
            shutil.rmtree(staging)
        with suppress(FileNotFoundError):
            lock_path.unlink()


# Concise compatibility alias for callers and older scripts.
load_bundle = load_artifact_bundle
