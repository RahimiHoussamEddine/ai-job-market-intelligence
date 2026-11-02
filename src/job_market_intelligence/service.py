"""Immutable application service shared by the CLI, API, and dashboard."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from scipy import sparse

from .artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactSafetyError,
    ArtifactSchemaError,
    LoadedArtifactBundle,
    ManifestValidationError,
    load_artifact_bundle,
)
from .config import load_config

SEARCH_DEFAULT_K: Final = 10
SEARCH_MAX_K: Final = 25
SKILL_COMPARISON_MAX: Final = 100
ARTIFACT_ENV: Final = "JOB_INTELLIGENCE_ARTIFACT_DIR"

OUTCOME_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "median_wage": "median_annual_wage_2024",
        "projected_growth": "employment_change_pct",
        "annual_openings": "annual_openings_2024_34",
    }
)


class ServiceInputError(ValueError):
    """Raised when a caller supplies an invalid query, code, or filter."""


class OccupationNotFoundError(ServiceInputError):
    """Raised when an O*NET-SOC code is not present in the frozen catalog."""


@dataclass(frozen=True, slots=True)
class Occupation:
    onet_soc_code: str
    base_soc_code: str
    soc_major_group: str
    canonical_title: str
    description: str
    job_zone: int | None
    core_task_count: int
    alias_count: int
    structured_observed_count: int


@dataclass(frozen=True, slots=True)
class MarketContext:
    onet_soc_code: str
    nem_code: str
    employment_2024: int | None
    employment_2034: int | None
    employment_change_2024_34: int | None
    employment_change_pct: float | None
    annual_openings_2024_34: int | None
    median_annual_wage_2024: int | None
    typical_education: str | None
    work_experience: str | None
    on_the_job_training: str | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    onet_soc_code: str
    canonical_title: str
    description: str
    similarity: float


@dataclass(frozen=True, slots=True)
class NeighborResult:
    rank: int
    onet_soc_code: str
    canonical_title: str
    description: str
    total_score: float
    lexical_score: float
    lsa_score: float
    structured_score: float | None
    development_distance: float | None
    explanation: str


@dataclass(frozen=True, slots=True)
class SkillDifference:
    feature_family: str
    element_id: str
    element_name: str
    scale_id: str
    scale_name: str
    left_value: float | None
    right_value: float | None
    difference: float | None
    left_metadata: SkillRatingMetadata | None
    right_metadata: SkillRatingMetadata | None


@dataclass(frozen=True, slots=True)
class SkillRatingMetadata:
    sample_size: int | None
    standard_error: float | None
    lower_ci_bound: float | None
    upper_ci_bound: float | None
    rating_date: date | None
    domain_source: str | None


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    onet_soc_code: str
    canonical_title: str
    nem_code: str
    development_distance: float
    outcome: str
    outcome_value: float
    employment_change_pct: float | None
    annual_openings_2024_34: int | None
    median_annual_wage_2024: int | None
    typical_education: str | None
    work_experience: str | None
    on_the_job_training: str | None


@dataclass(frozen=True, slots=True)
class ClusterMember:
    onet_soc_code: str
    canonical_title: str
    cluster_id: int
    cluster_label: str
    pca_x: float
    pca_y: float


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    cluster_id: int
    cluster_label: str
    occupation_count: int


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    artifact_version: str
    snapshot_date: str
    title_candidate_id: str
    transition_candidate_id: str
    repository_commit: str
    project_config_sha256: str
    source_manifest_sha256: str
    split_manifest_sha256: str
    locked_rankings_sha256: str
    locked_metrics_sha256: str
    environment_lock_sha256: str
    processed_manifest_sha256: str
    feature_manifest_sha256: str
    metric_contract_sha256: str
    candidate_manifest_sha256: str
    evaluation_universe_sha256: str
    implementation_manifest_sha256: str
    title_candidate_grid_sha256: str
    transition_candidate_grid_sha256: str
    fit_phase: str | None
    fit_codes_sha256: str | None
    fit_code_count: int | None
    fit_fold_manifest_sha256: str | None
    evidence_status: str
    source_hashes: tuple[tuple[str, str], ...]
    serving_file_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ServiceState:
    available: bool
    service: OccupationIntelligenceService | None
    reason_code: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class _SkillRating:
    feature_family: str
    element_id: str
    element_name: str
    scale_id: str
    scale_name: str
    value: float | None
    metadata: SkillRatingMetadata


class OccupationIntelligenceService:
    """Read-only operations over one completely validated serving bundle."""

    __slots__ = (
        "_bundle",
        "_catalog_codes",
        "_clusters",
        "_market_by_code",
        "_metadata",
        "_neighbors",
        "_occupations",
        "_skills",
        "_locked",
    )

    def __init__(self, bundle: LoadedArtifactBundle):
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "_bundle", bundle)

        occupations: dict[str, Occupation] = {}
        for row in bundle.tables["occupations"].to_pylist():
            occupation = Occupation(
                onet_soc_code=row["onet_soc_code"],
                base_soc_code=row["base_soc_code"],
                soc_major_group=row["soc_major_group"],
                canonical_title=row["canonical_title"],
                description=row["description"],
                job_zone=row["job_zone"],
                core_task_count=row["core_task_count"],
                alias_count=row["alias_count"],
                structured_observed_count=row["structured_observed_count"],
            )
            occupations[occupation.onet_soc_code] = occupation
        object.__setattr__(self, "_occupations", MappingProxyType(occupations))

        catalog_codes = tuple(
            row["onet_soc_code"] for row in bundle.tables["search_catalog"].to_pylist()
        )
        object.__setattr__(self, "_catalog_codes", catalog_codes)

        market_by_code: defaultdict[str, list[MarketContext]] = defaultdict(list)
        for row in bundle.tables["occupation_market"].to_pylist():
            context = MarketContext(**row)
            market_by_code[context.onet_soc_code].append(context)
        object.__setattr__(
            self,
            "_market_by_code",
            MappingProxyType(
                {
                    code: tuple(sorted(values, key=lambda value: value.nem_code))
                    for code, values in market_by_code.items()
                }
            ),
        )

        neighbors: defaultdict[str, list[NeighborResult]] = defaultdict(list)
        for row in bundle.tables["neighbors"].to_pylist():
            destination = occupations[row["destination_code"]]
            neighbors[row["query_code"]].append(
                NeighborResult(
                    rank=row["rank"],
                    onet_soc_code=destination.onet_soc_code,
                    canonical_title=destination.canonical_title,
                    description=destination.description,
                    total_score=row["total_score"],
                    lexical_score=row["lexical_score"],
                    lsa_score=row["lsa_score"],
                    structured_score=row["structured_score"],
                    development_distance=row["development_distance"],
                    explanation=row["explanation"],
                )
            )
        object.__setattr__(
            self,
            "_neighbors",
            MappingProxyType({code: tuple(values) for code, values in neighbors.items()}),
        )

        skills: defaultdict[str, dict[tuple[str, str, str], _SkillRating]] = defaultdict(dict)
        for row in bundle.tables["skill_ratings"].to_pylist():
            key = (row["feature_family"], row["element_id"], row["scale_id"])
            skills[row["onet_soc_code"]][key] = _SkillRating(
                feature_family=row["feature_family"],
                element_id=row["element_id"],
                element_name=row["element_name"],
                scale_id=row["scale_id"],
                scale_name=row["scale_name"],
                value=row["normalized_value"] if row["observed"] else None,
                metadata=SkillRatingMetadata(
                    sample_size=row["sample_size"],
                    standard_error=row["standard_error"],
                    lower_ci_bound=row["lower_ci_bound"],
                    upper_ci_bound=row["upper_ci_bound"],
                    rating_date=row["rating_date"],
                    domain_source=row["domain_source"],
                ),
            )
        object.__setattr__(
            self,
            "_skills",
            MappingProxyType({code: MappingProxyType(ratings) for code, ratings in skills.items()}),
        )

        clusters: list[ClusterMember] = []
        for row in bundle.tables["clusters"].to_pylist():
            occupation = occupations[row["onet_soc_code"]]
            clusters.append(
                ClusterMember(
                    onet_soc_code=occupation.onet_soc_code,
                    canonical_title=occupation.canonical_title,
                    cluster_id=row["cluster_id"],
                    cluster_label=row["cluster_label"],
                    pca_x=row["pca_x"],
                    pca_y=row["pca_y"],
                )
            )
        object.__setattr__(
            self,
            "_clusters",
            tuple(sorted(clusters, key=lambda item: (item.cluster_id, item.onet_soc_code))),
        )

        manifest = bundle.manifest
        provenance = manifest["provenance"]
        object.__setattr__(
            self,
            "_metadata",
            ArtifactMetadata(
                artifact_version=manifest["artifact_version"],
                snapshot_date=manifest["snapshot_date"],
                title_candidate_id=manifest["models"]["title_candidate_id"],
                transition_candidate_id=manifest["models"]["transition_candidate_id"],
                repository_commit=provenance["repository_commit"],
                project_config_sha256=provenance["project_config_sha256"],
                source_manifest_sha256=provenance["source_manifest_sha256"],
                split_manifest_sha256=provenance["split_manifest_sha256"],
                locked_rankings_sha256=provenance["locked_rankings_sha256"],
                locked_metrics_sha256=provenance["locked_metrics_sha256"],
                environment_lock_sha256=provenance["environment_lock_sha256"],
                processed_manifest_sha256=provenance["processed_manifest_sha256"],
                feature_manifest_sha256=provenance["feature_manifest_sha256"],
                metric_contract_sha256=provenance["metric_contract_sha256"],
                candidate_manifest_sha256=provenance["candidate_manifest_sha256"],
                evaluation_universe_sha256=provenance["evaluation_universe_sha256"],
                implementation_manifest_sha256=provenance["implementation_manifest_sha256"],
                title_candidate_grid_sha256=provenance["title_candidate_grid_sha256"],
                transition_candidate_grid_sha256=provenance["transition_candidate_grid_sha256"],
                fit_phase=manifest["models"].get("fit_phase"),
                fit_codes_sha256=manifest["models"].get("fit_codes_sha256"),
                fit_code_count=manifest["models"].get("fit_code_count"),
                fit_fold_manifest_sha256=manifest["models"].get("fit_fold_manifest_sha256"),
                evidence_status=manifest["evidence"]["status"],
                source_hashes=tuple(sorted(provenance["source_hashes"].items())),
                serving_file_hashes=tuple(
                    sorted(
                        (logical_name, entry["sha256"])
                        for logical_name, entry in manifest["files"].items()
                    )
                ),
            ),
        )
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("OccupationIntelligenceService is immutable")
        object.__setattr__(self, name, value)

    @property
    def metadata(self) -> ArtifactMetadata:
        return self._metadata

    @property
    def occupation_count(self) -> int:
        return len(self._occupations)

    @property
    def evidence_reports(self) -> Mapping[str, Mapping[str, Any]]:
        """Return checksum-bound, schema-validated release evidence."""

        return self._bundle.evidence_reports

    def occupation(self, onet_soc_code: str) -> Occupation:
        code = _validated_code(onet_soc_code)
        try:
            return self._occupations[code]
        except KeyError as exc:
            raise OccupationNotFoundError(f"unknown O*NET-SOC code: {code}") from exc

    def catalog(self) -> tuple[Occupation, ...]:
        """Return the frozen occupation catalog in O*NET-SOC code order."""

        return tuple(self._occupations[code] for code in self._catalog_codes)

    def market_context(self, onet_soc_code: str) -> tuple[MarketContext, ...]:
        occupation = self.occupation(onet_soc_code)
        return self._market_by_code.get(occupation.onet_soc_code, ())

    def search(self, query: str, *, k: int = SEARCH_DEFAULT_K) -> tuple[SearchResult, ...]:
        query = _validated_query(query)
        k = _validated_limit(k, maximum=SEARCH_MAX_K)
        try:
            transformed = self._bundle.search_transformer.transform([query])
        except Exception as exc:
            raise ServiceInputError(
                "the frozen title model could not transform this query"
            ) from exc
        query_vector = sparse.csr_matrix(transformed, dtype=np.float64)
        if query_vector.shape != (1, self._bundle.search_matrix.shape[1]):
            raise RuntimeError("validated title transformer changed output dimensionality")
        norm = float(sparse.linalg.norm(query_vector))
        if not np.isfinite(norm) or norm == 0:
            return ()
        query_vector = query_vector / norm
        raw_scores = self._bundle.search_matrix @ query_vector.T
        scores = (
            raw_scores.toarray().reshape(-1)
            if sparse.issparse(raw_scores)
            else np.asarray(raw_scores).reshape(-1)
        )
        if not np.isfinite(scores).all():
            raise RuntimeError("validated title index produced non-finite similarities")
        ranked_rows = np.lexsort((np.asarray(self._catalog_codes), -scores))
        order = tuple(row for row in ranked_rows if scores[row] > 0)[:k]
        results: list[SearchResult] = []
        for rank, row_index in enumerate(order, start=1):
            code = self._catalog_codes[int(row_index)]
            occupation = self._occupations[code]
            results.append(
                SearchResult(
                    rank=rank,
                    onet_soc_code=code,
                    canonical_title=occupation.canonical_title,
                    description=occupation.description,
                    similarity=float(scores[row_index]),
                )
            )
        return tuple(results)

    def neighbors(self, onet_soc_code: str, *, k: int = 10) -> tuple[NeighborResult, ...]:
        occupation = self.occupation(onet_soc_code)
        maximum = self._bundle.manifest["models"]["neighbor_display_max"]
        k = _validated_limit(k, maximum=maximum)
        return self._neighbors[occupation.onet_soc_code][:k]

    def compare_skills(
        self,
        left_code: str,
        right_code: str,
        *,
        feature_family: str = "transferable_skill",
        limit: int = 25,
    ) -> tuple[SkillDifference, ...]:
        left = self.occupation(left_code)
        right = self.occupation(right_code)
        family = _validated_token(feature_family, "feature_family", maximum_length=80)
        limit = _validated_limit(limit, maximum=SKILL_COMPARISON_MAX)
        left_ratings = self._skills.get(left.onet_soc_code, {})
        right_ratings = self._skills.get(right.onet_soc_code, {})
        keys = sorted(
            {
                key
                for key in set(left_ratings) | set(right_ratings)
                if key[0].casefold() == family.casefold()
            }
        )
        differences: list[SkillDifference] = []
        for key in keys:
            left_rating = left_ratings.get(key)
            right_rating = right_ratings.get(key)
            reference = left_rating or right_rating
            assert reference is not None
            left_value = left_rating.value if left_rating else None
            right_value = right_rating.value if right_rating else None
            difference = (
                right_value - left_value
                if left_value is not None and right_value is not None
                else None
            )
            differences.append(
                SkillDifference(
                    feature_family=reference.feature_family,
                    element_id=reference.element_id,
                    element_name=reference.element_name,
                    scale_id=reference.scale_id,
                    scale_name=reference.scale_name,
                    left_value=left_value,
                    right_value=right_value,
                    difference=difference,
                    left_metadata=left_rating.metadata if left_rating else None,
                    right_metadata=right_rating.metadata if right_rating else None,
                )
            )
        differences.sort(
            key=lambda item: (
                item.difference is None,
                -(abs(item.difference) if item.difference is not None else 0.0),
                item.element_name.casefold(),
                item.scale_id,
            )
        )
        return tuple(differences[:limit])

    def frontier(
        self,
        origin_code: str,
        *,
        outcome: str = "median_wage",
        max_development_distance: float | None = None,
        minimum_outcome: float | None = None,
        maximum_job_zone: int | None = None,
    ) -> tuple[FrontierPoint, ...]:
        origin = self.occupation(origin_code)
        if outcome not in OUTCOME_COLUMNS:
            allowed = ", ".join(OUTCOME_COLUMNS)
            raise ServiceInputError(f"outcome must be one of: {allowed}")
        outcome_column = OUTCOME_COLUMNS[outcome]
        if max_development_distance is not None:
            max_development_distance = _validated_nonnegative_number(
                max_development_distance, "max_development_distance"
            )
        if minimum_outcome is not None:
            minimum_outcome = _validated_number(minimum_outcome, "minimum_outcome")
        if maximum_job_zone is not None and (
            isinstance(maximum_job_zone, bool)
            or not isinstance(maximum_job_zone, int)
            or maximum_job_zone not in {2, 3, 4, 5}
        ):
            raise ServiceInputError(
                "maximum_job_zone must be 2 (combined Job Zone 1-2), 3, 4, or 5"
            )

        # One row per NEM occupation: shared national figures are never summed
        # across O*NET specialties.  Lowest development distance wins a tie.
        by_nem: dict[str, FrontierPoint] = {}
        for neighbor in self._neighbors[origin.onet_soc_code]:
            # Missing structured overlap means the development objective is
            # undefined; protocol forbids filling that gap with an imputation.
            if neighbor.development_distance is None:
                continue
            if (
                max_development_distance is not None
                and neighbor.development_distance > max_development_distance
            ):
                continue
            destination = self._occupations[neighbor.onet_soc_code]
            if maximum_job_zone is not None and (
                destination.job_zone is None or destination.job_zone > maximum_job_zone
            ):
                continue
            for market in self._market_by_code.get(destination.onet_soc_code, ()):
                raw_outcome = getattr(market, outcome_column)
                if raw_outcome is None:
                    continue
                outcome_value = float(raw_outcome)
                if minimum_outcome is not None and outcome_value < minimum_outcome:
                    continue
                point = FrontierPoint(
                    onet_soc_code=destination.onet_soc_code,
                    canonical_title=destination.canonical_title,
                    nem_code=market.nem_code,
                    development_distance=neighbor.development_distance,
                    outcome=outcome,
                    outcome_value=outcome_value,
                    employment_change_pct=market.employment_change_pct,
                    annual_openings_2024_34=market.annual_openings_2024_34,
                    median_annual_wage_2024=market.median_annual_wage_2024,
                    typical_education=market.typical_education,
                    work_experience=market.work_experience,
                    on_the_job_training=market.on_the_job_training,
                )
                existing = by_nem.get(market.nem_code)
                if existing is None or (
                    point.development_distance,
                    point.onet_soc_code,
                ) < (
                    existing.development_distance,
                    existing.onet_soc_code,
                ):
                    by_nem[market.nem_code] = point

        candidates = tuple(by_nem.values())
        efficient = [
            candidate
            for candidate in candidates
            if not any(
                other.nem_code != candidate.nem_code
                and other.development_distance <= candidate.development_distance
                and other.outcome_value >= candidate.outcome_value
                and (
                    other.development_distance < candidate.development_distance
                    or other.outcome_value > candidate.outcome_value
                )
                for other in candidates
            )
        ]
        efficient.sort(
            key=lambda item: (
                item.development_distance,
                -item.outcome_value,
                item.nem_code,
                item.onet_soc_code,
            )
        )
        return tuple(efficient)

    def clusters(self, *, cluster_id: int | None = None) -> tuple[ClusterMember, ...]:
        if cluster_id is None:
            return self._clusters
        if isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 0:
            raise ServiceInputError("cluster_id must be a non-negative integer")
        return tuple(member for member in self._clusters if member.cluster_id == cluster_id)

    def cluster_summaries(self) -> tuple[ClusterSummary, ...]:
        labels: dict[int, str] = {}
        counts: defaultdict[int, int] = defaultdict(int)
        for member in self._clusters:
            existing = labels.setdefault(member.cluster_id, member.cluster_label)
            if existing != member.cluster_label:
                raise RuntimeError("validated cluster labels are inconsistent")
            counts[member.cluster_id] += 1
        return tuple(
            ClusterSummary(
                cluster_id=cluster_id,
                cluster_label=labels[cluster_id],
                occupation_count=counts[cluster_id],
            )
            for cluster_id in sorted(labels)
        )

    def smoke_test(self) -> tuple[tuple[str, int], ...]:
        """Run target-free non-emptiness checks supplied by the release bundle."""

        results: list[tuple[str, int]] = []
        for query_id, query in self._bundle.smoke_queries:
            match_count = len(self.search(query, k=1))
            if match_count == 0:
                raise RuntimeError(f"target-free smoke query produced no results: {query_id}")
            results.append((query_id, match_count))
        return tuple(results)


def _validated_query(query: str) -> str:
    if not isinstance(query, str):
        raise ServiceInputError("query must be text")
    query = query.strip()
    if not query:
        raise ServiceInputError("query cannot be blank")
    if len(query) > 512:
        raise ServiceInputError("query cannot exceed 512 characters")
    return query


def _validated_code(code: str) -> str:
    return _validated_token(code, "onet_soc_code", maximum_length=32)


def _validated_token(value: str, name: str, *, maximum_length: int) -> str:
    if not isinstance(value, str):
        raise ServiceInputError(f"{name} must be text")
    value = value.strip()
    if not value:
        raise ServiceInputError(f"{name} cannot be blank")
    if len(value) > maximum_length:
        raise ServiceInputError(f"{name} cannot exceed {maximum_length} characters")
    return value


def _validated_limit(value: int, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ServiceInputError(f"limit must be an integer from 1 through {maximum}")
    return value


def _validated_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceInputError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value):
        raise ServiceInputError(f"{name} must be a finite number")
    return value


def _validated_nonnegative_number(value: float, name: str) -> float:
    value = _validated_number(value, name)
    if value < 0:
        raise ServiceInputError(f"{name} cannot be negative")
    return value


def default_artifact_path() -> Path:
    configured = os.environ.get(ARTIFACT_ENV)
    if configured:
        return Path(configured)
    project_root = os.environ.get("JOB_MARKET_PROJECT_ROOT")
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "project.toml"
    version = load_config(config_path).project.artifact_version if config_path.is_file() else "v2"
    return root / "artifacts" / version


@lru_cache(maxsize=8)
def _cached_service(bundle_path: str) -> OccupationIntelligenceService:
    return OccupationIntelligenceService(load_artifact_bundle(bundle_path))


def get_service(bundle_path: str | Path | None = None) -> OccupationIntelligenceService:
    """Return the one cached immutable service for a resolved artifact path."""

    path = Path(bundle_path) if bundle_path is not None else default_artifact_path()
    return _cached_service(str(path.expanduser().resolve(strict=False)))


def get_service_state(bundle_path: str | Path | None = None) -> ServiceState:
    """Describe artifact availability without making clients crash at import time."""

    try:
        return ServiceState(
            available=True,
            service=get_service(bundle_path),
            reason_code=None,
            reason=None,
        )
    except ArtifactError as exc:
        if isinstance(exc, ArtifactNotFoundError):
            reason_code = "artifact_missing"
        elif isinstance(exc, ArtifactIntegrityError):
            reason_code = "artifact_integrity_failed"
        elif isinstance(exc, ArtifactSafetyError):
            reason_code = "artifact_safety_rejected"
        elif isinstance(exc, (ArtifactSchemaError, ManifestValidationError)):
            reason_code = "artifact_schema_invalid"
        else:
            reason_code = "artifact_unavailable"
        return ServiceState(
            available=False,
            service=None,
            reason_code=reason_code,
            reason=str(exc),
        )


def clear_service_cache() -> None:
    """Clear only the process-local cache (primarily useful to isolated tests)."""

    _cached_service.cache_clear()
