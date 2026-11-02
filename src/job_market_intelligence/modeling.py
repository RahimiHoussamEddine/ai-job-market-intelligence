"""Frozen candidate grids, leakage gates, and one-SE model selection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .embeddings import TextLSAIndex
from .lexical import TfidfIndex, WeightedTextTransformer, cosine_scores
from .ranking_metrics import query_stratified_standard_error
from .retrieval import build_title_documents, simplex_weight_grid
from .skills import StructuredSimilarityIndex

PREDECLARED_LSA_DIMENSIONS: Final = (64, 128, 256)
PREDECLARED_WEIGHT_STEP: Final = 0.25
# A champion without an LSA component still needs one frozen representation
# for clustering. This target-free work-text fallback is fit on the same
# non-test occupations as locked evaluation and only then transforms the full
# serving catalog. PCA remains display-only.
CLUSTER_FALLBACK_LSA_DIMENSIONS: Final = 128
OPERATIONAL_COST_POLICY: Final = (
    "v1_deterministic_score_cells_x_active_components_plus_lsa_dimensions"
)
OPERATIONAL_COST_FORMULA: Final = (
    "score_cells * max(1, positive_weight_component_count) + lsa_dimensions * query_count"
)
STRUCTURED_ABLATION_REASON: Final = (
    "pure structured retrieval cannot guarantee full-catalog coverage without imputation"
)
SELECTION_COLUMNS: Final = (
    "candidate_id",
    "query_id",
    "primary_value",
    "soc_major_group",
    "latent_dimensions",
    "artifact_bytes",
    "operational_cost_units",
    "latency_ms",
    "deployment_eligible",
    "ineligibility_reason",
)

_FORBIDDEN_FEATURE_PATTERNS: Final = (
    re.compile(r"(?:^|_)qrels?(?:_|$)"),
    re.compile(r"(?:^|_)related(?:ness|_occupation)?(?:_|$)"),
    re.compile(r"(?:^|_)relevance(?:_|$)"),
    re.compile(r"(?:^|_)(?:judgment|label|target)(?:_|$)"),
    re.compile(r"(?:^|_)nem(?:_|$)"),
    re.compile(r"(?:^|_)(?:employment|wage|salary|openings|growth)(?:_|$)"),
    re.compile(r"(?:^|_)(?:education|experience|training)(?:_|$)"),
    re.compile(r"(?:^|_)(?:base_soc|soc_major|job_zone|fold|split_role)(?:_|$)"),
)


class ModelingError(ValueError):
    """Raised when candidates, feature schemas, or selection evidence drift."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    task: str
    representation: str
    analyzer: str | None
    lsa_dimensions: int
    weights: tuple[tuple[str, float], ...]

    @property
    def weight_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.weights))


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected_candidate_id: str
    metric_leader_id: str
    leader_estimate: float
    leader_bootstrap_standard_error: float
    one_se_threshold: float
    eligible_candidate_ids: tuple[str, ...]
    candidate_estimates: tuple[tuple[str, float], ...]
    candidate_tie_breaks: tuple[tuple[str, int, int, int], ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    unrestricted_metric_leader_id: str
    deployment_ineligible_candidates: tuple[tuple[str, str], ...]
    operational_cost_policy: str


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    proposed_candidate_id: str
    baseline_candidate_id: str
    release_candidate_id: str
    proposed_estimate: float
    baseline_estimate: float
    fallback_applied: bool


@dataclass(frozen=True, slots=True)
class TitleServingIndex:
    """Production-refit raw-title transformer and aligned catalog matrix."""

    candidate_id: str
    occupation_codes: tuple[str, ...]
    transformer: Any
    matrix: sparse.csr_matrix


@dataclass(frozen=True, slots=True)
class TransitionServingComponents:
    """Transparent, aligned production transition components."""

    candidate_id: str
    occupation_codes: tuple[str, ...]
    lexical_scores: np.ndarray
    lsa_scores: np.ndarray
    structured_scores: np.ndarray
    development_distances: np.ndarray
    weights: Mapping[str, float]
    cluster_embedding: np.ndarray
    term_matrix: sparse.csr_matrix
    feature_names: tuple[str, ...]


def _candidate_id(prefix: str, weights: Mapping[str, float]) -> str:
    encoded = "-".join(f"{name}{int(round(weight * 100)):03d}" for name, weight in weights.items())
    return f"{prefix}__{encoded}"


def title_candidate_grid(
    *, lsa_dimensions: Sequence[int] = PREDECLARED_LSA_DIMENSIONS
) -> tuple[CandidateSpec, ...]:
    """Return the preregistered alternate-title candidate grid."""

    dimensions = tuple(int(value) for value in lsa_dimensions)
    if dimensions != PREDECLARED_LSA_DIMENSIONS:
        raise ModelingError(f"title LSA dimensions must remain {PREDECLARED_LSA_DIMENSIONS!r}")
    candidates = [
        CandidateSpec("title_word", "title", "canonical_title", "word", 0, (("lexical", 1.0),)),
        CandidateSpec("title_char", "title", "canonical_title", "char", 0, (("lexical", 1.0),)),
        CandidateSpec(
            "title_word_char",
            "title",
            "canonical_title",
            "word_char",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec("profile_word", "title", "title_profile", "word", 0, (("lexical", 1.0),)),
    ]
    for dimension in dimensions:
        candidates.append(
            CandidateSpec(
                f"title_lsa_{dimension}",
                "title",
                "title_lsa",
                "word_char",
                dimension,
                (("lsa", 1.0),),
            )
        )
        for lexical_weight in (0.25, 0.5, 0.75):
            weights = {"lexical": lexical_weight, "lsa": 1.0 - lexical_weight}
            candidates.append(
                CandidateSpec(
                    _candidate_id(f"title_hybrid_lsa{dimension}", weights),
                    "title",
                    "title_hybrid",
                    "word_char",
                    dimension,
                    tuple(weights.items()),
                )
            )
    return tuple(candidates)


def transition_candidate_grid(
    *, lsa_dimensions: Sequence[int] = PREDECLARED_LSA_DIMENSIONS
) -> tuple[CandidateSpec, ...]:
    """Return lexical, LSA, structured, and fixed-grid transition candidates."""

    dimensions = tuple(int(value) for value in lsa_dimensions)
    if dimensions != PREDECLARED_LSA_DIMENSIONS:
        raise ModelingError(f"transition LSA dimensions must remain {PREDECLARED_LSA_DIMENSIONS!r}")
    candidates: list[CandidateSpec] = [
        CandidateSpec(
            "transition_title_word",
            "transition",
            "canonical_title",
            "word",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_title_char",
            "transition",
            "canonical_title",
            "char",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_title_word_char",
            "transition",
            "canonical_title",
            "word_char",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_work_word",
            "transition",
            "work_profile",
            "word",
            0,
            (("lexical", 1.0),),
        ),
        CandidateSpec(
            "transition_structured",
            "transition",
            "structured",
            None,
            0,
            (("structured", 1.0),),
        ),
    ]
    seen = {candidate.candidate_id for candidate in candidates}
    for dimension in dimensions:
        lsa = CandidateSpec(
            f"transition_work_lsa_{dimension}",
            "transition",
            "work_lsa",
            "word",
            dimension,
            (("lsa", 1.0),),
        )
        candidates.append(lsa)
        seen.add(lsa.candidate_id)
        for weights in simplex_weight_grid(("lexical", "lsa", "structured"), step=0.25):
            # Pure components are already present and do not need dimension-tagged duplicates.
            if sum(weight > 0 for weight in weights.values()) < 2:
                continue
            candidate_id = _candidate_id(f"transition_hybrid_lsa{dimension}", weights)
            if candidate_id in seen:  # defensive if grid generation ever changes
                continue
            seen.add(candidate_id)
            candidates.append(
                CandidateSpec(
                    candidate_id,
                    "transition",
                    "work_hybrid",
                    "word",
                    dimension,
                    tuple(weights.items()),
                )
            )
    return tuple(candidates)


def candidate_grid_digest(candidates: Sequence[CandidateSpec]) -> str:
    """Hash the full ordered candidate definitions before evaluation."""

    if not candidates or len({candidate.candidate_id for candidate in candidates}) != len(
        candidates
    ):
        raise ModelingError("candidate grid must be nonempty and unique by candidate ID")
    payload = json.dumps(
        [asdict(candidate) for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deployment_feasibility(candidate: CandidateSpec) -> tuple[bool, str]:
    """Apply the pre-metric full-catalog serving feasibility rule."""

    if candidate.task == "title":
        return True, ""
    if candidate.task != "transition":
        raise ModelingError(f"unknown candidate task: {candidate.task!r}")
    weights = candidate.weight_map
    if float(weights.get("lexical", 0.0)) > 0 or float(weights.get("lsa", 0.0)) > 0:
        return True, ""
    return False, STRUCTURED_ABLATION_REASON


def assert_feature_columns_safe(
    columns: Iterable[object], *, context: str = "feature schema"
) -> None:
    """Reject evaluation judgments, SOC shortcuts, and BLS context as features."""

    normalized = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().casefold()).strip("_") for column in columns
    ]
    offending = sorted(
        {
            name
            for name in normalized
            if any(pattern.search(name) is not None for pattern in _FORBIDDEN_FEATURE_PATTERNS)
        }
    )
    if offending:
        raise ModelingError(f"{context} contains forbidden target/context fields: {offending}")


def assert_artifact_payload_target_free(value: Any, *, context: str = "artifact") -> None:
    """Recursively inspect field names without scanning ordinary title prose."""

    visited: set[int] = set()

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 12 or id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, Mapping):
            assert_feature_columns_safe(item.keys(), context=path)
            for key, child in item.items():
                visit(child, f"{path}.{key}", depth + 1)
        elif isinstance(item, (list, tuple, set)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
        elif isinstance(item, pd.DataFrame):
            assert_feature_columns_safe(item.columns, context=path)

    visit(value, context, 0)


def _validated_selection_results(results: pd.DataFrame, *, required_role: str) -> pd.DataFrame:
    missing = sorted(set(SELECTION_COLUMNS) - set(results.columns))
    if missing:
        raise ModelingError(f"selection results are missing columns: {missing}")
    work = results.loc[
        :, list(SELECTION_COLUMNS) + (["split_role"] if "split_role" in results else [])
    ].copy()
    if work.empty or bool(work.duplicated(["candidate_id", "query_id"]).any()):
        raise ModelingError("candidate results must be nonempty and unique by candidate/query")
    for column in ("candidate_id", "query_id", "soc_major_group"):
        work[column] = work[column].astype("string").str.strip()
        if bool(work[column].isna().any() | work[column].eq("").any()):
            raise ModelingError(f"candidate results {column} contains blank values")
    for column in ("primary_value", "latency_ms"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if bool(work[column].isna().any() | ~np.isfinite(work[column]).all()):
            raise ModelingError(f"candidate results {column} must be finite")
    for column in ("latent_dimensions", "artifact_bytes", "operational_cost_units"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if bool(
            work[column].isna().any() or work[column].lt(0).any() or work[column].mod(1).ne(0).any()
        ):
            raise ModelingError(f"candidate results {column} must be nonnegative integers")
        work[column] = work[column].astype(int)
    if bool(work["latency_ms"].lt(0).any()):
        raise ModelingError("candidate latency cannot be negative")
    if not work["deployment_eligible"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ModelingError("deployment_eligible must contain booleans")
    work["deployment_eligible"] = work["deployment_eligible"].astype(bool)
    work["ineligibility_reason"] = work["ineligibility_reason"].astype("string").fillna("")
    invalid_reason = (work["deployment_eligible"] & work["ineligibility_reason"].ne("")) | (
        ~work["deployment_eligible"] & work["ineligibility_reason"].str.strip().eq("")
    )
    if bool(invalid_reason.any()):
        raise ModelingError("deployment eligibility reasons are inconsistent")
    if "split_role" in work and set(work["split_role"].astype(str)) != {required_role}:
        raise ModelingError(f"candidate results must contain only {required_role!r} rows")

    expected_queries: tuple[str, ...] | None = None
    expected_strata: dict[str, str] | None = None
    for candidate_id, group in work.groupby("candidate_id", sort=True):
        query_ids = tuple(sorted(group["query_id"].astype(str)))
        strata = dict(
            zip(group["query_id"].astype(str), group["soc_major_group"].astype(str), strict=True)
        )
        if expected_queries is None:
            expected_queries = query_ids
            expected_strata = strata
        elif query_ids != expected_queries or strata != expected_strata:
            raise ModelingError(
                f"candidate {candidate_id} was not evaluated on the identical aligned query set"
            )
        for column in (
            "latent_dimensions",
            "artifact_bytes",
            "operational_cost_units",
            "latency_ms",
            "deployment_eligible",
            "ineligibility_reason",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ModelingError(f"candidate {candidate_id} has inconsistent {column}")
    return work


def select_candidate_one_se(
    selection_results: pd.DataFrame,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260814,
) -> SelectionDecision:
    """Apply the frozen primary-metric and operational one-SE tie-break."""

    work = _validated_selection_results(selection_results, required_role="selection")
    estimates = (
        work.groupby("candidate_id", sort=True)["primary_value"].mean().astype(float).to_dict()
    )
    unrestricted_leader_id = sorted(
        estimates, key=lambda candidate: (-estimates[candidate], candidate)
    )[0]
    eligibility = (
        work.groupby("candidate_id", sort=True)
        .agg(
            deployment_eligible=("deployment_eligible", "first"),
            ineligibility_reason=("ineligibility_reason", "first"),
        )
        .to_dict("index")
    )
    deployable = sorted(
        candidate for candidate in estimates if bool(eligibility[candidate]["deployment_eligible"])
    )
    if not deployable:
        raise ModelingError("candidate results contain no deployment-eligible candidate")
    leader_id = sorted(
        deployable,
        key=lambda candidate: (-estimates[candidate], candidate),
    )[0]
    leader_rows = work.loc[work["candidate_id"].eq(leader_id)].sort_values(
        "query_id", kind="stable"
    )
    standard_error = query_stratified_standard_error(
        leader_rows["primary_value"],
        leader_rows["soc_major_group"],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    threshold = estimates[leader_id] - standard_error
    eligible = tuple(
        sorted(candidate for candidate in deployable if estimates[candidate] >= threshold - 1e-15)
    )
    metadata = (
        work.groupby("candidate_id", sort=True)
        .agg(
            latent_dimensions=("latent_dimensions", "first"),
            artifact_bytes=("artifact_bytes", "first"),
            operational_cost_units=("operational_cost_units", "first"),
        )
        .to_dict("index")
    )
    selected = sorted(
        eligible,
        key=lambda candidate: (
            int(metadata[candidate]["latent_dimensions"]),
            int(metadata[candidate]["artifact_bytes"]),
            int(metadata[candidate]["operational_cost_units"]),
            candidate,
        ),
    )[0]
    return SelectionDecision(
        selected_candidate_id=selected,
        metric_leader_id=leader_id,
        leader_estimate=float(estimates[leader_id]),
        leader_bootstrap_standard_error=float(standard_error),
        one_se_threshold=float(threshold),
        eligible_candidate_ids=eligible,
        candidate_estimates=tuple(sorted((key, float(value)) for key, value in estimates.items())),
        candidate_tie_breaks=tuple(
            sorted(
                (
                    candidate,
                    int(metadata[candidate]["latent_dimensions"]),
                    int(metadata[candidate]["artifact_bytes"]),
                    int(metadata[candidate]["operational_cost_units"]),
                )
                for candidate in estimates
            )
        ),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        unrestricted_metric_leader_id=unrestricted_leader_id,
        deployment_ineligible_candidates=tuple(
            sorted(
                (
                    candidate,
                    str(values["ineligibility_reason"]),
                )
                for candidate, values in eligibility.items()
                if not bool(values["deployment_eligible"])
            )
        ),
        operational_cost_policy=OPERATIONAL_COST_POLICY,
    )


def apply_confirmation_gate(
    confirmation_results: pd.DataFrame,
    *,
    proposed_candidate_id: str,
    baseline_candidate_id: str,
) -> ConfirmationDecision:
    """Fall back to title lexical baseline if the frozen champion underperforms."""

    work = _validated_selection_results(confirmation_results, required_role="confirmation")
    present = set(work["candidate_id"].astype(str))
    required = {proposed_candidate_id, baseline_candidate_id}
    if not required <= present:
        raise ModelingError(f"confirmation results lack candidates: {sorted(required - present)}")
    restricted = work.loc[work["candidate_id"].isin(required)].copy()
    counts = restricted.groupby("query_id")["candidate_id"].nunique()
    if bool(counts.ne(len(required)).any()):
        raise ModelingError("confirmation candidates are not paired on every query")
    estimates = restricted.groupby("candidate_id")["primary_value"].mean().to_dict()
    fallback = float(estimates[proposed_candidate_id]) < float(estimates[baseline_candidate_id])
    return ConfirmationDecision(
        proposed_candidate_id=proposed_candidate_id,
        baseline_candidate_id=baseline_candidate_id,
        release_candidate_id=baseline_candidate_id if fallback else proposed_candidate_id,
        proposed_estimate=float(estimates[proposed_candidate_id]),
        baseline_estimate=float(estimates[baseline_candidate_id]),
        fallback_applied=fallback,
    )


def development_decision_payload(
    selection: SelectionDecision,
    confirmation: ConfirmationDecision,
    *,
    experiment_version: str,
    project_config_sha256: str,
    fold_manifest_sha256: str,
    candidate_grid_sha256: str,
) -> dict[str, Any]:
    """Build and self-hash the frozen decision consumed by locked evaluation."""

    digests = {
        "project_config_sha256": project_config_sha256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "candidate_grid_sha256": candidate_grid_sha256,
    }
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests.values()):
        raise ModelingError("development decision inputs must be lowercase SHA-256 digests")
    core = {
        "schema_version": 1,
        "experiment_version": str(experiment_version),
        "release_candidate_id": confirmation.release_candidate_id,
        "selection": asdict(selection),
        "confirmation": asdict(confirmation),
        **digests,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "decision_sha256": hashlib.sha256(encoded).hexdigest()}


def multi_task_development_decision_payload(
    *,
    title_selection: SelectionDecision,
    title_confirmation: ConfirmationDecision,
    transition_selection: SelectionDecision,
    transition_confirmation: ConfirmationDecision,
    experiment_version: str,
    project_config_sha256: str,
    fold_manifest_sha256: str,
    feature_manifest_sha256: str,
    metric_contract_sha256: str,
    environment_lock_sha256: str,
    processed_manifest_sha256: str,
    candidate_manifest_sha256: str,
    evaluation_universe_sha256: str,
    implementation_manifest_sha256: str,
    title_candidate_grid_sha256: str,
    transition_candidate_grid_sha256: str,
) -> dict[str, Any]:
    """Freeze the independent title and transition champions in one decision."""

    digests = {
        "project_config_sha256": project_config_sha256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "metric_contract_sha256": metric_contract_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "processed_manifest_sha256": processed_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "evaluation_universe_sha256": evaluation_universe_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "title_candidate_grid_sha256": title_candidate_grid_sha256,
        "transition_candidate_grid_sha256": transition_candidate_grid_sha256,
    }
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests.values()):
        raise ModelingError("multi-task decision inputs must be lowercase SHA-256 digests")
    if not experiment_version or experiment_version != experiment_version.strip():
        raise ModelingError("experiment_version must be trimmed and nonblank")
    releases = {
        "title": title_confirmation.release_candidate_id,
        "transition": transition_confirmation.release_candidate_id,
    }
    core = {
        "schema_version": 2,
        "experiment_version": experiment_version,
        "release_candidates": releases,
        "title_selection": asdict(title_selection),
        "title_confirmation": asdict(title_confirmation),
        "transition_selection": asdict(transition_selection),
        "transition_confirmation": asdict(transition_confirmation),
        **digests,
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**core, "decision_sha256": hashlib.sha256(encoded).hexdigest()}


def _production_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "onet_soc_code",
        "canonical_title",
        "title_text",
        "work_text",
    }
    if missing := sorted(required - set(profiles.columns)):
        raise ModelingError(f"production profiles are missing columns: {missing}")
    result = profiles.copy()
    result["onet_soc_code"] = result["onet_soc_code"].astype("string").str.strip()
    if (
        result.empty
        or bool(result["onet_soc_code"].isna().any())
        or bool(result["onet_soc_code"].eq("").any())
        or bool(result["onet_soc_code"].duplicated().any())
    ):
        raise ModelingError("production profiles must be nonempty and unique by code")
    return result.sort_values("onet_soc_code", kind="stable").reset_index(drop=True)


def _fit_mask(codes: Sequence[str], fit_codes: Sequence[str]) -> np.ndarray:
    catalog = tuple(str(code) for code in codes)
    fitting = tuple(str(code) for code in fit_codes)
    if not fitting or len(fitting) != len(set(fitting)):
        raise ModelingError("fit codes must be nonempty and unique")
    if unknown := set(fitting) - set(catalog):
        raise ModelingError(f"fit codes are outside the production catalog: {sorted(unknown)[:5]}")
    if set(fitting) == set(catalog):
        raise ModelingError("locked-evidence refit must exclude held-out test codes")
    fitting_set = set(fitting)
    return np.asarray([code in fitting_set for code in catalog], dtype=bool)


def _unit_float64_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    result = sparse.csr_matrix(normalize(result, norm="l2", axis=1), dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    norms = np.sqrt(np.asarray(result.multiply(result).sum(axis=1)).ravel())
    if not np.isfinite(result.data).all() or np.any(norms == 0):
        raise ModelingError("production text representation contains an empty/non-finite row")
    return result


def fit_production_title_index(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    candidate: CandidateSpec,
    *,
    fit_codes: Sequence[str],
    min_df: int = 2,
    random_seed: int = 20260814,
) -> TitleServingIndex:
    """Refit on non-test text, then transform every serving occupation."""

    if candidate.task != "title":
        raise ModelingError("production title index requires a title candidate")
    catalog = _production_profiles(profiles)
    codes = tuple(catalog["onet_soc_code"].astype(str))
    fit_mask = _fit_mask(codes, fit_codes)
    canonical_documents = catalog["canonical_title"].astype(str).tolist()
    title_documents = build_title_documents(
        catalog,
        aliases,
        allowed_alias_codes=fit_codes,
    ).documents.set_index("onet_soc_code")
    profile_documents = title_documents.loc[list(codes), "title_text"].astype(str).tolist()

    transformer: Any
    documents: list[str]
    if candidate.representation == "canonical_title":
        transformer = TfidfIndex(
            analyzer=str(candidate.analyzer),  # type: ignore[arg-type]
            min_df=min_df,
            dtype=np.float64,
        )
        documents = canonical_documents
        transformer.fit(np.asarray(documents, dtype=object)[fit_mask].tolist())
        matrix = transformer.transform(documents)
    elif candidate.representation == "title_profile":
        transformer = TfidfIndex(
            analyzer=str(candidate.analyzer),  # type: ignore[arg-type]
            min_df=min_df,
            dtype=np.float64,
        )
        documents = profile_documents
        transformer.fit(np.asarray(documents, dtype=object)[fit_mask].tolist())
        matrix = transformer.transform(documents)
    elif candidate.representation == "title_lsa":
        transformer = TextLSAIndex(
            candidate.lsa_dimensions,
            analyzer="word_char",
            min_df=min_df,
            random_state=random_seed,
        )
        documents = profile_documents
        transformer.fit(np.asarray(documents, dtype=object)[fit_mask].tolist())
        matrix = transformer.transform(documents)
    elif candidate.representation == "title_hybrid":
        lexical = TfidfIndex(analyzer="word_char", min_df=min_df, dtype=np.float64)
        latent = TextLSAIndex(
            candidate.lsa_dimensions,
            analyzer="word_char",
            min_df=min_df,
            random_state=random_seed,
        )
        fit_documents = np.asarray(profile_documents, dtype=object)[fit_mask].tolist()
        lexical.fit(fit_documents)
        latent.fit(fit_documents)
        weights = candidate.weight_map
        transformer = WeightedTextTransformer(
            (lexical, latent),
            (weights["lexical"], weights["lsa"]),
        )
        documents = profile_documents
        matrix = transformer.transform(documents)
    else:
        raise ModelingError(
            f"unsupported production title representation: {candidate.representation}"
        )
    aligned = _unit_float64_csr(matrix)
    if aligned.shape[0] != len(codes):  # pragma: no cover - defensive contract
        raise ModelingError("production title matrix is not aligned to occupation codes")
    return TitleServingIndex(candidate.candidate_id, codes, transformer, aligned)


def fit_production_transition_components(
    profiles: pd.DataFrame,
    ratings: pd.DataFrame,
    candidate: CandidateSpec,
    *,
    fit_codes: Sequence[str],
    minimum_structured_overlap: int = 5,
    min_df: int = 2,
    random_seed: int = 20260814,
    diagnostic_lsa_dimensions: int = CLUSTER_FALLBACK_LSA_DIMENSIONS,
) -> TransitionServingComponents:
    """Refit transparent components on non-test text and transform all codes."""

    if candidate.task != "transition":
        raise ModelingError("production transition fit requires a transition candidate")
    catalog = _production_profiles(profiles)
    codes = tuple(catalog["onet_soc_code"].astype(str))
    fit_mask = _fit_mask(codes, fit_codes)
    canonical_documents = catalog["canonical_title"].astype(str).tolist()
    work_documents = catalog["work_text"].astype(str).tolist()

    if candidate.representation == "canonical_title":
        lexical_documents = canonical_documents
        lexical_analyzer = str(candidate.analyzer)
    else:
        lexical_documents = work_documents
        lexical_analyzer = "word"
    lexical_model = TfidfIndex(
        analyzer=lexical_analyzer,  # type: ignore[arg-type]
        min_df=min_df,
        dtype=np.float64,
    )
    lexical_model.fit(np.asarray(lexical_documents, dtype=object)[fit_mask].tolist())
    lexical_matrix = lexical_model.transform(lexical_documents)
    lexical_scores = cosine_scores(lexical_matrix, lexical_matrix)

    lsa_dimensions = (
        candidate.lsa_dimensions
        if candidate.representation in {"work_lsa", "work_hybrid"}
        else int(diagnostic_lsa_dimensions)
    )
    if lsa_dimensions < 1:
        raise ModelingError("diagnostic LSA dimensions must be positive")
    latent_model = TextLSAIndex(
        lsa_dimensions,
        analyzer="word",
        min_df=min_df,
        random_state=random_seed,
    )
    latent_model.fit(np.asarray(work_documents, dtype=object)[fit_mask].tolist())
    cluster_embedding = np.asarray(latent_model.transform(work_documents), dtype=np.float64)
    lsa_scores = cosine_scores(
        sparse.csr_matrix(cluster_embedding),
        sparse.csr_matrix(cluster_embedding),
    )

    structured = StructuredSimilarityIndex(
        ratings,
        codes,
        minimum_overlap=minimum_structured_overlap,
    )
    term_model = TfidfIndex(analyzer="word", min_df=min_df, dtype=np.float64)
    term_model.fit(np.asarray(work_documents, dtype=object)[fit_mask].tolist())
    term_matrix = term_model.transform(work_documents)
    feature_names = tuple(str(value) for value in term_model.get_feature_names_out())
    weights = MappingProxyType(
        {
            name: float(candidate.weight_map.get(name, 0.0))
            for name in ("lexical", "lsa", "structured")
        }
    )
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ModelingError("expanded production transition weights must sum to one")
    return TransitionServingComponents(
        candidate_id=candidate.candidate_id,
        occupation_codes=codes,
        lexical_scores=lexical_scores,
        lsa_scores=lsa_scores,
        structured_scores=np.asarray(structured.scores, dtype=np.float64),
        development_distances=np.asarray(structured.development_distance, dtype=np.float64),
        weights=weights,
        cluster_embedding=cluster_embedding,
        term_matrix=term_matrix,
        feature_names=feature_names,
    )


__all__ = [
    "CLUSTER_FALLBACK_LSA_DIMENSIONS",
    "CandidateSpec",
    "ConfirmationDecision",
    "ModelingError",
    "OPERATIONAL_COST_POLICY",
    "OPERATIONAL_COST_FORMULA",
    "PREDECLARED_LSA_DIMENSIONS",
    "PREDECLARED_WEIGHT_STEP",
    "SELECTION_COLUMNS",
    "SelectionDecision",
    "STRUCTURED_ABLATION_REASON",
    "TitleServingIndex",
    "TransitionServingComponents",
    "apply_confirmation_gate",
    "assert_artifact_payload_target_free",
    "assert_feature_columns_safe",
    "candidate_grid_digest",
    "development_decision_payload",
    "deployment_feasibility",
    "fit_production_title_index",
    "fit_production_transition_components",
    "multi_task_development_decision_payload",
    "select_candidate_one_se",
    "title_candidate_grid",
    "transition_candidate_grid",
]
