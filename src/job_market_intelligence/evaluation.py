"""Evaluation-only qrel boundary and one-shot locked-run integrity controls."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .embeddings import TextLSAIndex
from .lexical import TfidfIndex, WeightedTextTransformer, cosine_scores
from .modeling import (
    OPERATIONAL_COST_POLICY,
    CandidateSpec,
    ConfirmationDecision,
    SelectionDecision,
    apply_confirmation_gate,
    deployment_feasibility,
    select_candidate_one_se,
    title_candidate_grid,
    transition_candidate_grid,
)
from .ranking_metrics import (
    QREL_COLUMNS,
    RankingMetricError,
    macro_title_metrics,
    paired_stratified_bootstrap,
    stratified_bootstrap_interval,
    title_query_metrics,
    transition_metrics,
    transition_query_metrics,
)
from .retrieval import (
    assert_query_aliases_excluded,
    build_title_documents,
    candidate_codes_for_phase,
    combine_component_scores,
    fitting_codes_for_phase,
    prepare_alias_queries,
    rank_score_matrix,
)
from .skills import StructuredSimilarityIndex

LOCKED_STATE_SCHEMA_VERSION: Final = 1
LOCKED_RANKING_SCHEMA: Final = pa.schema(
    [
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("query_code", pa.string(), nullable=False),
        pa.field("candidate_code", pa.string(), nullable=False),
        pa.field("rank", pa.int64(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)
LOCKED_RANKING_COLUMNS: Final = tuple(field.name for field in LOCKED_RANKING_SCHEMA)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40,64}$")
STATE_FILENAME: Final = "locked_run_state.json"
RANKINGS_FILENAME: Final = "locked_rankings.parquet"
METRICS_FILENAME: Final = "locked_metrics.json"
REPORT_OUTPUT_FILENAMES: Final = (
    "title_benchmark.json",
    "transition_benchmark.json",
)
TRANSITION_EVALUATION_ROLES: Final = ("selection", "confirmation", "test")
TITLE_QUERY_POLICY: Final = (
    "development_role_scoped_locked_global_alias_ambiguity_deduplication_and_seeded_cap"
)
IMPLEMENTATION_ROOT_PREFIXES: Final = (
    "configs/",
    "docs/",
    "src/job_market_intelligence/",
)
IMPLEMENTATION_EXACT_PATHS: Final = {
    "Makefile",
    "pyproject.toml",
    "requirements.lock",
    "scripts/analyze.py",
    "scripts/benchmark.py",
    "scripts/evaluate.py",
    "scripts/train.py",
}


class EvaluationIntegrityError(RuntimeError):
    """Raised when a locked evaluation could be repeated, altered, or leaked."""


@dataclass(frozen=True, slots=True)
class GitSnapshot:
    repository: Path
    commit: str


@dataclass(frozen=True, slots=True)
class InputPin:
    logical_name: str
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class DevelopmentBenchmarkRun:
    """Synthetic-serializable evidence produced before any locked-test access."""

    alias_queries: pd.DataFrame
    title_target_ranks: pd.DataFrame
    transition_rankings: pd.DataFrame
    title_selection_results: pd.DataFrame
    title_confirmation_results: pd.DataFrame
    transition_selection_results: pd.DataFrame
    transition_confirmation_results: pd.DataFrame
    title_selection: SelectionDecision
    title_confirmation: ConfirmationDecision
    transition_selection: SelectionDecision
    transition_confirmation: ConfirmationDecision
    transition_query_universe: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class DevelopmentSelectionRun:
    alias_queries: pd.DataFrame
    title_target_ranks: pd.DataFrame
    transition_rankings: pd.DataFrame
    title_results: pd.DataFrame
    transition_results: pd.DataFrame
    title_selection: SelectionDecision
    transition_selection: SelectionDecision
    transition_query_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentConfirmationRun:
    alias_queries: pd.DataFrame
    title_target_ranks: pd.DataFrame
    transition_rankings: pd.DataFrame
    title_results: pd.DataFrame
    transition_results: pd.DataFrame
    title_confirmation: ConfirmationDecision
    transition_confirmation: ConfirmationDecision
    transition_query_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LockedRunSession:
    output_dir: Path

    @property
    def state_path(self) -> Path:
        return self.output_dir / STATE_FILENAME

    @property
    def rankings_path(self) -> Path:
        return self.output_dir / RANKINGS_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / METRICS_FILENAME

    def state(self) -> dict[str, Any]:
        return _load_state(self.state_path)

    def publish_rankings(self, rankings: pd.DataFrame) -> str:
        """Atomically write rankings once, before any aggregate metric report."""

        state = self.state()
        if state["status"] != "started":
            raise EvaluationIntegrityError("rankings can only be published from a started run")
        if self.rankings_path.exists() or self.metrics_path.exists():
            raise EvaluationIntegrityError(
                "locked output already exists; do not rerun or overwrite this experiment"
            )
        validated = validate_locked_rankings(rankings)
        table = pa.Table.from_pandas(
            validated,
            schema=LOCKED_RANKING_SCHEMA,
            preserve_index=False,
            safe=True,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{RANKINGS_FILENAME}.", suffix=".tmp", dir=self.output_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
                data_page_version="1.0",
            )
            _fsync_file(temporary)
            os.replace(temporary, self.rankings_path)
            _fsync_directory(self.output_dir)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest = sha256_file(self.rankings_path)
        state["rankings"] = {
            "path": RANKINGS_FILENAME,
            "sha256": digest,
            "byte_count": self.rankings_path.stat().st_size,
            "row_count": len(validated),
            "schema_sha256": hashlib.sha256(
                LOCKED_RANKING_SCHEMA.to_string().encode("utf-8")
            ).hexdigest(),
        }
        state["status"] = "rankings_written"
        _atomic_json(self.state_path, state, overwrite=True)
        return digest

    def load_rankings_for_reporting(self) -> pd.DataFrame:
        """Load only immutable ranking bytes; never call a model during resume."""

        state = self.state()
        if state["status"] not in {"rankings_written", "complete"}:
            raise EvaluationIntegrityError(
                "reporting resume requires a previously hashed rankings artifact"
            )
        entry = state.get("rankings")
        if not isinstance(entry, dict) or entry.get("path") != RANKINGS_FILENAME:
            raise EvaluationIntegrityError("locked state lacks rankings provenance")
        if not self.rankings_path.is_file() or self.rankings_path.is_symlink():
            raise EvaluationIntegrityError("locked rankings file is missing or unsafe")
        if sha256_file(self.rankings_path) != entry.get("sha256"):
            raise EvaluationIntegrityError("locked ranking bytes differ from the recorded hash")
        if self.rankings_path.stat().st_size != entry.get("byte_count"):
            raise EvaluationIntegrityError("locked ranking byte count differs from provenance")
        try:
            table = pq.read_table(self.rankings_path, schema=LOCKED_RANKING_SCHEMA)
        except Exception as exc:
            raise EvaluationIntegrityError("locked rankings Parquet cannot be read") from exc
        frame = table.to_pandas()
        validated = validate_locked_rankings(frame)
        if len(validated) != entry.get("row_count"):
            raise EvaluationIntegrityError("locked ranking row count differs from provenance")
        return validated

    def load_metrics_for_reporting(self) -> dict[str, Any]:
        """Reverify and load immutable aggregate evidence after completion."""

        state = self.state()
        if state["status"] != "complete":
            raise EvaluationIntegrityError("metrics are available only for a complete locked run")
        entry = state.get("metrics")
        expected_entry_keys = {
            "path",
            "sha256",
            "byte_count",
            "original_repository_commit",
            "reporting_repository_commit",
        }
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
            raise EvaluationIntegrityError("locked state lacks metrics provenance")
        if entry.get("path") != METRICS_FILENAME:
            raise EvaluationIntegrityError("locked metrics path differs from the contract")
        if not self.metrics_path.is_file() or self.metrics_path.is_symlink():
            raise EvaluationIntegrityError("locked metrics file is missing or unsafe")
        if self.metrics_path.stat().st_size != entry.get("byte_count"):
            raise EvaluationIntegrityError("locked metrics byte count differs from provenance")
        if sha256_file(self.metrics_path) != entry.get("sha256"):
            raise EvaluationIntegrityError("locked metrics bytes differ from the recorded hash")
        payload = _load_json_object(self.metrics_path, description="locked metrics")
        expected_payload_keys = {
            "schema_version",
            "experiment_version",
            "split",
            "rankings_sha256",
            "decision_sha256",
            "benchmarks",
            "original_repository_commit",
            "reporting_repository_commit",
        }
        if set(payload) != expected_payload_keys:
            raise EvaluationIntegrityError("locked metrics fields differ from the contract")
        if (
            payload["schema_version"] != 1
            or payload["split"] != "locked_test"
            or payload["experiment_version"] != state["experiment_version"]
            or payload["rankings_sha256"] != state["rankings"]["sha256"]
            or payload["decision_sha256"] != state["decision_sha256"]
            or payload["original_repository_commit"] != state["repository_commit"]
            or payload["original_repository_commit"] != entry["original_repository_commit"]
            or payload["reporting_repository_commit"] != entry["reporting_repository_commit"]
        ):
            raise EvaluationIntegrityError("locked metrics provenance differs from state")
        if not isinstance(payload["benchmarks"], Mapping) or not payload["benchmarks"]:
            raise EvaluationIntegrityError("locked metrics benchmarks are invalid")
        return payload

    def publish_metrics(
        self,
        report: Mapping[str, Any],
        *,
        repository: str | Path,
    ) -> str:
        """Publish a report at the exact code commit that generated rankings."""

        state = self.state()
        if state["status"] != "rankings_written":
            raise EvaluationIntegrityError("metrics can only be published once after rankings")
        rankings = self.load_rankings_for_reporting()
        if rankings.empty:  # pragma: no cover - validation already prevents this
            raise EvaluationIntegrityError("locked rankings are unexpectedly empty")
        git = require_reporting_git_state(
            repository,
            state=state,
            output_dir=self.output_dir,
        )
        verify_stored_inputs(state, repository=git.repository)
        payload = dict(report)
        required = {
            "schema_version",
            "experiment_version",
            "split",
            "rankings_sha256",
            "decision_sha256",
            "benchmarks",
        }
        if set(payload) != required:
            raise EvaluationIntegrityError(f"metric report keys must be exactly {sorted(required)}")
        if payload["schema_version"] != 1 or payload["split"] != "locked_test":
            raise EvaluationIntegrityError("metric report must be schema v1 locked_test evidence")
        if payload["experiment_version"] != state["experiment_version"]:
            raise EvaluationIntegrityError("metric report experiment version differs from state")
        if payload["rankings_sha256"] != state["rankings"]["sha256"]:
            raise EvaluationIntegrityError("metric report is not bound to locked ranking bytes")
        if payload["decision_sha256"] != state["decision_sha256"]:
            raise EvaluationIntegrityError("metric report is not bound to the frozen decision")
        if not isinstance(payload["benchmarks"], Mapping) or not payload["benchmarks"]:
            raise EvaluationIntegrityError("metric report benchmarks must be a nonempty object")
        _reject_nonfinite_json(payload)
        final_payload = {
            **payload,
            "original_repository_commit": state["repository_commit"],
            "reporting_repository_commit": git.commit,
        }
        if self.metrics_path.exists():
            # Recover only a byte-complete report left by a crash between the
            # report publication and the state transition to ``complete``.
            existing = _load_json_object(self.metrics_path, description="orphan locked metrics")
            existing_reporting_commit = existing.get("reporting_repository_commit")
            expected_without_reporting_commit = {
                key: value
                for key, value in final_payload.items()
                if key != "reporting_repository_commit"
            }
            actual_without_reporting_commit = {
                key: value
                for key, value in existing.items()
                if key != "reporting_repository_commit"
            }
            if (
                actual_without_reporting_commit != expected_without_reporting_commit
                or not isinstance(existing_reporting_commit, str)
                or COMMIT_PATTERN.fullmatch(existing_reporting_commit) is None
                or existing_reporting_commit != git.commit
            ):
                raise EvaluationIntegrityError(
                    "existing locked metrics cannot be safely adopted after interruption"
                )
            final_payload = existing
        else:
            _atomic_json(self.metrics_path, final_payload, overwrite=False)
        digest = sha256_file(self.metrics_path)
        state["metrics"] = {
            "path": METRICS_FILENAME,
            "sha256": digest,
            "byte_count": self.metrics_path.stat().st_size,
            "original_repository_commit": state["repository_commit"],
            "reporting_repository_commit": final_payload["reporting_repository_commit"],
        }
        state["status"] = "complete"
        _atomic_json(self.state_path, state, overwrite=True)
        return digest


@dataclass(frozen=True, slots=True)
class _CandidateRuntime:
    scores: np.ndarray
    artifact_bytes: int
    operational_cost_units: int
    latency_ms: float
    deployment_eligible: bool
    ineligibility_reason: str


def _serialized_size(value: Any) -> int:
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=0, protocol=5)
    return len(buffer.getbuffer())


def _measure_candidate_latency_ms(
    operation: Callable[[], np.ndarray | sparse.spmatrix],
    *,
    repeats: int = 9,
) -> float:
    """Measure candidate-specific fixed-batch transform/score/rank latency.

    This wall-clock value is retained as an honest diagnostic only. Selection
    uses deterministic ``operational_cost_units`` so timer noise cannot change
    a champion or decision hash.
    """

    for _ in range(2):
        operation()
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        raw = operation()
        sample = (
            np.asarray(raw.toarray(), dtype=np.float64)
            if sparse.issparse(raw)
            else np.asarray(raw, dtype=np.float64)
        )
        if sample.ndim != 2 or sample.size == 0:
            raise EvaluationIntegrityError("candidate latency operation returned no score rows")
        for row in sample:
            np.lexsort((np.arange(len(row)), -row))
        timings.append((time.perf_counter_ns() - started) / 1_000_000 / len(sample))
    return float(np.median(timings))


def _operational_cost_units(candidate: CandidateSpec, scores: np.ndarray) -> int:
    """Deterministic predeclared proxy for fixed-batch scoring work."""

    active_components = sum(weight > 0 for weight in candidate.weight_map.values())
    latent_work = max(0, int(candidate.lsa_dimensions)) * int(scores.shape[0])
    return int(scores.size * max(1, active_components) + latent_work)


def _profile_lookup(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "onet_soc_code",
        "canonical_title",
        "soc_major_group",
        "work_text",
    }
    if missing := sorted(required - set(profiles.columns)):
        raise EvaluationIntegrityError(f"occupation profiles are missing columns: {missing}")
    result = profiles.copy()
    result["onet_soc_code"] = result["onet_soc_code"].astype(str)
    if result.empty or bool(result["onet_soc_code"].duplicated().any()):
        raise EvaluationIntegrityError("occupation profiles must be nonempty and unique by code")
    return result.set_index("onet_soc_code", drop=False)


def _fit_title_candidate_runtimes(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    alias_queries: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    min_df: int,
    random_seed: int,
    phase: str,
) -> tuple[tuple[str, ...], dict[str, _CandidateRuntime]]:
    profile_by_code = _profile_lookup(profiles)
    catalog_codes = candidate_codes_for_phase(fold_manifest, phase)
    fit_codes = fitting_codes_for_phase(fold_manifest, phase)
    unknown = (set(catalog_codes) | set(fit_codes)) - set(profile_by_code.index)
    if unknown:
        raise EvaluationIntegrityError(
            f"split codes are absent from profiles: {sorted(unknown)[:5]}"
        )
    catalog_profiles = profile_by_code.loc[list(catalog_codes)]
    fit_mask = np.asarray([code in set(fit_codes) for code in catalog_codes], dtype=bool)
    canonical_documents = catalog_profiles["canonical_title"].astype(str).tolist()
    title_documents = build_title_documents(
        profiles.reset_index(drop=True),
        aliases,
        allowed_alias_codes=fit_codes,
    )
    title_by_code = title_documents.documents.set_index("onet_soc_code")
    profile_documents = title_by_code.loc[list(catalog_codes), "title_text"].astype(str).tolist()
    assert_query_aliases_excluded(title_documents, alias_queries)
    query_text = alias_queries["query_text"].astype(str).tolist()

    lexical_cache: dict[tuple[str, str], tuple[Any, np.ndarray]] = {}
    lsa_cache: dict[int, tuple[TextLSAIndex, np.ndarray]] = {}

    def lexical(kind: str, analyzer: str) -> tuple[TfidfIndex, np.ndarray]:
        key = (kind, analyzer)
        if key not in lexical_cache:
            documents = canonical_documents if kind == "canonical" else profile_documents
            transformer = TfidfIndex(
                analyzer=analyzer,  # type: ignore[arg-type]
                min_df=min_df,
                dtype=np.float64,
            )
            transformer.fit(np.asarray(documents, dtype=object)[fit_mask].tolist())
            lexical_cache[key] = (transformer, transformer.transform(documents))
        return lexical_cache[key]

    def lsa(dimension: int) -> tuple[TextLSAIndex, np.ndarray]:
        if dimension not in lsa_cache:
            transformer = TextLSAIndex(
                dimension,
                analyzer="word_char",
                min_df=min_df,
                random_state=random_seed,
            )
            transformer.fit(np.asarray(profile_documents, dtype=object)[fit_mask].tolist())
            lsa_cache[dimension] = (transformer, transformer.transform(profile_documents))
        return lsa_cache[dimension]

    runtimes: dict[str, _CandidateRuntime] = {}
    for candidate in candidates:
        if candidate.task != "title":
            raise EvaluationIntegrityError(
                f"non-title candidate in title grid: {candidate.candidate_id}"
            )
        if candidate.representation == "canonical_title":
            transformer, catalog_matrix = lexical("canonical", str(candidate.analyzer))
        elif candidate.representation == "title_profile":
            transformer, catalog_matrix = lexical("profile", str(candidate.analyzer))
        elif candidate.representation == "title_lsa":
            transformer, catalog_matrix = lsa(candidate.lsa_dimensions)
        elif candidate.representation == "title_hybrid":
            lexical_model, _ = lexical("profile", "word_char")
            lsa_model, _ = lsa(candidate.lsa_dimensions)
            weight_map = candidate.weight_map
            transformer = WeightedTextTransformer(
                (lexical_model, lsa_model),
                (weight_map["lexical"], weight_map["lsa"]),
            )
            catalog_matrix = transformer.transform(profile_documents)
        else:
            raise EvaluationIntegrityError(
                f"unsupported title representation: {candidate.representation}"
            )
        query_matrix = transformer.transform(query_text)
        scores = cosine_scores(
            sparse.csr_matrix(query_matrix, dtype=np.float64),
            sparse.csr_matrix(catalog_matrix, dtype=np.float64),
        )
        fixed_queries = query_text[: min(16, len(query_text))]

        def operation(
            transformer: Any = transformer,
            catalog_matrix: Any = catalog_matrix,
            fixed_queries: list[str] = fixed_queries,
        ) -> np.ndarray:
            fixed_matrix = transformer.transform(fixed_queries)
            return cosine_scores(
                sparse.csr_matrix(fixed_matrix, dtype=np.float64),
                sparse.csr_matrix(catalog_matrix, dtype=np.float64),
            )

        deployment_eligible, reason = deployment_feasibility(candidate)
        runtimes[candidate.candidate_id] = _CandidateRuntime(
            scores=scores,
            artifact_bytes=_serialized_size(transformer),
            operational_cost_units=_operational_cost_units(candidate, scores),
            latency_ms=_measure_candidate_latency_ms(operation),
            deployment_eligible=deployment_eligible,
            ineligibility_reason=reason,
        )
    return catalog_codes, runtimes


def _title_target_rank_rows(
    alias_queries: pd.DataFrame,
    catalog_codes: Sequence[str],
    runtimes: Mapping[str, _CandidateRuntime],
) -> pd.DataFrame:
    code_array = np.asarray(catalog_codes, dtype=object)
    code_index = {str(code): index for index, code in enumerate(catalog_codes)}
    rows: list[dict[str, Any]] = []
    for candidate_id, runtime in runtimes.items():
        for query_index, query in enumerate(alias_queries.itertuples(index=False)):
            target = str(query.target_code)
            if target not in code_index:
                continue
            target_index = code_index[target]
            values = runtime.scores[query_index]
            target_score = values[target_index]
            outranking = (values > target_score) | (
                (values == target_score) & (code_array.astype(str) < target)
            )
            rows.append(
                {
                    "phase": str(query.role),
                    "candidate_id": candidate_id,
                    "query_id": str(query.query_id),
                    "query_code": str(query.query_code),
                    "target_code": target,
                    "rank": int(np.count_nonzero(outranking)) + 1,
                    "target_score": float(target_score),
                    "soc_major_group": str(query.soc_major_group),
                    "artifact_bytes": runtime.artifact_bytes,
                    "operational_cost_units": runtime.operational_cost_units,
                    "latency_ms": runtime.latency_ms,
                    "deployment_eligible": runtime.deployment_eligible,
                    "ineligibility_reason": runtime.ineligibility_reason,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["phase", "candidate_id", "query_id"], kind="stable")
        .reset_index(drop=True)
    )


def _title_selection_frame(
    target_ranks: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    role: str,
) -> pd.DataFrame:
    specs = {candidate.candidate_id: candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []
    selected = target_ranks.loc[target_ranks["phase"].eq(role)].copy()
    selected["top1_accuracy"] = selected["rank"].eq(1).astype(float)
    for candidate_id, group in selected.groupby("candidate_id", sort=True):
        for query_code, occupation in group.groupby("target_code", sort=True):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "query_id": str(query_code),
                    "primary_value": float(occupation["top1_accuracy"].mean()),
                    "soc_major_group": str(occupation["soc_major_group"].iloc[0]),
                    "latent_dimensions": specs[candidate_id].lsa_dimensions,
                    "artifact_bytes": int(occupation["artifact_bytes"].iloc[0]),
                    "operational_cost_units": int(occupation["operational_cost_units"].iloc[0]),
                    "latency_ms": float(occupation["latency_ms"].iloc[0]),
                    "deployment_eligible": bool(occupation["deployment_eligible"].iloc[0]),
                    "ineligibility_reason": str(occupation["ineligibility_reason"].iloc[0]),
                    "split_role": role,
                }
            )
    return pd.DataFrame(rows)


def _fit_transition_candidate_runtimes(
    profiles: pd.DataFrame,
    ratings: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    minimum_structured_overlap: int,
    min_df: int,
    random_seed: int,
    phase: str,
) -> tuple[tuple[str, ...], dict[str, _CandidateRuntime]]:
    profile_by_code = _profile_lookup(profiles)
    catalog_codes = candidate_codes_for_phase(fold_manifest, phase)
    fit_codes = set(fitting_codes_for_phase(fold_manifest, phase))
    catalog = profile_by_code.loc[list(catalog_codes)]
    fit_mask = np.asarray([code in fit_codes for code in catalog_codes], dtype=bool)
    canonical_documents = catalog["canonical_title"].astype(str).tolist()
    work_documents = catalog["work_text"].astype(str).tolist()
    score_cache: dict[str, tuple[Any, np.ndarray]] = {}

    def text_scores(key: str, documents: list[str], analyzer: str) -> tuple[Any, np.ndarray]:
        if key not in score_cache:
            transformer = TfidfIndex(
                analyzer=analyzer,  # type: ignore[arg-type]
                min_df=min_df,
                dtype=np.float64,
            )
            transformer.fit(np.asarray(documents, dtype=object)[fit_mask].tolist())
            matrix = transformer.transform(documents)
            score_cache[key] = (transformer, cosine_scores(matrix, matrix))
        return score_cache[key]

    lsa_cache: dict[int, tuple[TextLSAIndex, np.ndarray]] = {}

    def lsa_scores(dimension: int) -> tuple[TextLSAIndex, np.ndarray]:
        if dimension not in lsa_cache:
            transformer = TextLSAIndex(
                dimension,
                analyzer="word",
                min_df=min_df,
                random_state=random_seed,
            )
            transformer.fit(np.asarray(work_documents, dtype=object)[fit_mask].tolist())
            matrix = transformer.transform(work_documents)
            lsa_cache[dimension] = (
                transformer,
                cosine_scores(sparse.csr_matrix(matrix), sparse.csr_matrix(matrix)),
            )
        return lsa_cache[dimension]

    structured_index = StructuredSimilarityIndex(
        ratings,
        catalog_codes,
        minimum_overlap=minimum_structured_overlap,
    )
    structured_scores = structured_index.scores
    runtimes: dict[str, _CandidateRuntime] = {}
    for candidate in candidates:
        if candidate.task != "transition":
            raise EvaluationIntegrityError(
                f"non-transition candidate in transition grid: {candidate.candidate_id}"
            )
        payload: Any
        if candidate.candidate_id == "transition_title_word":
            payload, scores = text_scores("title_word", canonical_documents, "word")
        elif candidate.candidate_id == "transition_title_char":
            payload, scores = text_scores("title_char", canonical_documents, "char")
        elif candidate.candidate_id == "transition_title_word_char":
            payload, scores = text_scores("title_word_char", canonical_documents, "word_char")
        elif candidate.representation == "work_profile":
            payload, scores = text_scores("work_word", work_documents, "word")
        elif candidate.representation == "structured":
            payload, scores = {"minimum_overlap": minimum_structured_overlap}, structured_scores
        elif candidate.representation == "work_lsa":
            payload, scores = lsa_scores(candidate.lsa_dimensions)
        elif candidate.representation == "work_hybrid":
            lexical_model, lexical_values = text_scores("work_word", work_documents, "word")
            lsa_model, lsa_values = lsa_scores(candidate.lsa_dimensions)
            components = {
                "lexical": lexical_values,
                "lsa": lsa_values,
                "structured": structured_scores,
            }
            scores = combine_component_scores(components, candidate.weight_map)
            payload = {
                "lexical": lexical_model,
                "lsa": lsa_model,
                "weights": dict(candidate.weight_map),
                "minimum_overlap": minimum_structured_overlap,
            }
        else:
            raise EvaluationIntegrityError(
                f"unsupported transition representation: {candidate.representation}"
            )
        batch_size = min(16, len(catalog_codes))
        if candidate.representation == "work_hybrid":

            def operation(
                lexical_values: np.ndarray = lexical_values,
                lsa_values: np.ndarray = lsa_values,
                structured_scores: np.ndarray = structured_scores,
                weights: Mapping[str, float] = candidate.weight_map,
                batch_size: int = batch_size,
            ) -> np.ndarray:
                combined = combine_component_scores(
                    {
                        "lexical": lexical_values[:batch_size],
                        "lsa": lsa_values[:batch_size],
                        "structured": structured_scores[:batch_size],
                    },
                    weights,
                )
                return np.nan_to_num(combined, nan=-np.inf)

        else:
            candidate_scores = np.asarray(scores, dtype=np.float64)

            def operation(
                scores: np.ndarray = candidate_scores,
                batch_size: int = batch_size,
            ) -> np.ndarray:
                return np.nan_to_num(scores[:batch_size].copy(), nan=-np.inf)

        numeric_scores = np.asarray(scores, dtype=np.float64)
        deployment_eligible, reason = deployment_feasibility(candidate)
        runtimes[candidate.candidate_id] = _CandidateRuntime(
            scores=numeric_scores,
            artifact_bytes=_serialized_size(payload),
            operational_cost_units=_operational_cost_units(candidate, numeric_scores),
            latency_ms=_measure_candidate_latency_ms(operation),
            deployment_eligible=deployment_eligible,
            ineligibility_reason=reason,
        )
    return catalog_codes, runtimes


def _transition_ranking_rows(
    fold_manifest: pd.DataFrame,
    catalog_codes: Sequence[str],
    runtimes: Mapping[str, _CandidateRuntime],
    *,
    role: str,
    limit: int = 20,
) -> pd.DataFrame:
    query_codes = tuple(
        fold_manifest.loc[fold_manifest["role"].eq(role), "onet_soc_code"]
        .astype(str)
        .sort_values(kind="stable")
    )
    catalog_index = {code: index for index, code in enumerate(catalog_codes)}
    unknown = set(query_codes) - set(catalog_index)
    if unknown:
        raise EvaluationIntegrityError(
            f"{role} queries are outside candidate catalog: {sorted(unknown)}"
        )
    query_indexes = [catalog_index[code] for code in query_codes]
    frames: list[pd.DataFrame] = []
    for candidate_id, runtime in runtimes.items():
        ranking = rank_score_matrix(
            query_codes,
            catalog_codes,
            runtime.scores[query_indexes],
            k=min(limit, len(catalog_codes) - 1),
            exclude_self=True,
        )
        ranking.insert(0, "candidate_id", candidate_id)
        ranking.insert(0, "phase", role)
        ranking["artifact_bytes"] = runtime.artifact_bytes
        ranking["operational_cost_units"] = runtime.operational_cost_units
        ranking["latency_ms"] = runtime.latency_ms
        ranking["deployment_eligible"] = runtime.deployment_eligible
        ranking["ineligibility_reason"] = runtime.ineligibility_reason
        frames.append(ranking)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["phase", "candidate_id", "query_code", "rank"], kind="stable"
    )


def transition_evaluation_query_codes(
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    role: str,
) -> tuple[str, ...]:
    """Freeze role queries with valid post-boundary qrels and tier coverage."""

    if role not in {"selection", "confirmation", "test"}:
        raise EvaluationIntegrityError(f"unsupported transition evaluation role: {role!r}")
    missing = sorted(set(QREL_COLUMNS) - set(qrels.columns))
    if missing:
        raise EvaluationIntegrityError(f"evaluation qrels are missing columns: {missing}")
    required_fold_columns = {"onet_soc_code", "role"}
    if missing := sorted(required_fold_columns - set(fold_manifest.columns)):
        raise EvaluationIntegrityError(f"fold manifest is missing columns: {missing}")
    role_codes = set(fold_manifest.loc[fold_manifest["role"].eq(role), "onet_soc_code"].astype(str))
    candidate_codes = set(fold_manifest["onet_soc_code"].astype(str))
    if role != "test":
        candidate_codes -= set(
            fold_manifest.loc[fold_manifest["role"].eq("test"), "onet_soc_code"].astype(str)
        )
    work = qrels.loc[
        qrels["query_onet_soc_code"].astype(str).isin(role_codes)
        & qrels["candidate_onet_soc_code"].astype(str).isin(candidate_codes),
        list(QREL_COLUMNS),
    ].copy()
    if bool(work.duplicated(["query_onet_soc_code", "candidate_onet_soc_code"]).any()):
        raise EvaluationIntegrityError("evaluation qrels repeat a query/candidate pair")
    valid_tiers = {"Primary-Short": 3, "Primary-Long": 2, "Supplemental": 1}
    if any(
        valid_tiers.get(str(row.relatedness_tier)) != row.relevance_grade
        for row in work.itertuples(index=False)
    ):
        raise EvaluationIntegrityError("qrel tier-to-grade mapping differs from the protocol")
    eligible: list[str] = []
    for query_code, group in work.groupby("query_onet_soc_code", sort=True):
        tiers = set(group["relatedness_tier"].astype(str))
        if "Primary-Short" in tiers and tiers & {"Primary-Short", "Primary-Long"}:
            eligible.append(str(query_code))
    if not eligible:
        raise EvaluationIntegrityError(f"{role} has no transition queries with required qrels")
    return tuple(sorted(eligible))


def build_transition_evaluation_universe(
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Preregister exact judged query IDs without exposing qrel destinations."""

    roles: dict[str, Any] = {}
    for role in TRANSITION_EVALUATION_ROLES:
        assigned = tuple(
            sorted(fold_manifest.loc[fold_manifest["role"].eq(role), "onet_soc_code"].astype(str))
        )
        eligible = transition_evaluation_query_codes(qrels, fold_manifest, role=role)
        roles[role] = {
            "assigned_query_count": len(assigned),
            "eligible_query_count": len(eligible),
            "query_codes": list(eligible),
            "query_codes_sha256": hashlib.sha256(
                json.dumps(list(eligible), sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "candidate_universe": "full_catalog" if role == "test" else "non_test",
        }
    return {
        "schema_version": 1,
        "query_policy": "post_boundary_primary_short_required",
        "roles": roles,
    }


def validate_transition_evaluation_universe(
    universe: Mapping[str, Any],
    fold_manifest: pd.DataFrame,
    *,
    qrels: pd.DataFrame | None = None,
) -> dict[str, tuple[str, ...]]:
    """Validate frozen eligible IDs and optionally recompute them from qrels."""

    if set(universe) != {"schema_version", "query_policy", "roles"}:
        raise EvaluationIntegrityError("transition evaluation universe fields differ")
    if (
        universe.get("schema_version") != 1
        or universe.get("query_policy") != "post_boundary_primary_short_required"
        or not isinstance(universe.get("roles"), Mapping)
        or set(universe["roles"]) != set(TRANSITION_EVALUATION_ROLES)
    ):
        raise EvaluationIntegrityError("transition evaluation universe contract is invalid")
    result: dict[str, tuple[str, ...]] = {}
    for role in TRANSITION_EVALUATION_ROLES:
        entry = universe["roles"][role]
        if not isinstance(entry, Mapping) or set(entry) != {
            "assigned_query_count",
            "eligible_query_count",
            "query_codes",
            "query_codes_sha256",
            "candidate_universe",
        }:
            raise EvaluationIntegrityError(f"transition {role} universe fields differ")
        codes = entry["query_codes"]
        if (
            not isinstance(codes, list)
            or any(not isinstance(code, str) or not code for code in codes)
            or codes != sorted(codes)
            or len(codes) != len(set(codes))
        ):
            raise EvaluationIntegrityError(f"transition {role} query IDs are not canonical")
        assigned = set(
            fold_manifest.loc[fold_manifest["role"].eq(role), "onet_soc_code"].astype(str)
        )
        if (
            isinstance(entry["assigned_query_count"], bool)
            or entry["assigned_query_count"] != len(assigned)
            or isinstance(entry["eligible_query_count"], bool)
            or entry["eligible_query_count"] != len(codes)
            or entry["query_codes_sha256"]
            != hashlib.sha256(
                json.dumps(codes, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            or not set(codes) <= assigned
            or entry["candidate_universe"] != ("full_catalog" if role == "test" else "non_test")
        ):
            raise EvaluationIntegrityError(f"transition {role} query universe is inconsistent")
        if qrels is not None and tuple(codes) != transition_evaluation_query_codes(
            qrels, fold_manifest, role=role
        ):
            raise EvaluationIntegrityError(
                f"transition {role} query universe differs from frozen qrel eligibility"
            )
        result[role] = tuple(codes)
    return result


def build_evaluation_universe(
    aliases: pd.DataFrame,
    profiles: pd.DataFrame,
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    maximum_aliases_per_occupation: int,
    seed: int,
    implementation_manifest_sha256: str,
) -> dict[str, Any]:
    """Freeze role-local title IDs and post-boundary transition query IDs."""

    title_roles: dict[str, Any] = {}
    for role in TRANSITION_EVALUATION_ROLES:
        role_queries = _role_alias_queries(
            aliases,
            profiles,
            fold_manifest,
            role=role,
            maximum_per_occupation=maximum_aliases_per_occupation,
            seed=seed,
        )
        query_ids = sorted(role_queries["query_id"].astype(str))
        title_roles[role] = {
            "assigned_occupation_count": int(fold_manifest["role"].eq(role).sum()),
            "eligible_occupation_count": int(role_queries["query_code"].nunique()),
            "eligible_query_count": len(query_ids),
            "query_ids": query_ids,
            "query_ids_sha256": hashlib.sha256(
                json.dumps(query_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    transition = build_transition_evaluation_universe(qrels, fold_manifest)
    if SHA256_PATTERN.fullmatch(implementation_manifest_sha256) is None:
        raise EvaluationIntegrityError("implementation manifest digest is invalid")
    return {
        "schema_version": 1,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "title": {
            "query_policy": TITLE_QUERY_POLICY,
            "maximum_queries_per_occupation": maximum_aliases_per_occupation,
            "roles": title_roles,
        },
        "transition": {
            "query_policy": transition["query_policy"],
            "roles": transition["roles"],
        },
    }


def validate_evaluation_universe(
    universe: Mapping[str, Any],
    fold_manifest: pd.DataFrame,
    *,
    aliases: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
    qrels: pd.DataFrame | None = None,
    seed: int | None = None,
) -> dict[str, Mapping[str, tuple[str, ...]]]:
    """Validate the combined target-free evaluation-universe contract."""

    if set(universe) != {
        "schema_version",
        "implementation_manifest_sha256",
        "title",
        "transition",
    }:
        raise EvaluationIntegrityError("evaluation universe fields differ")
    if (
        universe.get("schema_version") != 1
        or SHA256_PATTERN.fullmatch(str(universe.get("implementation_manifest_sha256", ""))) is None
    ):
        raise EvaluationIntegrityError("evaluation universe schema is unsupported")
    title = universe.get("title")
    if not isinstance(title, Mapping) or set(title) != {
        "query_policy",
        "maximum_queries_per_occupation",
        "roles",
    }:
        raise EvaluationIntegrityError("title evaluation universe fields differ")
    maximum = title["maximum_queries_per_occupation"]
    if (
        title["query_policy"] != TITLE_QUERY_POLICY
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 1
        or not isinstance(title["roles"], Mapping)
        or set(title["roles"]) != set(TRANSITION_EVALUATION_ROLES)
    ):
        raise EvaluationIntegrityError("title evaluation universe contract is invalid")
    title_result: dict[str, tuple[str, ...]] = {}
    for role in TRANSITION_EVALUATION_ROLES:
        entry = title["roles"][role]
        if not isinstance(entry, Mapping) or set(entry) != {
            "assigned_occupation_count",
            "eligible_occupation_count",
            "eligible_query_count",
            "query_ids",
            "query_ids_sha256",
        }:
            raise EvaluationIntegrityError(f"title {role} universe fields differ")
        query_ids = entry["query_ids"]
        if (
            not isinstance(query_ids, list)
            or any(not isinstance(value, str) or not value for value in query_ids)
            or query_ids != sorted(query_ids)
            or len(query_ids) != len(set(query_ids))
        ):
            raise EvaluationIntegrityError(f"title {role} query IDs are not canonical")
        assigned = int(fold_manifest["role"].eq(role).sum())
        if (
            entry["assigned_occupation_count"] != assigned
            or isinstance(entry["eligible_occupation_count"], bool)
            or not isinstance(entry["eligible_occupation_count"], int)
            or not 0 < entry["eligible_occupation_count"] <= assigned
            or entry["eligible_query_count"] != len(query_ids)
            or entry["query_ids_sha256"]
            != hashlib.sha256(
                json.dumps(query_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ):
            raise EvaluationIntegrityError(f"title {role} query universe is inconsistent")
        if aliases is not None or profiles is not None or seed is not None:
            if aliases is None or profiles is None or seed is None:
                raise EvaluationIntegrityError(
                    "title universe recomputation requires aliases, profiles, and seed"
                )
            recomputed = _role_alias_queries(
                aliases,
                profiles,
                fold_manifest,
                role=role,
                maximum_per_occupation=maximum,
                seed=seed,
            )
            if query_ids != sorted(recomputed["query_id"].astype(str)):
                raise EvaluationIntegrityError(
                    f"title {role} query IDs differ from frozen alias eligibility"
                )
            if entry["eligible_occupation_count"] != recomputed["query_code"].nunique():
                raise EvaluationIntegrityError(
                    f"title {role} occupation count differs from frozen eligibility"
                )
        title_result[role] = tuple(query_ids)
    transition = universe.get("transition")
    if not isinstance(transition, Mapping):
        raise EvaluationIntegrityError("transition evaluation universe is invalid")
    transition_result = validate_transition_evaluation_universe(
        {"schema_version": 1, **dict(transition)},
        fold_manifest,
        qrels=qrels,
    )
    return {"title": title_result, "transition": transition_result}


def _transition_selection_frame(
    rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    role: str,
    expected_query_codes: Sequence[str],
    runtimes: Mapping[str, _CandidateRuntime],
) -> pd.DataFrame:
    specs = {candidate.candidate_id: candidate for candidate in candidates}
    if set(runtimes) != set(specs):
        raise EvaluationIntegrityError(
            "transition candidate runtimes differ from the candidate grid"
        )
    major_by_code = fold_manifest.set_index("onet_soc_code")["soc_major_group"].astype(str)
    rows: list[dict[str, Any]] = []
    selected = rankings.loc[rankings["phase"].eq(role)]
    expected_queries = tuple(sorted(str(code) for code in expected_query_codes))
    for candidate_id in sorted(specs):
        group = selected.loc[selected["candidate_id"].eq(candidate_id)]
        metric_rows = group.loc[
            group["query_code"].astype(str).isin(expected_queries),
            ["query_code", "candidate_code", "rank"],
        ]
        try:
            metrics = transition_query_metrics(
                metric_rows,
                qrels,
                expected_query_codes=expected_queries,
            )
        except RankingMetricError as exc:
            raise EvaluationIntegrityError(str(exc)) from exc
        runtime = runtimes[candidate_id]
        for metric in metrics.itertuples(index=False):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "query_id": str(metric.query_code),
                    "primary_value": float(metric.ndcg_at_10),
                    "soc_major_group": str(major_by_code.loc[metric.query_code]),
                    "latent_dimensions": specs[candidate_id].lsa_dimensions,
                    "artifact_bytes": runtime.artifact_bytes,
                    "operational_cost_units": runtime.operational_cost_units,
                    "latency_ms": runtime.latency_ms,
                    "deployment_eligible": runtime.deployment_eligible,
                    "ineligibility_reason": runtime.ineligibility_reason,
                    "split_role": role,
                }
            )
    return pd.DataFrame(rows)


def _development_qrels_for_role(
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    role: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Remove every fold-zero query/edge before development metric code."""

    missing = sorted(set(QREL_COLUMNS) - set(qrels.columns))
    if missing:
        raise EvaluationIntegrityError(f"development qrels are missing columns: {missing}")
    non_test_codes = set(
        fold_manifest.loc[fold_manifest["role"].ne("test"), "onet_soc_code"].astype(str)
    )
    expected_queries = transition_evaluation_query_codes(
        qrels,
        fold_manifest,
        role=role,
    )
    query_codes = set(expected_queries)
    if not query_codes or not query_codes <= non_test_codes:
        raise EvaluationIntegrityError(f"development role {role!r} has an invalid query universe")
    result = qrels.loc[
        qrels["query_onet_soc_code"].astype(str).isin(query_codes)
        & qrels["candidate_onet_soc_code"].astype(str).isin(non_test_codes),
        list(QREL_COLUMNS),
    ].copy()
    if (
        set(result["query_onet_soc_code"].astype(str)) - query_codes
        or set(result["candidate_onet_soc_code"].astype(str)) - non_test_codes
    ):
        raise EvaluationIntegrityError("development qrel restriction retained a held-out edge")
    return result.reset_index(drop=True), expected_queries


def _role_alias_queries(
    aliases: pd.DataFrame,
    profiles: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    role: str,
    maximum_per_occupation: int,
    seed: int,
) -> pd.DataFrame:
    role_codes = tuple(
        fold_manifest.loc[fold_manifest["role"].eq(role), "onet_soc_code"].astype(str)
    )
    result = prepare_alias_queries(
        aliases,
        profiles,
        fold_manifest,
        maximum_per_occupation=maximum_per_occupation,
        seed=seed,
        allowed_query_codes=role_codes,
        ambiguity_scope="full_catalog" if role == "test" else "allowed_codes",
    ).reset_index(drop=True)
    if result.empty or set(result["role"].astype(str)) != {role}:
        raise EvaluationIntegrityError(
            f"{role} alias preparation lacks queries or crossed its frozen role boundary"
        )
    return result


def run_development_selection(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    ratings: pd.DataFrame,
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    title_candidates: Sequence[CandidateSpec] | None = None,
    transition_candidates: Sequence[CandidateSpec] | None = None,
    maximum_aliases_per_occupation: int = 25,
    minimum_structured_overlap: int = 5,
    min_df: int = 2,
    random_seed: int = 20260814,
    bootstrap_replicates: int = 2000,
) -> DevelopmentSelectionRun:
    """Evaluate the full candidate grids once on the selection role only."""

    title_specs = tuple(title_candidates or title_candidate_grid())
    transition_specs = tuple(transition_candidates or transition_candidate_grid())
    role_aliases = _role_alias_queries(
        aliases,
        profiles,
        fold_manifest,
        role="selection",
        maximum_per_occupation=maximum_aliases_per_occupation,
        seed=random_seed,
    )
    title_catalog, title_runtimes = _fit_title_candidate_runtimes(
        profiles,
        aliases,
        fold_manifest,
        role_aliases,
        title_specs,
        min_df=min_df,
        random_seed=random_seed,
        phase="selection",
    )
    title_ranks = _title_target_rank_rows(role_aliases, title_catalog, title_runtimes)
    title_results = _title_selection_frame(title_ranks, title_specs, role="selection")
    title_selection = select_candidate_one_se(
        title_results,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=random_seed,
    )
    transition_catalog, transition_runtimes = _fit_transition_candidate_runtimes(
        profiles,
        ratings,
        fold_manifest,
        transition_specs,
        minimum_structured_overlap=minimum_structured_overlap,
        min_df=min_df,
        random_seed=random_seed,
        phase="selection",
    )
    transition_rankings = _transition_ranking_rows(
        fold_manifest,
        transition_catalog,
        transition_runtimes,
        role="selection",
    )
    role_qrels, query_codes = _development_qrels_for_role(qrels, fold_manifest, role="selection")
    transition_results = _transition_selection_frame(
        transition_rankings,
        role_qrels,
        fold_manifest,
        transition_specs,
        role="selection",
        expected_query_codes=query_codes,
        runtimes=transition_runtimes,
    )
    transition_selection = select_candidate_one_se(
        transition_results,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=random_seed,
    )
    return DevelopmentSelectionRun(
        alias_queries=role_aliases,
        title_target_ranks=title_ranks,
        transition_rankings=transition_rankings,
        title_results=title_results,
        transition_results=transition_results,
        title_selection=title_selection,
        transition_selection=transition_selection,
        transition_query_codes=query_codes,
    )


def run_development_confirmation(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    ratings: pd.DataFrame,
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    title_proposed_candidate_id: str,
    transition_proposed_candidate_id: str,
    title_candidates: Sequence[CandidateSpec] | None = None,
    transition_candidates: Sequence[CandidateSpec] | None = None,
    title_baseline_id: str = "title_word_char",
    transition_baseline_id: str = "transition_title_word_char",
    maximum_aliases_per_occupation: int = 25,
    minimum_structured_overlap: int = 5,
    min_df: int = 2,
    random_seed: int = 20260814,
) -> DevelopmentConfirmationRun:
    """Evaluate only each frozen proposal and baseline on confirmation."""

    title_by_id = {
        candidate.candidate_id: candidate
        for candidate in (title_candidates or title_candidate_grid())
    }
    transition_by_id = {
        candidate.candidate_id: candidate
        for candidate in (transition_candidates or transition_candidate_grid())
    }

    def proposed_specs(
        by_id: Mapping[str, CandidateSpec], proposed_id: str, baseline_id: str
    ) -> tuple[CandidateSpec, ...]:
        required = {proposed_id, baseline_id}
        if missing := sorted(required - set(by_id)):
            raise EvaluationIntegrityError(f"confirmation grid lacks candidates: {missing}")
        return tuple(by_id[candidate_id] for candidate_id in sorted(required))

    title_specs = proposed_specs(title_by_id, title_proposed_candidate_id, title_baseline_id)
    transition_specs = proposed_specs(
        transition_by_id,
        transition_proposed_candidate_id,
        transition_baseline_id,
    )
    role_aliases = _role_alias_queries(
        aliases,
        profiles,
        fold_manifest,
        role="confirmation",
        maximum_per_occupation=maximum_aliases_per_occupation,
        seed=random_seed,
    )
    title_catalog, title_runtimes = _fit_title_candidate_runtimes(
        profiles,
        aliases,
        fold_manifest,
        role_aliases,
        title_specs,
        min_df=min_df,
        random_seed=random_seed,
        phase="confirmation",
    )
    title_ranks = _title_target_rank_rows(role_aliases, title_catalog, title_runtimes)
    title_results = _title_selection_frame(title_ranks, title_specs, role="confirmation")
    title_confirmation = apply_confirmation_gate(
        title_results,
        proposed_candidate_id=title_proposed_candidate_id,
        baseline_candidate_id=title_baseline_id,
    )
    transition_catalog, transition_runtimes = _fit_transition_candidate_runtimes(
        profiles,
        ratings,
        fold_manifest,
        transition_specs,
        minimum_structured_overlap=minimum_structured_overlap,
        min_df=min_df,
        random_seed=random_seed,
        phase="confirmation",
    )
    transition_rankings = _transition_ranking_rows(
        fold_manifest,
        transition_catalog,
        transition_runtimes,
        role="confirmation",
    )
    role_qrels, query_codes = _development_qrels_for_role(qrels, fold_manifest, role="confirmation")
    transition_results = _transition_selection_frame(
        transition_rankings,
        role_qrels,
        fold_manifest,
        transition_specs,
        role="confirmation",
        expected_query_codes=query_codes,
        runtimes=transition_runtimes,
    )
    transition_confirmation = apply_confirmation_gate(
        transition_results,
        proposed_candidate_id=transition_proposed_candidate_id,
        baseline_candidate_id=transition_baseline_id,
    )
    return DevelopmentConfirmationRun(
        alias_queries=role_aliases,
        title_target_ranks=title_ranks,
        transition_rankings=transition_rankings,
        title_results=title_results,
        transition_results=transition_results,
        title_confirmation=title_confirmation,
        transition_confirmation=transition_confirmation,
        transition_query_codes=query_codes,
    )


def run_development_benchmarks(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    ratings: pd.DataFrame,
    qrels: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    *,
    title_candidates: Sequence[CandidateSpec] | None = None,
    transition_candidates: Sequence[CandidateSpec] | None = None,
    title_baseline_id: str = "title_word_char",
    transition_baseline_id: str = "transition_title_word_char",
    maximum_aliases_per_occupation: int = 25,
    minimum_structured_overlap: int = 5,
    min_df: int = 2,
    random_seed: int = 20260814,
    bootstrap_replicates: int = 2000,
) -> DevelopmentBenchmarkRun:
    """Fit/evaluate every candidate using only the preregistered development roles.

    Related-occupation qrels enter only after all component score matrices have
    been fit.  Fold-zero aliases, queries, candidate occupations, and judgments
    are never touched by selection or confirmation.
    """

    selection = run_development_selection(
        profiles,
        aliases,
        ratings,
        qrels,
        fold_manifest,
        title_candidates=title_candidates,
        transition_candidates=transition_candidates,
        maximum_aliases_per_occupation=maximum_aliases_per_occupation,
        minimum_structured_overlap=minimum_structured_overlap,
        min_df=min_df,
        random_seed=random_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    confirmation = run_development_confirmation(
        profiles,
        aliases,
        ratings,
        qrels,
        fold_manifest,
        title_proposed_candidate_id=selection.title_selection.selected_candidate_id,
        transition_proposed_candidate_id=(selection.transition_selection.selected_candidate_id),
        title_candidates=title_candidates,
        transition_candidates=transition_candidates,
        title_baseline_id=title_baseline_id,
        transition_baseline_id=transition_baseline_id,
        maximum_aliases_per_occupation=maximum_aliases_per_occupation,
        minimum_structured_overlap=minimum_structured_overlap,
        min_df=min_df,
        random_seed=random_seed,
    )
    alias_queries = pd.concat(
        [selection.alias_queries, confirmation.alias_queries], ignore_index=True
    ).sort_values(["role", "query_id"], kind="stable")
    title_ranks = pd.concat(
        [selection.title_target_ranks, confirmation.title_target_ranks],
        ignore_index=True,
    ).sort_values(["phase", "candidate_id", "query_id"], kind="stable")
    transition_rankings = pd.concat(
        [selection.transition_rankings, confirmation.transition_rankings],
        ignore_index=True,
    ).sort_values(["phase", "candidate_id", "query_code", "rank"], kind="stable")
    return DevelopmentBenchmarkRun(
        alias_queries=alias_queries,
        title_target_ranks=title_ranks,
        transition_rankings=transition_rankings,
        title_selection_results=selection.title_results,
        title_confirmation_results=confirmation.title_results,
        transition_selection_results=selection.transition_results,
        transition_confirmation_results=confirmation.transition_results,
        title_selection=selection.title_selection,
        title_confirmation=confirmation.title_confirmation,
        transition_selection=selection.transition_selection,
        transition_confirmation=confirmation.transition_confirmation,
        transition_query_universe={
            "selection": selection.transition_query_codes,
            "confirmation": confirmation.transition_query_codes,
        },
    )


def generate_locked_rankings(
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
    ratings: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    decision: Mapping[str, Any],
    *,
    title_candidates: Sequence[CandidateSpec] | None = None,
    transition_candidates: Sequence[CandidateSpec] | None = None,
    title_baseline_id: str = "title_word_char",
    transition_baseline_id: str = "transition_title_word_char",
    maximum_aliases_per_occupation: int = 25,
    minimum_structured_overlap: int = 5,
    min_df: int = 2,
    random_seed: int = 20260814,
) -> pd.DataFrame:
    """Refit on non-test inputs and generate target-free compact locked rankings.

    Title evidence stores the exact full-catalog target rank (one row per alias),
    avoiding a multi-million-row redundant catalog expansion.  Transition
    evidence stores the complete top 20 required by every preregistered metric.
    No qrel object is accepted by this function.
    """

    verify_development_decision(decision)
    if decision.get("schema_version") != 2:
        raise EvaluationIntegrityError("locked multi-task generation requires decision schema v2")
    releases = decision["release_candidates"]
    title_specs_all = tuple(title_candidates or title_candidate_grid())
    transition_specs_all = tuple(transition_candidates or transition_candidate_grid())

    def selected_specs(
        candidates: Sequence[CandidateSpec], selected_id: str, baseline_id: str
    ) -> tuple[CandidateSpec, ...]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        required = {selected_id, baseline_id}
        if missing := sorted(required - set(by_id)):
            raise EvaluationIntegrityError(f"locked candidate grid lacks: {missing}")
        return tuple(by_id[candidate_id] for candidate_id in sorted(required))

    title_specs = selected_specs(title_specs_all, releases["title"], title_baseline_id)
    transition_specs = selected_specs(
        transition_specs_all, releases["transition"], transition_baseline_id
    )
    test_codes = tuple(
        fold_manifest.loc[fold_manifest["role"].eq("test"), "onet_soc_code"].astype(str)
    )
    test_aliases = prepare_alias_queries(
        aliases,
        profiles,
        fold_manifest,
        maximum_per_occupation=maximum_aliases_per_occupation,
        seed=random_seed,
        allowed_query_codes=test_codes,
        ambiguity_scope="full_catalog",
    )
    test_aliases = test_aliases.reset_index(drop=True)
    if test_aliases.empty:
        raise EvaluationIntegrityError("locked title fold has no eligible alternate-title queries")
    title_catalog, title_runtimes = _fit_title_candidate_runtimes(
        profiles,
        aliases,
        fold_manifest,
        test_aliases,
        title_specs,
        min_df=min_df,
        random_seed=random_seed,
        phase="locked_test",
    )
    target_ranks = _title_target_rank_rows(test_aliases, title_catalog, title_runtimes)
    title_locked = pd.DataFrame(
        {
            "benchmark": "title",
            "candidate_id": target_ranks["candidate_id"],
            "query_id": target_ranks["query_id"],
            "query_code": target_ranks["query_code"],
            "candidate_code": target_ranks["target_code"],
            "rank": target_ranks["rank"],
            "score": target_ranks["target_score"],
            "split": "locked_test",
        }
    )

    transition_catalog, transition_runtimes = _fit_transition_candidate_runtimes(
        profiles,
        ratings,
        fold_manifest,
        transition_specs,
        minimum_structured_overlap=minimum_structured_overlap,
        min_df=min_df,
        random_seed=random_seed,
        phase="locked_test",
    )
    transition_ranked = _transition_ranking_rows(
        fold_manifest,
        transition_catalog,
        transition_runtimes,
        role="test",
        limit=20,
    )
    transition_locked = pd.DataFrame(
        {
            "benchmark": "transition",
            "candidate_id": transition_ranked["candidate_id"],
            "query_id": transition_ranked["query_code"],
            "query_code": transition_ranked["query_code"],
            "candidate_code": transition_ranked["candidate_code"],
            "rank": transition_ranked["rank"],
            "score": transition_ranked["score"],
            "split": "locked_test",
        }
    )
    combined = pd.concat([title_locked, transition_locked], ignore_index=True)
    combined = combined.loc[:, list(LOCKED_RANKING_COLUMNS)].sort_values(
        ["benchmark", "candidate_id", "query_id", "rank"], kind="stable"
    )
    return validate_locked_rankings(combined.reset_index(drop=True))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_implementation_manifest(repository: str | Path) -> dict[str, Any]:
    """Hash every tracked file that can affect the frozen experiment."""

    root = Path(repository).resolve()
    top = Path(_git_output(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != root:
        raise EvaluationIntegrityError("implementation repository must be the Git top level")
    tracked = _git_output(root, "ls-files", "-z").stdout.split("\0")
    selected = sorted(
        path
        for path in tracked
        if path
        and (
            path in IMPLEMENTATION_EXACT_PATHS
            or any(path.startswith(prefix) for prefix in IMPLEMENTATION_ROOT_PREFIXES)
        )
    )
    if not selected:
        raise EvaluationIntegrityError("implementation manifest scope is empty")
    entries: list[dict[str, Any]] = []
    for relative in selected:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise EvaluationIntegrityError(f"implementation file is missing or unsafe: {relative}")
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
        )
    core = {
        "schema_version": 1,
        "scope": {
            "prefixes": list(IMPLEMENTATION_ROOT_PREFIXES),
            "exact_paths": sorted(IMPLEMENTATION_EXACT_PATHS),
        },
        "files": entries,
    }
    return {
        **core,
        "implementation_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def verify_implementation_manifest(manifest: Mapping[str, Any], repository: str | Path) -> str:
    """Reject code/config/protocol drift after the prepare boundary."""

    expected = build_implementation_manifest(repository)
    if dict(manifest) != expected:
        raise EvaluationIntegrityError("experiment implementation differs from preparation")
    return str(expected["implementation_sha256"])


def require_clean_git_commit(repository: str | Path) -> GitSnapshot:
    """Require a real commit with no tracked or untracked worktree changes."""

    root = Path(repository).resolve()
    if not root.is_dir():
        raise EvaluationIntegrityError(f"repository does not exist: {root}")

    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvaluationIntegrityError(
                f"Git integrity check failed: {' '.join(arguments)}"
            ) from exc
        return completed.stdout.strip()

    top_level = Path(run("rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise EvaluationIntegrityError("repository path must be the Git top level")
    commit = run("rev-parse", "HEAD")
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise EvaluationIntegrityError("Git HEAD is not a full commit hash")
    if run("status", "--porcelain=v1", "--untracked-files=all"):
        raise EvaluationIntegrityError("locked evaluation requires a completely clean worktree")
    # Also prove HEAD resolves to a commit object, not merely a hexadecimal ref.
    if run("cat-file", "-t", commit) != "commit":
        raise EvaluationIntegrityError("Git HEAD does not resolve to a commit object")
    return GitSnapshot(top_level, commit)


def _git_output(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationIntegrityError(
            f"Git integrity check failed: {' '.join(arguments)}"
        ) from exc


def _dirty_paths(repository: Path) -> set[str]:
    completed = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    tokens = [token for token in completed.stdout.split("\0") if token]
    paths: set[str] = set()
    for token in tokens:
        if len(token) < 4 or token[2] != " ":
            # Rename/copy continuation records and malformed status output are
            # never accepted at the reporting boundary.
            raise EvaluationIntegrityError("Git status contains an unsupported rename/copy record")
        status = token[:2]
        if "R" in status or "C" in status:
            raise EvaluationIntegrityError("renames/copies are forbidden during locked reporting")
        paths.add(token[3:])
    return paths


def require_reporting_git_state(
    repository: str | Path,
    *,
    state: Mapping[str, Any],
    output_dir: str | Path,
) -> GitSnapshot:
    """Allow only locked output dirt at the exact original code/input commit."""

    root = Path(repository).resolve()
    if not root.is_dir():
        raise EvaluationIntegrityError(f"repository does not exist: {root}")
    top_level = Path(_git_output(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top_level != root:
        raise EvaluationIntegrityError("repository path must be the Git top level")
    current_commit = _git_output(root, "rev-parse", "HEAD").stdout.strip()
    original_commit = str(state.get("repository_commit", ""))
    if (
        COMMIT_PATTERN.fullmatch(current_commit) is None
        or COMMIT_PATTERN.fullmatch(original_commit) is None
    ):
        raise EvaluationIntegrityError("locked reporting commit provenance is invalid")
    if current_commit != original_commit:
        raise EvaluationIntegrityError(
            "locked reporting requires HEAD to equal the original locked-run commit"
        )
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(root):
        raise EvaluationIntegrityError("locked output directory must be inside the repository")
    relative = destination.relative_to(root)
    allowed = {
        (relative / STATE_FILENAME).as_posix(),
        (relative / RANKINGS_FILENAME).as_posix(),
        (relative / METRICS_FILENAME).as_posix(),
        *((relative / filename).as_posix() for filename in REPORT_OUTPUT_FILENAMES),
    }
    dirty = _dirty_paths(root)
    unexpected = dirty - allowed
    if unexpected:
        raise EvaluationIntegrityError(
            "locked reporting found code/config/input changes outside output files: "
            + ", ".join(sorted(unexpected)[:5])
        )
    return GitSnapshot(root, current_commit)


def verify_input_pins(
    expected_inputs: Mapping[str, tuple[str | Path, str]],
    *,
    repository: str | Path,
) -> tuple[InputPin, ...]:
    """Verify every frozen input path and digest before the run intent is written."""

    root = Path(repository).resolve()
    if not expected_inputs:
        raise EvaluationIntegrityError("at least one frozen input pin is required")
    pins: list[InputPin] = []
    for logical_name in sorted(expected_inputs):
        if not logical_name or logical_name != logical_name.strip():
            raise EvaluationIntegrityError("input logical names must be trimmed and nonblank")
        raw_path, expected_digest = expected_inputs[logical_name]
        if SHA256_PATTERN.fullmatch(expected_digest) is None:
            raise EvaluationIntegrityError(f"input {logical_name} has an invalid SHA-256 pin")
        path = Path(raw_path).resolve()
        if not path.is_relative_to(root):
            raise EvaluationIntegrityError(f"input {logical_name} is outside the repository")
        if not path.is_file() or path.is_symlink():
            raise EvaluationIntegrityError(f"input {logical_name} is missing or unsafe")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise EvaluationIntegrityError(f"input checksum mismatch for {logical_name}")
        pins.append(InputPin(logical_name, path, actual, path.stat().st_size))
    return tuple(pins)


def verify_stored_inputs(
    state: Mapping[str, Any],
    *,
    repository: str | Path,
) -> tuple[InputPin, ...]:
    """Reverify state-bound path, byte count, and hash on every report/resume."""

    root = Path(repository).resolve()
    raw_inputs = state.get("inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise EvaluationIntegrityError("locked state has no frozen input provenance")
    pins: list[InputPin] = []
    for logical_name in sorted(raw_inputs):
        entry = raw_inputs[logical_name]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256", "byte_count"}:
            raise EvaluationIntegrityError(f"stored input {logical_name} has invalid provenance")
        relative_text = entry["path"]
        if (
            not isinstance(relative_text, str)
            or not relative_text
            or Path(relative_text).is_absolute()
        ):
            raise EvaluationIntegrityError(f"stored input {logical_name} has an unsafe path")
        path = (root / relative_text).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise EvaluationIntegrityError(f"stored input {logical_name} is missing or unsafe")
        digest = entry["sha256"]
        byte_count = entry["byte_count"]
        if (
            not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise EvaluationIntegrityError(f"stored input {logical_name} metadata is invalid")
        if path.stat().st_size != byte_count or sha256_file(path) != digest:
            raise EvaluationIntegrityError(f"stored input changed after ranking for {logical_name}")
        pins.append(InputPin(str(logical_name), path, digest, byte_count))
    return tuple(pins)


def verify_development_decision(decision: Mapping[str, Any]) -> str:
    """Verify the self-hash produced before any locked-test evaluation."""

    payload = dict(decision)
    digest = payload.pop("decision_sha256", None)
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise EvaluationIntegrityError("development decision lacks a valid decision_sha256")
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        required = {
            "schema_version",
            "experiment_version",
            "release_candidate_id",
            "selection",
            "confirmation",
            "project_config_sha256",
            "fold_manifest_sha256",
            "candidate_grid_sha256",
        }
        release_candidates: Mapping[str, Any] = {"default": payload.get("release_candidate_id")}
        digest_fields = (
            "project_config_sha256",
            "fold_manifest_sha256",
            "candidate_grid_sha256",
        )
    elif schema_version == 2:
        required = {
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
        }
        release_candidates = payload.get("release_candidates", {})
        digest_fields = (
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
        )
    else:
        raise EvaluationIntegrityError("development decision schema is unsupported")
    if set(payload) != required:
        raise EvaluationIntegrityError("development decision fields differ from its schema")
    if (
        not payload["experiment_version"]
        or not isinstance(release_candidates, Mapping)
        or not release_candidates
        or any(not isinstance(value, str) or not value for value in release_candidates.values())
    ):
        raise EvaluationIntegrityError("development decision identity fields are blank")
    if schema_version == 2 and set(release_candidates) != {"title", "transition"}:
        raise EvaluationIntegrityError("multi-task decision must freeze title and transition")
    for field in digest_fields:
        if SHA256_PATTERN.fullmatch(str(payload[field])) is None:
            raise EvaluationIntegrityError(f"development decision {field} is invalid")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise EvaluationIntegrityError("development decision self-hash does not match")
    if schema_version == 2:
        _verify_multi_task_decision_semantics(payload)
    return digest


def _verify_multi_task_decision_semantics(decision: Mapping[str, Any]) -> None:
    grids = {
        "title": {candidate.candidate_id: candidate for candidate in title_candidate_grid()},
        "transition": {
            candidate.candidate_id: candidate for candidate in transition_candidate_grid()
        },
    }
    baselines = {
        "title": "title_word_char",
        "transition": "transition_title_word_char",
    }
    selection_fields = {field.name for field in fields(SelectionDecision)}
    confirmation_fields = {field.name for field in fields(ConfirmationDecision)}
    releases = decision["release_candidates"]
    for task in ("title", "transition"):
        selection = decision[f"{task}_selection"]
        confirmation = decision[f"{task}_confirmation"]
        if (
            not isinstance(selection, Mapping)
            or set(selection) != selection_fields
            or not isinstance(confirmation, Mapping)
            or set(confirmation) != confirmation_fields
        ):
            raise EvaluationIntegrityError(f"{task} decision fields differ from the contract")
        candidate_ids = set(grids[task])
        estimates_payload = selection["candidate_estimates"]
        if not isinstance(estimates_payload, (list, tuple)) or not estimates_payload:
            raise EvaluationIntegrityError(f"{task} selection estimates are invalid")
        estimates: dict[str, float] = {}
        for pair in estimates_payload:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not pair[0]
                or isinstance(pair[1], bool)
                or not isinstance(pair[1], (int, float))
                or not np.isfinite(float(pair[1]))
                or not 0.0 <= float(pair[1]) <= 1.0
                or pair[0] in estimates
                or pair[0] not in candidate_ids
            ):
                raise EvaluationIntegrityError(f"{task} selection estimates are invalid")
            estimates[pair[0]] = float(pair[1])
        if list(estimates) != sorted(estimates):
            raise EvaluationIntegrityError(f"{task} selection estimates are not canonical")

        tie_break_payload = selection["candidate_tie_breaks"]
        if not isinstance(tie_break_payload, (list, tuple)):
            raise EvaluationIntegrityError(f"{task} candidate tie-break evidence is invalid")
        tie_breaks: dict[str, tuple[int, int, int]] = {}
        for entry in tie_break_payload:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 4
                or not isinstance(entry[0], str)
                or entry[0] in tie_breaks
                or entry[0] not in estimates
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in entry[1:]
                )
            ):
                raise EvaluationIntegrityError(f"{task} candidate tie-break evidence is invalid")
            candidate = grids[task][entry[0]]
            if entry[1] != candidate.lsa_dimensions or entry[2] < 1 or entry[3] < 1:
                raise EvaluationIntegrityError(
                    f"{task} candidate tie-break evidence differs from its grid"
                )
            tie_breaks[entry[0]] = (entry[1], entry[2], entry[3])
        if list(tie_breaks) != sorted(tie_breaks) or set(tie_breaks) != set(estimates):
            raise EvaluationIntegrityError(
                f"{task} candidate tie-break evidence does not cover the evaluated grid"
            )

        selected_id = selection["selected_candidate_id"]
        metric_leader_id = selection["metric_leader_id"]
        unrestricted_leader_id = selection["unrestricted_metric_leader_id"]
        if any(
            not isinstance(value, str) or value not in estimates
            for value in (selected_id, metric_leader_id, unrestricted_leader_id)
        ):
            raise EvaluationIntegrityError(f"{task} selection names an unevaluated candidate")
        if baselines[task] not in estimates:
            raise EvaluationIntegrityError(f"{task} selection omitted its lexical baseline")

        ineligible_payload = selection["deployment_ineligible_candidates"]
        if not isinstance(ineligible_payload, (list, tuple)):
            raise EvaluationIntegrityError(f"{task} deployment eligibility evidence is invalid")
        ineligible: dict[str, str] = {}
        for entry in ineligible_payload:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or entry[0] in ineligible
                or entry[0] not in estimates
            ):
                raise EvaluationIntegrityError(f"{task} deployment eligibility evidence is invalid")
            ineligible[entry[0]] = entry[1]
        expected_ineligible = {
            candidate_id: reason
            for candidate_id in estimates
            for eligible, reason in (deployment_feasibility(grids[task][candidate_id]),)
            if not eligible
        }
        if ineligible != expected_ineligible:
            raise EvaluationIntegrityError(f"{task} deployment eligibility evidence differs")
        deployable = set(estimates) - set(ineligible)
        if not deployable:
            raise EvaluationIntegrityError(f"{task} selection has no deployable candidate")
        expected_unrestricted_leader = min(
            estimates, key=lambda candidate_id: (-estimates[candidate_id], candidate_id)
        )
        expected_metric_leader = min(
            deployable, key=lambda candidate_id: (-estimates[candidate_id], candidate_id)
        )
        if (
            unrestricted_leader_id != expected_unrestricted_leader
            or metric_leader_id != expected_metric_leader
        ):
            raise EvaluationIntegrityError(f"{task} selection leader IDs differ from estimates")

        leader_estimate = selection["leader_estimate"]
        standard_error = selection["leader_bootstrap_standard_error"]
        threshold = selection["one_se_threshold"]
        replicates = selection["bootstrap_replicates"]
        bootstrap_seed = selection["bootstrap_seed"]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in (leader_estimate, standard_error, threshold)
            )
            or isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates < 2
            or (
                isinstance(bootstrap_seed, bool)
                or not isinstance(bootstrap_seed, int)
                or bootstrap_seed < 0
            )
        ):
            raise EvaluationIntegrityError(f"{task} one-SE evidence is invalid")
        expected_leader_estimate = estimates[metric_leader_id]
        expected_threshold = expected_leader_estimate - float(standard_error)
        if (
            float(leader_estimate) != expected_leader_estimate
            or float(standard_error) < 0
            or not np.isclose(float(threshold), expected_threshold, rtol=0, atol=1e-12)
        ):
            raise EvaluationIntegrityError(f"{task} one-SE evidence differs from estimates")
        eligible_payload = selection["eligible_candidate_ids"]
        if (
            not isinstance(eligible_payload, (list, tuple))
            or any(not isinstance(value, str) for value in eligible_payload)
            or list(eligible_payload) != sorted(set(eligible_payload))
        ):
            raise EvaluationIntegrityError(f"{task} one-SE eligible candidates are invalid")
        expected_eligible = sorted(
            candidate_id
            for candidate_id in deployable
            if estimates[candidate_id] >= float(threshold) - 1e-15
        )
        if list(eligible_payload) != expected_eligible or selected_id not in expected_eligible:
            raise EvaluationIntegrityError(f"{task} one-SE eligible set differs from estimates")
        expected_selected = min(
            expected_eligible,
            key=lambda candidate_id: (*tie_breaks[candidate_id], candidate_id),
        )
        if selected_id != expected_selected:
            raise EvaluationIntegrityError(f"{task} selection tie-break result is inconsistent")

        proposed_id = confirmation["proposed_candidate_id"]
        baseline_id = confirmation["baseline_candidate_id"]
        release_id = confirmation["release_candidate_id"]
        if selected_id != proposed_id:
            raise EvaluationIntegrityError(
                f"{task} confirmation proposal differs from frozen selection"
            )
        if baseline_id != baselines[task]:
            raise EvaluationIntegrityError(f"{task} confirmation baseline is not preregistered")
        if any(candidate_id not in grids[task] for candidate_id in (selected_id, release_id)):
            raise EvaluationIntegrityError(f"{task} decision names an unknown candidate")
        proposed_estimate = confirmation["proposed_estimate"]
        baseline_estimate = confirmation["baseline_estimate"]
        fallback = confirmation["fallback_applied"]
        if (
            isinstance(fallback, bool) is False
            or not isinstance(proposed_estimate, (int, float))
            or not isinstance(baseline_estimate, (int, float))
            or not np.isfinite(float(proposed_estimate))
            or not np.isfinite(float(baseline_estimate))
            or not 0.0 <= float(proposed_estimate) <= 1.0
            or not 0.0 <= float(baseline_estimate) <= 1.0
        ):
            raise EvaluationIntegrityError(f"{task} confirmation evidence is invalid")
        expected_fallback = float(proposed_estimate) < float(baseline_estimate)
        expected_release = baseline_id if expected_fallback else proposed_id
        if fallback != expected_fallback or release_id != expected_release:
            raise EvaluationIntegrityError(f"{task} fallback/release semantics are inconsistent")
        if releases[task] != release_id:
            raise EvaluationIntegrityError(f"{task} release cross-link differs from confirmation")
        deployable, reason = deployment_feasibility(grids[task][release_id])
        if not deployable:
            raise EvaluationIntegrityError(
                f"{task} release candidate is deployment-ineligible: {reason}"
            )
        if selection["operational_cost_policy"] != OPERATIONAL_COST_POLICY:
            raise EvaluationIntegrityError(f"{task} operational cost policy differs")


def begin_locked_run(
    output_dir: str | Path,
    *,
    repository: str | Path,
    experiment_version: str,
    decision: Mapping[str, Any],
    expected_inputs: Mapping[str, tuple[str | Path, str]],
) -> LockedRunSession:
    """Record one-shot run intent after verifying clean Git, decision, and inputs."""

    repository_path = Path(repository).resolve()
    early_destination = Path(output_dir).resolve()
    early_session = LockedRunSession(early_destination)
    if any(
        path.exists()
        for path in (
            early_session.state_path,
            early_session.rankings_path,
            early_session.metrics_path,
        )
    ):
        raise EvaluationIntegrityError(
            "locked run already started; only reporting-only resume is permitted"
        )
    git = require_clean_git_commit(repository_path)
    if not experiment_version or experiment_version != experiment_version.strip():
        raise EvaluationIntegrityError("experiment_version must be trimmed and nonblank")
    decision_digest = verify_development_decision(decision)
    if decision["experiment_version"] != experiment_version:
        raise EvaluationIntegrityError("decision and locked-run experiment versions differ")
    pins = verify_input_pins(expected_inputs, repository=git.repository)
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(git.repository):
        raise EvaluationIntegrityError("locked output directory must be inside the repository")
    if any(pin.path.is_relative_to(destination) for pin in pins):
        raise EvaluationIntegrityError("frozen inputs cannot be stored inside locked output")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise EvaluationIntegrityError("locked output directory cannot be a symlink")
    session = LockedRunSession(destination)
    if any(
        path.exists() for path in (session.state_path, session.rankings_path, session.metrics_path)
    ):
        raise EvaluationIntegrityError(
            "locked run already started; only reporting-only resume is permitted"
        )
    release_candidates = (
        dict(decision["release_candidates"])
        if decision["schema_version"] == 2
        else {"default": decision["release_candidate_id"]}
    )
    intent = {
        "schema_version": LOCKED_STATE_SCHEMA_VERSION,
        "experiment_version": experiment_version,
        "split": "locked_test",
        "repository_commit": git.commit,
        "decision_sha256": decision_digest,
        "release_candidates": release_candidates,
        "inputs": {
            pin.logical_name: {
                "path": pin.path.relative_to(git.repository).as_posix(),
                "sha256": pin.sha256,
                "byte_count": pin.byte_count,
            }
            for pin in pins
        },
    }
    intent_sha256 = hashlib.sha256(
        json.dumps(intent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    state = {
        **intent,
        "schema_version": LOCKED_STATE_SCHEMA_VERSION,
        "status": "started",
        "intent_sha256": intent_sha256,
        "rankings": None,
        "metrics": None,
    }
    _atomic_json(session.state_path, state, overwrite=False)
    return session


def resume_reporting_only(
    output_dir: str | Path,
    *,
    repository: str | Path,
) -> LockedRunSession:
    """Open an existing run only after validating immutable ranking bytes."""

    repository_path = Path(repository).resolve()
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(repository_path):
        raise EvaluationIntegrityError("locked output directory must be inside the repository")
    session = LockedRunSession(destination)
    state = session.state()
    git = require_reporting_git_state(
        repository_path,
        state=state,
        output_dir=destination,
    )
    verify_stored_inputs(state, repository=git.repository)
    if state["status"] not in {"rankings_written", "complete"}:
        raise EvaluationIntegrityError(
            "an interrupted pre-hash run cannot be resumed; create a new experiment version"
        )
    session.load_rankings_for_reporting()
    if state["status"] == "complete":
        session.load_metrics_for_reporting()
    return session


def validate_locked_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Validate exact target-free row-level ranking schema and stable sort."""

    if list(rankings.columns) != list(LOCKED_RANKING_COLUMNS):
        raise EvaluationIntegrityError(
            f"locked ranking columns must be exactly {LOCKED_RANKING_COLUMNS!r}"
        )
    frame = rankings.copy()
    if frame.empty:
        raise EvaluationIntegrityError("locked rankings must be nonempty")
    for column in (
        "benchmark",
        "candidate_id",
        "query_id",
        "query_code",
        "candidate_code",
        "split",
    ):
        frame[column] = frame[column].astype("string").str.strip()
        if bool(frame[column].isna().any() | frame[column].eq("").any()):
            raise EvaluationIntegrityError(f"locked rankings {column} contains blanks")
    if set(frame["benchmark"].astype(str)) - {"title", "transition"}:
        raise EvaluationIntegrityError("locked rankings contain an unknown benchmark")
    if set(frame["split"].astype(str)) != {"locked_test"}:
        raise EvaluationIntegrityError("locked rankings must contain only locked_test rows")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    if bool(
        frame["rank"].isna().any() or frame["rank"].lt(1).any() or frame["rank"].mod(1).ne(0).any()
    ):
        raise EvaluationIntegrityError("locked ranks must be positive integers")
    frame["rank"] = frame["rank"].astype("int64")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").astype("float64")
    if not np.isfinite(frame["score"]).all():
        raise EvaluationIntegrityError("locked ranking scores must be finite")
    group_key = ["benchmark", "candidate_id", "query_id"]
    if bool(frame.duplicated([*group_key, "rank"]).any()):
        raise EvaluationIntegrityError("locked rankings repeat a rank within a query")
    if bool(frame.duplicated([*group_key, "candidate_code"]).any()):
        raise EvaluationIntegrityError("locked rankings repeat a candidate within a query")
    for key, group in frame.groupby(group_key, sort=False):
        benchmark = str(group["benchmark"].iloc[0])
        if benchmark == "title":
            if len(group) != 1 or group["candidate_code"].iloc[0] != group["query_code"].iloc[0]:
                raise EvaluationIntegrityError(
                    f"locked title evidence for {key} must be one exact target-rank row"
                )
        else:
            if group["rank"].tolist() != list(range(1, len(group) + 1)):
                raise EvaluationIntegrityError(f"locked ranks for {key} are not contiguous/sorted")
            expected = group.sort_values(
                ["score", "candidate_code"], ascending=[False, True], kind="stable"
            )
            if group.index.tolist() != expected.index.tolist():
                raise EvaluationIntegrityError(
                    f"locked rankings for {key} violate score/code order"
                )
        if group["query_code"].nunique() != 1:
            raise EvaluationIntegrityError(f"locked query {key} maps to multiple occupation codes")
    expected_frame = frame.sort_values(
        ["benchmark", "candidate_id", "query_id", "rank"], kind="stable"
    )
    if frame.index.tolist() != expected_frame.index.tolist():
        raise EvaluationIntegrityError("locked ranking rows are not in canonical sort order")
    return frame.reset_index(drop=True)


def load_evaluation_qrels(path: str | Path, *, expected_sha256: str) -> pd.DataFrame:
    """Load Related Occupations only inside the evaluation module."""

    source = Path(path)
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise EvaluationIntegrityError("qrel checksum pin is invalid")
    if not source.is_file() or source.is_symlink() or sha256_file(source) != expected_sha256:
        raise EvaluationIntegrityError("qrel file is missing, unsafe, or differs from its pin")
    try:
        qrels = pd.read_parquet(source)
    except Exception as exc:
        raise EvaluationIntegrityError("qrel Parquet cannot be read") from exc
    missing = sorted(set(QREL_COLUMNS) - set(qrels.columns))
    if missing:
        raise EvaluationIntegrityError(f"qrels are missing columns: {missing}")
    return qrels


def evaluate_transition_rankings(
    locked_rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    candidate_id: str,
) -> dict[str, float | int]:
    """Aggregate one frozen transition candidate after ranking bytes exist."""

    validated = validate_locked_rankings(locked_rankings)
    selected = validated.loc[
        validated["benchmark"].eq("transition") & validated["candidate_id"].eq(candidate_id),
        ["query_code", "candidate_code", "rank"],
    ]
    if selected.empty:
        raise EvaluationIntegrityError(f"locked rankings lack transition candidate {candidate_id}")
    try:
        return transition_metrics(selected, qrels)
    except RankingMetricError as exc:
        raise EvaluationIntegrityError(str(exc)) from exc


def evaluate_title_rankings(
    locked_rankings: pd.DataFrame,
    *,
    candidate_id: str,
) -> dict[str, float | int]:
    """Macro-aggregate held-out aliases by occupation from immutable rankings."""

    validated = validate_locked_rankings(locked_rankings)
    selected = validated.loc[
        validated["benchmark"].eq("title") & validated["candidate_id"].eq(candidate_id)
    ].copy()
    if selected.empty:
        raise EvaluationIntegrityError(f"locked rankings lack title candidate {candidate_id}")
    queries = selected.loc[:, ["query_id", "query_code"]].drop_duplicates()
    if bool(queries["query_id"].duplicated().any()):
        raise EvaluationIntegrityError("a title query ID maps to multiple occupation codes")
    target_rows = selected.loc[
        selected["candidate_code"].eq(selected["query_code"]),
        ["query_id", "candidate_code", "rank"],
    ]
    if bool(target_rows["query_id"].duplicated().any()):
        raise EvaluationIntegrityError("a title ranking contains its target more than once")
    query_results = queries.merge(
        target_rows.loc[:, ["query_id", "rank"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    query_results["target_code"] = query_results["query_code"]
    try:
        return macro_title_metrics(
            query_results.loc[:, ["query_id", "query_code", "target_code", "rank"]]
        )
    except RankingMetricError as exc:
        raise EvaluationIntegrityError(str(exc)) from exc


def _title_query_frame(locked_rankings: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    validated = validate_locked_rankings(locked_rankings)
    selected = validated.loc[
        validated["benchmark"].eq("title") & validated["candidate_id"].eq(candidate_id)
    ].copy()
    if selected.empty:
        raise EvaluationIntegrityError(f"locked rankings lack title candidate {candidate_id}")
    queries = selected.loc[:, ["query_id", "query_code"]].drop_duplicates()
    if bool(queries["query_id"].duplicated().any()):
        raise EvaluationIntegrityError("a title query ID maps to multiple occupation codes")
    target_rows = selected.loc[
        selected["candidate_code"].eq(selected["query_code"]),
        ["query_id", "rank"],
    ]
    if bool(target_rows["query_id"].duplicated().any()):
        raise EvaluationIntegrityError("a title ranking contains its target more than once")
    query_results = queries.merge(target_rows, on="query_id", how="left", validate="one_to_one")
    query_results["target_code"] = query_results["query_code"]
    try:
        return title_query_metrics(
            query_results.loc[:, ["query_id", "query_code", "target_code", "rank"]]
        )
    except RankingMetricError as exc:
        raise EvaluationIntegrityError(str(exc)) from exc


def build_locked_descriptive_slices(
    locked_rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    profiles: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    title_candidate_id: str,
    transition_candidate_id: str,
    expected_transition_query_codes: Sequence[str],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260814,
) -> dict[str, list[dict[str, Any]]]:
    """Build locked, descriptive primary-metric slices at analysis-unit grain."""

    required_profiles = {
        "onet_soc_code",
        "soc_major_group",
        "job_zone",
        "structured_observed_count",
    }
    if missing := sorted(required_profiles - set(profiles.columns)):
        raise EvaluationIntegrityError(f"slice profiles are missing columns: {missing}")
    required_ratings = {"onet_soc_code", "observed", "rating_date"}
    if missing := sorted(required_ratings - set(ratings.columns)):
        raise EvaluationIntegrityError(f"slice ratings are missing columns: {missing}")
    metadata = profiles.loc[:, sorted(required_profiles)].copy()
    metadata["onet_soc_code"] = metadata["onet_soc_code"].astype("string")
    if metadata.empty or bool(metadata["onet_soc_code"].duplicated().any()):
        raise EvaluationIntegrityError("slice profiles must be unique by occupation code")
    metadata["soc_major_group"] = metadata["soc_major_group"].astype("string")

    def canonical_job_zone(value: object) -> str:
        if pd.isna(value):
            return "missing"
        numeric = float(value)
        if not numeric.is_integer() or int(numeric) not in {2, 3, 4, 5}:
            raise EvaluationIntegrityError("slice job zones must use official labels 2, 3, 4, 5")
        return str(int(numeric))

    metadata["job_zone"] = metadata["job_zone"].map(canonical_job_zone).astype("string")
    observed_counts = pd.to_numeric(metadata["structured_observed_count"], errors="coerce")
    if bool(observed_counts.isna().any() | observed_counts.lt(0).any()):
        raise EvaluationIntegrityError("slice structured counts must be nonnegative")
    metadata["profile_completeness"] = pd.cut(
        observed_counts,
        bins=[-1, 0, 49, float("inf")],
        labels=["none", "1-49", "50+"],
    ).astype("string")
    recency = ratings.loc[ratings["observed"].eq(True)].copy()  # noqa: E712
    recency["rating_date"] = pd.to_datetime(recency["rating_date"], errors="coerce")
    newest = recency.groupby("onet_soc_code")["rating_date"].max()

    def recency_bucket(value: object) -> str:
        if pd.isna(value):
            return "missing"
        year = int(value.year)  # type: ignore[union-attr]
        if year >= 2024:
            return "2024+"
        if year >= 2021:
            return "2021-2023"
        return "through-2020"

    metadata["rating_recency"] = metadata["onet_soc_code"].map(newest).map(recency_bucket)

    title = _title_query_frame(locked_rankings, title_candidate_id)
    title_units = (
        title.groupby("target_code", sort=True)["top1_accuracy"]
        .mean()
        .rename("primary_value")
        .reset_index()
        .rename(columns={"target_code": "query_code"})
    )
    transition_rows = validate_locked_rankings(locked_rankings).loc[
        lambda frame: (
            frame["benchmark"].eq("transition")
            & frame["candidate_id"].eq(transition_candidate_id)
            & frame["query_code"].isin(tuple(str(code) for code in expected_transition_query_codes))
        ),
        ["query_code", "candidate_code", "rank"],
    ]
    try:
        transition_units = transition_query_metrics(
            transition_rows,
            qrels,
            expected_query_codes=expected_transition_query_codes,
        ).loc[:, ["query_code", "ndcg_at_10"]]
    except RankingMetricError as exc:
        raise EvaluationIntegrityError(str(exc)) from exc
    transition_units = transition_units.rename(columns={"ndcg_at_10": "primary_value"})

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.merge(
            metadata,
            left_on="query_code",
            right_on="onet_soc_code",
            how="left",
            validate="one_to_one",
        )
        if bool(result["onet_soc_code"].isna().any()):
            raise EvaluationIntegrityError("slice units lack occupation metadata")
        return result

    dimensions = (
        "soc_major_group",
        "job_zone",
        "profile_completeness",
        "rating_recency",
    )

    def rows_for(units: pd.DataFrame, *, seed_offset: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        expected_units = len(units)
        for dimension_index, dimension in enumerate(dimensions):
            dimension_total = 0
            for value, group in units.groupby(dimension, dropna=False, sort=True):
                label = "missing" if pd.isna(value) else str(value)
                strata = group["soc_major_group"].astype("string").fillna("missing")
                interval = stratified_bootstrap_interval(
                    group["primary_value"],
                    strata,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + seed_offset + dimension_index * 100 + len(rows),
                )
                dimension_total += len(group)
                rows.append(
                    {
                        "dimension": dimension,
                        "value": label,
                        "estimate": interval.estimate,
                        "ci_lower": interval.ci_lower,
                        "ci_upper": interval.ci_upper,
                        "n_units": int(len(group)),
                        "descriptive_small_slice": bool(len(group) < 20),
                    }
                )
            if dimension_total != expected_units:
                raise EvaluationIntegrityError("slice dimension lost analysis units")
        return rows

    return {
        "title": rows_for(attach(title_units), seed_offset=0),
        "transition": rows_for(attach(transition_units), seed_offset=10_000),
    }


def build_title_benchmark_evidence(
    locked_rankings: pd.DataFrame,
    *,
    selected_candidate_id: str,
    baseline_candidate_id: str,
    slices: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260814,
) -> dict[str, Any]:
    """Build macro-occupation title estimates, intervals, and paired deltas."""

    selected_query = _title_query_frame(locked_rankings, selected_candidate_id)
    baseline_query = _title_query_frame(locked_rankings, baseline_candidate_id)
    selected_ids = set(selected_query["query_id"])
    if selected_ids != set(baseline_query["query_id"]):
        raise EvaluationIntegrityError("title champion and baseline use different alias queries")
    metric_names = ("top1_accuracy", "recall_at_5", "mrr")
    selected_units = selected_query.groupby("target_code", sort=True)[list(metric_names)].mean()
    baseline_units = baseline_query.groupby("target_code", sort=True)[list(metric_names)].mean()
    if not selected_units.index.equals(baseline_units.index):
        raise EvaluationIntegrityError("title champion and baseline cover different occupations")
    strata = pd.Series(
        [str(code)[:2] for code in selected_units.index],
        index=selected_units.index,
        dtype="string",
    )
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for offset, metric in enumerate(metric_names):
        interval = stratified_bootstrap_interval(
            selected_units[metric],
            strata,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + offset,
        )
        metrics.append(
            {
                "name": metric,
                "estimate": interval.estimate,
                "ci_lower": interval.ci_lower,
                "ci_upper": interval.ci_upper,
                "n_queries": int(len(selected_query)),
            }
        )
        if selected_candidate_id != baseline_candidate_id:
            delta = paired_stratified_bootstrap(
                selected_units[metric],
                baseline_units[metric],
                strata,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 100 + offset,
            )
            deltas.append(
                {
                    "baseline_candidate_id": baseline_candidate_id,
                    "metric": metric,
                    "delta": delta.estimate,
                    "ci_lower": delta.ci_lower,
                    "ci_upper": delta.ci_upper,
                }
            )
    return {
        "selected_candidate_id": selected_candidate_id,
        "primary_metric": "top1_accuracy",
        "metrics": metrics,
        "baseline_deltas": deltas,
        "slices": [dict(row) for row in slices],
    }


def build_transition_benchmark_evidence(
    locked_rankings: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    selected_candidate_id: str,
    baseline_candidate_id: str,
    expected_query_codes: Sequence[str],
    slices: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260814,
) -> dict[str, Any]:
    """Build query-macro transition estimates, intervals, and paired deltas."""

    validated = validate_locked_rankings(locked_rankings)

    def query_frame(candidate_id: str) -> pd.DataFrame:
        expected = tuple(sorted(str(code) for code in expected_query_codes))
        if not expected or len(expected) != len(set(expected)):
            raise EvaluationIntegrityError(
                "transition evidence query universe must be nonempty and unique"
            )
        selected = validated.loc[
            validated["benchmark"].eq("transition")
            & validated["candidate_id"].eq(candidate_id)
            & validated["query_code"].isin(expected),
            ["query_code", "candidate_code", "rank"],
        ]
        if selected.empty:
            raise EvaluationIntegrityError(
                f"locked rankings lack transition candidate {candidate_id}"
            )
        try:
            return transition_query_metrics(
                selected,
                qrels,
                expected_query_codes=expected,
            ).set_index("query_code")
        except RankingMetricError as exc:
            raise EvaluationIntegrityError(str(exc)) from exc

    selected_units = query_frame(selected_candidate_id)
    baseline_units = query_frame(baseline_candidate_id)
    if not selected_units.index.equals(baseline_units.index):
        raise EvaluationIntegrityError("transition champion and baseline use different queries")
    metric_names = (
        "ndcg_at_10",
        "primary_short_recall_at_5",
        "primary_recall_at_10",
        "ndcg_at_20",
        "mrr_at_10",
        "binary_primary_ndcg_at_10",
    )
    strata = pd.Series(
        [str(code)[:2] for code in selected_units.index],
        index=selected_units.index,
        dtype="string",
    )
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for offset, metric in enumerate(metric_names):
        interval = stratified_bootstrap_interval(
            selected_units[metric],
            strata,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + offset,
        )
        metrics.append(
            {
                "name": metric,
                "estimate": interval.estimate,
                "ci_lower": interval.ci_lower,
                "ci_upper": interval.ci_upper,
                "n_queries": int(len(selected_units)),
            }
        )
        if selected_candidate_id != baseline_candidate_id:
            delta = paired_stratified_bootstrap(
                selected_units[metric],
                baseline_units[metric],
                strata,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 100 + offset,
            )
            deltas.append(
                {
                    "baseline_candidate_id": baseline_candidate_id,
                    "metric": metric,
                    "delta": delta.estimate,
                    "ci_lower": delta.ci_lower,
                    "ci_upper": delta.ci_upper,
                }
            )
    return {
        "selected_candidate_id": selected_candidate_id,
        "primary_metric": "ndcg_at_10",
        "metrics": metrics,
        "baseline_deltas": deltas,
        "slices": [dict(row) for row in slices],
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvaluationIntegrityError("locked run state is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError("locked run state is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != LOCKED_STATE_SCHEMA_VERSION:
        raise EvaluationIntegrityError("locked run state schema is unsupported")
    expected = {
        "schema_version",
        "experiment_version",
        "split",
        "status",
        "intent_sha256",
        "repository_commit",
        "decision_sha256",
        "release_candidates",
        "inputs",
        "rankings",
        "metrics",
    }
    if set(value) != expected or value.get("split") != "locked_test":
        raise EvaluationIntegrityError("locked run state fields differ from the contract")
    if value.get("status") not in {"started", "rankings_written", "complete"}:
        raise EvaluationIntegrityError("locked run state has an unknown status")
    intent = {
        key: value[key]
        for key in (
            "schema_version",
            "experiment_version",
            "split",
            "repository_commit",
            "decision_sha256",
            "release_candidates",
            "inputs",
        )
    }
    expected_intent = hashlib.sha256(
        json.dumps(intent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if value.get("intent_sha256") != expected_intent:
        raise EvaluationIntegrityError("locked run intent hash does not match state")
    return value


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvaluationIntegrityError(f"{description} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIntegrityError(f"{description} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationIntegrityError(f"{description} must be a JSON object")
    _reject_nonfinite_json(value)
    return value


def _reject_constant(value: str) -> None:
    raise EvaluationIntegrityError(f"non-finite JSON constant is prohibited: {value}")


def _reject_nonfinite_json(value: Any) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise EvaluationIntegrityError("metric report contains a non-finite number")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite_json(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_nonfinite_json(child)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise EvaluationIntegrityError(f"refusing to overwrite locked file: {path.name}")
    _reject_nonfinite_json(value)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
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
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "DevelopmentBenchmarkRun",
    "DevelopmentConfirmationRun",
    "DevelopmentSelectionRun",
    "EvaluationIntegrityError",
    "GitSnapshot",
    "InputPin",
    "LOCKED_RANKING_COLUMNS",
    "LOCKED_RANKING_SCHEMA",
    "TITLE_QUERY_POLICY",
    "LockedRunSession",
    "begin_locked_run",
    "build_implementation_manifest",
    "build_locked_descriptive_slices",
    "build_evaluation_universe",
    "build_transition_evaluation_universe",
    "build_title_benchmark_evidence",
    "build_transition_benchmark_evidence",
    "evaluate_title_rankings",
    "evaluate_transition_rankings",
    "generate_locked_rankings",
    "load_evaluation_qrels",
    "require_clean_git_commit",
    "require_reporting_git_state",
    "run_development_benchmarks",
    "run_development_confirmation",
    "run_development_selection",
    "resume_reporting_only",
    "sha256_file",
    "transition_evaluation_query_codes",
    "validate_transition_evaluation_universe",
    "validate_locked_rankings",
    "validate_evaluation_universe",
    "verify_development_decision",
    "verify_input_pins",
    "verify_implementation_manifest",
    "verify_stored_inputs",
]
