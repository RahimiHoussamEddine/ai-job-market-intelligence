"""Deterministic occupation profiles and missing-aware structured vectors."""

from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from .crosswalk import (
    CrosswalkError,
    base_soc_code,
    canonical_onet_soc_code,
    soc_major_group,
)


class ProfileError(ValueError):
    """Raised when source rows cannot form leakage-safe deterministic profiles."""


SCALE_BOUNDS: Mapping[str, tuple[float, float]] = {
    "IM": (1.0, 5.0),
    "LV": (0.0, 7.0),
}


@dataclass(frozen=True, slots=True)
class StructuredMatrix:
    occupation_codes: tuple[str, ...]
    feature_keys: tuple[str, ...]
    values: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.occupation_codes), len(self.feature_keys))
        if self.values.shape != expected or self.observed.shape != expected:
            raise ProfileError(
                f"Structured matrix shapes must be {expected}; got "
                f"{self.values.shape} and {self.observed.shape}"
            )
        if self.observed.dtype != np.bool_:
            raise ProfileError("Structured missingness mask must have boolean dtype")
        if not np.isfinite(self.values).all():
            raise ProfileError("Structured matrix values must be finite; missingness uses the mask")
        if np.any((self.values[self.observed] < 0) | (self.values[self.observed] > 1)):
            raise ProfileError("Observed normalized structured values must lie in [0, 1]")


def normalize_text(value: object, *, casefold: bool = False) -> str:
    """NFKC/entity/whitespace normalization that preserves technical tokens."""

    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = "".join(character for character in text if unicodedata.category(character) != "Cf")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold() if casefold else text


def stable_text_id(*parts: object) -> str:
    payload = "\x1f".join(normalize_text(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(frame: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ProfileError(f"{table} is missing columns: {missing}")


def _canonical_codes(values: Iterable[object], table: str) -> list[str]:
    result: list[str] = []
    for value in values:
        try:
            result.append(canonical_onet_soc_code(value))
        except CrosswalkError as exc:
            raise ProfileError(f"{table}: {exc}") from exc
    return result


def _parse_date(value: object, field: str) -> date | None:
    text = normalize_text(value)
    if not text:
        return None
    for pattern in ("%Y-%m-%d", "%m/%Y", "%Y-%m"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ProfileError(f"{field}: invalid date {value!r}")


def build_aliases(job_titles: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        job_titles,
        ["onet_soc_code", "job_title", "short_title", "source"],
        "job_titles",
    )
    rows: list[dict[str, object]] = []
    for row in job_titles.to_dict("records"):
        code = _canonical_codes([row["onet_soc_code"]], "job_titles")[0]
        title = normalize_text(row["job_title"])
        if not title:
            raise ProfileError("job_titles: blank job title")
        normalized = normalize_text(title, casefold=True)
        short_title = normalize_text(row.get("short_title")) or None
        sources = normalize_text(row.get("source")) or None
        rows.append(
            {
                "alias_id": stable_text_id(code, normalized, sources or ""),
                "onet_soc_code": code,
                "job_title": title,
                "normalized_job_title": normalized,
                "short_title": short_title,
                "sources": sources,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "alias_id",
                "onet_soc_code",
                "job_title",
                "normalized_job_title",
                "short_title",
                "sources",
            ]
        )
    if result["alias_id"].duplicated().any():
        raise ProfileError("job_titles: duplicate deterministic alias IDs")
    return result.sort_values(
        ["onet_soc_code", "normalized_job_title", "job_title", "sources"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _natural_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", normalize_text(value))
    )


def build_tasks(task_statements: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        task_statements,
        [
            "onet_soc_code",
            "task_id",
            "task",
            "task_type",
            "incumbents_responding",
            "date",
            "domain_source",
        ],
        "task_statements",
    )
    rows: list[dict[str, object]] = []
    for row in task_statements.to_dict("records"):
        code = _canonical_codes([row["onet_soc_code"]], "task_statements")[0]
        task_id = normalize_text(row["task_id"])
        task = normalize_text(row["task"])
        task_type = normalize_text(row["task_type"])
        if not task_id or not task or not task_type:
            raise ProfileError("task_statements: task ID, text, and type are required")
        incumbents_text = normalize_text(row.get("incumbents_responding"))
        incumbents: int | None
        if incumbents_text:
            try:
                incumbents = int(incumbents_text.replace(",", ""))
            except ValueError as exc:
                raise ProfileError(
                    f"task_statements: invalid incumbents count {incumbents_text!r}"
                ) from exc
            if incumbents < 0:
                raise ProfileError("task_statements: incumbents count cannot be negative")
        else:
            incumbents = None
        rows.append(
            {
                "onet_soc_code": code,
                "task_id": task_id,
                "task": task,
                "normalized_task": normalize_text(task, casefold=True),
                "task_type": task_type,
                "incumbents_responding": incumbents,
                "task_date": _parse_date(row.get("date"), "task_statements.date"),
                "domain_source": normalize_text(row.get("domain_source")) or None,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ProfileError("task_statements: no task rows")
    if result.duplicated(["onet_soc_code", "task_id"]).any():
        raise ProfileError("task_statements: duplicate (occupation, task_id) keys")
    result["incumbents_responding"] = pd.array(result["incumbents_responding"], dtype="Int64")
    result["_task_order"] = result["task_id"].map(_natural_key)
    result = result.sort_values(["onet_soc_code", "_task_order"], kind="stable").drop(
        columns="_task_order"
    )
    return result.reset_index(drop=True)


def build_occupations(
    occupation_data: pd.DataFrame,
    job_zones: pd.DataFrame | None = None,
    *,
    job_zone_levels: int = 4,
) -> pd.DataFrame:
    _require_columns(occupation_data, ["onet_soc_code", "title", "description"], "occupation_data")
    rows: list[dict[str, object]] = []
    for row in occupation_data.to_dict("records"):
        code = _canonical_codes([row["onet_soc_code"]], "occupation_data")[0]
        title = normalize_text(row["title"])
        description = normalize_text(row["description"])
        if not title or not description:
            raise ProfileError(f"occupation_data: blank title/description for {code}")
        rows.append(
            {
                "onet_soc_code": code,
                "base_soc_code": base_soc_code(code),
                "soc_major_group": soc_major_group(code),
                "canonical_title": title,
                "description": description,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result["onet_soc_code"].duplicated().any():
        raise ProfileError("occupation_data must be nonempty and unique by O*NET-SOC code")

    if job_zones is None:
        result["job_zone"] = pd.array([None] * len(result), dtype="Int64")
        result["job_zone_date"] = [None] * len(result)
        result["job_zone_domain_source"] = pd.array([None] * len(result), dtype="string")
    else:
        _require_columns(
            job_zones,
            ["onet_soc_code", "job_zone", "date", "domain_source"],
            "job_zones",
        )
        zones: list[dict[str, object]] = []
        for row in job_zones.to_dict("records"):
            code = _canonical_codes([row["onet_soc_code"]], "job_zones")[0]
            zone_text = normalize_text(row["job_zone"])
            try:
                zone = int(zone_text)
            except ValueError as exc:
                raise ProfileError(f"job_zones: invalid job zone {row['job_zone']!r}") from exc
            if zone not in range(1, job_zone_levels + 1):
                raise ProfileError(f"job_zones: job zone outside 1..{job_zone_levels} for {code}")
            zones.append(
                {
                    "onet_soc_code": code,
                    "job_zone": zone,
                    "job_zone_date": _parse_date(row.get("date"), "job_zones.date"),
                    "job_zone_domain_source": normalize_text(row.get("domain_source")) or None,
                }
            )
        zone_frame = pd.DataFrame(zones)
        if zone_frame.empty:
            raise ProfileError("job_zones: no job-zone rows")
        if zone_frame["onet_soc_code"].duplicated().any():
            raise ProfileError("job_zones: duplicate occupation code")
        orphan = sorted(set(zone_frame["onet_soc_code"]) - set(result["onet_soc_code"]))
        if orphan:
            raise ProfileError(f"job_zones contains unknown occupation codes: {orphan[:10]}")
        result = result.merge(zone_frame, on="onet_soc_code", how="left", validate="one_to_one")
        result["job_zone"] = pd.array(result["job_zone"], dtype="Int64")
        result["job_zone_domain_source"] = pd.array(
            result["job_zone_domain_source"], dtype="string"
        )
    return result.sort_values("onet_soc_code", kind="stable").reset_index(drop=True)


def _flag(value: object, field: str) -> bool:
    text = normalize_text(value, casefold=True)
    if text in {"", "n", "no", "false", "0"}:
        return False
    if text in {"y", "yes", "true", "1"}:
        return True
    raise ProfileError(f"{field}: invalid flag {value!r}")


def _numeric(value: object, field: str) -> float | None:
    text = normalize_text(value)
    if text.casefold() in {"", "n/a", "na", "—", "–", "*"}:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ProfileError(f"{field}: invalid numeric value {value!r}") from exc
    if not math.isfinite(result):
        raise ProfileError(f"{field}: non-finite value")
    return result


def build_structured_ratings(
    transferable_skills: pd.DataFrame,
    knowledge: pd.DataFrame,
    *,
    scale_bounds: Mapping[str, tuple[float, float]] = SCALE_BOUNDS,
) -> pd.DataFrame:
    """Build a long, normalized rating table with an explicit observed mask."""

    required = [
        "onet_soc_code",
        "element_id",
        "element_name",
        "scale_id",
        "scale_name",
        "data_value",
        "n",
        "standard_error",
        "lower_ci_bound",
        "upper_ci_bound",
        "recommend_suppress",
        "not_relevant",
        "date",
        "domain_source",
    ]
    outputs: list[dict[str, object]] = []
    for family, frame in (
        ("transferable_skill", transferable_skills),
        ("knowledge", knowledge),
    ):
        _require_columns(frame, required, family)
        for row in frame.to_dict("records"):
            code = _canonical_codes([row["onet_soc_code"]], family)[0]
            element_id = normalize_text(row["element_id"])
            element_name = normalize_text(row["element_name"])
            scale_id = normalize_text(row["scale_id"]).upper()
            scale_name = normalize_text(row["scale_name"])
            if not element_id or not element_name or not scale_id or not scale_name:
                raise ProfileError(f"{family}: blank element/scale identity")
            if scale_id not in scale_bounds:
                raise ProfileError(
                    f"{family}: no published bounds registered for scale {scale_id!r}"
                )
            lower, upper = scale_bounds[scale_id]
            if not lower < upper:
                raise ProfileError(f"Invalid bounds for scale {scale_id!r}")
            raw_value = _numeric(row.get("data_value"), f"{family}.data_value")
            sample_n = _numeric(row.get("n"), f"{family}.n")
            standard_error = _numeric(row.get("standard_error"), f"{family}.standard_error")
            lower_ci = _numeric(row.get("lower_ci_bound"), f"{family}.lower_ci_bound")
            upper_ci = _numeric(row.get("upper_ci_bound"), f"{family}.upper_ci_bound")
            if sample_n is not None and (sample_n < 0 or not sample_n.is_integer()):
                raise ProfileError(f"{family}: n must be a nonnegative whole count")
            if standard_error is not None and standard_error < 0:
                raise ProfileError(f"{family}: standard error cannot be negative")
            if (lower_ci is None) != (upper_ci is None):
                raise ProfileError(f"{family}: confidence interval bounds must be jointly present")
            if lower_ci is not None and upper_ci is not None and lower_ci > upper_ci:
                raise ProfileError(f"{family}: lower confidence bound exceeds upper bound")
            if (
                raw_value is not None
                and lower_ci is not None
                and upper_ci is not None
                and not lower_ci <= raw_value <= upper_ci
            ):
                raise ProfileError(f"{family}: confidence interval does not contain data value")
            suppress = _flag(row.get("recommend_suppress"), f"{family}.recommend_suppress")
            not_relevant = _flag(row.get("not_relevant"), f"{family}.not_relevant")
            if raw_value is not None and not (lower <= raw_value <= upper):
                raise ProfileError(
                    f"{family}: {scale_id} value {raw_value} outside published [{lower}, {upper}]"
                )
            observed = raw_value is not None and not suppress and not not_relevant
            normalized = (raw_value - lower) / (upper - lower) if observed else None
            outputs.append(
                {
                    "onet_soc_code": code,
                    "feature_family": family,
                    "element_id": element_id,
                    "element_name": element_name,
                    "scale_id": scale_id,
                    "scale_name": scale_name,
                    "raw_value": raw_value,
                    "sample_size": None if sample_n is None else int(sample_n),
                    "standard_error": standard_error,
                    "lower_ci_bound": lower_ci,
                    "upper_ci_bound": upper_ci,
                    "normalized_value": normalized,
                    "observed": observed,
                    "recommend_suppress": suppress,
                    "not_relevant": not_relevant,
                    "rating_date": _parse_date(row.get("date"), f"{family}.date"),
                    "domain_source": normalize_text(row.get("domain_source")) or None,
                }
            )
    result = pd.DataFrame(outputs)
    key = ["onet_soc_code", "feature_family", "element_id", "scale_id"]
    if result.empty or result.duplicated(key).any():
        raise ProfileError("structured ratings must be nonempty and unique by feature key")
    element_names = result.groupby(["feature_family", "element_id"], sort=False, dropna=False)[
        "element_name"
    ].nunique(dropna=False)
    if bool(element_names.gt(1).any()):
        raise ProfileError("structured ratings map one element ID to conflicting names")
    scale_names = result.groupby("scale_id", sort=False, dropna=False)["scale_name"].nunique(
        dropna=False
    )
    if bool(scale_names.gt(1).any()):
        raise ProfileError("structured ratings map one scale ID to conflicting names")
    result["raw_value"] = pd.array(result["raw_value"], dtype="Float64")
    result["sample_size"] = pd.array(result["sample_size"], dtype="Int64")
    for column in ["standard_error", "lower_ci_bound", "upper_ci_bound"]:
        result[column] = pd.array(result[column], dtype="Float64")
    result["normalized_value"] = pd.array(result["normalized_value"], dtype="Float64")
    result["observed"] = result["observed"].astype(bool)
    result["recommend_suppress"] = result["recommend_suppress"].astype(bool)
    result["not_relevant"] = result["not_relevant"].astype(bool)
    result["domain_source"] = pd.array(result["domain_source"], dtype="string")
    invalid_mask = ~result["observed"] & result["normalized_value"].notna()
    if bool(invalid_mask.any()):
        raise ProfileError("unobserved structured ratings cannot carry normalized values")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def _deduplicated_text(items: Sequence[tuple[str, str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for normalized, display in sorted(items, key=lambda item: (item[0], item[1])):
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(display)
    return result


def build_occupation_profiles(
    occupations: pd.DataFrame,
    aliases: pd.DataFrame,
    tasks: pd.DataFrame,
    structured_ratings: pd.DataFrame,
    *,
    core_task_type: str = "Core",
) -> pd.DataFrame:
    """Build title/work views without accepting qrels or BLS outcome inputs."""

    _require_columns(
        occupations,
        [
            "onet_soc_code",
            "base_soc_code",
            "soc_major_group",
            "canonical_title",
            "description",
            "job_zone",
            "job_zone_date",
            "job_zone_domain_source",
        ],
        "occupations",
    )
    _require_columns(
        aliases,
        ["onet_soc_code", "job_title", "normalized_job_title"],
        "aliases",
    )
    _require_columns(
        tasks,
        ["onet_soc_code", "task_id", "task", "normalized_task", "task_type"],
        "tasks",
    )
    _require_columns(structured_ratings, ["onet_soc_code", "observed"], "structured_ratings")
    codes = set(occupations["onet_soc_code"])
    for name, frame in (
        ("aliases", aliases),
        ("tasks", tasks),
        ("structured_ratings", structured_ratings),
    ):
        orphan = sorted(set(frame["onet_soc_code"]) - codes)
        if orphan:
            raise ProfileError(f"{name} contains unknown occupation codes: {orphan[:10]}")

    alias_groups = {code: group for code, group in aliases.groupby("onet_soc_code", sort=False)}
    task_groups = {code: group for code, group in tasks.groupby("onet_soc_code", sort=False)}
    observed_counts = (
        structured_ratings.loc[structured_ratings["observed"]]
        .groupby("onet_soc_code", sort=False)
        .size()
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for occupation in occupations.sort_values("onet_soc_code", kind="stable").to_dict("records"):
        code = occupation["onet_soc_code"]
        canonical_title = normalize_text(occupation["canonical_title"])
        description = normalize_text(occupation["description"])
        alias_group = alias_groups.get(code)
        alias_items: list[tuple[str, str]] = []
        if alias_group is not None:
            alias_items = [
                (str(row.normalized_job_title), str(row.job_title))
                for row in alias_group.itertuples(index=False)
            ]
        title_items = [
            (normalize_text(canonical_title, casefold=True), canonical_title),
            *alias_items,
        ]
        title_parts = _deduplicated_text(title_items)

        core_items: list[tuple[str, str, tuple[object, ...]]] = []
        task_group = task_groups.get(code)
        if task_group is not None:
            for task in task_group.itertuples(index=False):
                if normalize_text(task.task_type) == core_task_type:
                    core_items.append(
                        (
                            str(task.normalized_task),
                            str(task.task),
                            _natural_key(task.task_id),
                        )
                    )
        core_items.sort(key=lambda item: (item[2], item[0], item[1]))
        seen_tasks: set[str] = set()
        core_tasks: list[str] = []
        for normalized, display, _ in core_items:
            if normalized not in seen_tasks:
                seen_tasks.add(normalized)
                core_tasks.append(display)
        work_parts = [part for part in [description, *core_tasks] if part]
        rows.append(
            {
                **occupation,
                "canonical_title": canonical_title,
                "description": description,
                "title_text": "\n".join(title_parts),
                "work_text": "\n".join(work_parts),
                "core_task_count": len(core_tasks),
                "alias_count": max(0, len(title_parts) - 1),
                "structured_observed_count": int(observed_counts.get(code, 0)),
            }
        )
    result = pd.DataFrame(rows)
    for column in occupations.columns:
        result[column] = result[column].astype(occupations[column].dtype)
    for column in ["core_task_count", "alias_count", "structured_observed_count"]:
        result[column] = pd.array(result[column], dtype="Int64")
    forbidden = {
        "relatedness_tier",
        "relatedness_index",
        "relevance_grade",
        "employment_2024",
        "employment_2034",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
    }
    leaked = forbidden & set(result.columns)
    if leaked:
        raise ProfileError(f"Forbidden judgment/outcome fields entered profiles: {sorted(leaked)}")
    return result.reset_index(drop=True)


def build_structured_matrix(
    structured_ratings: pd.DataFrame,
    occupation_codes: Iterable[str] | None = None,
) -> StructuredMatrix:
    _require_columns(
        structured_ratings,
        [
            "onet_soc_code",
            "feature_family",
            "element_id",
            "scale_id",
            "normalized_value",
            "observed",
        ],
        "structured_ratings",
    )
    key_columns = ["onet_soc_code", "feature_family", "element_id", "scale_id"]
    if structured_ratings.duplicated(key_columns).any():
        raise ProfileError("structured_ratings contains duplicate feature keys")
    codes = (
        tuple(sorted({canonical_onet_soc_code(code) for code in occupation_codes}))
        if occupation_codes is not None
        else tuple(sorted(structured_ratings["onet_soc_code"].unique()))
    )
    features = tuple(
        sorted(
            {
                f"{row.feature_family}:{row.element_id}:{row.scale_id}"
                for row in structured_ratings.itertuples(index=False)
            }
        )
    )
    code_index = {code: index for index, code in enumerate(codes)}
    feature_index = {feature: index for index, feature in enumerate(features)}
    values = np.zeros((len(codes), len(features)), dtype=np.float32)
    observed = np.zeros((len(codes), len(features)), dtype=np.bool_)
    for row in structured_ratings.itertuples(index=False):
        if row.onet_soc_code not in code_index:
            continue
        feature = f"{row.feature_family}:{row.element_id}:{row.scale_id}"
        i = code_index[row.onet_soc_code]
        j = feature_index[feature]
        is_observed = bool(row.observed)
        value = row.normalized_value
        if is_observed:
            if pd.isna(value):
                raise ProfileError("Observed structured rating has a missing normalized value")
            values[i, j] = float(value)
            observed[i, j] = True
        elif not pd.isna(value):
            raise ProfileError("Unobserved structured rating carries a normalized value")
    values.setflags(write=False)
    observed.setflags(write=False)
    return StructuredMatrix(codes, features, values, observed)


def missing_aware_cosine(
    left: StructuredMatrix,
    right: StructuredMatrix | None = None,
    *,
    minimum_overlap: int = 5,
) -> np.ndarray:
    """Cosine similarity using only features observed for both occupations."""

    if minimum_overlap < 1:
        raise ProfileError("minimum_overlap must be positive")
    other = right or left
    if left.feature_keys != other.feature_keys:
        raise ProfileError("Structured matrices must have identical ordered feature keys")
    result = np.full(
        (len(left.occupation_codes), len(other.occupation_codes)), np.nan, dtype=np.float32
    )
    for i in range(len(left.occupation_codes)):
        for j in range(len(other.occupation_codes)):
            overlap = left.observed[i] & other.observed[j]
            if int(overlap.sum()) < minimum_overlap:
                continue
            first = left.values[i, overlap].astype(np.float64, copy=False)
            second = other.values[j, overlap].astype(np.float64, copy=False)
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator > 0:
                result[i, j] = float(np.dot(first, second) / denominator)
    return result


__all__ = [
    "ProfileError",
    "SCALE_BOUNDS",
    "StructuredMatrix",
    "build_aliases",
    "build_occupation_profiles",
    "build_occupations",
    "build_structured_matrix",
    "build_structured_ratings",
    "build_tasks",
    "missing_aware_cosine",
    "normalize_text",
    "stable_text_id",
]
