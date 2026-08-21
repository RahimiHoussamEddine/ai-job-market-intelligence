"""Strict source schemas and deterministic tabular parsers.

Source headers are normalized only after they have been checked against a
versioned contract.  O*NET contracts are exact.  BLS workbooks are allowed to
move their header row and retain footnote suffixes, but every required semantic
column must resolve exactly once.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class SchemaError(ValueError):
    """Raised for schema drift, invalid keys, or ambiguous workbook columns."""


@dataclass(frozen=True, slots=True)
class TableContract:
    name: str
    columns: tuple[str, ...]
    business_key: tuple[str, ...]


RATING_COLUMNS = (
    "onet_soc_code",
    "title",
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
)

ONET_CONTRACTS: Mapping[str, TableContract] = {
    "occupation_data": TableContract(
        "occupation_data",
        ("onet_soc_code", "title", "description"),
        ("onet_soc_code",),
    ),
    "job_titles": TableContract(
        "job_titles",
        ("onet_soc_code", "title", "job_title", "short_title", "sources"),
        ("onet_soc_code", "job_title"),
    ),
    "task_statements": TableContract(
        "task_statements",
        (
            "onet_soc_code",
            "title",
            "task_id",
            "task",
            "task_type",
            "incumbents_responding",
            "date",
            "domain_source",
        ),
        ("onet_soc_code", "task_id"),
    ),
    "transferable_skills": TableContract(
        "transferable_skills", RATING_COLUMNS, ("onet_soc_code", "element_id", "scale_id")
    ),
    "knowledge": TableContract(
        "knowledge", RATING_COLUMNS, ("onet_soc_code", "element_id", "scale_id")
    ),
    "job_zones": TableContract(
        "job_zones",
        ("onet_soc_code", "title", "job_zone", "date", "domain_source"),
        ("onet_soc_code",),
    ),
    "related_occupations": TableContract(
        "related_occupations",
        (
            "onet_soc_code",
            "title",
            "related_onet_soc_code",
            "related_title",
            "relatedness_tier",
            "index",
        ),
        ("onet_soc_code", "related_onet_soc_code"),
    ),
}


BLS_OCCUPATION_COLUMNS = (
    "nem_title",
    "nem_code",
    "occupation_type",
    "employment_2024_thousands",
    "employment_2034_thousands",
    "employment_change_2024_34_thousands",
    "employment_change_pct",
    "annual_openings_2024_34_thousands",
    "median_annual_wage_2024",
    "typical_education",
    "work_experience",
    "on_the_job_training",
)

BLS_CROSSWALK_COLUMNS = (
    "onet_soc_code",
    "onet_soc_title",
    "nem_code",
    "nem_title",
)


def normalize_column_name(value: object) -> str:
    """Normalize a publisher header without discarding semantic words."""

    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("o*net", "onet").replace("o’net", "onet")
    text = text.replace("%", " percent ").replace("&", " and ")
    text = re.sub(r"\[(?:\d+|[a-z])\]", " ", text)
    text = re.sub(r"\((?:\d+|[a-z])\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "o_net_soc_code": "onet_soc_code",
        "source_s": "sources",
        "recommend_suppress": "recommend_suppress",
        "lower_ci_bound": "lower_ci_bound",
        "upper_ci_bound": "upper_ci_bound",
    }
    return aliases.get(text, text)


def normalized_columns(columns: Iterable[object]) -> tuple[str, ...]:
    return tuple(normalize_column_name(column) for column in columns)


def schema_fingerprint(columns: Sequence[str], dtypes: Sequence[object] | None = None) -> str:
    """Hash an ordered schema description for provenance manifests."""

    if dtypes is not None and len(columns) != len(dtypes):
        raise SchemaError("columns and dtypes must have equal length")
    description = [
        {"name": str(column), "dtype": str(dtypes[index]) if dtypes is not None else "source"}
        for index, column in enumerate(columns)
    ]
    payload = json.dumps(description, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_columns(
    contract: TableContract,
    columns: Iterable[object],
    *,
    exact: bool = True,
) -> tuple[str, ...]:
    actual = normalized_columns(columns)
    if len(actual) != len(set(actual)):
        duplicates = sorted({column for column in actual if actual.count(column) > 1})
        raise SchemaError(f"{contract.name}: duplicate normalized columns {duplicates}")
    expected = contract.columns
    if exact and actual != expected:
        raise SchemaError(
            f"{contract.name}: source schema drift; expected {expected!r}, got {actual!r}"
        )
    missing = [column for column in expected if column not in actual]
    if missing:
        raise SchemaError(f"{contract.name}: missing required columns {missing}")
    return actual


def validate_business_key(frame: pd.DataFrame, contract: TableContract) -> None:
    """Reject null or duplicate source business keys."""

    missing_columns = [column for column in contract.business_key if column not in frame.columns]
    if missing_columns:
        raise SchemaError(f"{contract.name}: key columns are absent: {missing_columns}")
    key = frame.loc[:, list(contract.business_key)]
    null_mask = key.isna() | key.astype("string").apply(lambda column: column.str.strip().eq(""))
    if bool(null_mask.any(axis=1).any()):
        rows = frame.index[null_mask.any(axis=1)].tolist()[:5]
        raise SchemaError(f"{contract.name}: null/blank business key at rows {rows}")
    duplicate_mask = key.duplicated(keep=False)
    if bool(duplicate_mask.any()):
        examples = key.loc[duplicate_mask].head(5).to_dict("records")
        raise SchemaError(f"{contract.name}: duplicate business keys: {examples}")


def read_onet_csv(path: str | Path, table: str) -> pd.DataFrame:
    """Read one locked O*NET CSV with exact header and key validation."""

    try:
        contract = ONET_CONTRACTS[table]
    except KeyError as exc:
        raise SchemaError(f"No O*NET schema contract for {table!r}") from exc
    source_path = Path(path)
    try:
        frame = pd.read_csv(
            source_path,
            dtype="string",
            encoding="utf-8-sig",
            keep_default_na=False,
            na_filter=False,
        )
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{table}: source is not UTF-8: {source_path}") from exc
    except Exception as exc:
        raise SchemaError(f"{table}: could not parse CSV {source_path}: {exc}") from exc
    canonical = validate_columns(contract, frame.columns, exact=True)
    frame.columns = canonical
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
    validate_business_key(frame, contract)
    return frame


def validate_onet_csv_bytes(payload: bytes, table: str) -> tuple[int, str]:
    """Cheap acquisition-time UTF-8/header validation.

    Full key validation is performed by :func:`read_onet_csv` during the atomic
    acquisition before the snapshot directory is published.
    """

    if not payload or b"\x00" in payload[:8192]:
        raise SchemaError(f"{table}: empty or binary payload where CSV was expected")
    try:
        first_line = payload.splitlines()[0].decode("utf-8-sig")
    except (IndexError, UnicodeDecodeError) as exc:
        raise SchemaError(f"{table}: invalid UTF-8 CSV header") from exc
    # Using pandas here would read the full payload twice.  The publisher's
    # headers contain no commas inside quoted labels, so a small csv parser is
    # sufficient and still handles future quoting correctly.
    import csv

    header = next(csv.reader([first_line]))
    contract = ONET_CONTRACTS[table]
    canonical = validate_columns(contract, header, exact=True)
    return max(0, len(payload.splitlines()) - 1), schema_fingerprint(canonical)


_BLS_OCCUPATION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "nem_title": (
        "2024_national_employment_matrix_title",
        "national_employment_matrix_title",
        "occupation_title",
        "nem_title",
    ),
    "nem_code": (
        "2024_national_employment_matrix_code",
        "national_employment_matrix_code",
        "occupation_code",
        "nem_code",
    ),
    "occupation_type": ("occupation_type",),
    "employment_2024_thousands": (
        "employment_2024",
        "employment_2024_thousands",
    ),
    "employment_2034_thousands": (
        "employment_2034",
        "employment_2034_thousands",
    ),
    "employment_change_2024_34_thousands": (
        "employment_change_numeric_2024_34",
        "employment_change_number_2024_34",
        "employment_change_2024_34_number",
        "employment_change_2024_34_numeric",
    ),
    "employment_change_pct": (
        "employment_change_percent_2024_34",
        "employment_change_2024_34_percent",
    ),
    "annual_openings_2024_34_thousands": (
        "occupational_openings_2024_34_annual_average",
        "annual_average_occupational_openings_2024_34",
    ),
    "median_annual_wage_2024": (
        "median_annual_wage_dollars_2024",
        "median_annual_wage_2024",
    ),
    "typical_education": (
        "typical_education_needed_for_entry",
        "typical_education",
    ),
    "work_experience": (
        "work_experience_in_a_related_occupation",
        "work_experience",
    ),
    "on_the_job_training": (
        "typical_on_the_job_training_needed_to_attain_competency_in_the_occupation",
        "on_the_job_training",
    ),
}

_BLS_CROSSWALK_ALIASES: Mapping[str, tuple[str, ...]] = {
    "onet_soc_code": (
        "onet_soc_2019_code",
        "2019_onet_soc_code",
        "onet_soc_code",
    ),
    "onet_soc_title": (
        "onet_soc_2019_title",
        "2019_onet_soc_title",
        "onet_soc_title",
    ),
    "nem_code": (
        "2024_national_employment_matrix_code",
        "national_employment_matrix_code",
        "nem_code",
    ),
    "nem_title": (
        "2024_national_employment_matrix_title",
        "national_employment_matrix_title",
        "nem_title",
        "ooh_occupation_title",
    ),
}


def _matches_alias(column: str, alias: str) -> bool:
    if column == alias:
        return True
    # Publisher footnote markers are stripped by normalize_column_name.  A
    # trailing unit marker is allowed, but semantically different prefixes are
    # not silently guessed.
    return column.startswith(f"{alias}_") and column[len(alias) + 1 :] in {
        "thousands",
        "dollars",
        "persons",
    }


def _resolve_columns(
    columns: Sequence[object], aliases: Mapping[str, tuple[str, ...]], table: str
) -> dict[object, str]:
    normalized = [normalize_column_name(column) for column in columns]
    result: dict[object, str] = {}
    used_indexes: set[int] = set()
    for canonical, candidates in aliases.items():
        matches = [
            index
            for index, column in enumerate(normalized)
            if any(_matches_alias(column, alias) for alias in candidates)
        ]
        if len(matches) != 1:
            raise SchemaError(
                f"{table}: expected exactly one column for {canonical!r}; "
                f"matched {[normalized[index] for index in matches]!r}"
            )
        index = matches[0]
        if index in used_indexes:
            raise SchemaError(f"{table}: one source column resolved to multiple meanings")
        used_indexes.add(index)
        result[columns[index]] = canonical
    return result


def _blank_mask(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    return strings.isna() | strings.str.strip().eq("").fillna(True)


def canonicalize_bls_occupation(frame: pd.DataFrame) -> pd.DataFrame:
    rename = _resolve_columns(frame.columns.tolist(), _BLS_OCCUPATION_ALIASES, "bls_occupation")
    result = frame.rename(columns=rename).loc[:, list(BLS_OCCUPATION_COLUMNS)].copy()
    result = result.dropna(how="all")
    # Publisher notes may follow the table in the title column. They are not
    # observations and have neither a NEM code nor an occupation type.
    result = result.loc[
        ~(_blank_mask(result["nem_code"]) & _blank_mask(result["occupation_type"]))
    ].copy()
    for column in ("nem_title", "nem_code", "occupation_type"):
        if bool(_blank_mask(result[column]).any()):
            raise SchemaError(f"bls_occupation: data row has blank {column}")
    if result.empty:
        raise SchemaError("bls_occupation: workbook contains no data rows")
    return result.reset_index(drop=True)


def canonicalize_bls_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    rename = _resolve_columns(frame.columns.tolist(), _BLS_CROSSWALK_ALIASES, "bls_crosswalk")
    result = frame.rename(columns=rename).loc[:, list(BLS_CROSSWALK_COLUMNS)].copy()
    result = result.dropna(how="all")
    result = result.loc[
        ~(_blank_mask(result["onet_soc_code"]) & _blank_mask(result["nem_code"]))
    ].copy()
    for column in BLS_CROSSWALK_COLUMNS:
        if bool(_blank_mask(result[column]).any()):
            raise SchemaError(f"bls_crosswalk: mapping row has blank {column}")
    if result.empty:
        raise SchemaError("bls_crosswalk: workbook contains no mapping rows")
    return result.reset_index(drop=True)


def _header_score(row: pd.Series, aliases: Mapping[str, tuple[str, ...]]) -> int:
    normalized = [normalize_column_name(value) for value in row.tolist() if pd.notna(value)]
    return sum(
        any(any(_matches_alias(column, alias) for alias in candidates) for column in normalized)
        for candidates in aliases.values()
    )


def read_bls_workbook(path: str | Path, kind: str) -> pd.DataFrame:
    """Read a BLS workbook with title/note rows before the semantic header."""

    if kind not in {"occupation", "crosswalk"}:
        raise SchemaError("BLS workbook kind must be 'occupation' or 'crosswalk'")
    aliases = _BLS_OCCUPATION_ALIASES if kind == "occupation" else _BLS_CROSSWALK_ALIASES
    try:
        preview = pd.read_excel(path, header=None, dtype=object, engine="openpyxl", nrows=40)
    except Exception as exc:
        raise SchemaError(f"bls_{kind}: could not parse workbook {path}: {exc}") from exc
    scores = [_header_score(row, aliases) for _, row in preview.iterrows()]
    if not scores or max(scores) != len(aliases):
        raise SchemaError(
            f"bls_{kind}: schema drift; no row uniquely contained all {len(aliases)} required headers"
        )
    header_candidates = [index for index, score in enumerate(scores) if score == len(aliases)]
    if len(header_candidates) != 1:
        raise SchemaError(
            f"bls_{kind}: schema drift; expected one semantic header row, found {header_candidates}"
        )
    header_index = header_candidates[0]
    try:
        frame = pd.read_excel(path, header=header_index, dtype=object, engine="openpyxl")
    except Exception as exc:
        raise SchemaError(f"bls_{kind}: could not read data rows: {exc}") from exc
    if kind == "occupation":
        return canonicalize_bls_occupation(frame)
    return canonicalize_bls_crosswalk(frame)


__all__ = [
    "BLS_CROSSWALK_COLUMNS",
    "BLS_OCCUPATION_COLUMNS",
    "ONET_CONTRACTS",
    "RATING_COLUMNS",
    "SchemaError",
    "TableContract",
    "canonicalize_bls_crosswalk",
    "canonicalize_bls_occupation",
    "normalize_column_name",
    "normalized_columns",
    "read_bls_workbook",
    "read_onet_csv",
    "schema_fingerprint",
    "validate_business_key",
    "validate_columns",
    "validate_onet_csv_bytes",
]
