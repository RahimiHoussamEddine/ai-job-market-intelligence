"""Official O*NET-to-BLS crosswalk and BLS unit normalization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import pandas as pd


class CrosswalkError(ValueError):
    """Raised when official codes, units, or crosswalk cardinality are invalid."""


_ONET_PATTERN = re.compile(r"^(\d{2})-(\d{4})\.(\d{2})$")
_BASE_SOC_PATTERN = re.compile(r"^(\d{2})-(\d{4})$")
_COMPACT_ONET_PATTERN = re.compile(r"^(\d{2})(\d{4})(\d{2})$")
_MISSING_VALUES = {"", "-", "--", "—", "–", "na", "n/a", "nan", "none"}


@dataclass(frozen=True, slots=True)
class CrosswalkCoverage:
    occupation_count: int
    mapped_occupation_count: int
    unmapped_occupation_count: int
    nem_count: int
    crosswalk_row_count: int

    @property
    def occupation_coverage(self) -> float:
        return (
            self.mapped_occupation_count / self.occupation_count if self.occupation_count else 0.0
        )


def _clean_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def canonical_onet_soc_code(value: object) -> str:
    """Return an exact ``XX-XXXX.XX`` O*NET-SOC code."""

    text = _clean_scalar(value)
    match = _ONET_PATTERN.fullmatch(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}.{match.group(3)}"
    compact = _COMPACT_ONET_PATTERN.fullmatch(text.replace(" ", ""))
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}.{compact.group(3)}"
    raise CrosswalkError(f"Invalid O*NET-SOC code: {value!r}")


def base_soc_code(value: object) -> str:
    return canonical_onet_soc_code(value)[:7]


def soc_major_group(value: object) -> str:
    return canonical_onet_soc_code(value)[:2]


def canonical_nem_code(value: object) -> str:
    """Return an exact detailed ``XX-XXXX`` National Employment Matrix code."""

    text = _clean_scalar(value)
    match = _BASE_SOC_PATTERN.fullmatch(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    compact = re.fullmatch(r"(\d{2})(\d{4})", text.replace(" ", ""))
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}"
    raise CrosswalkError(f"Invalid BLS NEM code: {value!r}")


def _decimal(value: object, field: str) -> Decimal | None:
    text = _clean_scalar(value).lower()
    if text in _MISSING_VALUES:
        return None
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = text.replace("≥", "").replace(">=", "").replace("<", "").strip()
    text = re.sub(r"\[(?:\d+|[a-z])\]$", "", text).strip()
    text = re.sub(r"\((?:\d+|[a-z])\)$", "", text).strip()
    try:
        result = Decimal(text)
    except InvalidOperation as exc:
        raise CrosswalkError(f"{field}: invalid numeric value {value!r}") from exc
    if not result.is_finite():
        raise CrosswalkError(f"{field}: non-finite numeric value {value!r}")
    return result


def parse_thousands_to_people(value: object, field: str = "value") -> int | None:
    """Convert a published thousands value to an exact integer person count."""

    parsed = _decimal(value, field)
    if parsed is None:
        return None
    people = parsed * Decimal(1000)
    if people != people.to_integral_value():
        raise CrosswalkError(
            f"{field}: thousands value {value!r} cannot be represented as whole people"
        )
    return int(people)


def parse_number(value: object, field: str = "value") -> float | None:
    parsed = _decimal(value, field)
    return None if parsed is None else float(parsed)


def parse_dollars(value: object, field: str = "wage") -> int | None:
    parsed = _decimal(value, field)
    if parsed is None:
        return None
    if parsed != parsed.to_integral_value() or parsed < 0:
        raise CrosswalkError(f"{field}: expected a nonnegative whole-dollar value, got {value!r}")
    return int(parsed)


def _normalize_label(value: object) -> str | None:
    text = re.sub(r"\s+", " ", _clean_scalar(value))
    return text or None


def _nullable_int(values: Iterable[int | None]) -> pd.Series:
    return pd.Series(list(values), dtype="Int64")


def normalize_bls_market(frame: pd.DataFrame, *, line_items_only: bool = True) -> pd.DataFrame:
    """Normalize Table 1.2 to unique detailed NEM rows and person units.

    The source publishes employment, numeric change, and annual openings in
    thousands.  Summary rows are never eligible for the occupation frontier.
    """

    required = {
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CrosswalkError(f"BLS occupation table is missing columns: {missing}")
    work = frame.copy()
    if line_items_only:
        mask = work["occupation_type"].map(_clean_scalar).str.casefold().eq("line item")
        work = work.loc[mask].copy()
    if work.empty:
        raise CrosswalkError("BLS occupation table contains no detailed Line item rows")

    codes: list[str] = []
    for value in work["nem_code"]:
        codes.append(canonical_nem_code(value))
    result = pd.DataFrame(
        {
            "nem_code": pd.Series(codes, dtype="string"),
            "nem_title": pd.Series(
                [_normalize_label(value) for value in work["nem_title"]], dtype="string"
            ),
            "occupation_type": pd.Series(
                [_normalize_label(value) for value in work["occupation_type"]], dtype="string"
            ),
            "employment_2024": _nullable_int(
                parse_thousands_to_people(value, "employment_2024")
                for value in work["employment_2024_thousands"]
            ),
            "employment_2034": _nullable_int(
                parse_thousands_to_people(value, "employment_2034")
                for value in work["employment_2034_thousands"]
            ),
            "employment_change_2024_34": _nullable_int(
                parse_thousands_to_people(value, "employment_change_2024_34")
                for value in work["employment_change_2024_34_thousands"]
            ),
            "employment_change_pct": pd.Series(
                [
                    parse_number(value, "employment_change_pct")
                    for value in work["employment_change_pct"]
                ],
                dtype="Float64",
            ),
            "annual_openings_2024_34": _nullable_int(
                parse_thousands_to_people(value, "annual_openings_2024_34")
                for value in work["annual_openings_2024_34_thousands"]
            ),
            "median_annual_wage_2024": _nullable_int(
                parse_dollars(value, "median_annual_wage_2024")
                for value in work["median_annual_wage_2024"]
            ),
            "typical_education": pd.Series(
                [_normalize_label(value) for value in work["typical_education"]], dtype="string"
            ),
            "work_experience": pd.Series(
                [_normalize_label(value) for value in work["work_experience"]], dtype="string"
            ),
            "on_the_job_training": pd.Series(
                [_normalize_label(value) for value in work["on_the_job_training"]], dtype="string"
            ),
        }
    )
    if bool(result["nem_title"].isna().any()):
        raise CrosswalkError("BLS detailed line item has a blank occupation title")
    duplicate = result["nem_code"].duplicated(keep=False)
    if bool(duplicate.any()):
        raise CrosswalkError(
            "BLS market table has duplicate NEM codes: "
            + ", ".join(sorted(result.loc[duplicate, "nem_code"].unique()))
        )
    numeric_nonnegative = [
        "employment_2024",
        "employment_2034",
        "annual_openings_2024_34",
        "median_annual_wage_2024",
    ]
    for column in numeric_nonnegative:
        if bool((result[column].dropna() < 0).any()):
            raise CrosswalkError(f"BLS {column} contains negative values")
    return result.sort_values("nem_code", kind="stable").reset_index(drop=True)


def normalize_crosswalk(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official many-to-many O*NET-SOC/NEM bridge."""

    required = {"onet_soc_code", "onet_soc_title", "nem_code", "nem_title"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CrosswalkError(f"BLS crosswalk is missing columns: {missing}")
    work = frame.copy()
    both_blank = work["onet_soc_code"].map(_clean_scalar).eq("") & work["nem_code"].map(
        _clean_scalar
    ).eq("")
    work = work.loc[~both_blank].copy()
    if work.empty:
        raise CrosswalkError("BLS crosswalk contains no mapping rows")
    rows: list[dict[str, object]] = []
    for record in work.to_dict("records"):
        try:
            onet_code = canonical_onet_soc_code(record["onet_soc_code"])
            nem_code = canonical_nem_code(record["nem_code"])
        except CrosswalkError as exc:
            raise CrosswalkError(f"Invalid official crosswalk row {record!r}: {exc}") from exc
        rows.append(
            {
                "onet_soc_code": onet_code,
                "onet_soc_title": _normalize_label(record["onet_soc_title"]),
                "nem_code": nem_code,
                "nem_title": _normalize_label(record["nem_title"]),
            }
        )
    result = pd.DataFrame(rows).astype(
        {
            "onet_soc_code": "string",
            "onet_soc_title": "string",
            "nem_code": "string",
            "nem_title": "string",
        }
    )
    if bool(result[["onet_soc_title", "nem_title"]].isna().any(axis=None)):
        raise CrosswalkError("Official crosswalk mapping titles must be nonblank")
    key = ["onet_soc_code", "nem_code"]
    duplicate = result.duplicated(key, keep=False)
    if bool(duplicate.any()):
        examples = result.loc[duplicate, key].head().to_dict("records")
        raise CrosswalkError(f"Official crosswalk has duplicate code pairs: {examples}")
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def join_market_context(
    crosswalk: pd.DataFrame,
    market: pd.DataFrame,
    *,
    valid_onet_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Join without aggregating or duplicating-and-summing NEM measures."""

    if crosswalk.duplicated(["onet_soc_code", "nem_code"]).any():
        raise CrosswalkError("Crosswalk code pairs must be unique before the market join")
    if market["nem_code"].duplicated().any():
        raise CrosswalkError("BLS market rows must be unique by NEM code")
    missing_market = sorted(set(crosswalk["nem_code"]) - set(market["nem_code"]))
    if missing_market:
        raise CrosswalkError(
            f"Crosswalk references NEM codes absent from detailed BLS line items: {missing_market[:10]}"
        )
    if valid_onet_codes is not None:
        valid = {canonical_onet_soc_code(code) for code in valid_onet_codes}
        missing_onet = sorted(set(crosswalk["onet_soc_code"]) - valid)
        if missing_onet:
            raise CrosswalkError(
                f"Crosswalk references O*NET codes absent from occupation data: {missing_onet[:10]}"
            )
    joined = crosswalk.merge(
        market,
        on="nem_code",
        how="left",
        validate="many_to_one",
        suffixes=("_crosswalk", ""),
        sort=False,
    )
    joined = joined.drop(columns=["nem_title_crosswalk"], errors="ignore")
    key = ["onet_soc_code", "nem_code"]
    if joined.duplicated(key).any():
        raise CrosswalkError("Market join changed official crosswalk cardinality")
    return joined.sort_values(key, kind="stable").reset_index(drop=True)


def crosswalk_coverage(
    occupation_codes: Iterable[str], crosswalk: pd.DataFrame
) -> CrosswalkCoverage:
    occupations = {canonical_onet_soc_code(code) for code in occupation_codes}
    mapped = occupations & set(crosswalk["onet_soc_code"])
    return CrosswalkCoverage(
        occupation_count=len(occupations),
        mapped_occupation_count=len(mapped),
        unmapped_occupation_count=len(occupations - mapped),
        nem_count=int(crosswalk["nem_code"].nunique()),
        crosswalk_row_count=len(crosswalk),
    )


__all__ = [
    "CrosswalkCoverage",
    "CrosswalkError",
    "base_soc_code",
    "canonical_nem_code",
    "canonical_onet_soc_code",
    "crosswalk_coverage",
    "join_market_context",
    "normalize_bls_market",
    "normalize_crosswalk",
    "parse_dollars",
    "parse_number",
    "parse_thousands_to_people",
    "soc_major_group",
]
