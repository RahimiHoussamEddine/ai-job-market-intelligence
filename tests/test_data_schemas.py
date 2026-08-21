from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from job_market_intelligence.schemas import (
    BLS_CROSSWALK_COLUMNS,
    BLS_OCCUPATION_COLUMNS,
    SchemaError,
    canonicalize_bls_crosswalk,
    canonicalize_bls_occupation,
    read_bls_workbook,
    read_onet_csv,
)

OCCUPATION_HEADERS = [
    "2024 National Employment Matrix title",
    "2024 National Employment Matrix code",
    "Occupation type",
    "Employment, 2024",
    "Employment, 2034",
    "Employment change, 2024–34 (Numeric)",
    "Employment change, 2024–34 (Percent)",
    "Occupational openings, 2024–34 annual average",
    "Median annual wage, dollars, 2024[1]",
    "Typical education needed for entry",
    "Work experience in a related occupation",
    "Typical on-the-job training needed to attain competency in the occupation",
]


def test_onet_job_title_schema_uses_singular_optional_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "job_titles.csv"
    path.write_text(
        "O*NET-SOC Code,Title,Job Title,Short Title,Source\n"
        "11-1011.00,Chief Executives,Chief Executive Officer,,\n",
        encoding="utf-8",
    )

    frame = read_onet_csv(path, "job_titles")

    assert tuple(frame.columns) == (
        "onet_soc_code",
        "title",
        "job_title",
        "short_title",
        "source",
    )
    assert pd.isna(frame.loc[0, "source"])


def test_onet_job_title_schema_rejects_plural_sources_header(tmp_path: Path) -> None:
    path = tmp_path / "job_titles.csv"
    path.write_text(
        "O*NET-SOC Code,Title,Job Title,Short Title,Sources\n"
        "11-1011.00,Chief Executives,Chief Executive Officer,,Employer\n",
        encoding="utf-8",
    )

    with pytest.raises(SchemaError, match="source schema drift"):
        read_onet_csv(path, "job_titles")


def test_onet_schema_rejects_column_reordering(tmp_path: Path) -> None:
    path = tmp_path / "occupation_data.csv"
    path.write_text(
        "Title,O*NET-SOC Code,Description\nChief Executives,11-1011.00,Lead.\n",
        encoding="utf-8",
    )

    with pytest.raises(SchemaError, match="source schema drift"):
        read_onet_csv(path, "occupation_data")


def test_bls_occupation_canonicalization_drops_publisher_note_rows() -> None:
    frame = pd.DataFrame(
        [
            [
                "Chief executives",
                "11-1011",
                "Line item",
                313.2,
                329.1,
                15.9,
                5.1,
                19.8,
                206680,
                "Bachelor's degree",
                "5 years or more",
                "None",
            ],
            [
                "Footnote: wage estimates are national.",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        ],
        columns=OCCUPATION_HEADERS,
    )

    result = canonicalize_bls_occupation(frame)

    assert tuple(result.columns) == BLS_OCCUPATION_COLUMNS
    assert len(result) == 1
    assert result.loc[0, "nem_code"] == "11-1011"


def test_bls_crosswalk_rejects_partial_mapping_rows() -> None:
    frame = pd.DataFrame(
        {
            "O*NET-SOC 2019 code": ["11-1011.00"],
            "O*NET-SOC 2019 title": ["Chief Executives"],
            "2024 National Employment Matrix code": [None],
            "2024 National Employment Matrix title": ["Chief executives"],
        }
    )

    with pytest.raises(SchemaError, match="blank nem_code"):
        canonicalize_bls_crosswalk(frame)


def test_bls_occupation_reader_selects_table_1_2_in_multisheet_workbook(
    tmp_path: Path,
) -> None:
    path = tmp_path / "occupation.xlsx"
    workbook = Workbook()
    index = workbook.active
    index.title = "Index"
    index.append(OCCUPATION_HEADERS)
    index.append(["This is not a data row"])
    sheet = workbook.create_sheet("Table 1.2")
    sheet.append(["Employment by occupation"])
    sheet.append(OCCUPATION_HEADERS)
    sheet.append(
        [
            "Chief executives",
            "11-1011",
            "Line item",
            313.2,
            329.1,
            15.9,
            5.1,
            19.8,
            206680,
            "Bachelor's degree",
            "5 years or more",
            "None",
        ]
    )
    workbook.save(path)

    result = read_bls_workbook(path, "occupation")

    assert len(result) == 1
    assert result.loc[0, "nem_code"] == "11-1011"


@pytest.mark.parametrize("data_sheet_name", [None, "Table 1.1"])
def test_bls_occupation_reader_rejects_missing_or_wrong_sheet(
    tmp_path: Path, data_sheet_name: str | None
) -> None:
    path = tmp_path / "occupation.xlsx"
    workbook = Workbook()
    workbook.active.title = "Index"
    if data_sheet_name is not None:
        sheet = workbook.create_sheet(data_sheet_name)
        sheet.append(["Employment by occupation"])
        sheet.append(OCCUPATION_HEADERS)
    workbook.save(path)

    with pytest.raises(SchemaError, match="required 'Table 1.2'"):
        read_bls_workbook(path, "occupation")


def test_bls_occupation_reader_rejects_header_row_drift(tmp_path: Path) -> None:
    path = tmp_path / "occupation.xlsx"
    workbook = Workbook()
    workbook.active.title = "Index"
    sheet = workbook.create_sheet("Table 1.2")
    sheet.append(["Employment by occupation"])
    sheet.append(["Unexpected publisher note"])
    sheet.append(OCCUPATION_HEADERS)
    workbook.save(path)

    with pytest.raises(SchemaError, match="row 1 contained 0 of 12 required headers"):
        read_bls_workbook(path, "occupation")


def test_bls_crosswalk_reader_requires_official_first_sheet_and_header_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crosswalk.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "ONET to SOC crosswalk"
    for _ in range(4):
        sheet.append(["Publisher title"])
    sheet.append(
        [
            "O*NET-SOC 2019 code",
            "O*NET-SOC 2019 title",
            "2024 National Employment Matrix code",
            "2024 National Employment Matrix title",
        ]
    )
    sheet.append(["11-1011.00", "Chief Executives", "11-1011", "Chief executives"])
    workbook.save(path)

    result = read_bls_workbook(path, "crosswalk")

    assert len(result) == 1
    assert result.loc[0, "onet_soc_code"] == "11-1011.00"


def test_bls_crosswalk_reader_rejects_nonofficial_first_sheet(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.xlsx"
    workbook = Workbook()
    workbook.active.title = "Index"
    sheet = workbook.create_sheet("ONET to SOC crosswalk")
    for _ in range(4):
        sheet.append(["Publisher title"])
    sheet.append(
        [
            "O*NET-SOC 2019 code",
            "O*NET-SOC 2019 title",
            "2024 National Employment Matrix code",
            "2024 National Employment Matrix title",
        ]
    )
    workbook.save(path)

    with pytest.raises(SchemaError, match="as the first worksheet"):
        read_bls_workbook(path, "crosswalk")


def test_crosswalk_columns_resolve_official_2019_and_2024_labels() -> None:
    frame = pd.DataFrame(
        {
            "O*NET-SOC 2019 code": ["11-1011.00"],
            "O*NET-SOC 2019 title": ["Chief Executives"],
            "2024 National Employment Matrix code": ["11-1011"],
            "2024 National Employment Matrix title": ["Chief executives"],
        }
    )

    result = canonicalize_bls_crosswalk(frame)

    assert tuple(result.columns) == BLS_CROSSWALK_COLUMNS
