from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from job_market_intelligence.config import load_config
from job_market_intelligence.sources import (
    AcquisitionError,
    ManifestError,
    PinMismatchError,
    acquire_sources,
    build_source_specs,
    verify_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def snapshot_clock() -> datetime:
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


ONET_ROWS = {
    "occupation_data": (
        "O*NET-SOC Code,Title,Description\n"
        "11-1011.00,Chief Executives,Determine organizational direction.\n"
    ),
    "job_titles": (
        "O*NET-SOC Code,Title,Job Title,Short Title,Sources\n"
        "11-1011.00,Chief Executives,Chief Executive Officer,,Employer\n"
    ),
    "task_statements": (
        "O*NET-SOC Code,Title,Task ID,Task,Task Type,Incumbents Responding,Date,Domain Source\n"
        "11-1011.00,Chief Executives,1,Direct organizational policy.,Core,10,07/2026,Analyst\n"
    ),
    "transferable_skills": (
        "O*NET-SOC Code,Title,Element ID,Element Name,Scale ID,Scale Name,Data Value,N,"
        "Standard Error,Lower CI Bound,Upper CI Bound,Recommend Suppress,Not Relevant,Date,"
        "Domain Source\n"
        "11-1011.00,Chief Executives,2.A.1,Leadership,IM,Importance,3,10,.1,2.8,3.2,N,N,"
        "07/2026,Analyst\n"
    ),
    "knowledge": (
        "O*NET-SOC Code,Title,Element ID,Element Name,Scale ID,Scale Name,Data Value,N,"
        "Standard Error,Lower CI Bound,Upper CI Bound,Recommend Suppress,Not Relevant,Date,"
        "Domain Source\n"
        "11-1011.00,Chief Executives,2.C.1,Economics,IM,Importance,3,10,.1,2.8,3.2,N,N,"
        "07/2026,Analyst\n"
    ),
    "job_zones": (
        "O*NET-SOC Code,Title,Job Zone,Date,Domain Source\n"
        "11-1011.00,Chief Executives,4,07/2026,Analyst\n"
    ),
    "related_occupations": (
        "O*NET-SOC Code,Title,Related O*NET-SOC Code,Related Title,Relatedness Tier,Index\n"
        "11-1011.00,Chief Executives,13-2011.00,Accountants and Auditors,Primary-Short,1\n"
    ),
}


def _xlsx_bytes(headers: list[str], row: list[object]) -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Publisher title"])
    sheet.append(headers)
    sheet.append(row)
    payload = BytesIO()
    workbook.save(payload)
    return payload.getvalue()


def _source_payloads() -> dict[str, bytes]:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    payloads = {
        spec.requested_url: ONET_ROWS[spec.table].encode("utf-8")
        for spec in build_source_specs(config)
        if spec.family == "onet"
    }
    payloads[config.bls.occupation_url] = _xlsx_bytes(
        [
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
        ],
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
    )
    payloads[config.bls.crosswalk_url] = _xlsx_bytes(
        [
            "O*NET-SOC 2019 code",
            "O*NET-SOC 2019 title",
            "2024 National Employment Matrix code",
            "2024 National Employment Matrix title",
        ],
        ["11-1011.00", "Chief Executives", "11-1011", "Chief executives"],
    )
    return payloads


class _Response:
    def __init__(self, url: str, payload: bytes) -> None:
        self.url = url
        self.status_code = 200
        self.history: tuple[object, ...] = ()
        self.headers = {
            "Content-Type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if url.endswith(".xlsx")
                else "application/octet-stream"
            ),
            "Content-Length": str(len(payload)),
            "ETag": '"fixture"',
        }
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for offset in range(0, len(self._payload), chunk_size):
            yield self._payload[offset : offset + chunk_size]

    def close(self) -> None:
        return None


class _Session:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str, **_: Any) -> _Response:
        self.calls.append(url)
        return _Response(url, self.payloads[url])


def test_acquisition_is_atomic_content_addressed_and_reusable_offline(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    session = _Session(_source_payloads())

    manifest = acquire_sources(config, tmp_path, session=session, clock=snapshot_clock)

    assert len(session.calls) == 9
    assert manifest.project_snapshot_date == "2026-08-21"
    assert len(manifest.sources) == 9
    assert all(
        source.object_path == f"objects/sha256/{source.sha256}" for source in manifest.sources
    )
    assert all(source.row_count == 1 for source in manifest.sources)
    assert all(source.archive_members for source in manifest.sources if source.media_type == "xlsx")
    snapshot = tmp_path / "snapshots" / "v1"
    assert verify_snapshot(snapshot, config=config) == manifest

    offline = _Session({})
    assert acquire_sources(config, tmp_path, session=offline, clock=snapshot_clock) == manifest
    assert offline.calls == []


def test_snapshot_rejects_unknown_files_and_manifest_row_count_tampering(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    acquire_sources(config, tmp_path, session=_Session(_source_payloads()), clock=snapshot_clock)
    snapshot = tmp_path / "snapshots" / "v1"
    extra = snapshot / "unexpected.txt"
    extra.write_text("not declared", encoding="utf-8")

    with pytest.raises(ManifestError, match="file set differs"):
        verify_snapshot(snapshot, config=config)
    extra.unlink()

    manifest_path = snapshot / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["sources"][0]["row_count"] = 2
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestError, match="Row-count provenance mismatch"):
        verify_snapshot(snapshot, config=config)


def test_snapshot_rejects_same_size_source_object_tampering(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    manifest = acquire_sources(
        config,
        tmp_path,
        session=_Session(_source_payloads()),
        clock=snapshot_clock,
    )
    snapshot = tmp_path / "snapshots" / "v1"
    occupation_source = manifest.sources[0]
    object_path = snapshot / occupation_source.object_path
    payload = bytearray(object_path.read_bytes())
    index = payload.index(ord("D"))
    payload[index] = ord("S")
    object_path.chmod(0o644)
    object_path.write_bytes(payload)

    assert object_path.stat().st_size == occupation_source.byte_count
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        verify_snapshot(snapshot, config=config)


def test_refresh_rejects_changed_publisher_bytes_without_overwriting_pin(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    payloads = _source_payloads()
    original = acquire_sources(config, tmp_path, session=_Session(payloads), clock=snapshot_clock)
    changed = dict(payloads)
    occupation_url = build_source_specs(config)[0].requested_url
    changed[occupation_url] = changed[occupation_url].replace(
        b"Determine organizational direction.", b"Set organizational direction."
    )

    with pytest.raises(PinMismatchError, match="onet.occupation_data"):
        acquire_sources(
            config,
            tmp_path,
            refresh=True,
            session=_Session(changed),
            clock=snapshot_clock,
        )

    assert verify_snapshot(tmp_path / "snapshots" / "v1", config=config) == original


def test_new_snapshot_requires_reviewed_config_version(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")

    with pytest.raises(AcquisitionError, match="configured artifact_version"):
        acquire_sources(
            config,
            tmp_path,
            snapshot_id="v2",
            new_snapshot=True,
            session=_Session(_source_payloads()),
            clock=snapshot_clock,
        )
