from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from data_test_support import canonical_frames

import job_market_intelligence.warehouse as warehouse_module
from job_market_intelligence.config import ProjectConfig, load_config
from job_market_intelligence.sources import (
    EXPECTED_SOURCE_LAYOUT,
    SourceManifest,
    SourceObject,
)
from job_market_intelligence.warehouse import (
    TABLE_COLUMNS,
    WarehouseError,
    _build_stage,
    _verified_source_manifest,
    audit_frames,
    audit_onet_release,
    verify_processed_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_source_manifest_is_checked_against_verified_raw_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(root=tmp_path, snapshot_id="v1")
    source_snapshot = tmp_path / "data" / "raw" / "snapshots" / "v1"
    source_snapshot.mkdir(parents=True)
    raw_manifest = source_snapshot / "manifest.json"
    raw_manifest.write_bytes(b'{"snapshot_id":"v1"}\n')
    public_manifest = tmp_path / "reports" / f"source_manifest.{config.snapshot_id}.json"
    public_manifest.parent.mkdir(parents=True)
    public_manifest.write_bytes(raw_manifest.read_bytes())
    digest = hashlib.sha256(public_manifest.read_bytes()).hexdigest()
    public_manifest.with_suffix(".sha256").write_text(
        f"{digest}  {public_manifest.name}\n", encoding="ascii"
    )
    verified = object()

    def verify_snapshot(path: Path, *, config: object) -> object:
        assert path == source_snapshot
        assert config is not None
        return verified

    monkeypatch.setattr(warehouse_module, "verify_snapshot", verify_snapshot)
    assert _verified_source_manifest(public_manifest, config) is verified

    public_manifest.write_bytes(b'{"snapshot_id":"tampered"}\n')
    with pytest.raises(WarehouseError, match="differs from the verified raw snapshot"):
        _verified_source_manifest(public_manifest, config)


def _source_manifest(config: ProjectConfig) -> SourceManifest:
    sources = []
    for index, (source_id, identity) in enumerate(EXPECTED_SOURCE_LAYOUT.items()):
        family, table, filename, media_type = identity
        digest = f"{index + 1:064x}"
        sources.append(
            SourceObject(
                source_id=source_id,
                family=family,
                table=table,
                publisher="Official fixture publisher",
                vintage="fixture-v1",
                filename=filename,
                media_type=media_type,
                requested_url="https://example.com/source",
                resolved_url="https://example.com/source",
                redirect_chain=(),
                retrieved_at_utc="2026-08-21T12:00:00Z",
                status_code=200,
                content_type="application/octet-stream",
                etag=None,
                last_modified=None,
                response_date=None,
                content_disposition=None,
                byte_count=1,
                sha256=digest,
                object_path=f"objects/sha256/{digest}",
                row_count=1,
                schema_fingerprint="f" * 64,
                parser_name="fixture",
                parser_version="fixture==1",
                archive_members=(),
            )
        )
    return SourceManifest(
        manifest_version=1,
        snapshot_id=config.snapshot_id,
        created_at_utc="2026-08-21T12:00:00Z",
        project_name="ai-job-market-intelligence",
        project_artifact_version=config.project.artifact_version,
        project_snapshot_date="2026-08-21",
        config_file_sha256=config.fingerprint,
        config_semantic_sha256=config.semantic_fingerprint,
        repository_commit=None,
        sources=tuple(sources),
    )


def test_canonical_frame_contract_and_quality_audit_are_strict() -> None:
    frames = canonical_frames()

    report = audit_frames(frames)

    assert report.ok
    assert report.issues == ()
    assert all(tuple(frames[name].columns) == TABLE_COLUMNS[name] for name in TABLE_COLUMNS)
    assert "sample_size" in frames["skill_ratings"].columns
    assert "n" not in frames["skill_ratings"].columns


def test_audit_rejects_market_values_not_equal_to_unsummed_official_join() -> None:
    frames = canonical_frames()
    frames["occupation_market"] = frames["occupation_market"].copy()
    frames["occupation_market"].loc[0, "employment_2024"] += 1

    report = audit_frames(frames)

    assert not report.ok
    assert {issue.check for issue in report.issues} == {"occupation_market.exact_join"}


def test_audit_rejects_tier_grade_disagreement() -> None:
    frames = canonical_frames()
    frames["related_qrels"] = frames["related_qrels"].copy()
    frames["related_qrels"].loc[0, "relevance_grade"] = 2

    report = audit_frames(frames)

    assert not report.ok
    assert "related_qrels.tier_grade_mapping" in {issue.check for issue in report.issues}


def test_audit_allows_null_task_type_but_rejects_unknown_labels() -> None:
    frames = canonical_frames()
    frames["tasks"] = frames["tasks"].copy()
    frames["tasks"].loc[0, "task_type"] = None

    assert audit_frames(frames).ok

    frames["tasks"].loc[0, "task_type"] = "Inferred"
    report = audit_frames(frames)
    assert {issue.check for issue in report.issues} == {"tasks.task_type"}


def test_processed_snapshot_carries_exact_schemas_hashes_and_rejects_unknown_files(
    tmp_path: Path,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text("{}", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    _build_stage(
        canonical_frames(),
        processed,
        source_manifest_path,
        _source_manifest(config),
        config,
    )

    result = verify_processed_snapshot(processed)

    assert result.snapshot_id == config.snapshot_id
    assert result.table_rows["occupations"] == 2
    extra = processed / "debug.csv"
    extra.write_text("not declared", encoding="utf-8")
    with pytest.raises(WarehouseError, match="file set differs"):
        verify_processed_snapshot(processed)


def test_processed_snapshot_rejects_parquet_byte_tampering(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text("{}", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    _build_stage(
        canonical_frames(),
        processed,
        source_manifest_path,
        _source_manifest(config),
        config,
    )
    parquet = processed / "tables" / "occupations.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")

    with pytest.raises(WarehouseError, match="byte-count mismatch"):
        verify_processed_snapshot(processed)


def test_processed_snapshot_rejects_same_size_warehouse_tampering(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text("{}", encoding="utf-8")
    processed = tmp_path / "processed"
    processed.mkdir()
    _build_stage(
        canonical_frames(),
        processed,
        source_manifest_path,
        _source_manifest(config),
        config,
    )
    warehouse = processed / "warehouse.duckdb"
    payload = bytearray(warehouse.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    warehouse.write_bytes(payload)

    with pytest.raises(WarehouseError, match="warehouse checksum mismatch"):
        verify_processed_snapshot(processed)


def test_release_audit_refuses_unregistered_partial_data() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "project.toml")
    partial = {
        table: pd.DataFrame()
        for table in (
            "occupation_data",
            "job_titles",
            "task_statements",
            "transferable_skills",
            "knowledge",
            "job_zones",
            "related_occupations",
        )
    }

    with pytest.raises(WarehouseError, match="release audit failed"):
        audit_onet_release(partial, config)
