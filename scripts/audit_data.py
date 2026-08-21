#!/usr/bin/env python3
"""Run offline checksum, schema, provenance, and quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from job_market_intelligence.config import ConfigError, load_config
from job_market_intelligence.sources import AcquisitionError, verify_snapshot
from job_market_intelligence.warehouse import (
    WarehouseError,
    audit_warehouse,
    verify_processed_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify immutable raw and processed data snapshots without network access."
    )
    parser.add_argument("--config", type=Path, help="Path to configs/project.toml")
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--processed-snapshot", type=Path)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Verify source objects and provenance before a processed snapshot exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        source_snapshot = args.source_snapshot or (
            config.root / "data" / "raw" / "snapshots" / config.snapshot_id
        )
        source_manifest = verify_snapshot(source_snapshot, config=config)
        payload: dict[str, object] = {
            "ok": True,
            "snapshot_id": source_manifest.snapshot_id,
            "snapshot_date": source_manifest.project_snapshot_date,
            "source_rows": {
                source.source_id: source.row_count for source in source_manifest.sources
            },
            "source_sha256": {
                source.source_id: source.sha256 for source in source_manifest.sources
            },
        }
        if not args.source_only:
            processed_snapshot = args.processed_snapshot or (
                config.root / "data" / "processed" / "snapshots" / config.snapshot_id
            )
            result = verify_processed_snapshot(
                processed_snapshot,
                config=config,
                source_manifest_path=source_snapshot,
            )
            audit = audit_warehouse(result.warehouse_path)
            payload["warehouse_sha256"] = result.warehouse_sha256
            payload["processed_rows"] = dict(result.table_rows)
            payload["quality_audit"] = audit.to_dict()
    except (ConfigError, AcquisitionError, WarehouseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
