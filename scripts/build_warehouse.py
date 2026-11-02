#!/usr/bin/env python3
"""Build and verify the immutable DuckDB/Parquet processed snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from job_market_intelligence.config import ConfigError, load_config
from job_market_intelligence.sources import AcquisitionError
from job_market_intelligence.warehouse import WarehouseError, build_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic canonical tables from a verified source snapshot."
    )
    parser.add_argument("--config", type=Path, help="Path to configs/project.toml")
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="Source snapshot directory or manifest.json (default: data/raw/snapshots/<version>)",
    )
    parser.add_argument("--processed-root", type=Path, help="Override data/processed root")
    parser.add_argument("--snapshot-id", help="Must equal the source snapshot ID")
    parser.add_argument(
        "--publish-source-manifest",
        type=Path,
        help="Public manifest path (default: reports/source_manifest.<version>.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        source_snapshot = args.source_snapshot or (
            config.root / "data" / "raw" / "snapshots" / config.snapshot_id
        )
        result = build_snapshot(
            config,
            source_snapshot,
            processed_root=args.processed_root,
            snapshot_id=args.snapshot_id,
            publish_manifest_path=args.publish_source_manifest,
        )
    except (ConfigError, AcquisitionError, WarehouseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {
        "snapshot_id": result.snapshot_id,
        "source_manifest_sha256": result.source_manifest_sha256,
        "warehouse_sha256": result.warehouse_sha256,
        "table_rows": dict(result.table_rows),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
