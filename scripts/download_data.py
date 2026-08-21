#!/usr/bin/env python3
"""Acquire or verify the locked official-data snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from job_market_intelligence.config import ConfigError, load_config
from job_market_intelligence.sources import (
    AcquisitionError,
    acquire_sources,
    build_source_specs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the nine locked O*NET 30.3/BLS 2024-34 source files into an "
            "immutable checksum-addressed snapshot. Existing snapshots are verified "
            "offline unless --refresh is supplied."
        )
    )
    parser.add_argument("--config", type=Path, help="Path to configs/project.toml")
    parser.add_argument("--raw-root", type=Path, help="Override data/raw output root")
    parser.add_argument("--snapshot-id", help="Must match [project].artifact_version")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download and compare all bytes with an existing immutable pin",
    )
    parser.add_argument(
        "--new-snapshot",
        action="store_true",
        help="Assert that the explicitly named, config-matching snapshot is new",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the exact request plan without creating directories or using the network",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    return parser


def _plan(config: Any) -> dict[str, Any]:
    return {
        "mode": "plan",
        "project": config.project.name,
        "artifact_version": config.project.artifact_version,
        "snapshot_date": config.project.snapshot_date.isoformat(),
        "config_file_sha256": config.fingerprint,
        "config_semantic_sha256": config.semantic_fingerprint,
        "sources": [
            {
                "source_id": spec.source_id,
                "publisher": spec.publisher,
                "vintage": spec.vintage,
                "url": spec.requested_url,
                "filename": spec.filename,
                "media_type": spec.media_type,
            }
            for spec in build_source_specs(config)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        print("error: timeout values must be positive", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        if args.plan:
            if args.refresh or args.new_snapshot:
                raise AcquisitionError("--plan cannot be combined with mutation flags")
            payload = _plan(config)
        else:
            manifest = acquire_sources(
                config,
                raw_root=args.raw_root,
                snapshot_id=args.snapshot_id,
                new_snapshot=args.new_snapshot,
                refresh=args.refresh,
                timeout=(args.connect_timeout, args.read_timeout),
            )
            payload = {
                "mode": "verified" if not args.refresh else "refreshed_and_verified",
                "snapshot_id": manifest.snapshot_id,
                "snapshot_date": manifest.project_snapshot_date,
                "created_at_utc": manifest.created_at_utc,
                "config_file_sha256": manifest.config_file_sha256,
                "source_count": len(manifest.sources),
                "sources": [
                    {
                        "source_id": source.source_id,
                        "sha256": source.sha256,
                        "byte_count": source.byte_count,
                        "row_count": source.row_count,
                        "schema_fingerprint": source.schema_fingerprint,
                    }
                    for source in manifest.sources
                ],
            }
    except (ConfigError, AcquisitionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
