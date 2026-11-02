"""Command-line adapter for the validated occupation-intelligence service."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError
from .service import (
    OccupationIntelligenceService,
    OccupationNotFoundError,
    ServiceInputError,
    get_service,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-intelligence",
        description="Query one frozen, checksum-validated occupation serving artifact.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Serving bundle directory (otherwise use JOB_INTELLIGENCE_ARTIFACT_DIR).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify-artifact", help="Validate all bytes and run smoke queries."
    )
    verify.add_argument("--skip-smoke", action="store_true")

    search = subparsers.add_parser("search", help="Search occupations with arbitrary text.")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10)

    occupation = subparsers.add_parser("occupation", help="Show an occupation and BLS contexts.")
    occupation.add_argument("onet_soc_code")

    neighbors = subparsers.add_parser("neighbors", help="Show frozen adjacent occupations.")
    neighbors.add_argument("onet_soc_code")
    neighbors.add_argument("-k", type=int, default=10)

    skills = subparsers.add_parser("compare-skills", help="Compare occupation-level ratings.")
    skills.add_argument("left_code")
    skills.add_argument("right_code")
    skills.add_argument("--feature-family", default="transferable_skill")
    skills.add_argument("--limit", type=int, default=25)

    frontier = subparsers.add_parser("frontier", help="Compute a transparent Pareto frontier.")
    frontier.add_argument("origin_code")
    frontier.add_argument(
        "--outcome",
        choices=("median_wage", "projected_growth", "annual_openings"),
        default="median_wage",
    )
    frontier.add_argument("--max-development-distance", type=float)
    frontier.add_argument("--minimum-outcome", type=float)
    frontier.add_argument(
        "--maximum-job-zone",
        type=int,
        help="2 means combined Job Zone 1-2; other official O*NET 30.3 labels are 3, 4, and 5.",
    )

    clusters = subparsers.add_parser("clusters", help="List exploratory cluster navigation.")
    clusters.add_argument("--cluster-id", type=int)
    clusters.add_argument("--summaries-only", action="store_true")

    subparsers.add_parser(
        "evidence", help="Show checksum-bound release evidence and lineage metadata."
    )
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _execute(args: argparse.Namespace, service: OccupationIntelligenceService) -> dict[str, Any]:
    if args.command == "verify-artifact":
        checks = () if args.skip_smoke else service.smoke_test()
        return {
            "status": "ok",
            "occupation_count": service.occupation_count,
            "artifact": _jsonable(service.metadata),
            "smoke_checks": [
                {"query_id": query_id, "result_count": count} for query_id, count in checks
            ],
        }
    if args.command == "search":
        return {
            "query": args.query,
            "results": _jsonable(service.search(args.query, k=args.k)),
            "interpretation": "Similarity is not a match percentage.",
        }
    if args.command == "occupation":
        return {
            "occupation": _jsonable(service.occupation(args.onet_soc_code)),
            "market_context": _jsonable(service.market_context(args.onet_soc_code)),
        }
    if args.command == "neighbors":
        return {
            "origin": _jsonable(service.occupation(args.onet_soc_code)),
            "neighbors": _jsonable(service.neighbors(args.onet_soc_code, k=args.k)),
            "interpretation": "Occupation similarity is not evidence of individual mobility or qualification.",
        }
    if args.command == "compare-skills":
        return {
            "left": _jsonable(service.occupation(args.left_code)),
            "right": _jsonable(service.occupation(args.right_code)),
            "differences": _jsonable(
                service.compare_skills(
                    args.left_code,
                    args.right_code,
                    feature_family=args.feature_family,
                    limit=args.limit,
                )
            ),
        }
    if args.command == "frontier":
        return {
            "origin": _jsonable(service.occupation(args.origin_code)),
            "outcome": args.outcome,
            "frontier": _jsonable(
                service.frontier(
                    args.origin_code,
                    outcome=args.outcome,
                    max_development_distance=args.max_development_distance,
                    minimum_outcome=args.minimum_outcome,
                    maximum_job_zone=args.maximum_job_zone,
                )
            ),
            "interpretation": "National projections are scenarios and annual openings are not live vacancies.",
        }
    if args.command == "clusters":
        result: dict[str, Any] = {
            "summaries": _jsonable(service.cluster_summaries()),
            "interpretation": "Clusters are exploratory navigation aids, not occupational truth.",
        }
        if not args.summaries_only:
            result["members"] = _jsonable(service.clusters(cluster_id=args.cluster_id))
        return result
    if args.command == "evidence":
        return {
            "status": service.metadata.evidence_status,
            "artifact": _jsonable(service.metadata),
            "reports": _jsonable(service.evidence_reports),
        }
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        service = get_service(args.artifact_dir)
        payload = _execute(args, service)
    except (ServiceInputError, OccupationNotFoundError) as exc:
        print(json.dumps({"status": "invalid_request", "error": str(exc)}), file=sys.stderr)
        return 2
    except ArtifactError as exc:
        print(
            json.dumps(
                {
                    "status": "artifact_unavailable",
                    "error": str(exc),
                    "hint": "Build or mount the checksum-validated serving bundle.",
                }
            ),
            file=sys.stderr,
        )
        return 3
    except RuntimeError as exc:
        print(json.dumps({"status": "service_error", "error": str(exc)}), file=sys.stderr)
        return 4
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
