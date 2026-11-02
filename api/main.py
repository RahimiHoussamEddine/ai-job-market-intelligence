"""FastAPI adapter for the immutable occupation-intelligence service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from job_market_intelligence.service import (
    OccupationIntelligenceService,
    OccupationNotFoundError,
    ServiceInputError,
    get_service_state,
)

app = FastAPI(
    title="AI Job Market Intelligence API",
    version="1.0.0",
    description=(
        "Aggregate U.S. occupation search and transparent career-transition context. "
        "Similarity is not an individual match or qualification assessment."
    ),
)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=512)
    k: int = Field(default=10, ge=1, le=25)


async def _service() -> OccupationIntelligenceService:
    state = get_service_state()
    if not state.available or state.service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "artifact_unavailable",
                "message": "The validated serving artifact is not available.",
                "reason_code": state.reason_code,
            },
        )
    return state.service


ServiceDependency = Annotated[OccupationIntelligenceService, Depends(_service)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@app.exception_handler(OccupationNotFoundError)
async def _not_found_handler(_request: Any, exc: OccupationNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": {"code": "occupation_not_found", "message": str(exc)}},
    )


@app.exception_handler(ServiceInputError)
async def _input_handler(_request: Any, exc: ServiceInputError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "invalid_request", "message": str(exc)}},
    )


@app.get("/health", tags=["operations"])
async def health(deep: bool = False) -> JSONResponse:
    state = get_service_state()
    if not state.available or state.service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "reason_code": state.reason_code,
                "message": "The validated serving artifact is not available.",
                "artifact_loaded": False,
            },
        )
    service = state.service
    smoke_checks: list[dict[str, Any]] | None = None
    if deep:
        try:
            smoke_checks = [
                {"query_id": query_id, "result_count": count}
                for query_id, count in service.smoke_test()
            ]
        except RuntimeError:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "reason_code": "artifact_smoke_failed",
                    "message": "The loaded artifact failed a target-free smoke check.",
                    "artifact_loaded": True,
                },
            )
    return JSONResponse(
        content={
            "status": "ok",
            "artifact_loaded": True,
            "occupation_count": service.occupation_count,
            "artifact": asdict(service.metadata),
            "smoke_checks": smoke_checks,
        }
    )


@app.post("/v1/search", tags=["discovery"])
async def search(
    service: ServiceDependency,
    request: SearchRequest,
) -> dict[str, Any]:
    return {
        "query": request.query,
        "results": [asdict(result) for result in service.search(request.query, k=request.k)],
        "interpretation": "Similarity reflects the frozen occupation text model; it is not a match percentage.",
    }


@app.get("/v1/occupations/{onet_soc_code}", tags=["occupations"])
async def occupation(onet_soc_code: str, service: ServiceDependency) -> dict[str, Any]:
    return {
        "occupation": asdict(service.occupation(onet_soc_code)),
        "market_context": [asdict(context) for context in service.market_context(onet_soc_code)],
    }


@app.get("/v1/occupations/{onet_soc_code}/neighbors", tags=["transitions"])
async def neighbors(
    onet_soc_code: str,
    service: ServiceDependency,
    k: Annotated[int, Query(ge=1, le=25)] = 10,
) -> dict[str, Any]:
    return {
        "origin": asdict(service.occupation(onet_soc_code)),
        "neighbors": [asdict(result) for result in service.neighbors(onet_soc_code, k=k)],
        "interpretation": (
            "These are occupation-level similarities, not worker mobility estimates or "
            "qualification determinations."
        ),
    }


@app.get("/v1/skills/compare", tags=["skills"])
async def compare_skills(
    service: ServiceDependency,
    left: Annotated[str, Query(min_length=1, max_length=32)],
    right: Annotated[str, Query(min_length=1, max_length=32)],
    feature_family: Annotated[str, Query(min_length=1, max_length=80)] = "transferable_skill",
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, Any]:
    return {
        "left": asdict(service.occupation(left)),
        "right": asdict(service.occupation(right)),
        "differences": [
            asdict(result)
            for result in service.compare_skills(
                left,
                right,
                feature_family=feature_family,
                limit=limit,
            )
        ],
        "interpretation": (
            "Values are occupation-level O*NET ratings. Missing or suppressed ratings are "
            "shown as missing, never imputed."
        ),
    }


@app.get("/v1/occupations/{onet_soc_code}/frontier", tags=["transitions"])
async def frontier(
    onet_soc_code: str,
    service: ServiceDependency,
    outcome: Annotated[
        str,
        Query(pattern="^(median_wage|projected_growth|annual_openings)$"),
    ] = "median_wage",
    max_development_distance: Annotated[float | None, Query(ge=0)] = None,
    minimum_outcome: float | None = None,
    maximum_job_zone: Annotated[
        int | None,
        Query(
            ge=2,
            le=5,
            description=(
                "Official O*NET 30.3 stored label: 2 is the combined Job Zone 1-2 category; "
                "other values are 3, 4, and 5."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    return {
        "origin": asdict(service.occupation(onet_soc_code)),
        "outcome": outcome,
        "frontier": [
            asdict(point)
            for point in service.frontier(
                onet_soc_code,
                outcome=outcome,
                max_development_distance=max_development_distance,
                minimum_outcome=minimum_outcome,
                maximum_job_zone=maximum_job_zone,
            )
        ],
        "interpretation": (
            "Each row is Pareto efficient across lower observed development distance and the "
            "selected national BLS outcome. Projections are scenarios; openings are not vacancies."
        ),
    }


@app.get("/v1/clusters", tags=["navigation"])
async def clusters(
    service: ServiceDependency,
    cluster_id: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    members = service.clusters(cluster_id=cluster_id)
    return {
        "summaries": [asdict(summary) for summary in service.cluster_summaries()],
        "members": [asdict(member) for member in members],
        "interpretation": "Clusters are exploratory navigation aids, not occupational truth.",
    }


@app.get("/v1/evidence", tags=["evidence"])
async def evidence(service: ServiceDependency) -> dict[str, Any]:
    """Expose only checksum-bound reports accepted by the artifact loader."""

    return {
        "status": service.metadata.evidence_status,
        "artifact": asdict(service.metadata),
        "reports": _jsonable(service.evidence_reports),
    }
