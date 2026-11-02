from __future__ import annotations

import asyncio
import json
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

import job_market_intelligence.artifacts as artifact_module
from api.main import _service, app
from job_market_intelligence.artifacts import (
    ArtifactIntegrityError,
    ArtifactSafetyError,
    ArtifactSchemaError,
    load_artifact_bundle,
    write_artifact_bundle,
)
from job_market_intelligence.service import (
    ServiceInputError,
    clear_service_cache,
    get_service,
)


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, manifest: dict[str, object]) -> None:
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_from_loaded_fixture(source_path: Path, output_path: Path, **overrides: object) -> Path:
    source = load_artifact_bundle(source_path)
    raw_manifest = _manifest(source_path)
    provenance = dict(raw_manifest["provenance"])  # type: ignore[arg-type]
    provenance.pop("project_config_sha256")
    arguments: dict[str, object] = {
        **dict(source.tables),
        "artifact_version": "writer-roundtrip-v1",
        "snapshot_date": "2026-08-14",
        "title_vectorizer": source.search_transformer,
        "title_matrix": source.search_matrix,
        "smoke_queries": [
            {"id": query_id, "query": query} for query_id, query in source.smoke_queries
        ],
        "evidence_reports": {},
        "model_metadata": {
            "title_candidate_id": source.manifest["models"]["title_candidate_id"],
            "transition_candidate_id": source.manifest["models"]["transition_candidate_id"],
        },
        "provenance": provenance,
    }
    arguments.update(overrides)
    return write_artifact_bundle(output_path, **arguments)


def test_writer_round_trips_and_never_overwrites(synthetic_bundle: Path) -> None:
    output = synthetic_bundle.parent / "writer-roundtrip-v1"

    published = _write_from_loaded_fixture(synthetic_bundle, output)
    loaded = load_artifact_bundle(published)

    assert published == output.resolve()
    assert loaded.manifest["artifact_version"] == "writer-roundtrip-v1"
    assert loaded.manifest["files"]["title_matrix"]["shape"] == tuple(loaded.search_matrix.shape)
    assert loaded.smoke_queries == (("arbitrary-text", "software applications"),)
    assert not list(output.parent.glob(".writer-roundtrip-v1.*"))
    assert published.stat().st_mode & 0o777 == 0o755
    for candidate in published.rglob("*"):
        expected_mode = 0o755 if candidate.is_dir() else 0o644
        assert candidate.stat().st_mode & 0o777 == expected_mode

    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        _write_from_loaded_fixture(synthetic_bundle, output)


def test_writer_cleans_staging_when_full_validation_fails(synthetic_bundle: Path) -> None:
    source = load_artifact_bundle(synthetic_bundle)
    source.search_transformer.related_qrels_ = {"must": "never ship"}
    output = synthetic_bundle.parent / "writer-rejected-v1"

    with pytest.raises(ArtifactSafetyError, match="evaluation-only"):
        _write_from_loaded_fixture(
            synthetic_bundle,
            output,
            title_vectorizer=source.search_transformer,
        )

    assert not output.exists()
    assert not list(output.parent.glob(".writer-rejected-v1.*"))


def test_bundle_and_shared_service_preserve_release_semantics(synthetic_bundle: Path) -> None:
    bundle = load_artifact_bundle(synthetic_bundle)
    service = get_service(synthetic_bundle)

    assert bundle.evidence_reports == {}
    assert get_service(synthetic_bundle) is service
    assert service.metadata.evidence_status == "not_generated"
    assert {
        "title_vectorizer",
        "title_matrix",
        "neighbors",
        "smoke_queries",
    } <= dict(service.metadata.serving_file_hashes).keys()
    assert service.occupation_count == 3
    assert service.smoke_test() == (("arbitrary-text", 1),)
    assert service.search("software applications", k=2)[0].onet_soc_code == "15-1252.00"
    assert len(service.neighbors("15-1252.00", k=25)) == 2
    with pytest.raises(ServiceInputError, match="1 through 25"):
        service.neighbors("15-1252.00", k=26)

    differences = service.compare_skills("15-1252.00", "15-2051.00", limit=10)
    programming = next(item for item in differences if item.element_name == "Programming")
    assert programming.left_value == pytest.approx(0.96)
    assert programming.right_value is None
    assert programming.left_metadata is not None
    assert programming.left_metadata.sample_size == 120

    frontier = service.frontier("15-1252.00", outcome="median_wage")
    assert [point.onet_soc_code for point in frontier] == ["15-2051.00"]
    assert (
        service.frontier("15-1252.00", outcome="median_wage", maximum_job_zone=3)[0].onet_soc_code
        == "29-1141.00"
    )
    with pytest.raises(ServiceInputError, match="combined Job Zone 1-2"):
        service.frontier("15-1252.00", outcome="median_wage", maximum_job_zone=1)
    with pytest.raises(ServiceInputError, match="combined Job Zone 1-2"):
        service.frontier("15-1252.00", outcome="median_wage", maximum_job_zone=2.0)  # type: ignore[arg-type]

    with pytest.raises((FrozenInstanceError, AttributeError)):
        service.metadata.artifact_version = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        service._metadata = service.metadata  # type: ignore[misc]


def test_checksum_failure_prevents_model_deserialization(
    synthetic_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    occupation_path = synthetic_bundle / "occupations.parquet"
    occupation_path.write_bytes(occupation_path.read_bytes() + b"tampered")
    deserialized = False

    def forbidden_load(_path: Path) -> object:
        nonlocal deserialized
        deserialized = True
        raise AssertionError("model deserialization must not run")

    monkeypatch.setattr(artifact_module.joblib, "load", forbidden_load)
    with pytest.raises(ArtifactIntegrityError, match="size mismatch"):
        load_artifact_bundle(synthetic_bundle)
    assert deserialized is False


def test_config_matrix_and_reports_are_checked_before_joblib(
    synthetic_bundle: Path,
    resign_artifact_file,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deserialized = False

    def forbidden_load(_path: Path) -> object:
        nonlocal deserialized
        deserialized = True
        raise AssertionError("model deserialization must not run")

    monkeypatch.setattr(artifact_module.joblib, "load", forbidden_load)

    matrix_path = synthetic_bundle / "title_matrix.npz"
    sparse.save_npz(matrix_path, sparse.eye(2, dtype="float64", format="csr"))
    resign_artifact_file("title_matrix")
    manifest = _manifest(synthetic_bundle)
    manifest["files"]["title_matrix"]["shape"] = [2, 2]  # type: ignore[index]
    _write_manifest(synthetic_bundle, manifest)
    with pytest.raises(ArtifactSchemaError, match="search_catalog"):
        load_artifact_bundle(synthetic_bundle)
    assert deserialized is False

    # Restore only what is needed to prove report validation precedes joblib.
    def reject_reports(*_args: object, **_kwargs: object) -> object:
        raise ArtifactSchemaError("report contract rejected")

    original_matrix = sparse.csr_matrix(
        [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]], dtype="float64"
    )
    sparse.save_npz(matrix_path, original_matrix)
    resign_artifact_file("title_matrix")
    manifest = _manifest(synthetic_bundle)
    manifest["models"]["title_feature_count"] = 2  # type: ignore[index]
    manifest["files"]["title_matrix"]["shape"] = [3, 2]  # type: ignore[index]
    _write_manifest(synthetic_bundle, manifest)
    monkeypatch.setattr(artifact_module, "_validate_evidence_reports", reject_reports)
    with pytest.raises(ArtifactSchemaError, match="report contract"):
        load_artifact_bundle(synthetic_bundle)
    assert deserialized is False


def test_literal_config_and_foreign_keys_are_checked_before_joblib(
    synthetic_bundle: Path,
    resign_artifact_file,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deserialized = False

    def forbidden_load(_path: Path) -> object:
        nonlocal deserialized
        deserialized = True
        raise AssertionError("model deserialization must not run")

    monkeypatch.setattr(artifact_module.joblib, "load", forbidden_load)
    config_path = synthetic_bundle.parent.parent / "configs" / "project.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="project_config_sha256"):
        load_artifact_bundle(synthetic_bundle)
    assert deserialized is False

    manifest = _manifest(synthetic_bundle)
    manifest["provenance"]["project_config_sha256"] = artifact_module.sha256_file(  # type: ignore[index]
        config_path
    )
    _write_manifest(synthetic_bundle, manifest)
    neighbor_path = synthetic_bundle / "neighbors.parquet"
    rows = pq.read_table(neighbor_path).to_pylist()
    rows[0]["destination_code"] = "99-9999.00"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=artifact_module.CORE_TABLE_SCHEMAS["neighbors"]),
        neighbor_path,
    )
    resign_artifact_file("neighbors")
    with pytest.raises(ArtifactSchemaError, match="unknown occupation"):
        load_artifact_bundle(synthetic_bundle)
    assert deserialized is False


def test_qrel_names_and_undeclared_files_are_rejected(synthetic_bundle: Path) -> None:
    manifest = _manifest(synthetic_bundle)
    manifest["files"]["related_qrels"] = dict(  # type: ignore[index]
        manifest["files"]["smoke_queries"]  # type: ignore[index]
    )
    _write_manifest(synthetic_bundle, manifest)
    with pytest.raises(ArtifactSafetyError, match="evaluation-only"):
        load_artifact_bundle(synthetic_bundle)


def test_parquet_metadata_cannot_hide_judgment_payloads(
    synthetic_bundle: Path, resign_artifact_file
) -> None:
    catalog_path = synthetic_bundle / "search_catalog.parquet"
    table = pq.read_table(catalog_path).replace_schema_metadata(
        {b"human_judgments": b"must never ship"}
    )
    pq.write_table(table, catalog_path)
    resign_artifact_file("search_catalog")

    with pytest.raises(ArtifactSchemaError, match="schema"):
        load_artifact_bundle(synthetic_bundle)


def test_matrix_archive_cannot_hide_extra_payloads(
    synthetic_bundle: Path, resign_artifact_file
) -> None:
    matrix_path = synthetic_bundle / "title_matrix.npz"
    with zipfile.ZipFile(matrix_path, mode="a") as archive:
        archive.writestr("human_judgments.npy", b"must never ship")
    resign_artifact_file("title_matrix")

    with pytest.raises(ArtifactSafetyError, match="only the canonical CSR"):
        load_artifact_bundle(synthetic_bundle)


def test_hidden_judgment_metadata_in_model_is_rejected(
    synthetic_bundle: Path, resign_artifact_file
) -> None:
    transformer = load_artifact_bundle(synthetic_bundle).search_transformer
    transformer.release_metadata_ = {"human_judgments": {"must": "never ship"}}
    joblib.dump(transformer, synthetic_bundle / "title_vectorizer.joblib")
    resign_artifact_file("title_vectorizer")

    with pytest.raises(ArtifactSafetyError, match="evaluation-only"):
        load_artifact_bundle(synthetic_bundle)


def test_locked_evidence_schema_binds_candidates_metrics_and_tables(tmp_path: Path) -> None:
    metrics = lambda names: [  # noqa: E731 - compact deterministic report factory
        {
            "name": name,
            "estimate": 0.5,
            "ci_lower": 0.4,
            "ci_upper": 0.6,
            "n_queries": 20,
        }
        for name in names
    ]
    deltas = lambda baseline, names, estimate: [  # noqa: E731
        {
            "baseline_candidate_id": baseline,
            "metric": name,
            "delta": estimate,
            "ci_lower": estimate - 0.08,
            "ci_upper": estimate + 0.08,
        }
        for name in names
    ]
    slices = [
        {
            "dimension": dimension,
            "value": value,
            "estimate": 0.5,
            "ci_lower": 0.4,
            "ci_upper": 0.6,
            "n_units": 20,
            "descriptive_small_slice": False,
        }
        for dimension, value in (
            ("soc_major_group", "15"),
            ("job_zone", "4"),
            ("profile_completeness", "1-49"),
            ("rating_recency", "missing"),
        )
    ]
    payloads = {
        "title_benchmark": {
            "selected_candidate_id": "title-v1",
            "primary_metric": "top1_accuracy",
            "metrics": metrics(("top1_accuracy", "recall_at_5", "mrr")),
            "slices": slices,
            "baseline_deltas": deltas(
                "title_word_char", ("top1_accuracy", "recall_at_5", "mrr"), 0.1
            ),
        },
        "transition_benchmark": {
            "selected_candidate_id": "transition-v1",
            "primary_metric": "ndcg_at_10",
            "metrics": metrics(
                (
                    "ndcg_at_10",
                    "primary_short_recall_at_5",
                    "primary_recall_at_10",
                    "ndcg_at_20",
                    "mrr_at_10",
                    "binary_primary_ndcg_at_10",
                )
            ),
            "slices": slices,
            "baseline_deltas": deltas(
                "transition_title_word_char",
                (
                    "ndcg_at_10",
                    "primary_short_recall_at_5",
                    "primary_recall_at_10",
                    "ndcg_at_20",
                    "mrr_at_10",
                    "binary_primary_ndcg_at_10",
                ),
                0.05,
            ),
        },
        "cluster_stability": {
            "selected_k": 12,
            "stability_metric": "pairwise_adjusted_rand_stability",
            "stability_estimate": 0.72,
            "silhouette": 0.31,
            "seeds": [20260814, 20260815, 20260816, 20260817, 20260818],
        },
        "data_coverage": {
            "counts": {
                "occupation_count": 2,
                "occupations_with_bls": 2,
                "nem_occupation_count": 2,
                "crosswalk_rows": 2,
            },
            "rates": {
                "bls_coverage": 1.0,
                "rated_occupation_coverage": 0.5,
                "job_zone_coverage": 0.5,
            },
            "notes": ["Coverage is measured at serving-table grain."],
        },
        "limitations": {
            "items": [
                "Agreement with O*NET lists is not independent semantic truth.",
                "National projections are not live vacancies.",
            ]
        },
    }
    logical_names = {report_type: f"report_{report_type}" for report_type in payloads}
    paths: dict[str, Path] = {}
    for report_type, payload in payloads.items():
        logical_name = logical_names[report_type]
        path = tmp_path / f"{logical_name}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "report_type": report_type,
                    "evidence_status": "locked",
                    "split": (
                        "locked_test"
                        if report_type in {"title_benchmark", "transition_benchmark"}
                        else "release_context"
                    ),
                    "artifact_version": "v1",
                    "payload": payload,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        paths[logical_name] = path

    manifest = {
        "artifact_version": "v1",
        "models": {
            "title_candidate_id": "title-v1",
            "transition_candidate_id": "transition-v1",
        },
        "evidence": {"status": "locked", "report_files": logical_names},
    }
    tables = {
        "occupations": pa.table(
            {
                "onet_soc_code": ["01-0001.00", "02-0002.00"],
                "structured_observed_count": [1, 0],
                "job_zone": [2, None],
            }
        ),
        "occupation_market": pa.table(
            {
                "onet_soc_code": ["01-0001.00", "02-0002.00"],
                "nem_code": ["01-0001", "02-0002"],
            }
        ),
        "clusters": pa.table({"cluster_id": list(range(12))}),
    }
    reports = artifact_module._validate_evidence_reports(manifest, paths, tables)
    assert set(reports) == set(payloads)

    payloads["title_benchmark"]["selected_candidate_id"] = "unbound-candidate"
    title_path = paths[logical_names["title_benchmark"]]
    wrapper = json.loads(title_path.read_text(encoding="utf-8"))
    wrapper["payload"] = payloads["title_benchmark"]
    title_path.write_text(json.dumps(wrapper, sort_keys=True), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="does not match the serving model"):
        artifact_module._validate_evidence_reports(manifest, paths, tables)


def test_locked_transition_rankings_separate_assigned_and_eligible_universes(
    tmp_path: Path,
) -> None:
    codes = [f"11-{index:04d}.00" for index in range(1, 6)]
    test_codes = codes[:2]
    fold_manifest = pd.DataFrame(
        {
            "onet_soc_code": codes,
            "role": ["test", "test", "discovery", "selection", "confirmation"],
        }
    )
    manifest = {
        "models": {
            "title_candidate_id": "title_word_char",
            "transition_candidate_id": "transition_title_word_char",
        }
    }
    universe = {
        "title": {
            "roles": {
                "test": {
                    "query_ids": ["alias-a", "alias-b"],
                    "eligible_occupation_count": 2,
                }
            }
        },
        # Only one of the two assigned test occupations has a usable qrel.
        "transition": {"roles": {"test": {"query_codes": [test_codes[0]]}}},
    }
    neighbor_rows: list[dict[str, object]] = []
    locked_rows: list[dict[str, object]] = [
        {
            "benchmark": "title",
            "candidate_id": "title_word_char",
            "query_id": "alias-a",
            "query_code": test_codes[0],
            "candidate_code": test_codes[0],
            "rank": 1,
            "score": 0.9,
            "split": "locked_test",
        },
        {
            "benchmark": "title",
            "candidate_id": "title_word_char",
            "query_id": "alias-b",
            "query_code": test_codes[1],
            "candidate_code": test_codes[1],
            "rank": 1,
            "score": 0.8,
            "split": "locked_test",
        },
    ]
    for query_code in test_codes:
        destinations = [code for code in codes if code != query_code]
        for rank, destination_code in enumerate(destinations, start=1):
            score = 1.0 - rank / 10
            neighbor_rows.append(
                {
                    "query_code": query_code,
                    "destination_code": destination_code,
                    "rank": rank,
                    "total_score": score,
                }
            )
            locked_rows.append(
                {
                    "benchmark": "transition",
                    "candidate_id": "transition_title_word_char",
                    "query_id": query_code,
                    "query_code": query_code,
                    "candidate_code": destination_code,
                    "rank": rank,
                    "score": score,
                    "split": "locked_test",
                }
            )
    locked_rows.sort(
        key=lambda row: (
            str(row["benchmark"]),
            str(row["candidate_id"]),
            str(row["query_id"]),
            int(row["rank"]),
        )
    )
    ranking_path = tmp_path / "locked_rankings.parquet"
    pq.write_table(
        pa.Table.from_pylist(locked_rows, schema=artifact_module.SAFE_LOCKED_RANKING_SCHEMA),
        ranking_path,
    )
    tables = {
        "occupations": pa.table({"onet_soc_code": codes}),
        "neighbors": pa.Table.from_pylist(neighbor_rows),
    }

    validated = artifact_module._validate_locked_rankings(
        ranking_path,
        manifest=manifest,
        fold_manifest=fold_manifest,
        universe=universe,
        tables=tables,
    )
    assert set(validated.loc[validated["benchmark"].eq("transition"), "query_code"]) == set(
        test_codes
    )

    incomplete = validated.loc[
        ~(validated["benchmark"].eq("transition") & validated["query_code"].eq(test_codes[1]))
    ]
    pq.write_table(
        pa.Table.from_pylist(
            incomplete.to_dict("records"),
            schema=artifact_module.SAFE_LOCKED_RANKING_SCHEMA,
        ),
        ranking_path,
    )
    with pytest.raises(ArtifactIntegrityError, match="every assigned test query"):
        artifact_module._validate_locked_rankings(
            ranking_path,
            manifest=manifest,
            fold_manifest=fold_manifest,
            universe=universe,
            tables=tables,
        )


def test_benchmark_deltas_are_conditional_on_truthful_baseline_comparison() -> None:
    metrics = [
        {
            "name": name,
            "estimate": 0.5,
            "ci_lower": 0.4,
            "ci_upper": 0.6,
            "n_queries": 20,
        }
        for name in ("top1_accuracy", "recall_at_5", "mrr")
    ]
    manifest = {"models": {"title_candidate_id": "title_word_char"}}
    fallback_payload = {
        "selected_candidate_id": "title_word_char",
        "primary_metric": "top1_accuracy",
        "metrics": metrics,
        "slices": [
            {
                "dimension": dimension,
                "value": value,
                "estimate": 0.5,
                "ci_lower": 0.4,
                "ci_upper": 0.6,
                "n_units": 20,
                "descriptive_small_slice": False,
            }
            for dimension, value in (
                ("soc_major_group", "15"),
                ("job_zone", "4"),
                ("profile_completeness", "1-49"),
                ("rating_recency", "missing"),
            )
        ],
        "baseline_deltas": [],
    }
    artifact_module._validate_benchmark_payload(
        "title_benchmark", fallback_payload, manifest, "evidence.title"
    )

    fallback_payload["slices"][0]["descriptive_small_slice"] = True
    with pytest.raises(ArtifactSchemaError, match=r"must equal \(n_units < 20\)"):
        artifact_module._validate_benchmark_payload(
            "title_benchmark", fallback_payload, manifest, "evidence.title"
        )
    fallback_payload["slices"][0]["descriptive_small_slice"] = False

    fallback_payload["baseline_deltas"] = [
        {
            "baseline_candidate_id": "title_word_char",
            "metric": "top1_accuracy",
            "delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    ]
    with pytest.raises(ArtifactSchemaError, match="must be empty"):
        artifact_module._validate_benchmark_payload(
            "title_benchmark", fallback_payload, manifest, "evidence.title"
        )

    manifest["models"]["title_candidate_id"] = "title-profile-champion"
    fallback_payload["selected_candidate_id"] = "title-profile-champion"
    fallback_payload["baseline_deltas"] = []
    with pytest.raises(ArtifactSchemaError, match="must be non-empty"):
        artifact_module._validate_benchmark_payload(
            "title_benchmark", fallback_payload, manifest, "evidence.title"
        )

    fallback_payload["baseline_deltas"] = [
        {
            "baseline_candidate_id": "unregistered-comparator",
            "metric": "top1_accuracy",
            "delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    ]
    with pytest.raises(ArtifactSchemaError, match="preregistered 'title_word_char'"):
        artifact_module._validate_benchmark_payload(
            "title_benchmark", fallback_payload, manifest, "evidence.title"
        )


def test_undeclared_file_is_rejected(synthetic_bundle: Path) -> None:
    (synthetic_bundle / "unexpected.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="not declared"):
        load_artifact_bundle(synthetic_bundle)


def test_api_is_strict_shared_and_does_not_offer_get_search(synthetic_bundle: Path) -> None:
    service = get_service(synthetic_bundle)

    async def override_service():
        return service

    app.dependency_overrides[_service] = override_service

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health?deep=true")
            assert health.status_code == 200
            assert health.json()["smoke_checks"] == [
                {"query_id": "arbitrary-text", "result_count": 1}
            ]

            response = await client.post(
                "/v1/search", json={"query": "software applications", "k": 2}
            )
            assert response.status_code == 200
            assert response.json()["results"][0]["onet_soc_code"] == "15-1252.00"
            get_response = await client.get("/v1/search", params={"query": "software"})
            assert get_response.status_code == 405
            extra_response = await client.post(
                "/v1/search",
                json={"query": "software", "k": 2, "unexpected": True},
            )
            assert extra_response.status_code == 422
            coercion_response = await client.post(
                "/v1/search", json={"query": "software", "k": "2"}
            )
            assert coercion_response.status_code == 422

            occupation = await client.get("/v1/occupations/15-1252.00")
            assert occupation.status_code == 200
            assert occupation.json()["occupation"]["canonical_title"] == "Software Developers"
            neighbors = await client.get("/v1/occupations/15-1252.00/neighbors", params={"k": 2})
            assert neighbors.status_code == 200
            assert len(neighbors.json()["neighbors"]) == 2
            skills = await client.get(
                "/v1/skills/compare",
                params={"left": "15-1252.00", "right": "15-2051.00", "limit": 10},
            )
            assert skills.status_code == 200
            assert skills.json()["differences"]
            frontier = await client.get(
                "/v1/occupations/15-1252.00/frontier",
                params={"outcome": "median_wage", "maximum_job_zone": 4},
            )
            assert frontier.status_code == 200
            assert frontier.json()["frontier"]
            invalid_zone = await client.get(
                "/v1/occupations/15-1252.00/frontier", params={"maximum_job_zone": 1}
            )
            assert invalid_zone.status_code == 422
            clusters = await client.get("/v1/clusters")
            assert clusters.status_code == 200
            assert clusters.json()["summaries"]
            missing_occupation = await client.get("/v1/occupations/99-9999.00")
            assert missing_occupation.status_code == 404
            assert missing_occupation.json()["detail"]["code"] == "occupation_not_found"

            evidence = await client.get("/v1/evidence")
            assert evidence.status_code == 200
            assert evidence.json()["status"] == "not_generated"
            assert evidence.json()["reports"] == {}

    try:
        asyncio.run(exercise())
    finally:
        app.dependency_overrides.clear()


def test_missing_artifact_health_does_not_leak_server_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "private" / "missing-artifact"
    monkeypatch.setenv("JOB_INTELLIGENCE_ARTIFACT_DIR", str(missing))
    clear_service_cache()

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 503
            body = response.json()
            assert body["reason_code"] == "artifact_missing"
            assert str(missing) not in response.text
            search = await client.post("/v1/search", json={"query": "software", "k": 1})
            assert search.status_code == 503
            assert str(missing) not in search.text

    asyncio.run(exercise())


def test_dashboard_and_release_automation_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "app" / "main.py").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert '"Evidence & lineage"' in app_source
    assert '"Shared strengths"' in app_source
    assert '"Positive target − source gaps"' in app_source
    assert "options=(2, 3, 4, 5)" in app_source
    assert "USER app" in dockerfile
    assert "requirements.lock" in workflow
    assert "verify-artifact" in workflow
    assert "health?deep=true" in workflow
    assert "_stcore/health" in workflow
    assert "ReadonlyRootfs" in workflow
    assert "docker compose up --detach --no-build --wait" in workflow
    assert "locked-test" not in workflow.casefold()
    assert " train" not in workflow.casefold()
