# AI Job Market Intelligence

> Evaluated U.S. occupation search and transparent career-transition frontiers
> using O*NET® 30.3 and BLS 2024–34 national projections.

**Status: v2 experiment contract locked; release results are not generated yet.**
No sample, synthetic, or development metric will be presented as a final result.

Most career tools collapse similarity, opportunity, and pay into one unexplained
score. This project keeps them separate: reproducible information-retrieval
experiments identify related work, O*NET ratings explain occupation-level
development differences, BLS projections add national context, and Pareto
frontiers expose trade-offs without pretending to predict an individual's
career outcome.

## What this is—and is not

This is an aggregate occupation-intelligence demonstrator for career services,
workforce planning, and personal exploration. It supports:

- free-text occupation discovery;
- evidence-backed adjacent-occupation retrieval;
- occupation-level transferable-skill comparisons;
- national wage, projected growth, annual openings, and entry-requirement views;
- transparent multi-objective filtering and Pareto frontiers.

It does **not** rank people, screen applicants, estimate employability or hiring
probability, ingest CVs, report live vacancies, or establish that a person is
qualified for a regulated occupation. Similarity is not a match percentage.

## Evaluation contract

The preregistered protocol is in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). Two tasks are kept
separate:

1. held-out alternate-title discovery, evaluated with top-1 accuracy, Recall@5,
   and reciprocal rank;
2. adjacent-occupation retrieval, evaluated by agreement with O*NET's
   expert-reviewed Related Occupations lists using macro nDCG@10 and supporting
   retrieval metrics.

Related-occupation links are evaluation judgments only and are prohibited from
all searchable features. BLS outcomes are decision context, never model labels
or retrieval features. The locked test is inspected once after the candidate
grid, split manifest, metrics, and reporting code are committed.

## Reproducibility boundary

Raw official downloads and build intermediates stay out of Git. The public v2
release will contain a validated, checksum-addressed serving bundle, locked
rankings and metrics, model/data cards, and a real dashboard capture. The app,
API, and CLI will share the same immutable service artifact and will not fit or
download models at request time.

Source data attribution and modification disclosures are in
[`NOTICE.md`](NOTICE.md). Software is MIT licensed.

## Serving the frozen release

All three interfaces use one checksum-validated, immutable artifact. Nothing is
downloaded or fitted at request time, and a missing artifact produces an
explicit unavailable state rather than sample data.

```bash
make install
make data-plan      # inspect the acquisition plan without network or writes
make data-download  # the only network-capable Make target
make data-build
make data-audit
make quality
make experiment-prepare
make experiment-select
make experiment-confirm
make locked-rank
make locked-report
make package-release  # only after the one-shot locked report is complete
make verify-artifact
make api      # FastAPI on 127.0.0.1:8000
make app      # Streamlit on 127.0.0.1:8501
```

These stateful release stages use the tracked, public
`reports/source_manifest.v2.json` as their canonical source-provenance input and
must be run in the displayed order. The v1 source manifest remains unchanged at
`reports/source_manifest.json`. CI validates code and committed release artifacts
but never starts an experiment or locked run.

The CLI includes `verify-artifact`, search, occupation, neighbors, skill
comparison, frontier, cluster, and evidence views. The search API accepts strict
JSON via `POST /v1/search`; the health endpoint is `GET /health?deep=true`.
Container hardening and artifact-mount instructions are documented in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). The exact file, schema, atomic
publication, evidence, and deserialization boundary is documented in
[`docs/ARTIFACT_CONTRACT.md`](docs/ARTIFACT_CONTRACT.md).
