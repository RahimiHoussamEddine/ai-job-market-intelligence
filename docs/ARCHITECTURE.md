# Architecture and trust boundaries

The repository has four deliberately separate boundaries. This separation is a
statistical control as much as a software design choice.

```text
official bytes
    │  acquisition + checksum/schema validation
    ▼
deterministic snapshot ──► evaluation-only qrels
    │                         │
    │ permitted features      │ metrics only
    ▼                         ▼
fit/select/freeze ───────► locked rankings + reports
    │
    │ target-free derived assets
    ▼
validated v4 bundle ─────► shared service ─────► CLI / API / dashboard
```

## 1. Official-source boundary

Acquisition is the only component allowed to access the network. It downloads
the configured O*NET 30.3 CSVs and BLS 2024–34 workbooks into ignored immutable
storage, records HTTP and cryptographic provenance, and validates known schemas
before transformation. Build, experiment, application, and API commands are
offline by default.

The deterministic snapshot retains source-grain keys. In particular, the
O*NET/NEM bridge is many-to-many and BLS quantities are never multiplied and
summed across detailed O*NET specialties.

## 2. Evaluation boundary

Related Occupations is loaded through an evaluation-only interface. Neither its
tier/index nor any graph derivative can enter profile text, structured ratings,
retrieval artifacts, clustering, or serving. The split and feature manifests
are checksum-addressed, and candidate selection cannot see locked-test query
judgments.

Development selection, the one-shot confirmation gate, and locked test execution
are distinct commands/states. Locked row-level rankings are written atomically
and hashed before aggregate reporting so a reporting defect cannot silently
cause a second model evaluation.

## 3. Artifact boundary

The public serving bundle contains only occupation profiles, unsummed market
context, observed skill ratings, a fitted free-text search index, precomputed
transition neighbors with component explanations, stable clusters, and
target-free smoke queries. Job Zones retain the official O*NET 30.3 encoding
without recoding: stored value 2 represents the redesigned combined Job Zone
1–2 category (very little to some preparation), followed by 3, 4, and 5. The
bundle contains no relevance judgments, expected search targets, personal data,
or query logs.

The manifest binds every file's digest, bytes, ordered schema/dtype, row count,
sort/key contract, exact selected-candidate specs and fit-code digest,
source/config/environment/split hashes, build commit, and evidence reports. A
locked bundle includes exact safe copies of the target-free rankings, metrics,
decision, fold/contracts, public source/processed manifests, evaluation
universe, and implementation manifest. The loader rehashes and cross-links them
before deserialization, then recomputes transition scores/ranks and enforces
relational, numerical, and fitted-transformer semantic invariants.
The complete schema and validation order are in
[`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md).

## 4. Product boundary

`OccupationIntelligenceService` is the only application-facing interface. The
CLI, FastAPI application, and Streamlit dashboard share one immutable instance;
none may fit models, download data, mutate artifacts, or log queries. Search
scores remain technical ranking evidence—not percentages or outcomes.

Deployment runs as an unprivileged user with a read-only offline artifact. A
missing or invalid bundle yields a clear degraded health/setup state rather than
falling back to sample data.
