# Serving artifact contract

Schema version 1 is a fail-closed release boundary. `manifest.json` declares
every file; undeclared files, symlinks, unsafe paths, unknown logical names, and
duplicate JSON keys are rejected.

## Required logical files

| Logical name | Format | Purpose |
|---|---|---|
| `occupations` | Parquet | Canonical profile catalog and O*NET 30.3 labels: stored 2 means combined Job Zone 1–2, followed by 3, 4, and 5 |
| `occupation_market` | Parquet | Unsummed O*NET-SOC/NEM-grain BLS context |
| `skill_ratings` | Parquet | Observed ratings, masks, N, SE, CI, date, and domain source |
| `clusters` | Parquet | Frozen cluster assignment and PCA display coordinates |
| `neighbors` | Parquet | Complete stored top-250 transition candidates per occupation |
| `search_catalog` | Parquet | Zero-based row order for the search matrix |
| `title_vectorizer` | Joblib | Fitted, offline free-text transformer |
| `title_matrix` | SciPy NPZ | Canonical float64 CSR matrix with unit-normalized rows |
| `smoke_queries` | JSON | Arbitrary target-free query text with no expected labels |

Every Parquet entry binds bytes, SHA-256, row count, exact ordered Arrow fields,
primary key, and sort key. Matrix entries additionally bind shape and dtype. The
manifest binds the literal project-config and environment-lock digests, source
and split manifests, locked ranking and metric digests, source-object hashes,
selected candidate IDs and exact specs, non-test fit-code/fold digests, build
commit, transition components, and display/storage limits.

A locked release additionally carries exact-byte, checksum-declared safe
lineage under `lineage/`:

| Logical name | Format | Independent binding |
|---|---|---|
| `locked_metrics` | JSON | Aggregate locked evidence and rankings/decision hashes |
| `locked_rankings` | Parquet | Target-free selected/baseline query IDs, ranks, and scores |
| `development_decision` | JSON | Self-hashed selection/confirmation releases and input digests |
| `fold_manifest` | CSV | Exact grouped role assignment for the serving catalog |
| `fold_digest` | JSON | Semantic hash of the canonical fold CSV |
| `processed_manifest` | JSON | Processed table/source/config provenance (not table bytes) |
| `feature_manifest` | JSON | Frozen permitted feature policy and structured keys |
| `metric_contract` | JSON | Preregistered metrics, bootstrap, selection, and confirmation policy |
| `candidate_manifest` | JSON | Complete exact title/transition candidate grids |
| `source_manifest` | JSON | Tracked public official-source manifest and object hashes |
| `evaluation_universe` | JSON | Target-free role-local title query IDs and transition query codes |
| `implementation_manifest` | JSON | Self-hashed implementation-file inventory fixed before scoring |

The bundle never contains qrel table bytes, relevance tiers, or grades. A source
or processed manifest may truthfully name the official evaluation source/table;
that provenance is not a serving feature or a copy of its judgments.

## Validation order

The loader performs these stages in order:

1. parse a unique-key, finite-number JSON manifest with exact fields;
2. resolve canonical relative paths and reject symlinks or undeclared files;
3. verify every byte count and SHA-256;
4. verify the checked-in `configs/project.toml` and `requirements.lock` bytes;
5. read exact Parquet schemas and enforce keys, sorts, values, foreign keys,
   catalog alignment, unsummed shared NEM values, and full transition depth;
6. validate target-free smoke-query and sparse-matrix contracts;
7. before Joblib deserialization, parse all locked safe-lineage files; rehash the
   exact rankings, metrics, decision, source, processed, feature, metric,
   candidate, universe, and implementation bytes; recompute the fold and
   implementation semantic hashes; and verify every decision/provenance link;
8. validate the target-free locked-ranking schema/query universes, require the
   evidence wrappers to equal the benchmark payloads in `locked_metrics`, and
   bind the released model IDs/specs and non-test fit-code digest;
9. recompute every transition total score and rank from the stored component
   columns using the selected missing-aware weights;
10. only then deserialize the checksum-verified Joblib transformer, reject
    evaluation-field metadata, and verify its exact class/analyzer/ngrams,
    minimum document frequency, LSA dimension/seed, and hybrid weights against
    the selected candidate and literal configuration;
11. validate dimensions and run every target-free smoke query against the
    catalog.

Joblib is pickle-based, so the manifest must come from the trusted release
process. Checksums detect drift; they are not a signature for an untrusted
third-party bundle.

## Production publication

`write_artifact_bundle(...)` is the sole production writer. It accepts the six
exact serving tables, fitted title transformer and aligned matrix, target-free
smoke queries, frozen model identifiers, release provenance, and either no
evidence (pre-lock) or all five complete locked report wrappers plus all twelve
exact safe-lineage paths. It derives the selected specs, locked-test fit-code
binding, source pins, and lineage digests rather than trusting caller
assertions. It computes file sizes, hashes, schemas, keys, sorts, matrix
metadata, and the literal configuration hash; callers cannot substitute those
derived values.

The writer creates a same-filesystem staging directory under an exclusive
version publication lock, runs the complete loader—including report checks and
smoke inference—against that stage, and only then renames the directory into
place. An existing version is never overwritten, and a failed validation
removes the stage without exposing a partial manifest. The release runner must
therefore select a new artifact version rather than mutate `artifacts/v1`.
Before publication, directories are normalized to mode `0755` and files to
`0644`, so the immediately published local bundle is readable by the
unprivileged container user when bind-mounted read-only.

## Locked evidence

A pre-release artifact uses `evidence.status = "not_generated"` and contains no
report files. A release uses `"locked"` and must checksum-bind all five reports:

- alternate-title benchmark, including top-1, Recall@5, MRR, paired baseline
  deltas, query counts, and confidence intervals;
- adjacent-occupation benchmark, including graded nDCG@10, all preregistered
  supporting metrics, the binary-Primary nDCG@10 sensitivity analysis, paired
  deltas, counts, and intervals;
- selected cluster stability, silhouette, k, and preregistered seeds;
- serving-table data-coverage counts/rates;
- a non-empty limitations list.

Benchmark reports must identify the exact selected serving candidates and the
`locked_test` split. Paired deltas compare only with the preregistered lexical
baselines (`title_word_char` and `transition_title_word_char`). They are required
when the selected candidate differs from its baseline and must be empty when the
confirmation fallback selects that baseline; a fabricated self-delta is never
valid. Context reports use `release_context`. Development or synthetic evidence
cannot satisfy this contract.

Each benchmark wrapper also checksum-binds its descriptive primary-metric
slices by SOC major group, official Job Zone, profile completeness, and rating
recency. Slice rows bind the estimate, interval, analysis-unit count, and the
predeclared `n_units < 20` descriptive-only flag. Every dimension must cover the
same analysis units; the slice rows are evidence, not an unbound side report.

Pure structured similarity remains a documented ablation, but it is not
release-eligible. The source audit has usable structured ratings for 894 of
1,016 occupations; that occupation-level count does not imply that every pair
has structured evidence. Pair-level structured cosine uses observed skill plus
knowledge ratings and is unavailable unless transferable-skill development
distance is also defined. Hybrids renormalize only over finite,
positively-weighted components and never impute. A releasable transition
candidate must therefore give positive weight to an always-finite lexical or
LSA text component and produce the exact stored depth—`min(250, catalog size -
1)`—for every query.

Related-occupation qrels, tiers, relevance grades, expected smoke targets,
judgment fields, BLS-derived retrieval inputs, and arbitrary extra files are not
permitted in the serving bundle. Truthful source provenance may still name the
official O*NET Related Occupations file.
