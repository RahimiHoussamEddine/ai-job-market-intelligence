# Preregistered experiment protocol

Protocol version: 2.0
Locked on: 2026-11-02, before v2 selection and locked evaluation

## Decision and claim boundary

The product supports aggregate U.S. occupation exploration. It answers which
occupations are textually or structurally related, which occupation-level skill
ratings differ, and how destinations compare on national BLS projections and
typical requirements. It does not estimate real worker mobility, individual
fit, employability, qualification, hiring probability, local demand, or active
vacancies.

The primary NLP claim is deliberately narrow: a frozen retrieval system can be
measured for **agreement with O*NET's expert-reviewed related-occupation list**.
Those judgments were partly developed from task, knowledge, and title
similarity, so they are a useful but circular proxy—not independent semantic
truth. Unlisted occupations are unjudged rather than proven irrelevant.

## Frozen data contract

- O*NET® Database 30.3, May 2026: occupations, alternate titles, task
  statements, transferable skills, knowledge, job zones, and Related
  Occupations.
- BLS Employment Projections 2024–2034 occupational table, released
  2025-08-28.
- Official BLS NEM/O*NET-SOC crosswalk.
- Acquisition date: 2026-08-21. Each response, archive member, and deterministic
  derived table receives a byte size and SHA-256 digest in the source manifest.

Raw files are immutable and ignored by Git. Acquisition rejects changed bytes
in locked mode, path traversal, unexpected content types, schema drift, and
duplicate business keys. An explicit new snapshot must use a new version rather
than overwrite v1.

Release 30.3 uses O*NET's current four-level Job Zone framework: the combined
Job Zone 1–2 category (encoded as `2` in the database), followed by Job Zones 3,
4, and 5. Validation and application filters therefore accept stored values
`2–5`; they do not recode the official values or treat them as the legacy
five-level scheme. See O*NET's official
[Job Zone transition paper](https://www.onetcenter.org/dl_files/Job_Zone_Transition.pdf).

## Leakage controls and split

The query unit is an O*NET occupation. Specialty occupations are grouped by
base `XX-XXXX` SOC code and stratified by two-digit SOC major group with
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260814)`.

- folds 3 and 4: representation discovery/fitting;
- fold 2: candidate and hyperparameter selection;
- fold 1: one-shot confirmation gate;
- fold 0: locked final test.

The code-to-fold manifest is written and hashed before any metric is computed.
Alternate-title ambiguity is role-scoped for development so fold-zero aliases
cannot remove selection or confirmation queries. For locked test, ambiguity is
resolved across the full catalog first, then queries are restricted to fold 0;
this matches the global search catalog without feeding test aliases into the
indexed documents. All aliases for an occupation stay within its assigned
role. Vocabulary, IDF, SVD, scalers, and normalization rules fit only on
discovery occupations during selection.
Locked-test occupations are excluded from the selection/confirmation candidate
pool, so reciprocal relatedness edges cannot expose their judgments. After the
champion and gate are frozen, the selected representation may be refit on
discovery, selection, and confirmation inputs—never test judgments—before one
test evaluation against the full catalog.

Related Occupations data are loaded only by the evaluation boundary. Related
tier/index, reverse links, graph features, SOC hierarchy, and BLS outcomes are
forbidden retrieval features. Automated tests inspect feature schemas and
artifacts for these fields.

## Task A: alternate-title discovery

Each query is a held-out normalized O*NET alternate title; the relevant result
is its canonical occupation. Exact normalized duplicates are deduplicated,
ambiguous aliases mapping to multiple occupations are removed, and at most 25
stable-hash-selected aliases per occupation are retained. Test aliases are not
included in indexed alias text.

Primary metric: macro occupation-level top-1 accuracy. Supporting metrics:
Recall@5 and reciprocal rank. Query-weighted metrics are reported only as a
diagnostic so occupations with many aliases do not dominate selection.

Predeclared systems:

- canonical-title word/character TF-IDF;
- profile word TF-IDF;
- dense latent semantic analysis (TruncatedSVD) with 64, 128, or 256 dimensions;
- nonnegative lexical/LSA hybrids on a 0.25 weight grid.

## Task B: adjacent-occupation retrieval

The query profile contains canonical title, description, and O*NET Core tasks.
Candidate profiles contain the same permitted fields plus separately normalized
transferable-skill and knowledge ratings. The query occupation itself is always
excluded.

O*NET tiers are mapped to a documented evaluation convention:
`Primary-Short=3`, `Primary-Long=2`, and `Supplemental=1`. Primary metric is
macro query-level nDCG@10. Supporting metrics are Recall@5 for Primary-Short,
Recall@10 for all primary relations, nDCG@20, and MRR@10. Binary-primary
relevance is a preregistered sensitivity analysis.

Predeclared systems:

- title word/character TF-IDF;
- work-profile word TF-IDF;
- work-profile LSA with 64, 128, or 256 dimensions;
- missing-aware cosine over normalized transferable-skill and knowledge
  ratings;
- nonnegative interpretable hybrids over lexical, LSA, and structured
  similarities, using simplex weights in 0.25 increments.

The same-major-SOC ordering is a sanity diagnostic and is never champion
eligible.

## Model selection and uncertainty

Candidates rank by the selection fold's primary metric. Candidates within one
stratified query-bootstrap standard error of the deployment-eligible leader use
the fixed tie-break: fewer latent dimensions, smaller serialized artifact,
lower deterministic operational cost, then lexicographic candidate ID. The cost
is `score_cells * max(1, positive_weight_component_count) + lsa_dimensions *
query_count`; measured latency is diagnostic only. The pure structured ablation
is evaluated but ineligible for release because it cannot guarantee complete
catalog coverage without imputing missing values. No test result can change
these rules.

A query with no finite candidate scores has an empty ranking. Its expected
query remains in the evaluation universe and receives zero for each ranked
relevance metric; no fallback score or candidate is invented.

The frozen title champion is compared with `title_word_char` on confirmation;
the transition champion is compared with `transition_title_word_char`. If a
champion underperforms its own baseline on the task's primary metric, that
baseline becomes the release model; equality retains the champion. No retuning
occurs.
Final metrics use 2,000 paired query bootstrap replicates stratified by SOC major
group. Report paired deltas against preregistered baselines and slices by SOC
family, job zone, profile completeness, and source-rating recency where coverage
permits. Small slices are labelled descriptive.

## Clustering

Clusters are exploratory navigation, not occupational truth. K-means is fit in
the frozen dense representation for `k in {12,18,24}` with five predeclared
seeds. Selection uses pairwise adjusted-Rand stability after 80% occupation
bootstrap refits, then original-space silhouette, then smaller `k`. PCA is only
a two-dimensional display. Cluster labels are deterministic top-term summaries,
not generated claims. Adjusted mutual information against SOC major groups is
descriptive context only. If the selected transition candidate has no LSA
component, a 128-dimensional work-profile LSA is fit on non-test occupations
for clustering; its vocabulary and SVD never use locked-test documents to fit.

## BLS overlay and Pareto decisions

BLS fields remain visible and separate: 2024 employment, projected 2034
employment, projected percentage change, average annual openings for 2024–34,
2024 median annual wage, typical education, work experience, and on-the-job
training. Projections are conditional national scenarios, annual openings are
not active vacancies, and wages are not personalized salary estimates.

The official crosswalk is many-to-many. Frontier rows are computed at unique
BLS NEM occupation granularity; shared figures are never summed across O*NET
specialties. A destination is Pareto efficient if no other eligible destination
has both lower occupation-level development distance and a better selected BLS
outcome, with at least one strict improvement. No opaque opportunity score is
allowed.

## Locked release protocol

Before the test command, commit and hash source/config contracts, acquisition
manifests, the fold manifest, feature schemas, candidate grid, metrics,
selection rule, environment lock, and report schemas. The locked command must:

1. require a clean, tracked Git commit and bind code, config, data, query
   universes, and the development decision to it;
2. verify every input checksum and the frozen development decision;
3. write row-level rankings atomically before aggregate metrics;
4. hash rankings and persist the hash in all downstream provenance;
5. prohibit a second test run. A reporting-only resume is allowed only from the
   same exact clean Git commit and after it validates identical immutable
   ranking bytes;
6. build a target-free serving smoke fixture and checksum-addressed artifact;
7. label every result by split and prevent development metrics appearing as
   locked evidence.

Any change to data, features, candidates, judgments, thresholds, or metric code
requires a new experiment version. Reporting does not run from a descendant
commit: a reporting defect requires a new experiment version and a fresh locked
evaluation under its own committed protocol.
