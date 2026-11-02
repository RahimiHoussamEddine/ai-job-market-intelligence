# Model card — v4

The release bundle's `reports/title_benchmark.json` and
`reports/transition_benchmark.json` contain locked estimates, uncertainty
intervals, paired baseline deltas, and descriptive slices. The committed source
reports are versioned under `reports/locked/v4/`.

The machine-readable reports are the authoritative numerical evidence. This
card documents their claim boundaries and methodology; it does not duplicate
metric values that could drift from the checksum-bound release report.

Locked benchmark evidence will bind descriptive primary-metric slices for SOC
major group, Job Zone, profile completeness, and rating recency. Any slice with
fewer than 20 analysis units is labelled descriptive; no unbound slice file or
small-slice claim is treated as confirmatory evidence.

## Pre-metric release feasibility

The source audit finds usable structured ratings for 894 of 1,016 occupations.
That is occupation-level coverage, not pair-level availability. Structured
cosine uses observed skill plus knowledge ratings, but a pair is treated as
unavailable unless its transferable-skill development distance is also defined.
Hybrids renormalize over finite positively-weighted components and never impute.
The pure structured candidate is therefore an ablation only and cannot be
selected for deployment. Every release-eligible transition candidate must give
positive weight to an always-finite lexical or LSA text component and must
materialize the complete stored candidate depth, `min(250, catalog size - 1)`,
for every occupation.

This is an operational feasibility rule fixed before development or locked-test
metrics are read; it is not a performance claim.

## Frozen selection and production policy

Each task applies the preregistered one-standard-error rule independently and
confirms its proposal against its own lexical baseline: `title_word_char` for
alternate-title discovery and `transition_title_word_char` for transitions. If
more than one release-eligible candidate remains within the one-SE threshold,
the deterministic simplicity order is: fewer latent dimensions
(`lsa_dimensions`, with non-LSA candidates recorded as zero), fewer serialized
artifact bytes, lower operational cost, then candidate ID. Operational cost is
defined—not timed—as
`score_cells * max(1, positive_weight_component_count) + lsa_dimensions * query_count`.
Measured latency is retained only as a diagnostic and never breaks a tie.

If the selected transition representation has no LSA component, exploratory
clustering uses the preregistered 128-dimensional work-profile LSA fallback. Its
vocabulary, IDF, and SVD are fit only on non-test roles before transforming the
full catalog, exactly like the production retrieval representations.

See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) for the frozen design.
