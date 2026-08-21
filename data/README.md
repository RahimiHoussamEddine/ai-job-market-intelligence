# Data boundary

The repository does not track raw official downloads or deterministic build
intermediates. Run the acquisition and snapshot scripts to populate `data/raw/`
and `data/processed/`; both directories are ignored except for `.gitkeep`.

The v1 source contract was frozen on 2026-08-14; the delayed, actual retrieval
date is 2026-08-21 and is recorded independently from each response's UTC
timestamp in the public source manifest.

The public release may redistribute audited, derived O*NET profiles and vectors
under CC BY 4.0 with the attribution and modification disclosure in
[`../NOTICE.md`](../NOTICE.md). BLS inputs are U.S. federal public-domain
materials, but are still cited and versioned.

Acquisition must record requested/resolved URLs, UTC retrieval time, source
vintage, HTTP metadata, byte count, SHA-256, schema fingerprint, row count, key
coverage, parser version, and repository commit. The first reviewed snapshot
digest becomes this project's pin; it is not represented as a publisher-issued
checksum.
