# Reproducibility and release sequence

The release is intentionally staged so final evidence cannot influence model
choice.

1. Install the Python 3.12 lock, acquire the exact official snapshot, then build
   and audit the deterministic processed tables.
2. Finalize and **commit the implementation**: configuration, environment lock,
   schemas, feature code, candidate grids, metrics, selection protocol, and
   reporting code. `experiment-prepare` requires this clean commit.
3. Run `experiment-prepare`. It freezes the fold, implementation, candidate,
   feature, metric, and target-free evaluation-universe contracts, but computes
   no candidate scores. Review and **commit those prepared contracts**.
4. From that clean commit, run `experiment-select` exactly once. Review and
   **commit the selection outputs**; do not modify the prepared contracts.
5. From the new clean commit, run `experiment-confirm` exactly once. It applies
   the held-out confirmation gate and freezes the multi-task development
   decision. Review and **commit that decision and its bound outputs**.
6. From the resulting clean commit, run `locked-rank` once, then
   `locked-report`. Reporting reads only the immutable, checksum-recorded locked
   rankings; it does not refit or rerank.
7. Without changing the locked code or inputs, run `package-release`. It replays
   the selected locked rankings with the production-fitted objects, validates
   all exact lineage cross-links, and atomically publishes the no-overwrite
   serving bundle.
8. Validate installed-package inference, API/dashboard flows, and non-root
   container operation. Then commit the release evidence and artifact, push the
   exact SHA, and require green remote CI.

Raw inputs and processed build files are local and ignored. Public release
assets are derived, attributed, compact, and independently integrity checked.
An acquisition refresh or any change to features, judgments, candidates,
selection, or metrics creates a new experiment version; it never overwrites v1.

The serving and quality commands are now fixed:

```bash
make install
make data-plan      # offline: print the exact official request plan
make data-download  # explicit network acquisition or pinned-byte verification
make data-build     # offline deterministic DuckDB/Parquet snapshot
make data-audit     # offline checksum/schema/provenance/quality audit
make quality
make experiment-prepare  # freeze folds, grids, features, metrics, and query universes
make experiment-select   # run the preregistered development selection stage
make experiment-confirm  # run the held-out confirmation gate exactly once
make locked-rank          # create immutable target-free locked rankings
make locked-report        # aggregate only the already-written locked rankings
make package-release  # offline post-lock refit and immutable publication
make artifact-gate
make verify-artifact  # mandatory once artifacts/v1/manifest.json exists
```

`make artifact-gate` accepts absence only while the repository explicitly has no
v1 manifest. Once present, the complete checksum/schema/relationship/model/report
validation and target-free smoke suite must pass. `make package-release` consumes
`data/processed/snapshots/v1`, `reports/experiments/v1`,
`reports/locked/v1`, and the tracked public `reports/source_manifest.json`; it
writes the no-overwrite `artifacts/v1` bundle. These evidence, source, and output
paths are canonical and enforced by the commands. Acquisition, candidate
selection, and the one-shot locked evaluation remain separate release commands.
Run the five experiment/locked targets above in order at the stated commit
boundaries. CI intentionally invokes none of these stateful commands. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the offline API/dashboard boundary.
