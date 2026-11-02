# Offline serving and deployment

The API, CLI, and dashboard read the same immutable `artifacts/v4` bundle. They
never acquire data, fit a representation, or write query logs. Startup validates
every declared byte and contract before the bundle becomes available.

## Local processes

Install the exact Python 3.12 environment and verify the release artifact:

```bash
make install
make quality
make verify-artifact
```

Run the API and dashboard in separate terminals:

```bash
make api
make app
```

The API listens on `127.0.0.1:8000`; its `/health?deep=true` route reruns the
target-free serving smoke queries. The dashboard listens on `127.0.0.1:8501`.
A missing or invalid bundle produces HTTP 503 and a setup state—never fallback
data or placeholder metrics.

## Containers

`Dockerfile` installs only the exact lock plus this package and switches to the
unprivileged `app` user. Compose mounts the artifact read-only, drops Linux
capabilities, enables `no-new-privileges`, and makes the container filesystem
read-only apart from a small temporary filesystem.

```bash
make docker-build
make compose-check
docker compose up --wait
```

To mount a release bundle elsewhere, set `JOB_INTELLIGENCE_ARTIFACT_DIR` to the
host directory before `docker compose up`. The in-container path remains
`/opt/artifacts/v4`. Stop the deployment with `docker compose down`.

Runtime serving is offline by construction: the artifact contains the fitted
free-text transformer, normalized sparse catalog matrix, precomputed top-250
transition candidates, unsummed BLS contexts, skill precision metadata,
clusters, evidence reports, and target-free smoke queries. No endpoint performs
network acquisition or model fitting.

## Release checks

CI installs `requirements.lock`, checks the installed dependency graph, runs
Ruff and the full test suite, validates an included release artifact (or proves
its explicit pre-release absence), validates Compose, builds the image, and
asserts that its configured user is not root. CI has no acquisition, experiment,
training, or locked-test command.
