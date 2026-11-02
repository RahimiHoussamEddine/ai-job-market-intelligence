"""Frozen grouped occupation splits and their deterministic manifest.

The split is a data contract, not a convenience train/test helper.  Every
O*NET specialty sharing a base SOC code is assigned together, and the emitted
manifest carries enough metadata to make a later digest meaningful.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_MANIFEST_VERSION: Final = 1
SPLIT_METHOD: Final = "StratifiedGroupKFold"
DEFAULT_ROLE_FOLDS: Final[Mapping[str, tuple[int, ...]]] = {
    "discovery": (3, 4),
    "selection": (2,),
    "confirmation": (1,),
    "test": (0,),
}
MANIFEST_COLUMNS: Final = (
    "onet_soc_code",
    "base_soc_code",
    "soc_major_group",
    "fold",
    "role",
    "split_method",
    "n_splits",
    "random_seed",
)


class SplitManifestError(ValueError):
    """Raised when a split would violate the preregistered grouping contract."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SplitManifestError(f"{context} is missing columns: {missing}")


def _normalized_role_folds(
    role_folds: Mapping[str, Iterable[int]], n_splits: int
) -> dict[str, tuple[int, ...]]:
    expected_roles = {"discovery", "selection", "confirmation", "test"}
    if set(role_folds) != expected_roles:
        raise SplitManifestError(
            "role_folds must contain exactly discovery, selection, confirmation, and test"
        )
    normalized = {role: tuple(int(fold) for fold in folds) for role, folds in role_folds.items()}
    flattened = [fold for folds in normalized.values() for fold in folds]
    if sorted(flattened) != list(range(n_splits)) or len(flattened) != len(set(flattened)):
        raise SplitManifestError("role_folds must assign every configured fold exactly once")
    if any(not folds for folds in normalized.values()):
        raise SplitManifestError("every split role must contain at least one fold")
    return normalized


def build_fold_manifest(
    occupations: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_seed: int = 20260814,
    role_folds: Mapping[str, Iterable[int]] = DEFAULT_ROLE_FOLDS,
) -> pd.DataFrame:
    """Assign occupation codes using the exact preregistered grouped splitter.

    The input is sorted before scikit-learn sees it, removing source row order
    as a source of split drift.  Group labels must map to exactly one SOC major
    group; silently choosing a stratum for an inconsistent group would destroy
    the stated stratification unit.
    """

    if n_splits < 2:
        raise SplitManifestError("n_splits must be at least two")
    roles = _normalized_role_folds(role_folds, n_splits)
    _require_columns(
        occupations,
        ("onet_soc_code", "base_soc_code", "soc_major_group"),
        "occupations",
    )
    source = occupations.loc[:, ["onet_soc_code", "base_soc_code", "soc_major_group"]].copy()
    if source.empty:
        raise SplitManifestError("occupations must be nonempty")
    for column in source.columns:
        source[column] = source[column].astype("string").str.strip()
        if bool(source[column].isna().any() | source[column].eq("").any()):
            raise SplitManifestError(f"occupations.{column} contains missing/blank values")
    if bool(source["onet_soc_code"].duplicated().any()):
        raise SplitManifestError("occupations.onet_soc_code must be unique")
    group_strata = source.groupby("base_soc_code", sort=False)["soc_major_group"].nunique()
    inconsistent = group_strata[group_strata.ne(1)]
    if not inconsistent.empty:
        raise SplitManifestError(
            "base SOC groups span multiple SOC major groups: "
            + ", ".join(inconsistent.index.astype(str)[:5])
        )
    if source["base_soc_code"].nunique() < n_splits:
        raise SplitManifestError("fewer base SOC groups than configured folds")

    source = source.sort_values(
        ["onet_soc_code", "base_soc_code", "soc_major_group"], kind="stable"
    ).reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )
    assignments = np.full(len(source), -1, dtype=np.int8)
    placeholder = np.zeros((len(source), 1), dtype=np.uint8)
    try:
        for fold, (_, validation_indexes) in enumerate(
            splitter.split(
                placeholder,
                y=source["soc_major_group"].to_numpy(),
                groups=source["base_soc_code"].to_numpy(),
            )
        ):
            if np.any(assignments[validation_indexes] != -1):
                raise SplitManifestError("splitter assigned an occupation more than once")
            assignments[validation_indexes] = fold
    except ValueError as exc:
        raise SplitManifestError(f"grouped split could not be constructed: {exc}") from exc
    if np.any(assignments < 0):
        raise SplitManifestError("splitter did not assign every occupation")

    fold_to_role = {fold: role for role, folds in roles.items() for fold in folds}
    result = source.assign(
        fold=assignments.astype(np.int64),
        role=[fold_to_role[int(fold)] for fold in assignments],
        split_method=SPLIT_METHOD,
        n_splits=int(n_splits),
        random_seed=int(random_seed),
    )
    result = result.loc[:, list(MANIFEST_COLUMNS)]
    assert_fold_manifest(
        result,
        expected_codes=source["onet_soc_code"],
        role_folds=roles,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    return result


def assert_fold_manifest(
    manifest: pd.DataFrame,
    *,
    expected_codes: Iterable[object] | None = None,
    aliases: pd.DataFrame | None = None,
    role_folds: Mapping[str, Iterable[int]] = DEFAULT_ROLE_FOLDS,
    n_splits: int = 5,
    random_seed: int = 20260814,
) -> None:
    """Assert grouping, role, coverage, and optional alias invariants."""

    roles = _normalized_role_folds(role_folds, n_splits)
    _require_columns(manifest, MANIFEST_COLUMNS, "fold manifest")
    if list(manifest.columns) != list(MANIFEST_COLUMNS):
        raise SplitManifestError(f"fold manifest column order must be {MANIFEST_COLUMNS!r}")
    if manifest.empty or bool(manifest["onet_soc_code"].duplicated().any()):
        raise SplitManifestError("fold manifest must be nonempty and unique by occupation code")
    if manifest["onet_soc_code"].tolist() != sorted(manifest["onet_soc_code"].tolist()):
        raise SplitManifestError("fold manifest must be sorted by occupation code")
    if set(manifest["fold"].astype(int)) != set(range(n_splits)):
        raise SplitManifestError("fold manifest does not contain every configured fold")
    if set(manifest["n_splits"].astype(int)) != {n_splits}:
        raise SplitManifestError("fold manifest n_splits metadata differs from the contract")
    if set(manifest["random_seed"].astype(int)) != {random_seed}:
        raise SplitManifestError("fold manifest random_seed metadata differs from the contract")
    if set(manifest["split_method"].astype(str)) != {SPLIT_METHOD}:
        raise SplitManifestError("fold manifest split_method metadata differs from the contract")
    fold_to_role = {fold: role for role, folds in roles.items() for fold in folds}
    observed_pairs = set(
        zip(manifest["fold"].astype(int), manifest["role"].astype(str), strict=True)
    )
    expected_pairs = {(fold, role) for fold, role in fold_to_role.items()}
    if observed_pairs != expected_pairs:
        raise SplitManifestError("fold-to-role mapping differs from the frozen contract")
    group_fold_counts = manifest.groupby("base_soc_code", sort=False)["fold"].nunique()
    if bool(group_fold_counts.ne(1).any()):
        raise SplitManifestError("a base SOC group appears in more than one fold")
    group_strata = manifest.groupby("base_soc_code", sort=False)["soc_major_group"].nunique()
    if bool(group_strata.ne(1).any()):
        raise SplitManifestError("a base SOC group spans more than one SOC major group")

    if expected_codes is not None:
        expected = {str(value).strip() for value in expected_codes}
        observed = set(manifest["onet_soc_code"].astype(str))
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            raise SplitManifestError(
                f"fold manifest code coverage differs; missing={missing}, extra={extra}"
            )

    if aliases is not None:
        _require_columns(aliases, ("onet_soc_code",), "aliases")
        alias_codes = set(aliases["onet_soc_code"].dropna().astype(str))
        unknown = alias_codes - set(manifest["onet_soc_code"].astype(str))
        if unknown:
            raise SplitManifestError(
                "aliases reference occupation codes absent from the fold manifest: "
                + ", ".join(sorted(unknown)[:5])
            )
        if "fold" in aliases.columns:
            expected_folds = manifest.set_index("onet_soc_code")["fold"].astype(int)
            joined = aliases.loc[:, ["onet_soc_code", "fold"]].copy()
            joined["expected_fold"] = joined["onet_soc_code"].map(expected_folds)
            mismatch = joined["fold"].astype(int).ne(joined["expected_fold"])
            if bool(mismatch.any()):
                raise SplitManifestError("an alias carries a fold different from its occupation")


def fold_manifest_digest(manifest: pd.DataFrame) -> str:
    """Return a semantic SHA-256 over the exact ordered manifest contract."""

    # Infer metadata here so callers cannot hash a malformed manifest and treat
    # the resulting digest as approval.  Multiple metadata values are invalid.
    if manifest.empty:
        raise SplitManifestError("cannot hash an empty fold manifest")
    n_values = set(manifest.get("n_splits", pd.Series(dtype=int)).astype(int))
    seed_values = set(manifest.get("random_seed", pd.Series(dtype=int)).astype(int))
    if len(n_values) != 1 or len(seed_values) != 1:
        raise SplitManifestError("fold manifest metadata must be constant before hashing")
    n_splits = next(iter(n_values))
    random_seed = next(iter(seed_values))
    assert_fold_manifest(manifest, n_splits=n_splits, random_seed=random_seed)
    rows = [
        {
            "onet_soc_code": str(row.onet_soc_code),
            "base_soc_code": str(row.base_soc_code),
            "soc_major_group": str(row.soc_major_group),
            "fold": int(row.fold),
            "role": str(row.role),
            "split_method": str(row.split_method),
            "n_splits": int(row.n_splits),
            "random_seed": int(row.random_seed),
        }
        for row in manifest.itertuples(index=False)
    ]
    payload = json.dumps(
        {"manifest_version": SPLIT_MANIFEST_VERSION, "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_fold_manifest(
    manifest: pd.DataFrame,
    path: str | Path,
    *,
    digest_path: str | Path | None = None,
) -> str:
    """Atomically publish the immutable CSV and optional semantic-hash sidecar.

    Repeating an identical interrupted publication is idempotent.  Existing
    bytes that differ are never overwritten; that requires a new experiment
    version.
    """

    digest = fold_manifest_digest(manifest)
    destination = Path(path)
    csv_bytes = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _publish_immutable_bytes(destination, csv_bytes)
    if digest_path is not None:
        checksum = Path(digest_path)
        checksum_payload = (
            json.dumps(
                {
                    "manifest_version": SPLIT_MANIFEST_VERSION,
                    "path": destination.name,
                    "semantic_sha256": digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _publish_immutable_bytes(checksum, checksum_payload)
    return digest


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise SplitManifestError(
                f"refusing to overwrite different locked split bytes: {path.name}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard link publishes the complete inode and has
            # O_EXCL-like no-overwrite behavior if another writer won the race.
            os.link(temporary_name, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise SplitManifestError(
                    f"concurrent writer published different split bytes: {path.name}"
                ) from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def codes_for_roles(manifest: pd.DataFrame, roles: Iterable[str]) -> tuple[str, ...]:
    """Return sorted occupation codes assigned to the requested split roles."""

    requested = set(roles)
    unknown = requested - {"discovery", "selection", "confirmation", "test"}
    if unknown:
        raise SplitManifestError(f"unknown split roles: {sorted(unknown)}")
    return tuple(
        manifest.loc[manifest["role"].isin(requested), "onet_soc_code"]
        .astype(str)
        .sort_values(kind="stable")
    )


__all__ = [
    "DEFAULT_ROLE_FOLDS",
    "MANIFEST_COLUMNS",
    "SPLIT_MANIFEST_VERSION",
    "SPLIT_METHOD",
    "SplitManifestError",
    "assert_fold_manifest",
    "build_fold_manifest",
    "codes_for_roles",
    "fold_manifest_digest",
    "write_fold_manifest",
]
