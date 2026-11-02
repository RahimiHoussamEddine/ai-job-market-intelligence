from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from job_market_intelligence.splits import (
    SplitManifestError,
    assert_fold_manifest,
    build_fold_manifest,
    fold_manifest_digest,
    write_fold_manifest,
)


def _occupations() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for major in ("11", "15"):
        for group in range(10):
            base = f"{major}-{1000 + group:04d}"
            for specialty in range(2):
                rows.append(
                    {
                        "onet_soc_code": f"{base}.{specialty:02d}",
                        "base_soc_code": base,
                        "soc_major_group": major,
                    }
                )
    return pd.DataFrame(rows)


def test_grouped_fold_manifest_is_deterministic_and_keeps_siblings_together() -> None:
    occupations = _occupations()
    first = build_fold_manifest(occupations.sample(frac=1.0, random_state=7))
    second = build_fold_manifest(occupations.sample(frac=1.0, random_state=13))

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("base_soc_code")["fold"].nunique().eq(1).all()
    assert set(first["fold"]) == set(range(5))
    assert fold_manifest_digest(first) == fold_manifest_digest(second)

    aliases = pd.DataFrame(
        {
            "onet_soc_code": first["onet_soc_code"],
            "fold": first["fold"],
        }
    )
    assert_fold_manifest(first, expected_codes=occupations["onet_soc_code"], aliases=aliases)
    aliases.loc[0, "fold"] = (int(aliases.loc[0, "fold"]) + 1) % 5
    with pytest.raises(SplitManifestError, match="alias carries"):
        assert_fold_manifest(first, aliases=aliases)


def test_fold_manifest_publication_is_immutable_and_interruption_safe(tmp_path: Path) -> None:
    first = build_fold_manifest(_occupations(), random_seed=20260814)
    csv_path = tmp_path / "folds.csv"
    digest_path = tmp_path / "folds.sha256.json"

    digest = write_fold_manifest(first, csv_path, digest_path=digest_path)
    assert write_fold_manifest(first, csv_path, digest_path=digest_path) == digest
    sidecar = json.loads(digest_path.read_text(encoding="utf-8"))
    assert sidecar == {
        "manifest_version": 1,
        "path": "folds.csv",
        "semantic_sha256": digest,
    }

    different = build_fold_manifest(_occupations(), random_seed=20260815)
    with pytest.raises(SplitManifestError, match="refusing to overwrite"):
        write_fold_manifest(different, csv_path, digest_path=digest_path)

    # If publication was interrupted after the CSV link, retry may safely add
    # the identical missing sidecar without touching the manifest inode.
    standalone = tmp_path / "standalone.csv"
    write_fold_manifest(first, standalone)
    recovered_sidecar = tmp_path / "standalone.sha256.json"
    write_fold_manifest(first, standalone, digest_path=recovered_sidecar)
    assert recovered_sidecar.is_file()


def test_manifest_rejects_a_group_split_across_folds() -> None:
    manifest = build_fold_manifest(_occupations())
    sibling_indexes = manifest.index[manifest["base_soc_code"].eq(manifest.loc[0, "base_soc_code"])]
    new_fold = (int(manifest.loc[sibling_indexes[0], "fold"]) + 1) % 5
    manifest.loc[sibling_indexes[0], "fold"] = new_fold
    manifest.loc[sibling_indexes[0], "role"] = {
        0: "test",
        1: "confirmation",
        2: "selection",
        3: "discovery",
        4: "discovery",
    }[new_fold]

    with pytest.raises(SplitManifestError, match="base SOC group"):
        assert_fold_manifest(manifest)
