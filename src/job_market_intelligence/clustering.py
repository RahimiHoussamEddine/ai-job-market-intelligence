"""Exploratory, stability-selected occupation clusters with PCA display only."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

PREDECLARED_K_VALUES: Final = (12, 18, 24)
PREDECLARED_SEEDS: Final = (20260814, 20260815, 20260816, 20260817, 20260818)


class ClusteringError(ValueError):
    """Raised when exploratory clustering cannot follow the frozen design."""


@dataclass(frozen=True, slots=True)
class ClusterResult:
    selected_k: int
    selected_seed: int
    assignments: pd.DataFrame
    diagnostics: pd.DataFrame
    stability_metric: str = "mean_pairwise_adjusted_rand"


def _dense_features(features: np.ndarray | sparse.spmatrix) -> np.ndarray:
    values = features.toarray() if sparse.issparse(features) else np.asarray(features)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ClusteringError("cluster features must have at least 3 rows and 2 columns")
    if not np.isfinite(values).all():
        raise ClusteringError("cluster features contain non-finite values")
    return values


def _bootstrap_labels(
    features: np.ndarray,
    *,
    k: int,
    seed: int,
    fraction: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_size = max(k, int(math.ceil(fraction * len(features))))
    indexes = rng.choice(len(features), size=sample_size, replace=True)
    # A bootstrap can exceptionally contain fewer than k unique rows.  Draw
    # deterministically again rather than accepting an ill-defined empty cluster.
    for _ in range(100):
        if len(np.unique(indexes)) >= k:
            break
        indexes = rng.choice(len(features), size=sample_size, replace=True)
    else:  # pragma: no cover - impossible for release-scale data
        raise ClusteringError(f"bootstrap for k={k} did not contain k unique occupations")
    model = KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd")
    model.fit(features[indexes])
    if len(np.unique(model.labels_)) != k:
        raise ClusteringError(f"bootstrap KMeans produced fewer than {k} nonempty clusters")
    return model.predict(features)


def _full_fit(features: np.ndarray, *, k: int, seeds: Sequence[int]) -> tuple[KMeans, int]:
    fitted: list[tuple[float, int, KMeans]] = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd").fit(features)
        if len(np.unique(model.labels_)) != k:
            raise ClusteringError(f"full KMeans produced fewer than {k} nonempty clusters")
        fitted.append((float(model.inertia_), int(seed), model))
    _, seed, model = min(fitted, key=lambda item: (item[0], item[1]))
    return model, seed


def evaluate_cluster_grid(
    features: np.ndarray | sparse.spmatrix,
    *,
    k_values: Sequence[int] = PREDECLARED_K_VALUES,
    seeds: Sequence[int] = PREDECLARED_SEEDS,
    bootstrap_fraction: float = 0.8,
    soc_major_groups: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Evaluate bootstrap-refit stability, then original-space silhouette."""

    values = _dense_features(features)
    ks = tuple(int(value) for value in k_values)
    seed_values = tuple(int(value) for value in seeds)
    if not ks or len(ks) != len(set(ks)) or any(k < 2 or k >= len(values) for k in ks):
        raise ClusteringError("k values must be unique and between 2 and n_occupations-1")
    if len(seed_values) < 2 or len(seed_values) != len(set(seed_values)):
        raise ClusteringError("at least two unique clustering seeds are required")
    if not 0.0 < bootstrap_fraction <= 1.0:
        raise ClusteringError("bootstrap_fraction must lie in (0, 1]")
    if soc_major_groups is not None and len(soc_major_groups) != len(values):
        raise ClusteringError("SOC major-group labels do not align with feature rows")

    rows: list[dict[str, object]] = []
    for k in ks:
        bootstrap_predictions = [
            _bootstrap_labels(
                values,
                k=k,
                seed=seed,
                fraction=bootstrap_fraction,
            )
            for seed in seed_values
        ]
        pairwise = [
            adjusted_rand_score(left, right)
            for left, right in itertools.combinations(bootstrap_predictions, 2)
        ]
        full_model, selected_seed = _full_fit(values, k=k, seeds=seed_values)
        silhouette = float(silhouette_score(values, full_model.labels_, metric="euclidean"))
        row: dict[str, object] = {
            "k": k,
            "mean_pairwise_adjusted_rand": float(np.mean(pairwise)),
            "minimum_pairwise_adjusted_rand": float(np.min(pairwise)),
            "silhouette": silhouette,
            "selected_seed": selected_seed,
            "inertia": float(full_model.inertia_),
        }
        if soc_major_groups is not None:
            row["soc_major_adjusted_mutual_info"] = float(
                adjusted_mutual_info_score(
                    np.asarray(soc_major_groups, dtype=str), full_model.labels_
                )
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)


def select_cluster_configuration(diagnostics: pd.DataFrame) -> tuple[int, int]:
    """Select stability first, silhouette second, and smaller k third."""

    required = {
        "k",
        "mean_pairwise_adjusted_rand",
        "silhouette",
        "selected_seed",
    }
    if missing := sorted(required - set(diagnostics.columns)):
        raise ClusteringError(f"cluster diagnostics are missing columns: {missing}")
    if diagnostics.empty or bool(
        diagnostics[["mean_pairwise_adjusted_rand", "silhouette"]]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any()
        .any()
    ):
        raise ClusteringError("cluster diagnostics must be nonempty and finite")
    ordered = diagnostics.sort_values(
        ["mean_pairwise_adjusted_rand", "silhouette", "k"],
        ascending=[False, False, True],
        kind="stable",
    )
    winner = ordered.iloc[0]
    return int(winner["k"]), int(winner["selected_seed"])


def _cluster_labels(
    cluster_ids: np.ndarray,
    term_matrix: np.ndarray | sparse.spmatrix,
    feature_names: Sequence[object],
    *,
    top_terms: int,
) -> dict[int, str]:
    if top_terms < 1:
        raise ClusteringError("top_terms must be positive")
    if term_matrix.shape[0] != len(cluster_ids) or term_matrix.shape[1] != len(feature_names):
        raise ClusteringError("term matrix, feature names, and cluster rows are misaligned")
    names = np.asarray([str(value).strip() for value in feature_names], dtype=object)
    if any(not value for value in names) or len(names) != len(set(names.tolist())):
        raise ClusteringError("feature names must be nonblank and unique")
    labels: dict[int, str] = {}
    for cluster_id in sorted(set(cluster_ids.tolist())):
        member_rows = np.flatnonzero(cluster_ids == cluster_id)
        if sparse.issparse(term_matrix):
            mean_scores = np.asarray(term_matrix[member_rows].mean(axis=0)).ravel()
        else:
            mean_scores = np.asarray(term_matrix, dtype=np.float64)[member_rows].mean(axis=0)
        order = np.lexsort((names.astype(str), -mean_scores))
        selected = [str(names[index]) for index in order if mean_scores[index] > 0][:top_terms]
        labels[cluster_id] = " · ".join(selected) if selected else f"cluster {cluster_id}"
    return labels


def fit_exploratory_clusters(
    features: np.ndarray | sparse.spmatrix,
    occupation_codes: Sequence[str],
    term_matrix: np.ndarray | sparse.spmatrix,
    feature_names: Sequence[object],
    *,
    k_values: Sequence[int] = PREDECLARED_K_VALUES,
    seeds: Sequence[int] = PREDECLARED_SEEDS,
    bootstrap_fraction: float = 0.8,
    soc_major_groups: Sequence[object] | None = None,
    top_terms: int = 3,
) -> ClusterResult:
    """Fit selected clusters and a separate two-dimensional PCA display."""

    values = _dense_features(features)
    codes = tuple(str(code) for code in occupation_codes)
    if (
        len(codes) != len(values)
        or len(codes) != len(set(codes))
        or any(not code for code in codes)
    ):
        raise ClusteringError("occupation codes must align one-to-one with feature rows")
    diagnostics = evaluate_cluster_grid(
        values,
        k_values=k_values,
        seeds=seeds,
        bootstrap_fraction=bootstrap_fraction,
        soc_major_groups=soc_major_groups,
    )
    selected_k, selected_seed = select_cluster_configuration(diagnostics)
    model = KMeans(
        n_clusters=selected_k,
        n_init=20,
        random_state=selected_seed,
        algorithm="lloyd",
    ).fit(values)
    raw_labels = np.asarray(model.labels_, dtype=int)
    summaries = _cluster_labels(
        raw_labels,
        term_matrix,
        feature_names,
        top_terms=top_terms,
    )
    # Canonicalize arbitrary KMeans label numbers using content summaries and
    # the smallest member code as a deterministic second key.
    label_order = sorted(
        summaries,
        key=lambda cluster_id: (
            summaries[cluster_id],
            min(code for code, label in zip(codes, raw_labels, strict=True) if label == cluster_id),
        ),
    )
    relabel = {old: new for new, old in enumerate(label_order)}
    canonical_labels = np.asarray([relabel[value] for value in raw_labels], dtype=int)
    canonical_summaries = {relabel[old]: summaries[old] for old in label_order}
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(values)
    assignments = pd.DataFrame(
        {
            "onet_soc_code": codes,
            "cluster_id": canonical_labels,
            "cluster_label": [canonical_summaries[value] for value in canonical_labels],
            "pca_x": coordinates[:, 0].astype(float),
            "pca_y": coordinates[:, 1].astype(float),
        }
    ).sort_values("onet_soc_code", kind="stable")
    return ClusterResult(
        selected_k=selected_k,
        selected_seed=selected_seed,
        assignments=assignments.reset_index(drop=True),
        diagnostics=diagnostics,
    )


__all__ = [
    "ClusterResult",
    "ClusteringError",
    "PREDECLARED_K_VALUES",
    "PREDECLARED_SEEDS",
    "evaluate_cluster_grid",
    "fit_exploratory_clusters",
    "select_cluster_configuration",
]
