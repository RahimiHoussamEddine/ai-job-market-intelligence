from __future__ import annotations

import numpy as np

from job_market_intelligence.clustering import (
    evaluate_cluster_grid,
    fit_exploratory_clusters,
    select_cluster_configuration,
)


def test_cluster_selection_uses_stability_then_silhouette_and_emits_pca() -> None:
    rng = np.random.default_rng(20260814)
    left = rng.normal(loc=(-3.0, -3.0, 0.0), scale=0.15, size=(20, 3))
    right = rng.normal(loc=(3.0, 3.0, 0.0), scale=0.15, size=(20, 3))
    features = np.vstack([left, right])
    term_matrix = np.column_stack(
        [
            np.r_[np.ones(20), np.zeros(20)],
            np.r_[np.zeros(20), np.ones(20)],
            np.full(40, 0.2),
        ]
    )
    codes = [f"15-{index:04d}.00" for index in range(40)]
    seeds = (7, 8, 9)
    diagnostics = evaluate_cluster_grid(
        features,
        k_values=(2, 3),
        seeds=seeds,
        soc_major_groups=["11"] * 20 + ["15"] * 20,
    )
    selected_k, _ = select_cluster_configuration(diagnostics)
    assert selected_k in {2, 3}

    result = fit_exploratory_clusters(
        features,
        codes,
        term_matrix,
        ["management", "technology", "general"],
        k_values=(2, 3),
        seeds=seeds,
        soc_major_groups=["11"] * 20 + ["15"] * 20,
    )
    assert result.selected_k == selected_k
    assert len(result.assignments) == 40
    assert result.assignments["onet_soc_code"].tolist() == sorted(codes)
    assert np.isfinite(result.assignments[["pca_x", "pca_y"]]).all().all()
    assert result.assignments.groupby("cluster_id")["cluster_label"].nunique().eq(1).all()
