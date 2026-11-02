from __future__ import annotations

import pandas as pd
import pytest

from job_market_intelligence.modeling import (
    OPERATIONAL_COST_POLICY,
    STRUCTURED_ABLATION_REASON,
    apply_confirmation_gate,
    candidate_grid_digest,
    select_candidate_one_se,
    title_candidate_grid,
    transition_candidate_grid,
)
from job_market_intelligence.ranking_metrics import (
    deterministic_rank,
    macro_title_metrics,
    paired_stratified_bootstrap,
    transition_metrics,
)


def test_deterministic_ranking_and_macro_title_weighting() -> None:
    ranked = deterministic_rank(["B", "A", "C"], [0.5, 0.5, 0.2])
    assert [item.candidate_id for item in ranked] == ["A", "B", "C"]

    queries = pd.DataFrame(
        [
            {"query_id": "a1", "query_code": "A", "target_code": "A", "rank": 1},
            {"query_id": "a2", "query_code": "A", "target_code": "A", "rank": 1},
            {"query_id": "b1", "query_code": "B", "target_code": "B", "rank": pd.NA},
        ]
    )
    metrics = macro_title_metrics(queries)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["query_weighted_top1_accuracy"] == pytest.approx(2 / 3)


def _qrels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query in ("Q1", "Q2"):
        for index, (candidate, tier, grade) in enumerate(
            [("A", "Primary-Short", 3), ("B", "Primary-Long", 2), ("C", "Supplemental", 1)],
            start=1,
        ):
            rows.append(
                {
                    "query_onet_soc_code": query,
                    "candidate_onet_soc_code": f"{query}-{candidate}",
                    "relatedness_tier": tier,
                    "relevance_grade": grade,
                    "relatedness_index": index,
                }
            )
    return pd.DataFrame(rows)


def test_transition_metrics_follow_graded_and_primary_conventions() -> None:
    rankings = pd.DataFrame(
        [
            {"query_code": query, "candidate_code": f"{query}-{candidate}", "rank": rank}
            for query in ("Q1", "Q2")
            for rank, candidate in enumerate(("A", "B", "C"), start=1)
        ]
    )
    metrics = transition_metrics(rankings, _qrels())
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["primary_short_recall_at_5"] == pytest.approx(1.0)
    assert metrics["primary_recall_at_10"] == pytest.approx(1.0)
    assert metrics["mrr_at_10"] == pytest.approx(1.0)


def test_paired_stratified_bootstrap_is_reproducible() -> None:
    champion = pd.Series([0.8, 0.5, 0.9, 0.4], index=list("abcd"))
    baseline = pd.Series([0.7, 0.4, 0.7, 0.5], index=list("abcd"))
    strata = pd.Series(["11", "11", "15", "15"], index=list("abcd"))

    first = paired_stratified_bootstrap(champion, baseline, strata, replicates=200, seed=7)
    second = paired_stratified_bootstrap(champion, baseline, strata, replicates=200, seed=7)
    assert first == second
    assert first.estimate == pytest.approx(0.075)
    assert first.ci_lower <= first.estimate <= first.ci_upper


def _candidate_results(role: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(20):
        leader_value = 1.0 if index % 2 == 0 else 0.0
        for candidate, value, components, size, latency in (
            ("leader", leader_value, 128, 1000, 2.0),
            ("simple", leader_value - 0.04, 0, 200, 1.0),
            ("poor", leader_value - 0.40, 0, 100, 0.5),
        ):
            rows.append(
                {
                    "candidate_id": candidate,
                    "query_id": f"q{index:02d}",
                    "primary_value": value,
                    "soc_major_group": "11" if index < 10 else "15",
                    "latent_dimensions": components,
                    "artifact_bytes": size,
                    "operational_cost_units": int(latency * 100),
                    "latency_ms": latency,
                    "deployment_eligible": True,
                    "ineligibility_reason": "",
                    "split_role": role,
                }
            )
    return pd.DataFrame(rows)


def test_one_se_prefers_simpler_candidate_and_confirmation_can_fallback() -> None:
    decision = select_candidate_one_se(
        _candidate_results("selection"), bootstrap_replicates=200, bootstrap_seed=9
    )
    assert decision.metric_leader_id == "leader"
    assert decision.selected_candidate_id == "simple"
    assert "poor" not in decision.eligible_candidate_ids

    confirmation = _candidate_results("confirmation")
    confirmation.loc[confirmation["candidate_id"].eq("simple"), "primary_value"] -= 0.10
    gate = apply_confirmation_gate(
        confirmation,
        proposed_candidate_id="simple",
        baseline_candidate_id="leader",
    )
    assert gate.fallback_applied
    assert gate.release_candidate_id == "leader"


def test_ineligible_metric_leader_cannot_win_and_latency_is_diagnostic_only() -> None:
    results = _candidate_results("selection")
    structured = results.loc[results["candidate_id"].eq("leader")].copy()
    structured["candidate_id"] = "transition_structured"
    structured["primary_value"] += 0.25
    structured["deployment_eligible"] = False
    structured["ineligibility_reason"] = STRUCTURED_ABLATION_REASON
    combined = pd.concat([results, structured], ignore_index=True)

    first = select_candidate_one_se(combined, bootstrap_replicates=200, bootstrap_seed=9)
    changed_latency = combined.copy()
    changed_latency["latency_ms"] = changed_latency["latency_ms"] * 1000 + 17
    second = select_candidate_one_se(changed_latency, bootstrap_replicates=200, bootstrap_seed=9)

    assert first == second
    assert first.unrestricted_metric_leader_id == "transition_structured"
    assert first.selected_candidate_id != "transition_structured"
    assert first.deployment_ineligible_candidates == (
        ("transition_structured", STRUCTURED_ABLATION_REASON),
    )
    assert first.operational_cost_policy == OPERATIONAL_COST_POLICY


def test_candidate_grids_are_frozen_and_hashable() -> None:
    title = title_candidate_grid()
    transition = transition_candidate_grid()
    assert {candidate.lsa_dimensions for candidate in title if candidate.lsa_dimensions} == {
        64,
        128,
        256,
    }
    assert {candidate.lsa_dimensions for candidate in transition if candidate.lsa_dimensions} == {
        64,
        128,
        256,
    }
    assert candidate_grid_digest(title) == candidate_grid_digest(title_candidate_grid())
    assert len({candidate.candidate_id for candidate in transition}) == len(transition)
