"""Focused Streamlit views backed by the same immutable serving service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from job_market_intelligence.service import (
    OccupationIntelligenceService,
    ServiceInputError,
    get_service_state,
)


def _plain(value: Any) -> Any:
    """Convert immutable artifact mappings/tuples into display-only containers."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _precision_text(metadata: Mapping[str, Any] | None) -> str:
    if metadata is None:
        return "No observed rating metadata"
    sample_size = metadata.get("sample_size")
    standard_error = metadata.get("standard_error")
    lower = metadata.get("lower_ci_bound")
    upper = metadata.get("upper_ci_bound")
    rating_date = metadata.get("rating_date")
    domain_source = metadata.get("domain_source")
    interval = "missing" if lower is None or upper is None else f"[{lower:.3f}, {upper:.3f}]"
    return (
        f"N={sample_size if sample_size is not None else 'missing'}; "
        f"SE={standard_error if standard_error is not None else 'missing'}; "
        f"CI={interval}; date={rating_date or 'missing'}; source={domain_source or 'missing'}"
    )


def _occupation_options(service: OccupationIntelligenceService) -> tuple[dict[str, str], list[str]]:
    labels = {
        item.onet_soc_code: f"{item.canonical_title} · {item.onet_soc_code}"
        for item in service.catalog()
    }
    return labels, list(labels)


def _render_discovery(service: OccupationIntelligenceService) -> None:
    st.subheader("Discover occupations")
    st.caption(
        "Search the frozen occupation catalog with ordinary text. Similarity is not a match "
        "percentage and says nothing about an individual's qualification."
    )
    with st.form("search-form", clear_on_submit=False):
        query = st.text_input(
            "Occupation, title, or kind of work",
            placeholder="e.g. design accessible software interfaces",
            max_chars=512,
        )
        limit = st.slider("Results", 1, 25, 10)
        submitted = st.form_submit_button("Search", type="primary")
    if not submitted:
        return
    try:
        results = service.search(query, k=limit)
    except ServiceInputError as exc:
        st.warning(str(exc))
        return
    if not results:
        st.info("The frozen vocabulary did not produce a non-zero query representation.")
        return
    frame = pd.DataFrame(asdict(result) for result in results)
    frame = frame.rename(
        columns={
            "canonical_title": "Occupation",
            "onet_soc_code": "O*NET-SOC",
            "similarity": "Text similarity",
            "description": "Description",
            "rank": "Rank",
        }
    )
    st.dataframe(
        frame[["Rank", "Occupation", "O*NET-SOC", "Text similarity", "Description"]],
        hide_index=True,
        width="stretch",
        column_config={"Text similarity": st.column_config.NumberColumn(format="%.3f")},
    )


def _render_transitions(
    service: OccupationIntelligenceService, labels: dict[str, str], codes: list[str]
) -> None:
    st.subheader("Adjacent occupations and trade-offs")
    st.caption(
        "Adjacent occupations reflect a frozen aggregate retrieval model. They do not estimate "
        "worker mobility, employability, or qualification."
    )
    origin = st.selectbox(
        "Starting occupation",
        codes,
        format_func=labels.__getitem__,
        key="transition-origin",
    )
    neighbor_limit = st.slider("Adjacent occupations", 1, 25, 10)
    neighbors = service.neighbors(origin, k=neighbor_limit)
    neighbor_frame = pd.DataFrame(asdict(result) for result in neighbors)
    if not neighbor_frame.empty:
        st.dataframe(
            neighbor_frame.rename(
                columns={
                    "rank": "Rank",
                    "canonical_title": "Occupation",
                    "onet_soc_code": "O*NET-SOC",
                    "total_score": "Total",
                    "lexical_score": "Lexical",
                    "lsa_score": "LSA",
                    "structured_score": "Structured",
                    "development_distance": "Development distance",
                    "explanation": "Explanation",
                }
            )[
                [
                    "Rank",
                    "Occupation",
                    "O*NET-SOC",
                    "Total",
                    "Lexical",
                    "LSA",
                    "Structured",
                    "Development distance",
                    "Explanation",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Pareto frontier")
    outcome = st.selectbox(
        "National BLS outcome",
        ("median_wage", "projected_growth", "annual_openings"),
        format_func={
            "median_wage": "2024 median annual wage",
            "projected_growth": "Projected 2024–34 change (%)",
            "annual_openings": "Average annual openings, 2024–34",
        }.__getitem__,
    )
    left, right = st.columns(2)
    with left:
        use_distance = st.checkbox("Limit development distance")
        max_distance = st.number_input(
            "Maximum distance",
            min_value=0.0,
            value=1.0,
            disabled=not use_distance,
        )
    with right:
        use_zone = st.checkbox("Limit typical job zone")
        max_zone = st.select_slider(
            "Maximum job zone (official O*NET 30.3 labels)",
            options=(2, 3, 4, 5),
            value=5,
            format_func={
                2: "Job Zone 1–2 · very little to some preparation",
                3: "Job Zone 3 · medium preparation",
                4: "Job Zone 4 · high preparation",
                5: "Job Zone 5 · extensive preparation",
            }.__getitem__,
            disabled=not use_zone,
        )
    points = service.frontier(
        origin,
        outcome=outcome,
        max_development_distance=max_distance if use_distance else None,
        maximum_job_zone=max_zone if use_zone else None,
    )
    if not points:
        st.info("No candidate has complete values for both selected objectives and filters.")
    else:
        frontier_frame = pd.DataFrame(asdict(point) for point in points)
        figure = px.scatter(
            frontier_frame,
            x="development_distance",
            y="outcome_value",
            hover_name="canonical_title",
            hover_data=["onet_soc_code", "nem_code"],
            labels={
                "development_distance": "Occupation-level development distance (lower)",
                "outcome_value": "Selected national outcome (higher)",
            },
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(frontier_frame, hide_index=True, width="stretch")
    st.caption(
        "BLS projections are conditional national scenarios; annual openings are not active "
        "vacancies, and wages are not personalized salary estimates. O*NET stores the redesigned "
        "combined Job Zone 1–2 category as 2. Shared NEM values are not summed."
    )


def _render_skills(
    service: OccupationIntelligenceService, labels: dict[str, str], codes: list[str]
) -> None:
    st.subheader("Compare transferable-skill ratings")
    st.caption(
        "This view compares occupation-level O*NET ratings. Suppressed, not-relevant, and missing "
        "ratings stay missing rather than being filled."
    )
    left_column, right_column = st.columns(2)
    with left_column:
        left_code = st.selectbox(
            "First occupation", codes, format_func=labels.__getitem__, key="skill-left"
        )
    with right_column:
        default_right = 1 if len(codes) > 1 else 0
        right_code = st.selectbox(
            "Second occupation",
            codes,
            index=default_right,
            format_func=labels.__getitem__,
            key="skill-right",
        )
    differences = service.compare_skills(left_code, right_code, limit=100)
    if not differences:
        st.info("No transferable-skill ratings are available for this pair.")
        return
    rows: list[dict[str, Any]] = []
    for item in differences:
        raw = asdict(item)
        left_metadata = raw.pop("left_metadata")
        right_metadata = raw.pop("right_metadata")
        raw["label"] = f"{raw['element_name']} · {raw['scale_name']}"
        raw["shared_strength"] = (
            min(raw["left_value"], raw["right_value"])
            if raw["left_value"] is not None and raw["right_value"] is not None
            else None
        )
        raw["left_precision"] = _precision_text(left_metadata)
        raw["right_precision"] = _precision_text(right_metadata)
        raw["left_metadata"] = left_metadata
        raw["right_metadata"] = right_metadata
        rows.append(raw)
    frame = pd.DataFrame(rows)

    view = st.radio(
        "Comparison lens",
        ("Shared strengths", "Positive target − source gaps"),
        horizontal=True,
        help=(
            "Shared strengths rank elements by the lower of the two observed ratings. "
            "Positive gaps show only ratings where the target exceeds the source."
        ),
    )
    if view == "Shared strengths":
        selected = frame.dropna(subset=["shared_strength"]).sort_values(
            ["shared_strength", "label"], ascending=[False, True]
        )
    else:
        selected = frame.loc[frame["difference"].fillna(0) > 0].sort_values(
            ["difference", "label"], ascending=[False, True]
        )
    selected = selected.head(12).copy()
    if selected.empty:
        st.info("No observed rating pair is available for this comparison lens.")
        return

    source_title = service.occupation(left_code).canonical_title
    target_title = service.occupation(right_code).canonical_title
    line_x: list[float | None] = []
    line_y: list[str | None] = []
    for row in selected.itertuples(index=False):
        line_x.extend((row.left_value, row.right_value, None))
        line_y.extend((row.label, row.label, None))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "#94a3b8", "width": 3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["left_value"],
            y=selected["label"],
            mode="markers",
            name=f"Source · {source_title}",
            marker={"size": 11, "color": "#2563eb"},
            customdata=selected[["left_precision"]],
            hovertemplate="%{y}<br>rating=%{x:.3f}<br>%{customdata[0]}<extra>Source</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected["right_value"],
            y=selected["label"],
            mode="markers",
            name=f"Target · {target_title}",
            marker={"size": 11, "color": "#ea580c"},
            customdata=selected[["right_precision"]],
            hovertemplate="%{y}<br>rating=%{x:.3f}<br>%{customdata[0]}<extra>Target</extra>",
        )
    )
    figure.update_layout(
        xaxis_title="Normalized occupation-level O*NET rating",
        yaxis_title=None,
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(selected["label"].tolist())),
        },
        legend={"orientation": "h", "y": 1.08},
        height=max(440, 42 * len(selected) + 160),
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
    )
    st.plotly_chart(figure, width="stretch")

    precision_rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        for role, metadata in (
            ("Source", row["left_metadata"]),
            ("Target", row["right_metadata"]),
        ):
            metadata = metadata or {}
            precision_rows.append(
                {
                    "Element · scale": row["label"],
                    "Occupation role": role,
                    "N": metadata.get("sample_size"),
                    "SE": metadata.get("standard_error"),
                    "Raw-scale lower CI": metadata.get("lower_ci_bound"),
                    "Raw-scale upper CI": metadata.get("upper_ci_bound"),
                    "Rating date": metadata.get("rating_date"),
                    "Domain source": metadata.get("domain_source"),
                }
            )
    with st.expander("Rating precision and lineage"):
        st.dataframe(pd.DataFrame(precision_rows), hide_index=True, width="stretch")
    observed_pairs = int(frame["difference"].notna().sum())
    st.caption(
        f"{observed_pairs} of {len(frame)} element-scale pairs have both ratings observed. "
        "The dumbbell uses normalized ratings; N, SE, raw publisher-scale confidence bounds, "
        "rating date, and domain source are shown when O*NET supplies them."
    )


def _render_clusters(service: OccupationIntelligenceService) -> None:
    st.subheader("Explore stable clusters")
    st.caption(
        "Clusters are exploratory navigation, not occupational truth. PCA coordinates are only a "
        "two-dimensional display of the frozen representation."
    )
    members = service.clusters()
    frame = pd.DataFrame(asdict(member) for member in members)
    figure = px.scatter(
        frame,
        x="pca_x",
        y="pca_y",
        color=frame["cluster_id"].astype(str),
        hover_name="canonical_title",
        hover_data=["onet_soc_code", "cluster_label"],
        labels={"color": "Cluster", "pca_x": "PCA 1", "pca_y": "PCA 2"},
    )
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        pd.DataFrame(asdict(summary) for summary in service.cluster_summaries()),
        hide_index=True,
        width="stretch",
    )


def _render_benchmark(title: str, report: Mapping[str, Any]) -> None:
    st.markdown(f"#### {title}")
    payload = _plain(report["payload"])
    st.caption(
        f"Locked test · selected candidate `{payload['selected_candidate_id']}` · "
        f"primary metric `{payload['primary_metric']}`"
    )
    metrics = pd.DataFrame(payload["metrics"]).rename(
        columns={
            "name": "Metric",
            "estimate": "Estimate",
            "ci_lower": "CI lower",
            "ci_upper": "CI upper",
            "n_queries": "Queries",
        }
    )
    st.dataframe(metrics, hide_index=True, width="stretch")
    if payload["baseline_deltas"]:
        deltas = pd.DataFrame(payload["baseline_deltas"]).rename(
            columns={
                "baseline_candidate_id": "Baseline",
                "metric": "Metric",
                "delta": "Paired delta",
                "ci_lower": "CI lower",
                "ci_upper": "CI upper",
            }
        )
        st.dataframe(deltas, hide_index=True, width="stretch")
    else:
        st.info(
            "The confirmation gate selected the preregistered lexical baseline; "
            "a self-comparison delta is intentionally not reported."
        )

    st.markdown("##### Descriptive primary-metric slices")
    slices = pd.DataFrame(payload["slices"]).rename(
        columns={
            "dimension": "Dimension",
            "value": "Slice",
            "estimate": "Estimate",
            "ci_lower": "CI lower",
            "ci_upper": "CI upper",
            "n_units": "Analysis units",
            "descriptive_small_slice": "Small slice (descriptive)",
        }
    )
    st.dataframe(slices, hide_index=True, width="stretch")
    st.caption(
        "Slices use the benchmark's primary metric. Rows with fewer than 20 analysis units are "
        "descriptive and are not confirmatory evidence."
    )


def _render_evidence(service: OccupationIntelligenceService) -> None:
    st.subheader("Evidence & lineage")
    st.caption(
        "Only locked, checksum-bound reports accepted by the serving-artifact validator appear here."
    )
    metadata = asdict(service.metadata)
    source_hashes = metadata.pop("source_hashes")
    serving_file_hashes = metadata.pop("serving_file_hashes")
    lineage_fields = {
        "Repository commit": metadata["repository_commit"],
        "Project config SHA-256": metadata["project_config_sha256"],
        "Source manifest SHA-256": metadata["source_manifest_sha256"],
        "Split manifest SHA-256": metadata["split_manifest_sha256"],
        "Locked rankings SHA-256": metadata["locked_rankings_sha256"],
        "Locked metrics SHA-256": metadata["locked_metrics_sha256"],
        "Environment lock SHA-256": metadata["environment_lock_sha256"],
        "Processed manifest SHA-256": metadata["processed_manifest_sha256"],
        "Feature manifest SHA-256": metadata["feature_manifest_sha256"],
        "Metric contract SHA-256": metadata["metric_contract_sha256"],
        "Candidate manifest SHA-256": metadata["candidate_manifest_sha256"],
        "Evaluation universe SHA-256": metadata["evaluation_universe_sha256"],
        "Implementation manifest SHA-256": metadata["implementation_manifest_sha256"],
        "Title candidate grid SHA-256": metadata["title_candidate_grid_sha256"],
        "Transition candidate grid SHA-256": metadata["transition_candidate_grid_sha256"],
    }
    if metadata["fit_codes_sha256"] is not None:
        lineage_fields.update(
            {
                "Production fit phase": metadata["fit_phase"],
                "Production fit-code SHA-256": metadata["fit_codes_sha256"],
                "Production fit-code count": metadata["fit_code_count"],
                "Production fit fold SHA-256": metadata["fit_fold_manifest_sha256"],
            }
        )
    lineage_frame = pd.DataFrame(
        {"Lineage object": lineage_fields.keys(), "Digest / commit": lineage_fields.values()}
    )
    lineage_frame["Digest / commit"] = lineage_frame["Digest / commit"].astype(str)
    st.dataframe(
        lineage_frame,
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### Official-source hashes")
    st.dataframe(
        pd.DataFrame(source_hashes, columns=["Source", "SHA-256"]),
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### Serving, model, and report hashes")
    st.dataframe(
        pd.DataFrame(serving_file_hashes, columns=["Logical artifact", "SHA-256"]),
        hide_index=True,
        width="stretch",
    )

    reports = service.evidence_reports
    if service.metadata.evidence_status != "locked" or not reports:
        st.info(
            "Locked evidence has not been generated. Development, placeholder, and synthetic "
            "metrics are intentionally not displayed."
        )
        return

    _render_benchmark("Alternate-title discovery", reports["title_benchmark"])
    _render_benchmark("Adjacent-occupation agreement", reports["transition_benchmark"])

    cluster = _plain(reports["cluster_stability"]["payload"])
    st.markdown("#### Exploratory cluster stability")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Selected k": cluster["selected_k"],
                    "Stability metric": cluster["stability_metric"],
                    "Stability": cluster["stability_estimate"],
                    "Silhouette": cluster["silhouette"],
                    "Seeds": ", ".join(str(seed) for seed in cluster["seeds"]),
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    coverage = _plain(reports["data_coverage"]["payload"])
    coverage_columns = st.columns(2)
    with coverage_columns[0]:
        st.markdown("#### Data coverage counts")
        st.dataframe(
            pd.DataFrame(
                {"Measure": coverage["counts"].keys(), "Count": coverage["counts"].values()}
            ),
            hide_index=True,
            width="stretch",
        )
    with coverage_columns[1]:
        st.markdown("#### Data coverage rates")
        st.dataframe(
            pd.DataFrame({"Measure": coverage["rates"].keys(), "Rate": coverage["rates"].values()}),
            hide_index=True,
            width="stretch",
        )
    for note in coverage.get("notes", []):
        st.caption(note)

    limitations = _plain(reports["limitations"]["payload"])["items"]
    st.markdown("#### Validated limitations")
    for item in limitations:
        st.markdown(f"- {item}")


def render_app() -> None:
    st.set_page_config(page_title="AI Job Market Intelligence", layout="wide")
    st.title("AI Job Market Intelligence")
    st.write(
        "Transparent U.S. occupation discovery and career-transition trade-offs from a frozen, "
        "checksum-validated artifact."
    )
    state = get_service_state()
    if not state.available or state.service is None:
        st.error("The validated serving artifact is not available.")
        st.info(
            "Build or mount the v1 artifact, then set JOB_INTELLIGENCE_ARTIFACT_DIR to its directory. "
            f"State: {state.reason_code or 'artifact_unavailable'}."
        )
        st.stop()
    service = state.service
    labels, codes = _occupation_options(service)

    with st.sidebar:
        st.subheader("Release artifact")
        st.json(asdict(service.metadata))
        st.caption(f"{service.occupation_count:,} occupations")
        st.warning(
            "Aggregate decision support only. Do not use this tool to rank people, screen applicants, "
            "or infer qualification, employability, or hiring probability."
        )

    discovery, transitions, skills, clusters, evidence = st.tabs(
        ("Discover", "Transitions", "Skills", "Clusters", "Evidence & lineage")
    )
    with discovery:
        _render_discovery(service)
    with transitions:
        _render_transitions(service, labels, codes)
    with skills:
        _render_skills(service, labels, codes)
    with clusters:
        _render_clusters(service)
    with evidence:
        _render_evidence(service)


if __name__ == "__main__":
    render_app()
