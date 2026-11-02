from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from job_market_intelligence.embeddings import EmbeddingError, LSAEmbedder
from job_market_intelligence.lexical import TfidfIndex, cosine_scores
from job_market_intelligence.modeling import (
    CandidateSpec,
    ModelingError,
    assert_feature_columns_safe,
    fit_production_title_index,
    fit_production_transition_components,
)
from job_market_intelligence.retrieval import (
    RetrievalError,
    assert_query_aliases_excluded,
    build_title_documents,
    candidate_codes_for_phase,
    combine_component_scores,
    fitting_codes_for_phase,
    prepare_alias_queries,
    rank_alias_queries,
    simplex_weight_grid,
)
from job_market_intelligence.skills import (
    StructuredSimilarityIndex,
    reject_structured_leakage,
)


def _catalog() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occupations = pd.DataFrame(
        [
            {
                "onet_soc_code": "11-1001.00",
                "canonical_title": "Alpha Manager",
                "soc_major_group": "11",
            },
            {
                "onet_soc_code": "15-1001.00",
                "canonical_title": "Beta Developer",
                "soc_major_group": "15",
            },
            {
                "onet_soc_code": "15-1002.00",
                "canonical_title": "Gamma Analyst",
                "soc_major_group": "15",
            },
        ]
    )
    aliases = pd.DataFrame(
        [
            {
                "onet_soc_code": "11-1001.00",
                "job_title": "Team Lead",
                "normalized_job_title": "team lead",
            },
            {
                "onet_soc_code": "11-1001.00",
                "job_title": "Lead, Team",
                "normalized_job_title": "team lead",
            },
            {
                "onet_soc_code": "11-1001.00",
                "job_title": "Shared Role",
                "normalized_job_title": "shared role",
            },
            {
                "onet_soc_code": "15-1001.00",
                "job_title": "Code Builder",
                "normalized_job_title": "code builder",
            },
            {
                "onet_soc_code": "15-1001.00",
                "job_title": "Shared Role",
                "normalized_job_title": "shared role",
            },
            {
                "onet_soc_code": "15-1002.00",
                "job_title": "Data Investigator",
                "normalized_job_title": "data investigator",
            },
        ]
    )
    manifest = pd.DataFrame(
        {
            "onet_soc_code": occupations["onet_soc_code"],
            "fold": [3, 2, 1],
            "role": ["discovery", "selection", "confirmation"],
        }
    )
    return occupations, aliases, manifest


def test_alias_queries_drop_ambiguity_and_are_absent_from_evaluation_documents() -> None:
    occupations, aliases, manifest = _catalog()
    queries = prepare_alias_queries(aliases, occupations, manifest, maximum_per_occupation=25)

    assert "shared role" not in set(queries["normalized_query"])
    assert queries.groupby(["query_code", "normalized_query"]).size().eq(1).all()
    evaluated = queries.loc[queries["role"].isin(["selection", "confirmation"])]
    documents = build_title_documents(
        occupations,
        aliases,
        allowed_alias_codes={"11-1001.00"},
    )
    assert_query_aliases_excluded(documents, evaluated)
    assert (
        documents.documents.set_index("onet_soc_code").at["11-1001.00", "indexed_alias_count"] == 2
    )

    leaking = build_title_documents(occupations, aliases)
    with pytest.raises(RetrievalError, match="held-out aliases"):
        assert_query_aliases_excluded(leaking, evaluated)


def test_locked_alias_ambiguity_uses_full_catalog_before_test_filtering() -> None:
    occupations, aliases, manifest = _catalog()
    test_code = "15-1002.00"
    manifest = manifest.copy()
    manifest.loc[manifest["onet_soc_code"].eq(test_code), "role"] = "test"
    disputed = pd.DataFrame(
        [
            {
                "onet_soc_code": test_code,
                "job_title": "Platform Artisan",
                "normalized_job_title": "platform artisan",
            },
            {
                "onet_soc_code": "15-1001.00",
                "job_title": "Platform Artisan",
                "normalized_job_title": "platform artisan",
            },
        ]
    )
    aliases = pd.concat([aliases, disputed], ignore_index=True)

    development = prepare_alias_queries(
        aliases,
        occupations,
        manifest,
        allowed_query_codes={"15-1001.00"},
        ambiguity_scope="allowed_codes",
    )
    locked = prepare_alias_queries(
        aliases,
        occupations,
        manifest,
        allowed_query_codes={test_code},
        ambiguity_scope="full_catalog",
    )

    assert "platform artisan" in set(development["normalized_query"])
    assert "platform artisan" not in set(locked["normalized_query"])
    assert set(locked["role"]) == {"test"}


def test_tfidf_transformer_is_serving_compatible_and_lsa_is_deterministic(tmp_path) -> None:
    documents = [
        "C++ software developer builds systems",
        "Data analyst studies systems and models",
        "Nurse provides patient care",
        "Manager leads software teams",
    ]
    index = TfidfIndex(analyzer="word_char", min_df=1)
    catalog = index.fit_transform(documents)
    assert np.allclose(np.sqrt(catalog.multiply(catalog).sum(axis=1)).A1, 1.0)
    query = index.transform(["C++ developer"])
    scores = cosine_scores(query, catalog)
    assert scores.shape == (1, 4)
    assert int(np.argmax(scores[0])) == 0

    path = tmp_path / "title_vectorizer.joblib"
    joblib.dump(index, path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        restored = joblib.load(path)
    assert np.allclose(restored.transform(["C++ developer"]).toarray(), query.toarray())

    lsa = LSAEmbedder(2, random_state=7)
    embedded = lsa.fit_transform(catalog)
    assert embedded.shape == (4, 2)
    assert np.allclose(embedded, LSAEmbedder(2, random_state=7).fit_transform(catalog))
    with pytest.raises(EmbeddingError, match="requires more"):
        LSAEmbedder(catalog.shape[1]).fit(catalog)


def test_alias_ranking_and_phase_boundaries() -> None:
    occupations, aliases, manifest = _catalog()
    queries = prepare_alias_queries(aliases, occupations, manifest)
    documents = build_title_documents(
        occupations, aliases, excluded_alias_codes=set(occupations["onet_soc_code"])
    )
    transformer = TfidfIndex(analyzer="char", min_df=1)
    matrix = transformer.fit_transform(documents.documents["title_text"].tolist())
    results = rank_alias_queries(
        transformer,
        matrix,
        documents.documents["onet_soc_code"].tolist(),
        queries,
    )
    assert set(results.columns) == {"query_id", "query_code", "target_code", "rank"}
    assert candidate_codes_for_phase(manifest, "selection") == tuple(
        sorted(manifest["onet_soc_code"])
    )
    assert fitting_codes_for_phase(manifest, "selection") == ("11-1001.00",)
    assert fitting_codes_for_phase(manifest, "confirmation") == (
        "11-1001.00",
        "15-1001.00",
    )


def test_simplex_and_missing_component_combination_do_not_impute() -> None:
    grid = simplex_weight_grid(("lexical", "lsa", "structured"), step=0.25)
    assert len(grid) == 15
    assert all(sum(weights.values()) == pytest.approx(1.0) for weights in grid)
    scores = combine_component_scores(
        {
            "lexical": np.array([[0.4, 0.2]]),
            "structured": np.array([[np.nan, 0.8]]),
        },
        {"lexical": 0.5, "structured": 0.5},
    )
    assert scores[0, 0] == pytest.approx(0.4)
    assert scores[0, 1] == pytest.approx(0.5)
    all_missing = combine_component_scores(
        {"structured": np.array([[np.nan]])}, {"structured": 1.0}
    )
    assert np.isnan(all_missing[0, 0])


def _ratings() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "A": [0.2, 0.4, 0.6],
        "B": [0.4, 0.4, 0.8],
        "C": [0.9, None, None],
    }
    codes = {"A": "11-1001.00", "B": "15-1001.00", "C": "15-1002.00"}
    for label, ratings in values.items():
        for index, value in enumerate(ratings):
            rows.append(
                {
                    "onet_soc_code": codes[label],
                    "feature_family": "transferable_skill" if index < 2 else "knowledge",
                    "element_id": f"E{index}",
                    "scale_id": "IM",
                    "normalized_value": value,
                    "observed": value is not None,
                }
            )
    return pd.DataFrame(rows)


def test_structured_similarity_and_positive_gap_preserve_missingness() -> None:
    codes = ["15-1001.00", "11-1001.00", "15-1002.00"]
    index = StructuredSimilarityIndex(_ratings(), codes, minimum_overlap=2)

    assert index.occupation_codes == tuple(codes)
    assert np.isfinite(index.row("11-1001.00")[0])
    assert np.isnan(index.row("11-1001.00")[2])
    # Moving A -> B has positive gaps, while B -> A has none on shared skills.
    assert index.distance_row("11-1001.00")[0] > 0
    assert index.distance_row("15-1001.00")[1] == pytest.approx(0.0)

    leaked = _ratings().assign(employment_change_pct=10.0)
    with pytest.raises(Exception, match="forbidden"):
        reject_structured_leakage(leaked)
    with pytest.raises(ModelingError, match="forbidden"):
        assert_feature_columns_safe(["work_text", "relevance_grade"])


def test_structured_similarity_uses_same_skill_gap_validity_mask_as_serving() -> None:
    ratings = pd.DataFrame(
        [
            {
                "onet_soc_code": code,
                "feature_family": family,
                "element_id": element,
                "scale_id": "IM",
                "normalized_value": value,
                "observed": True,
            }
            for code, skill, knowledge in (
                ("11-1001.00", 0.2, 0.4),
                ("15-1001.00", 0.8, 0.6),
            )
            for family, element, value in (
                ("transferable_skill", "S1", skill),
                ("knowledge", "K1", knowledge),
            )
        ]
    )
    index = StructuredSimilarityIndex(ratings, ["11-1001.00", "15-1001.00"], minimum_overlap=2)
    # Skill+knowledge would meet two-feature similarity overlap, but the
    # transferable-skill gap has only one feature. Both are frozen unavailable.
    assert np.isnan(index.scores[0, 1])
    assert np.isnan(index.development_distance[0, 1])
    assert np.array_equal(
        np.isfinite(index.scores),
        np.isfinite(index.development_distance),
    )


def test_production_refit_aligns_title_transform_and_expands_transition_weights() -> None:
    codes = ["11-1001.00", "15-1001.00", "15-1002.00", "29-1001.00"]
    profiles = pd.DataFrame(
        {
            "onet_soc_code": codes,
            "canonical_title": [
                "Operations Manager",
                "Software Developer",
                "Data Analyst",
                "Registered Nurse Specialist",
            ],
            "title_text": ["unused"] * 4,
            "work_text": [
                "lead operations teams budgets",
                "build software systems code",
                "analyze data reports models",
                "provide patient clinical care",
            ],
        }
    )
    aliases = pd.DataFrame(
        {
            "onet_soc_code": codes,
            "job_title": ["Ops Lead", "Code Builder", "Data Investigator", "Clinical Nurse"],
            "normalized_job_title": [
                "ops lead",
                "code builder",
                "data investigator",
                "clinical nurse",
            ],
        }
    )
    ratings = pd.DataFrame(
        [
            {
                "onet_soc_code": code,
                "feature_family": family,
                "element_id": f"E{feature_index}",
                "scale_id": "IM",
                "normalized_value": (code_index + feature_index + 1) / 10,
                "observed": True,
            }
            for code_index, code in enumerate(codes)
            for feature_index, family in enumerate(
                ("transferable_skill", "transferable_skill", "knowledge")
            )
        ]
    )
    title_spec = CandidateSpec(
        "title_word_char",
        "title",
        "canonical_title",
        "word_char",
        0,
        (("lexical", 1.0),),
    )
    fit_codes = tuple(codes[:3])
    title = fit_production_title_index(
        profiles,
        aliases,
        title_spec,
        fit_codes=fit_codes,
        min_df=1,
    )
    assert title.occupation_codes == tuple(codes)
    assert title.matrix.dtype == np.float64
    assert title.transformer.transform(["software developer"]).shape[1] == title.matrix.shape[1]

    transition_spec = CandidateSpec(
        "synthetic-hybrid",
        "transition",
        "work_hybrid",
        "word",
        2,
        (("lexical", 0.5), ("lsa", 0.5), ("structured", 0.0)),
    )
    transition = fit_production_transition_components(
        profiles,
        ratings,
        transition_spec,
        fit_codes=fit_codes,
        minimum_structured_overlap=1,
        min_df=1,
        diagnostic_lsa_dimensions=2,
    )
    assert tuple(transition.weights) == ("lexical", "lsa", "structured")
    assert transition.weights["structured"] == 0.0
    assert transition.lexical_scores.shape == (4, 4)
    assert transition.cluster_embedding.shape == (4, 2)

    mutated_profiles = profiles.copy()
    heldout = mutated_profiles["onet_soc_code"].eq(codes[-1])
    mutated_profiles.loc[heldout, "canonical_title"] = "Alien Nurse Specialist"
    mutated_profiles.loc[heldout, "title_text"] = "alien heldout specialist"
    mutated_profiles.loc[heldout, "work_text"] = "alien heldout clinical token"
    mutated_aliases = aliases.copy()
    mutated_aliases.loc[
        mutated_aliases["onet_soc_code"].eq(codes[-1]),
        ["job_title", "normalized_job_title"],
    ] = ["Alien Heldout", "alien heldout"]
    title_mutated = fit_production_title_index(
        mutated_profiles,
        mutated_aliases,
        title_spec,
        fit_codes=fit_codes,
        min_df=1,
    )
    transition_mutated = fit_production_transition_components(
        mutated_profiles,
        ratings,
        transition_spec,
        fit_codes=fit_codes,
        minimum_structured_overlap=1,
        min_df=1,
        diagnostic_lsa_dimensions=2,
    )
    assert np.array_equal(
        title.transformer.get_feature_names_out(),
        title_mutated.transformer.get_feature_names_out(),
    )
    assert np.allclose(title.matrix[:3].toarray(), title_mutated.matrix[:3].toarray())
    assert transition.feature_names == transition_mutated.feature_names
    assert np.allclose(transition.cluster_embedding[:3], transition_mutated.cluster_embedding[:3])
    assert title_mutated.matrix.shape[0] == len(codes)
    assert transition_mutated.cluster_embedding.shape[0] == len(codes)
