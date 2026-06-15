#!/usr/bin/env python3
"""Swiss Open Government Data Portal - Fuzzy HCIR Search System.

A research prototype for the Master Thesis:
"Improving Access to Swiss Open Government Data through Fuzzy Human-Centered Information Retrieval"

This file is intentionally kept small and acts as the Streamlit entrypoint.
Implementation details live in:
- `code/prototype/api/` (CKAN client)
- `code/prototype/ranking/` (fuzzy + baselines)
- `code/prototype/ui/` (Streamlit components + CSS)

Run:
- `streamlit run code/prototype/app.py`
- or: `streamlit run code/prototype/swiss_ogd_portal.py`
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional

import streamlit as st

try:
    from code.prototype.api.client import OpenDataSwissClient
    from code.prototype.demo_data import get_demo_data
    from code.prototype.ranking.baselines import (
        BM25Ranker,
        PortalDefaultRanker,
        deduplicate_datasets,
        deduplicate_display_datasets,
        deduplicate_ranked_results,
    )
    from code.prototype.ranking.fuzzy import FuzzyHCIRRanker, MultilingualQueryProcessor
    from code.prototype.ranking.models import normalize_text, safe_html_text
    from code.prototype.ui.components import (
        render_comparison_view,
        render_header,
        render_result_card,
        render_sidebar,
    )
    from code.prototype.ui.pagination import render_pagination_controls, render_pagination_summary
except Exception:  # Streamlit runs with script dir on sys.path
    from api.client import OpenDataSwissClient
    from demo_data import get_demo_data
    from ranking.baselines import (
        BM25Ranker,
        PortalDefaultRanker,
        deduplicate_datasets,
        deduplicate_display_datasets,
        deduplicate_ranked_results,
    )
    from ranking.fuzzy import FuzzyHCIRRanker, MultilingualQueryProcessor
    from ranking.models import normalize_text, safe_html_text
    from ui.components import render_comparison_view, render_header, render_result_card, render_sidebar
    from ui.pagination import render_pagination_controls, render_pagination_summary

try:
    from code.prototype.visual_explanations import build_top3_radar_figure
except Exception:  # Streamlit runs with script dir on sys.path
    from visual_explanations import build_top3_radar_figure


def configure_page() -> None:
    st.set_page_config(
        page_title="opendata.swiss | Fuzzy Search Research",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://handbook.opendata.swiss/",
            "Report a bug": None,
            "About": """
        ## Swiss OGD Fuzzy HCIR Research Prototype

        Master Thesis Project - University of Fribourg

        **Research Questions:**
        - RQ1: Fuzzy modeling of vague query intent
        - RQ2: Comparison with keyword baselines
        - RQ3: Explainability and user trust
        - RQ4: Advantages over AI/semantic approaches
        """,
        },
    )


def _build_fq(settings: Dict) -> Optional[str]:
    fq_parts: List[str] = []

    if settings.get("theme"):
        fq_parts.append(f"groups:{settings['theme']}")
    if settings.get("organizations"):
        orgs = " OR ".join(settings["organizations"])
        fq_parts.append(f"organization:({orgs})")
    if settings.get("formats"):
        fmts = " OR ".join([str(fmt).upper() for fmt in settings["formats"]])
        fq_parts.append(f"res_format:({fmts})")
    if settings.get("licenses"):
        lics = " OR ".join(settings["licenses"])
        fq_parts.append(f"license_id:({lics})")

    return " AND ".join(fq_parts) if fq_parts else None


def _update_facet_options(api_client: OpenDataSwissClient, *, query: str, fq: Optional[str]) -> None:
    # Update sidebar facet options for the current query context.
    try:
        api_client._rate_limit()
        facet_params = {
            "q": query,
            "rows": 0,
            "facet": "true",
            "facet.field": ["organization", "res_format"],
            "facet.limit": 30,
        }
        if fq:
            facet_params["fq"] = fq
        facet_resp = api_client.session.get(
            f"{api_client.BASE_URL}/package_search",
            params=facet_params,
            timeout=30,
        )
        facet_data = facet_resp.json() if facet_resp.ok else {}
        facets = (facet_data.get("result") or {}).get("facets") or {}
        org_f = facets.get("organization") or {}
        fmt_f = facets.get("res_format") or {}
        st.session_state["facet_org_options"] = [
            k for k, _ in sorted(org_f.items(), key=lambda kv: kv[1], reverse=True)
        ]
        st.session_state["facet_format_options"] = [
            k for k, _ in sorted(fmt_f.items(), key=lambda kv: kv[1], reverse=True)
        ]
    except Exception:
        pass


def _update_license_options_from_results(raw_results: List[Dict]) -> None:
    try:
        license_ids: List[str] = []
        for dataset in raw_results:
            lid = dataset.get("license_id")
            if lid and lid not in license_ids:
                license_ids.append(lid)
        st.session_state["facet_license_options"] = license_ids[:30]
    except Exception:
        pass


def main() -> None:
    configure_page()
    render_header()
    settings = render_sidebar()

    api_client = OpenDataSwissClient(error_handler=st.error)
    fuzzy_ranker = FuzzyHCIRRanker()
    portal_ranker = PortalDefaultRanker()
    bm25_ranker = BM25Ranker()

    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
                "Search Swiss Open Government Data",
                placeholder="e.g., 'recent air quality data', 'complete transport statistics'",
                key="search_query",
            )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.markdown("**Try these:**")
    example_cols = st.columns(4)
    example_queries = [
        "recent air quality data",
        "complete transport statistics",
        "environment pollution monitoring",
        "well-documented climate data",
    ]

    for col, example in zip(example_cols, example_queries):
        with col:
            if st.button(f"{example}", key=f"example_{example}"):
                query = example
                search_clicked = True

    st.markdown("</div>", unsafe_allow_html=True)

    if not (query and search_clicked):
        return

    with st.spinner("Searching opendata.swiss..."):
        display_limit = int(settings.get("num_results", 20) or 20)

        if "Demo" in str(settings.get("data_source", "")):
            raw_results = get_demo_data()
            total_count = len(raw_results)
            fq = None
        else:
            fq = _build_fq(settings)

            scoring_corpus_size = 50
            raw_results, total_count = api_client.search(
                query,
                rows=max(scoring_corpus_size, display_limit),
                fq=fq,
            )

            _update_facet_options(api_client, query=query, fq=fq)

        raw_results = deduplicate_datasets(raw_results)
        raw_results = deduplicate_display_datasets(raw_results)
        _update_license_options_from_results(raw_results)

        if not raw_results:
            st.warning("No results found. Try adjusting your query or filters.")
            return

        query_processor = MultilingualQueryProcessor()
        query_info = query_processor.process(query)

        if query_info.get("has_vague_terms"):
            vague_found = [k for k, v in (query_info.get("vague_predicates") or {}).items() if v]
            st.info(
                "Fuzzy query detected! Found vague terms: " + ", ".join(vague_found)
                + ". The fuzzy system will interpret these contextually."
            )

        ranking_method = settings.get("ranking_method")

        if ranking_method == "Compare All":
            results_fuzzy = deduplicate_ranked_results(
                fuzzy_ranker.rank(
                    raw_results.copy(),
                    query,
                    factor_weights=settings.get("factor_weights"),
                )
            )
            results_portal = deduplicate_display_datasets(portal_ranker.rank(raw_results.copy(), query))
            results_bm25 = deduplicate_display_datasets(bm25_ranker.rank(raw_results.copy(), query))

            render_comparison_view(results_fuzzy, results_portal, results_bm25, query)

            st.markdown("---")
            st.markdown("### Fuzzy HCIR Results")
            for result in results_fuzzy[:display_limit]:
                render_result_card(result, settings, query, "Compare All")
            return

        if ranking_method == "Fuzzy HCIR (Research)":
            results = deduplicate_ranked_results(
                fuzzy_ranker.rank(
                    raw_results,
                    query,
                    factor_weights=settings.get("factor_weights"),
                )
            )

            results = [r for r in results if r and hasattr(r, "title") and r.title]

            if settings.get("show_explanations") and len(results) >= 3:
                with st.expander("Visual explanation: compare top 3", expanded=False):
                    fig = build_top3_radar_figure(results[:3])
                    st.plotly_chart(fig, use_container_width=True)

            page_size = display_limit
            fq_sig = fq or ""
            weights_sig = json.dumps(settings.get("factor_weights", {}), sort_keys=True)
            signature = hashlib.md5(
                (query + "|" + str(ranking_method) + "|" + fq_sig + "|" + str(page_size) + "|" + weights_sig).encode(
                    "utf-8"
                )
            ).hexdigest()

            if st.session_state.get("pagination_signature") != signature:
                st.session_state["pagination_signature"] = signature
                st.session_state["page_index"] = 0

            total_pages = max(1, math.ceil(len(results) / page_size))
            page_index = int(st.session_state.get("page_index", 0))
            page_index = max(0, min(page_index, total_pages - 1))
            st.session_state["page_index"] = page_index

            start = page_index * page_size
            end = start + page_size
            page_results = results[start:end]

            # Enhanced pagination header
            render_pagination_summary(total_count, len(page_results), page_index, total_pages)

            # Top pagination bar
            should_prev, should_next, new_page = render_pagination_controls(
                page_index, total_pages, key_suffix="top", show_page_jump=True
            )
            if should_prev or should_next or new_page != page_index:
                st.session_state["page_index"] = new_page
                st.rerun()

            st.markdown("---")

            for result in page_results:
                render_result_card(result, settings, query, str(ranking_method or "Fuzzy HCIR"))
            
            # Bottom pagination bar
            st.markdown("---")
            should_prev, should_next, new_page = render_pagination_controls(
                page_index, total_pages, key_suffix="bottom", show_page_jump=total_pages <= 20
            )
            if should_prev or should_next or new_page != page_index:
                st.session_state["page_index"] = new_page
                st.rerun()
            
            return

        if ranking_method == "Portal Default":
            portal_results = deduplicate_display_datasets(portal_ranker.rank(raw_results, query))
            st.markdown(f"### Found {total_count} datasets (Portal Default Ranking)")

            for i, ds in enumerate(portal_results[:display_limit]):
                title = ds.get("title", {})
                title_str = normalize_text(title, "Untitled")

                safe_title = safe_html_text(title_str, "Untitled")
                st.markdown(f"**#{i + 1}** [{safe_title}](https://opendata.swiss/en/dataset/{ds.get('name', '')})")
            return

        if ranking_method == "BM25 Keyword":
            bm25_results = deduplicate_display_datasets(bm25_ranker.rank(raw_results, query))
            st.markdown(f"### Found {total_count} datasets (BM25 Keyword Ranking)")

            for i, ds in enumerate(bm25_results[:display_limit]):
                title = ds.get("title", {})
                title_str = normalize_text(title, "Untitled")

                safe_title = safe_html_text(title_str, "Untitled")
                st.markdown(f"**#{i + 1}** [{safe_title}](https://opendata.swiss/en/dataset/{ds.get('name', '')})")
            return

        st.error("Unknown ranking method.")


# Backwards-compatible re-exports (useful for notebooks/tests)
__all__ = [
    "OpenDataSwissClient",
    "FuzzyHCIRRanker",
    "MultilingualQueryProcessor",
    "PortalDefaultRanker",
    "BM25Ranker",
    "main",
]


if __name__ == "__main__":
    main()
