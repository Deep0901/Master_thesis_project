from __future__ import annotations

import hashlib
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    from code.prototype.feedback_logger import append_feedback_event, build_event
except Exception:  # Streamlit script-dir execution
    from feedback_logger import append_feedback_event, build_event

try:
    from code.prototype.ranking.models import DatasetResult, safe_html_text
except Exception:  # Streamlit script-dir execution
    from ranking.models import DatasetResult, safe_html_text

from .style import SWISS_GOV_CSS


def render_header() -> None:
    """Render Swiss government style header."""
    st.markdown(SWISS_GOV_CSS, unsafe_allow_html=True)

    st.markdown(
        """
    <div class="swiss-header">
        <div style="display: flex; align-items: center;">
            <div class="swiss-cross"></div>
            <div>
                <h1>🔍 opendata.swiss | Fuzzy Search</h1>
                <p>Research Prototype - Human-Centered Information Retrieval</p>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict:
    """Render sidebar with settings, filters, and advanced options."""
    with st.sidebar:
        st.markdown("### 🔧 Search Settings")

        data_source = st.radio(
            "Data Source",
            ["🌐 Live API (opendata.swiss)", "📦 Demo Data"],
            index=0,
        )

        st.markdown("---")

        st.markdown("### 📊 Ranking Method")
        ranking_method = st.selectbox(
            "Select ranking algorithm",
            ["Fuzzy HCIR (Research)", "Portal Default", "BM25 Keyword", "Compare All"],
            index=0,
        )

        st.markdown("---")

        st.markdown("### 🏷️ Filters")

        themes = [
            "All Themes",
            "environment",
            "mobility",
            "economy",
            "population",
            "health",
            "energy",
            "education",
            "agriculture",
            "geography",
            "culture",
            "politics",
            "crime",
            "construction",
            "finances",
        ]
        selected_theme = st.selectbox("Theme", themes)

        languages = ["All", "English", "German", "French", "Italian"]
        selected_lang = st.selectbox("Interface Language", languages)

        st.markdown("---")

        st.markdown("### 🎯 Faceted Search")
        st.caption("Narrow down results before ranking is applied")

        org_options = st.session_state.get("facet_org_options", [])
        fmt_options = st.session_state.get(
            "facet_format_options",
            ["CSV", "GEOJSON", "JSON", "XML", "XLSX", "PDF", "API", "ZIP"],
        )
        license_options = st.session_state.get("facet_license_options", [])

        with st.expander("🏢 **Organizations**", expanded=False):
            selected_orgs = st.multiselect(
                "Filter by organization",
                options=org_options,
                default=[],
                help="Show only datasets from selected organizations.",
                key="orgs_filter",
                label_visibility="collapsed",
            )
        
        if not selected_orgs:
            selected_orgs = []

        with st.expander("📄 **Data Formats**", expanded=False):
            selected_formats = st.multiselect(
                "Filter by data format",
                options=fmt_options,
                default=[],
                help="Show only datasets with selected formats (e.g., CSV, GeoJSON, JSON).",
                key="formats_filter",
                label_visibility="collapsed",
            )
        
        if not selected_formats:
            selected_formats = []

        with st.expander("⚖️ **Licenses**", expanded=False):
            selected_licenses = st.multiselect(
                "Filter by license",
                options=license_options,
                default=[],
                help="Show only datasets with selected licenses.",
                key="licenses_filter",
                label_visibility="collapsed",
            )
        
        if not selected_licenses:
            selected_licenses = []

        # Display active filters summary
        active_filters = []
        if selected_orgs:
            active_filters.append(f"📊 Orgs: {', '.join(selected_orgs[:2])}" + (f" +{len(selected_orgs)-2}" if len(selected_orgs) > 2 else ""))
        if selected_formats:
            active_filters.append(f"📋 Formats: {', '.join(selected_formats[:2])}" + (f" +{len(selected_formats)-2}" if len(selected_formats) > 2 else ""))
        if selected_licenses:
            active_filters.append(f"⚖️ Licenses: {', '.join(selected_licenses[:2])}" + (f" +{len(selected_licenses)-2}" if len(selected_licenses) > 2 else ""))
        
        if active_filters:
            st.info("🔍 Active filters:\n" + "\n".join(active_filters))

        st.markdown("---")

        with st.expander("⚙️ Advanced Options"):
            num_results = st.slider("Results per page", 10, 50, 20)
            show_explanations = st.checkbox("Show ranking explanations", value=True)
            show_factors = st.checkbox("Show factor breakdown", value=True)

            st.markdown("**Factor weights**")
            w_similarity = st.slider("Similarity", 0.0, 2.0, 1.0, 0.05)
            w_recency = st.slider("Recency", 0.0, 2.0, 1.0, 0.05)
            w_completeness = st.slider("Completeness", 0.0, 2.0, 1.0, 0.05)
            w_resources = st.slider("Resources", 0.0, 2.0, 1.0, 0.05)

        st.markdown("---")

        st.markdown(
            """
        ### 📖 About This Prototype

        This is a research prototype for the Master Thesis:

        *"Improving Access to Swiss Open Government Data through
        Fuzzy Human-Centered Information Retrieval"*

        **University of Fribourg**
        Human-IST Institute

        **Author:** Deep Shukla
        **Supervisor:** Janick Spycher
        **Examiner:** Prof. Dr. Edy Portmann
        """
        )

        return {
            "data_source": data_source,
            "ranking_method": ranking_method,
            "theme": selected_theme if selected_theme != "All Themes" else None,
            "language": selected_lang,
            "organizations": selected_orgs,
            "formats": selected_formats,
            "licenses": selected_licenses,
            "num_results": num_results if "num_results" in dir() else 20,
            "show_explanations": show_explanations if "show_explanations" in dir() else True,
            "show_factors": show_factors if "show_factors" in dir() else True,
            "factor_weights": {
                "recency": w_recency,
                "completeness": w_completeness,
                "resources": w_resources,
                "similarity": w_similarity,
            },
        }


def get_format_badge_class(fmt: str) -> str:
    fmt_upper = fmt.upper()
    format_classes = {
        "CSV": "format-csv",
        "JSON": "format-json",
        "XML": "format-xml",
        "PDF": "format-pdf",
        "API": "format-api",
        "GEOJSON": "format-geojson",
        "XLSX": "format-xlsx",
        "XLS": "format-xlsx",
    }
    return format_classes.get(fmt_upper, "format-default")


def render_result_card(result: DatasetResult, settings: Dict, query: str, ranking_method: str):
    if result is None:
        return
    if not hasattr(result, "title") or result.title is None:
        return

    score_pct = int(result.relevance_score * 100)
    if score_pct >= 70:
        relevance_class = "high-relevance"
        badge_class = "score-excellent"
    elif score_pct >= 50:
        relevance_class = "medium-relevance"
        badge_class = "score-good"
    elif score_pct >= 30:
        relevance_class = "medium-relevance"
        badge_class = "score-moderate"
    else:
        relevance_class = "low-relevance"
        badge_class = "score-low"

    raw_title = result.title
    if isinstance(raw_title, dict):
        raw_title = (
            raw_title.get("en")
            or raw_title.get("de")
            or raw_title.get("fr")
            or raw_title.get("it")
            or next(iter(raw_title.values()), "Untitled")
        )
    elif not raw_title or str(raw_title).strip() == "":
        raw_title = "Untitled"

    title = safe_html_text(raw_title, "Untitled")

    organization = safe_html_text(result.organization, "Unknown")
    description = safe_html_text(result.description)

    format_badges = " ".join(
        [
            f'<span class="format-badge {get_format_badge_class(fmt)}">{safe_html_text(fmt)}</span>'
            for fmt in result.format_list
        ]
    )

    tags_html = " ".join(
        [
            f'<span class="tag-pill">{safe_html_text(tag)}</span>'
            for tag in result.tags[:5]
            if tag is not None and str(tag).strip()
        ]
    )

    card_html = f"""
    <div class="result-card {relevance_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem; font-weight: 600; color: #333;">#{result.rank}</span>
                    <span class="score-badge {badge_class}">{score_pct}% Relevant</span>
                    {format_badges}
                </div>
                <h3 style="margin: 0 0 8px 0; font-size: 1.15rem;">
                    <a href="{result.url}" target="_blank" style="color: #006699; text-decoration: none;">
                        {title}
                    </a>
                </h3>
                <p style="color: #555; font-size: 0.9rem; margin: 0 0 10px 0; line-height: 1.5;">
                    {description[:250]}{'...' if len(description) > 250 else ''}
                </p>
                <div style="display: flex; flex-wrap: wrap; gap: 15px; font-size: 0.85rem; color: #666;">
                    <span class="org-badge">
                        <svg width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M8.707 1.5a1 1 0 0 0-1.414 0L.646 8.146a.5.5 0 0 0 .708.708L8 2.207l6.646 6.647a.5.5 0 0 0 .708-.708L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.707 1.5Z"/>
                            <path d="m8 3.293 6 6V13.5a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 13.5V9.293l6-6Z"/>
                        </svg>
                        {organization}
                    </span>
                    <span>📅 Modified: {result.days_since_modified} days ago</span>
                    <span>📁 {len(result.resources)} resources</span>
                </div>
                <div style="margin-top: 8px;">
                    {tags_html}
                </div>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    feedback_key = f"fb:{ranking_method}:{hashlib.md5((query + '|' + result.id).encode('utf-8')).hexdigest()}"
    voted = st.session_state.get("feedback_votes", {}).get(feedback_key)

    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
    with fb_col1:
        if st.button("👍", key=f"{feedback_key}:up", disabled=bool(voted), help="Was this result helpful?"):
            st.session_state.setdefault("feedback_votes", {})[feedback_key] = True
            event = build_event(
                query=query,
                dataset_id=result.id,
                rank=int(result.rank or 0),
                helpful=True,
                ranking_method=ranking_method,
                data_source=str(settings.get("data_source", "")),
                relevance_score=float(result.relevance_score or 0.0),
                metadata_scores={
                    "recency": float(getattr(result.factors, "recency_score", 0.0)) if result.factors else 0.0,
                    "completeness": float(getattr(result.factors, "completeness_score", 0.0)) if result.factors else 0.0,
                    "resources": float(getattr(result.factors, "resource_score", 0.0)) if result.factors else 0.0,
                    "similarity": float(getattr(result.factors, "similarity_score", 0.0)) if result.factors else 0.0,
                },
                extra={"license_id": result.license, "organization": result.organization},
            )
            path = append_feedback_event(event)
            st.toast(f"Feedback saved to {path}")
    with fb_col2:
        if st.button("👎", key=f"{feedback_key}:down", disabled=bool(voted), help="Was this result not helpful?"):
            st.session_state.setdefault("feedback_votes", {})[feedback_key] = True
            event = build_event(
                query=query,
                dataset_id=result.id,
                rank=int(result.rank or 0),
                helpful=False,
                ranking_method=ranking_method,
                data_source=str(settings.get("data_source", "")),
                relevance_score=float(result.relevance_score or 0.0),
                metadata_scores={
                    "recency": float(getattr(result.factors, "recency_score", 0.0)) if result.factors else 0.0,
                    "completeness": float(getattr(result.factors, "completeness_score", 0.0)) if result.factors else 0.0,
                    "resources": float(getattr(result.factors, "resource_score", 0.0)) if result.factors else 0.0,
                    "similarity": float(getattr(result.factors, "similarity_score", 0.0)) if result.factors else 0.0,
                },
                extra={"license_id": result.license, "organization": result.organization},
            )
            path = append_feedback_event(event)
            st.toast(f"Feedback saved to {path}")
    with fb_col3:
        st.caption("Was this result helpful? (👍 / 👎)")

    explanation_text = (result.explanation or "").strip()
    if settings.get("show_explanations") and result.factors and explanation_text and explanation_text.lower() != "none":
        with st.expander("📊 Why this ranking?", expanded=False):
            # Import the radar chart builder
            try:
                from code.prototype.visual_explanations import build_individual_factor_radar
            except Exception:  # Streamlit runs with script dir on sys.path
                from visual_explanations import build_individual_factor_radar

            # Show individual radar chart in Swiss colors
            radar_fig = build_individual_factor_radar(result, title=f"Rank #{result.rank} - Factor Profile", color="#006699")
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Factor Scores:**")
                st.markdown(f"🎯 **Similarity:** {result.factors.similarity_term}")
                st.progress(result.factors.similarity_score)

                st.markdown(f"📅 **Recency:** {result.factors.recency_term}")
                st.progress(result.factors.recency_score)

            with col2:
                st.markdown(f"📋 **Completeness:** {result.factors.completeness_term}")
                st.progress(result.factors.completeness_score)

                st.markdown(f"📁 **Resources:** {result.factors.resource_term}")
                st.progress(result.factors.resource_score)

            st.markdown("---")
            st.markdown(explanation_text)


def render_comparison_view(
    results_fuzzy: List[DatasetResult],
    results_portal: List[Dict],
    results_bm25: List[Dict],
    query: str,
) -> None:
    st.markdown("### 📊 Ranking Comparison")
    st.markdown("Compare how different algorithms rank the same datasets.")

    comparison_data = []

    fuzzy_ranks = {r.id: r.rank for r in results_fuzzy}
    portal_ranks = {r.get("name", ""): i + 1 for i, r in enumerate(results_portal)}
    bm25_ranks = {r.get("name", ""): i + 1 for i, r in enumerate(results_bm25)}

    for result in results_fuzzy[:10]:
        title = result.title.get("en") or result.title.get("de") or "Untitled"
        comparison_data.append(
            {
                "Dataset": title[:50] + ("..." if len(title) > 50 else ""),
                "Fuzzy HCIR": fuzzy_ranks.get(result.id, "-"),
                "Portal Default": portal_ranks.get(result.id, "-"),
                "BM25": bm25_ranks.get(result.id, "-"),
                "Fuzzy Score": f"{int(result.relevance_score * 100)}%",
            }
        )

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    st.markdown("### 📈 Method Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="color: #4A9F35;">Fuzzy HCIR</div>
            <div class="metric-label">Handles vague queries<br/>Explainable rankings<br/>Quality-aware</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="color: #006699;">Portal Default</div>
            <div class="metric-label">Solr-based scoring<br/>Optimized for keywords<br/>Fast retrieval</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="color: #F29400;">BM25</div>
            <div class="metric-label">Probabilistic model<br/>Term frequency based<br/>Document length normalized</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
