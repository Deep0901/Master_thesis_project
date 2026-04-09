#!/usr/bin/env python3
"""
Swiss OGD Analytics Dashboard

Research analytics dashboard for analyzing prototype performance,
portal statistics, and evaluation metrics.

Run: streamlit run code/prototype/analytics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import time

# Page config
st.set_page_config(
    page_title="Swiss OGD Analytics | Research Dashboard",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
}
.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}
.chart-container {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_portal_statistics():
    """Load portal statistics from opendata.swiss."""
    try:
        response = requests.get(
            "https://opendata.swiss/api/3/action/package_search",
            params={"rows": 0},
            timeout=10
        )
        data = response.json()
        total_datasets = data['result']['count']
        
        # Get theme counts
        response = requests.get(
            "https://opendata.swiss/api/3/action/group_list",
            params={"all_fields": True},
            timeout=10
        )
        themes = response.json()['result'] if response.json()['success'] else []
        
        # Get organization counts
        response = requests.get(
            "https://opendata.swiss/api/3/action/organization_list",
            params={"all_fields": True},
            timeout=10
        )
        orgs = response.json()['result'] if response.json()['success'] else []
        
        return {
            "total_datasets": total_datasets,
            "themes": themes,
            "organizations": orgs
        }
    except Exception as e:
        st.warning(f"Could not fetch live data: {e}")
        return {
            "total_datasets": 14254,
            "themes": [],
            "organizations": []
        }


@st.cache_data(ttl=3600)
def load_sample_metadata(n: int = 100):
    """Load sample dataset metadata for analysis."""
    try:
        response = requests.get(
            "https://opendata.swiss/api/3/action/package_search",
            params={"rows": n, "sort": "metadata_modified desc"},
            timeout=30
        )
        if response.json()['success']:
            return response.json()['result']['results']
        return []
    except:
        return []


def load_evaluation_results():
    """Load evaluation results if available."""
    eval_path = "evaluation/results"
    results = {}
    
    # Try to load experiment results
    exp_file = os.path.join(eval_path, "experiment_results.json")
    if os.path.exists(exp_file):
        with open(exp_file) as f:
            results['experiment'] = json.load(f)
    
    return results


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_recency(datasets: List[Dict]) -> pd.DataFrame:
    """Analyze recency distribution."""
    recency_data = []
    now = datetime.now()
    
    for ds in datasets:
        try:
            mod_str = ds.get('metadata_modified', '')
            if mod_str:
                mod_date = datetime.fromisoformat(mod_str.replace('Z', '+00:00'))
                days = (now - mod_date.replace(tzinfo=None)).days
                recency_data.append({
                    'dataset': ds.get('name', ''),
                    'days_since_modified': days,
                    'year': mod_date.year
                })
        except:
            pass
    
    return pd.DataFrame(recency_data)


def analyze_completeness(datasets: List[Dict]) -> pd.DataFrame:
    """Analyze metadata completeness."""
    completeness_data = []
    
    fields_to_check = ['title', 'description', 'resources', 'tags', 
                       'groups', 'organization', 'license_id']
    
    for ds in datasets:
        filled = sum(1 for f in fields_to_check if ds.get(f))
        completeness = filled / len(fields_to_check)
        
        # Resource count
        resources = ds.get('resources', [])
        res_count = len(resources) if resources else 0
        
        # Format diversity
        formats = set()
        for r in resources:
            fmt = r.get('format', '').upper()
            if fmt:
                formats.add(fmt)
        
        completeness_data.append({
            'dataset': ds.get('name', ''),
            'completeness': completeness,
            'resource_count': res_count,
            'format_diversity': len(formats)
        })
    
    return pd.DataFrame(completeness_data)


def compute_fuzzy_distribution(values: List[float], variable: str) -> Dict[str, float]:
    """Compute fuzzy term distribution for values."""
    from code.fuzzy_system.production_engine import CalibratedOGDVariables
    
    if variable == 'recency':
        var = CalibratedOGDVariables.create_recency_variable()
    elif variable == 'completeness':
        var = CalibratedOGDVariables.create_completeness_variable()
    elif variable == 'resources':
        var = CalibratedOGDVariables.create_resources_variable()
    else:
        return {}
    
    term_counts = {term: 0 for term in var.terms}
    
    for v in values:
        memberships = var.fuzzify(v)
        dominant = max(memberships, key=memberships.get)
        term_counts[dominant] += 1
    
    total = len(values)
    return {k: v/total for k, v in term_counts.items()}


# ============================================================================
# DASHBOARD PAGES
# ============================================================================

def render_overview_page():
    """Render the overview dashboard page."""
    st.title("📊 Swiss OGD Analytics Dashboard")
    st.markdown("Real-time analytics for opendata.swiss portal and fuzzy ranking research")
    
    # Load data
    stats = load_portal_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['total_datasets']:,}</div>
            <div class="metric-label">Total Datasets</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{len(stats['organizations'])}</div>
            <div class="metric-label">Organizations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{len(stats['themes'])}</div>
            <div class="metric-label">Themes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">4</div>
            <div class="metric-label">Languages</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Theme distribution
    if stats['themes']:
        st.subheader("📁 Theme Distribution")
        theme_data = []
        for theme in stats['themes']:
            theme_data.append({
                'Theme': theme.get('display_name', theme.get('name', '')),
                'Datasets': theme.get('package_count', 0)
            })
        df_themes = pd.DataFrame(theme_data).sort_values('Datasets', ascending=True)
        
        fig = px.bar(df_themes, x='Datasets', y='Theme', orientation='h',
                    color='Datasets', color_continuous_scale='viridis')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top organizations
    if stats['organizations']:
        st.subheader("🏛️ Top Publishing Organizations")
        org_data = []
        for org in stats['organizations'][:20]:
            org_data.append({
                'Organization': org.get('title', org.get('name', '')),
                'Datasets': org.get('package_count', 0)
            })
        df_orgs = pd.DataFrame(org_data).sort_values('Datasets', ascending=False).head(10)
        
        fig = px.bar(df_orgs, x='Organization', y='Datasets',
                    color='Datasets', color_continuous_scale='plasma')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_metadata_analysis_page():
    """Render metadata quality analysis page."""
    st.title("📋 Metadata Quality Analysis")
    st.markdown("Analysis of dataset metadata completeness and quality distribution")
    
    # Load sample data
    with st.spinner("Loading sample datasets..."):
        n_samples = st.sidebar.slider("Sample Size", 50, 500, 100)
        datasets = load_sample_metadata(n_samples)
    
    if not datasets:
        st.warning("Could not load dataset samples")
        return
    
    st.success(f"Loaded {len(datasets)} datasets for analysis")
    
    # Analyze recency
    st.subheader("📅 Recency Distribution")
    recency_df = analyze_recency(datasets)
    
    if not recency_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(recency_df, x='days_since_modified', nbins=30,
                             title="Days Since Last Modification")
            fig.update_layout(xaxis_title="Days", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics
            st.markdown("**Recency Statistics:**")
            st.write(f"- Mean: {recency_df['days_since_modified'].mean():.0f} days")
            st.write(f"- Median: {recency_df['days_since_modified'].median():.0f} days")
            st.write(f"- Std Dev: {recency_df['days_since_modified'].std():.0f} days")
            st.write(f"- Min: {recency_df['days_since_modified'].min()} days")
            st.write(f"- Max: {recency_df['days_since_modified'].max()} days")
            
            # Percentiles
            st.markdown("**Percentiles:**")
            for p in [10, 25, 50, 75, 90]:
                val = recency_df['days_since_modified'].quantile(p/100)
                st.write(f"- P{p}: {val:.0f} days")
        
        # Year distribution
        st.subheader("📆 Updates by Year")
        yearly = recency_df.groupby('year').size().reset_index(name='count')
        fig = px.bar(yearly, x='year', y='count', title="Dataset Updates by Year")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyze completeness
    st.subheader("📊 Completeness Analysis")
    completeness_df = analyze_completeness(datasets)
    
    if not completeness_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(completeness_df, x='completeness', nbins=20,
                             title="Metadata Completeness Distribution")
            fig.add_vline(x=completeness_df['completeness'].mean(), 
                         line_dash="dash", line_color="red",
                         annotation_text="Mean")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(completeness_df, x='resource_count', nbins=20,
                             title="Resource Count Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        st.subheader("🔗 Resource Count vs Completeness")
        fig = px.scatter(completeness_df, x='resource_count', y='completeness',
                        color='format_diversity', size='format_diversity',
                        title="Resources vs Completeness (colored by format diversity)")
        st.plotly_chart(fig, use_container_width=True)


def render_fuzzy_analysis_page():
    """Render fuzzy system analysis page."""
    st.title("🧠 Fuzzy System Analysis")
    st.markdown("Analyze fuzzy membership function calibration and inference behavior")
    
    # Import fuzzy components
    try:
        from code.fuzzy_system.production_engine import (
            CalibratedOGDVariables, create_ogd_fuzzy_engine
        )
    except ImportError:
        st.error("Could not import fuzzy engine. Run from thesis root directory.")
        return
    
    # Create variables
    recency_var = CalibratedOGDVariables.create_recency_variable()
    completeness_var = CalibratedOGDVariables.create_completeness_variable()
    resources_var = CalibratedOGDVariables.create_resources_variable()
    similarity_var = CalibratedOGDVariables.create_similarity_variable()
    
    # Plot membership functions
    st.subheader("📈 Membership Function Visualization")
    
    variable_choice = st.selectbox(
        "Select Variable",
        ["Recency (days)", "Completeness (ratio)", "Resources (count)", "Similarity (score)"]
    )
    
    if "Recency" in variable_choice:
        var = recency_var
        x_range = np.linspace(0, 4500, 500)
        x_label = "Days Since Modification"
    elif "Completeness" in variable_choice:
        var = completeness_var
        x_range = np.linspace(0, 1, 500)
        x_label = "Completeness Ratio"
    elif "Resources" in variable_choice:
        var = resources_var
        x_range = np.linspace(0, 50, 500)
        x_label = "Number of Resources"
    else:
        var = similarity_var
        x_range = np.linspace(0, 1, 500)
        x_label = "Similarity Score"
    
    # Create membership function plot
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y_values = [mf.evaluate(x) for x in x_range]
        fig.add_trace(go.Scatter(
            x=x_range, y=y_values,
            mode='lines',
            name=term,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=f"Membership Functions: {var.name}",
        xaxis_title=x_label,
        yaxis_title="Membership Degree",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive inference
    st.subheader("🔮 Interactive Fuzzy Inference")
    st.markdown("Adjust inputs to see how the fuzzy system computes relevance scores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_recency = st.slider("Recency (days)", 0, 3650, 100)
        input_completeness = st.slider("Completeness", 0.0, 1.0, 0.75)
    
    with col2:
        input_resources = st.slider("Resource Count", 0, 20, 4)
        input_similarity = st.slider("Similarity", 0.0, 1.0, 0.7)
    
    # Run inference
    engine = create_ogd_fuzzy_engine()
    result = engine.infer({
        "recency": input_recency,
        "completeness": input_completeness,
        "resources": input_resources,
        "similarity": input_similarity
    })
    
    # Display result
    st.markdown("### Inference Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_pct = int(result.crisp_output * 100)
        if score_pct >= 70:
            color = "#4CAF50"
        elif score_pct >= 50:
            color = "#2196F3"
        elif score_pct >= 30:
            color = "#FF9800"
        else:
            color = "#f44336"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: {color}; border-radius: 10px; color: white;">
            <div style="font-size: 3rem; font-weight: bold;">{score_pct}%</div>
            <div>Relevance Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Dominant Linguistic Terms:**")
        for var_name, term, degree in result.dominant_factors:
            st.write(f"- {var_name}: **{term}** ({degree:.2f})")
    
    with col3:
        st.markdown("**Active Rules:**")
        st.write(f"Total: {len(result.active_rules)}")
        for rule_act in result.get_top_rules(3):
            st.write(f"- Rule {rule_act.rule.id}: {rule_act.firing_strength:.2f}")


def render_evaluation_page():
    """Render evaluation results page."""
    st.title("📈 Evaluation Results")
    st.markdown("Analysis of ranking system evaluation experiments")
    
    # Check for evaluation results
    eval_results = load_evaluation_results()
    
    if not eval_results:
        st.info("No evaluation results found. Run the evaluation experiment first.")
        
        st.markdown("""
        ### How to Generate Results
        
        Run the experiment runner:
        ```bash
        python -m evaluation.experiment_runner
        ```
        
        This will:
        1. Load benchmark queries
        2. Run all ranking systems
        3. Compute IR metrics
        4. Save results to `evaluation/results/`
        """)
        
        # Show placeholder metrics
        st.subheader("📊 Expected Metrics")
        
        placeholder_data = {
            'System': ['Portal Default', 'BM25', 'Metadata Quality', 'Fuzzy HCIR'],
            'MAP': [0.988, 0.799, 0.982, 0.837],
            'nDCG@10': [0.813, 0.739, 0.763, 0.580],
            'P@10': [0.993, 0.867, 0.993, 0.713],
            'MRR': [1.000, 0.950, 1.000, 0.867]
        }
        
        df = pd.DataFrame(placeholder_data)
        
        # Metrics comparison chart
        fig = px.bar(df, x='System', y=['MAP', 'nDCG@10', 'P@10'],
                    title="Ranking System Performance Comparison",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df.set_index('System'))
        return
    
    # Display actual results
    if 'experiment' in eval_results:
        exp = eval_results['experiment']
        
        st.subheader("📊 System Performance Metrics")
        
        # Create comparison dataframe
        systems = list(exp.get('results', {}).keys())
        metrics_data = []
        
        for sys in systems:
            sys_results = exp['results'][sys]
            metrics_data.append({
                'System': sys,
                'MAP': sys_results.get('map', 0),
                'nDCG@10': sys_results.get('ndcg@10', 0),
                'P@10': sys_results.get('p@10', 0),
                'MRR': sys_results.get('mrr', 0)
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Bar chart
        fig = px.bar(df, x='System', y=['MAP', 'nDCG@10', 'P@10'],
                    title="Ranking System Performance",
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(df.set_index('System').style.highlight_max(axis=0))


def render_about_page():
    """Render about page."""
    st.title("ℹ️ About This Research")
    
    st.markdown("""
    ## Improving Access to Swiss Open Government Data through Fuzzy Human-Centered Information Retrieval
    
    ### Research Context
    
    This dashboard is part of a Master Thesis project at the **University of Fribourg, Human-IST Institute**.
    
    ### Research Questions
    
    1. **RQ1:** How can fuzzy logic effectively model vague human query intent and quality preferences?
    2. **RQ2:** How does the fuzzy approach compare to keyword-based baselines in terms of retrieval effectiveness?
    3. **RQ3:** How do explanations affect user trust and satisfaction with ranking results?
    4. **RQ4:** What advantages does the fuzzy logic approach offer over emerging AI-based semantic search?
    
    ### Methodology
    
    This research follows **Design Science Research (DSR)** methodology:
    - Problem identification → Swiss OGD accessibility challenges
    - Solution design → Fuzzy ranking with explainability
    - Development → Prototype implementation
    - Evaluation → Automated metrics + User study
    - Communication → Thesis + Publications
    
    ### Key Components
    
    - **Fuzzy Inference Engine:** Mamdani fuzzy system with calibrated membership functions
    - **Data-Driven Calibration:** Parameters derived from actual opendata.swiss statistics
    - **Explainable Rankings:** Human-readable explanations for every result
    - **Comparison Framework:** Portal default, BM25, and fuzzy approaches
    
    ### Team
    
    - **Author:** Deep Shukla
    - **Supervisor:** Janick Spycher
    - **Examiner:** Prof. Dr. Edy Portmann
    
    ### Timeline
    
    March 2026 - August 2026 (Submission deadline: September 9, 2026)
    
    ### Contact
    
    - Email: deep.shukla@unifr.ch
    - Institution: Human-IST Institute, University of Fribourg
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main dashboard application."""
    # Sidebar navigation
    st.sidebar.title("🇨🇭 Swiss OGD Research")
    
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Overview", "📋 Metadata Analysis", "🧠 Fuzzy System", 
         "📈 Evaluation", "ℹ️ About"]
    )
    
    # Render selected page
    if page == "📊 Overview":
        render_overview_page()
    elif page == "📋 Metadata Analysis":
        render_metadata_analysis_page()
    elif page == "🧠 Fuzzy System":
        render_fuzzy_analysis_page()
    elif page == "📈 Evaluation":
        render_evaluation_page()
    else:
        render_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <small>
    **University of Fribourg**<br>
    Human-IST Institute<br>
    Master Thesis 2026
    </small>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
