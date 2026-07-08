"""Visual explainability helpers (radar/spider charts)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def build_top3_radar_figure(results: Sequence, title: str = "Top 3 factor comparison", color_scheme: str = "swiss"):
    """Build a Plotly radar chart for the top 3 results.

    `results` items are expected to have `.factors` with 4 factor scores:
    recency_score, completeness_score, resource_score, similarity_score.
    
    Args:
        results: Sequence of result objects with .factors and .rank attributes.
        title: Chart title.
        color_scheme: Color scheme ("swiss" for Swiss government blue/green, or "default").
    """
    import plotly.graph_objects as go

    categories = ["Recency", "Completeness", "Resources", "Similarity"]

    # Swiss government color palette
    colors_swiss = ["#006699", "#4A9F35", "#E32C1F"]  # Blue, Green, Red-Orange
    colors_default = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors = colors_swiss if color_scheme == "swiss" else colors_default

    fig = go.Figure()

    for idx, r in enumerate(list(results)[:3]):
        if not getattr(r, "factors", None):
            continue
        values = [
            _clamp01(getattr(r.factors, "recency_score", 0.0)),
            _clamp01(getattr(r.factors, "completeness_score", 0.0)),
            _clamp01(getattr(r.factors, "resource_score", 0.0)),
            _clamp01(getattr(r.factors, "similarity_score", 0.0)),
        ]
        # Close the loop
        values_loop = values + [values[0]]
        categories_loop = categories + [categories[0]]

        label = f"#{getattr(r, 'rank', '?')}"
        fig.add_trace(
            go.Scatterpolar(
                r=values_loop,
                theta=categories_loop,
                fill="toself",
                name=label,
                line=dict(color=colors[idx % len(colors)]),
                fillcolor=colors[idx % len(colors)],
                opacity=0.6,
                hovertemplate="%{theta}: %{r:.2f}<extra>" + label + "</extra>",
            )
        )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
            ),
        ),
        showlegend=True,
        hovermode="closest",
        margin=dict(l=50, r=50, t=80, b=50),
        height=420,
        font=dict(family="Arial, sans-serif", size=11),
    )

    return fig


def build_individual_factor_radar(result, title: str = "Factor Breakdown", color: str = "#006699"):
    """Build a single radar chart for a specific result's factors.
    
    Useful for detailed inspection of one dataset's scoring profile.
    """
    import plotly.graph_objects as go

    if not getattr(result, "factors", None):
        return None

    categories = ["Recency", "Completeness", "Resources", "Similarity"]
    values = [
        _clamp01(getattr(result.factors, "recency_score", 0.0)),
        _clamp01(getattr(result.factors, "completeness_score", 0.0)),
        _clamp01(getattr(result.factors, "resource_score", 0.0)),
        _clamp01(getattr(result.factors, "similarity_score", 0.0)),
    ]
    values_loop = values + [values[0]]
    categories_loop = categories + [categories[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_loop,
            theta=categories_loop,
            fill="toself",
            line=dict(color=color),
            fillcolor=color,
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
            ),
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=50, r=50, t=80, b=50),
        height=380,
        font=dict(family="Arial, sans-serif", size=11),
    )

    return fig
