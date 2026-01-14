"""Chart visualizations for cause areas and method comparison.

This module provides:
- Bar charts for cause area frequency
- Comparison charts for NLP vs LLM extraction
- Pie charts for category breakdown
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_cause_area_bar_chart(
    cause_areas: list[str],
    mention_counts: dict[str, int],
    top_n: int = 25,
    title: str = "Most Mentioned Cause Areas",
    color: str = "#2E86AB",
) -> go.Figure:
    """Create horizontal bar chart of cause area mentions.

    Args:
        cause_areas: List of cause areas (sorted by count)
        mention_counts: Dict mapping cause area to count
        top_n: Number of items to show
        title: Chart title
        color: Bar color
    Returns:
        Plotly Figure
    """
    # Get top N
    top_areas = cause_areas[:top_n]
    counts = [mention_counts.get(area, 0) for area in top_areas]
    # order by count
    top_areas, counts = zip(*sorted(zip(top_areas, counts), key=lambda x: x[1], reverse=False))
    # # Reverse for horizontal display (highest at top)
    # top_areas = top_areas[::-1]
    # counts = counts[::-1]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=top_areas,
            orientation="h",
            marker_color=color,
            text=counts,
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Number of Mentions",
        yaxis_title="",
        height=max(400, top_n * 25),
        margin=dict(l=250, r=50, t=50, b=50),
        showlegend=False,
    )

    return fig


def create_extraction_comparison_chart(
    comparison_df: pd.DataFrame,
    title: str = "NLP vs LLM Extraction Comparison",
) -> go.Figure:
    """Create chart comparing NLP and LLM extraction totals.

    Args:
        comparison_df: DataFrame with category, nlp_total, llm_total, overlap columns
        title: Chart title
    Returns:
        Plotly Figure
    """
    categories = comparison_df["category"].tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="NLP (spaCy)",
            x=categories,
            y=comparison_df["nlp_total"],
            marker_color="#2E86AB",
        )
    )

    fig.add_trace(
        go.Bar(
            name="LLM (Ollama)",
            x=categories,
            y=comparison_df["llm_total"],
            marker_color="#E85D04",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Overlap",
            x=categories,
            y=comparison_df["overlap"],
            marker_color="#52B788",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Total Extractions",
        height=400,
        legend=dict(x=0.7, y=0.95),
    )

    return fig


def create_comparison_scatter(
    comparison_df: pd.DataFrame,
    category: str = "cause_areas",
) -> go.Figure:
    """Create scatter plot comparing NLP vs LLM counts per row.

    Args:
        comparison_df: Per-row comparison results
        category: Which category to plot
    Returns:
        Plotly Figure
    """
    nlp_col = f"{category}_nlp" if f"{category}_nlp" in comparison_df.columns else f"nlp_{category}"
    llm_col = f"{category}_llm" if f"{category}_llm" in comparison_df.columns else f"llm_{category}"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comparison_df[nlp_col],
            y=comparison_df[llm_col],
            mode="markers",
            marker=dict(
                size=8,
                color="#2E86AB",
                opacity=0.6,
            ),
            hovertemplate=("NLP: %{x}<br>LLM: %{y}<br><extra></extra>"),
        )
    )

    # Add diagonal reference line
    max_val = max(comparison_df[nlp_col].max(), comparison_df[llm_col].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Equal",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{category.replace('_', ' ').title()}: NLP vs LLM Counts",
        xaxis_title="NLP Extractions",
        yaxis_title="LLM Extractions",
        height=400,
        width=500,
    )

    return fig


def create_category_breakdown_pie(
    category_counts: dict[str, int],
    title: str = "Cause Area Categories",
) -> go.Figure:
    """Create pie chart of category breakdown.

    Args:
        category_counts: Dict mapping category to count
        title: Chart title
    Returns:
        Plotly Figure
    """
    labels = list(category_counts.keys())
    values = list(category_counts.values())

    colors = px.colors.qualitative.Set2[: len(labels)]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo="percent+label",
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        legend=dict(x=1.05, y=0.5),
        height=500,
    )

    return fig


def _create_mentions_df(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        pd.concat(
            [
                df["expertise_parsed"].explode().value_counts().rename("expertise"),
                df["interests_parsed"].explode().value_counts().rename("interest"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"index": "keyword"})
    )
    df["ratio"] = df["interest"] / df["expertise"]
    df["total"] = df["expertise"] + df["interest"]
    return df


def create_expertise_vs_interest_chart(
    df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Create comparison of expertise vs interest areas.

    Args:
        expertise_counts: Dict of expertise keyword counts
        interest_counts: Dict of interest keyword counts
        top_n: Number of items to show
    Returns:
        Plotly Figure
    """
    # Get all keywords
    keyword_df = _create_mentions_df(df)

    keyword_df["total"] = keyword_df["expertise"] + keyword_df["interest"]
    keyword_df = keyword_df.nlargest(top_n, "total").sort_values("expertise", ascending=True)

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Bar(
            y=keyword_df["keyword"].str.title(),
            x=keyword_df["expertise"],
            name="Expertise",
            orientation="h",
            marker_color="#E85D04",
        )
    )

    fig.add_trace(
        go.Bar(
            y=keyword_df["keyword"].str.title(),
            x=keyword_df["interest"],
            name="Interest",
            orientation="h",
            marker_color="#0077B6",
        )
    )

    fig.update_layout(
        title="Expertise vs Interest Areas",
        barmode="group",
        height=max(400, top_n * 30),
        xaxis_title="Count",
        legend=dict(x=0.79, y=0.02),
        margin=dict(l=200, pad=10),
    )

    return fig


def create_undervalued_chart(
    df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Create chart of undervalued areas (high interest, low expertise).

    Args:
        expertise_counts: Dict of expertise counts
        interest_counts: Dict of interest counts
        min_expertise: Minimum expertise to include
    Returns:
        Plotly Figure showing interest/expertise ratio
    """
    keyword_df = _create_mentions_df(df)
    median_ratio = keyword_df["ratio"].median()
    display_df = keyword_df.sort_values("ratio", ascending=False)
    display_df = display_df.nlargest(top_n, "ratio")[::-1]

    colors = ["#DC2F02" if r > median_ratio else "#023E8A" for r in display_df["ratio"]]

    fig = go.Figure(
        go.Bar(
            x=display_df["ratio"],
            y=display_df["keyword"].str.title(),
            name="Value",
            marker_color=colors,
            orientation="h",
            showlegend=False,
            text=display_df["total"],  # show interest counts on bars
            textposition="outside",
        )
    )

    # add annotation text stating that numbers printed on the bars are the total number of mentions (between expertise and interest)
    # fig.update_layout(
    #     title=dict(
    #         text="Undervalued Areas (Interest/Expertise Ratio > 1)",
    #         x=0.5,
    #         xanchor="center",
    #         yanchor="top",
    #         font=dict(size=20),
    #         # subtitle is set below
    #     ),
    #     xaxis_title="Interest/Expertise Ratio",
    #     yaxis_title="",
    #     height=max(400, top_n * 25),
    #     margin=dict(l=200, pad=10),
    #     xaxis=dict(rangemode="tozero"),
    #     legend=dict(x=0.64, y=0.02),
    #     # Add subtitle
    #     titlefont=dict(size=20),
    # )
    # fig.update_layout(
    #     title={
    #         "text": "Undervalued Areas (Interest/Expertise Ratio > 1)<br><span style='font-size:14px;font-weight:normal'>Numbers printed on the bars are the total number of mentions (between expertise and interest)</span>",
    #         "x": 0.5,
    #         "xanchor": "center",
    #     }
    # )

    fig.add_vline(
        x=median_ratio,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"<b>Median:</b> {median_ratio:.2f}",
        annotation_position="top",
    )

    # Add color legends for "Undervalued" (red) and "Saturated" (blue)
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker_color="#DC2F02",
            name="Undervalued (red)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            marker_color="#023E8A",
            name="Saturated (blue)",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text="Undervalued Areas (Interest/Expertise Ratio > 1)<br><span style='font-size:14px;font-weight:normal'>Numbers printed on the bars are the total number of mentions (between expertise and interest)</span>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Interest/Expertise Ratio",
        yaxis_title="",
        height=max(400, top_n * 25),
        margin=dict(l=200, pad=10),
        xaxis=dict(rangemode="tozero"),
        legend=dict(x=0.64, y=0.02),
    )

    return fig
