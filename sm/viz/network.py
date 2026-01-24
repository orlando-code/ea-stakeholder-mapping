"""Semantic network visualization for cause area clustering.

This module provides:
- 2D scatter plot showing cause areas positioned by semantic similarity
- Node size represents mention frequency
- Node color represents cluster membership
- Treemap and heatmap visualizations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sm.analysis.semantic import SemanticAnalysisResult

# Cluster color palette (visually distinct, colorblind-friendly)
CLUSTER_COLORS = [
    "#2E86AB",  # Blue
    "#E85D04",  # Orange
    "#52B788",  # Green
    "#9D4EDD",  # Purple
    "#F72585",  # Pink
    "#4CC9F0",  # Cyan
    "#FFBE0B",  # Yellow
    "#3A86FF",  # Light blue
    "#8338EC",  # Violet
    "#FF006E",  # Magenta
    "#06D6A0",  # Teal
    "#EF476F",  # Red
    "#faefaf",  # pale yellow
    "#aefaea",  # light blue
]


def create_semantic_network(
    result: SemanticAnalysisResult,
    min_mentions: int = 1,
    max_node_size: float = 50,
    min_node_size: float = 8,
    title: str = "Cause Area Semantic Landscape",
    show_labels: bool = True,
    showtitle: bool = True,
) -> go.Figure:
    """Create 2D semantic network visualization.

    Nodes are positioned by semantic similarity (using t-SNE reduction).
    Node size represents mention frequency.
    Node color represents cluster membership.

    Args:
        result: SemanticAnalysisResult from SemanticAnalyzer
        min_mentions: Minimum mentions to include a node
        max_node_size: Maximum node size in pixels
        min_node_size: Minimum node size in pixels
        title: Chart title
        show_labels: Whether to show text labels
    Returns:
        Plotly Figure
    """
    if not result.cause_areas or result.coordinates_2d is None:
        return go.Figure().update_layout(title="No data available")

    df = result.to_dataframe()

    # Filter by minimum mentions
    df = df[df["mentions"] >= min_mentions].copy()

    if df.empty:
        return go.Figure().update_layout(title="No data meeting minimum mentions threshold")

    # Scale node sizes
    max_mentions = df["mentions"].max()
    min_mentions_val = df["mentions"].min()

    if max_mentions > min_mentions_val:
        df["node_size"] = (df["mentions"] - min_mentions_val) / (
            max_mentions - min_mentions_val
        ) * (max_node_size - min_node_size) + min_node_size
    else:
        df["node_size"] = (max_node_size + min_node_size) / 2

    # Assign colors by cluster
    df["color"] = df["cluster"].apply(
        lambda x: CLUSTER_COLORS[int(x) % len(CLUSTER_COLORS)]
        if pd.notna(x) and x >= 0
        else "#CCCCCC"
    )

    # Create figure
    fig = go.Figure()

    # Add traces per cluster for legend
    clusters_added = set()

    for _, row in df.iterrows():
        cluster_id = row.get("cluster", -1)
        cluster_name = row.get("cluster_name", "Other")

        # Show legend entry only once per cluster
        show_legend = cluster_id not in clusters_added
        clusters_added.add(cluster_id)

        fig.add_trace(
            go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    size=row["node_size"],
                    color=row["color"],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=row["cause_area"] if show_labels else "",
                textposition="top center",
                textfont=dict(size=9),
                name=cluster_name if show_legend else "",
                legendgroup=str(cluster_id),
                showlegend=show_legend,
                hovertemplate=(
                    f"<b>{row['cause_area']}</b><br>"
                    f"Mentions: {row['mentions']}<br>"
                    f"Cluster: {cluster_name}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)) if showtitle else "",
        showlegend=True,
        legend=dict(
            title="Clusters",
            x=1.02,
            y=0.5,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        autosize=True,
        margin=dict(l=0, b=0, t=50, r=0),
        plot_bgcolor="rgba(245,245,245,0.5)",
    )

    return fig


def create_cluster_treemap(
    result: SemanticAnalysisResult,
    title: str = "Cause Area Clusters by Mentions",
    max_members_per_cluster: int = 10,
) -> go.Figure:
    """Create treemap visualization of cause area clusters.

    Args:
        result: SemanticAnalysisResult from SemanticAnalyzer
        title: Chart title
        max_members_per_cluster: Maximum members to show per cluster
    Returns:
        Plotly Figure
    """
    if not result.clusters:
        return go.Figure().update_layout(title="No clusters available")

    # Build treemap data
    labels = ["All Cause Areas"]
    parents = [""]
    values = [0]
    colors = ["#FFFFFF"]

    for i, cluster in enumerate(result.clusters):
        cluster_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        # Add cluster node
        labels.append(cluster.name)
        parents.append("All Cause Areas")
        values.append(cluster.total_mentions)
        colors.append(cluster_color)

        # Add member nodes
        for member in cluster.members[:max_members_per_cluster]:
            mentions = result.mention_counts.get(member, 1)
            labels.append(member)
            parents.append(cluster.name)
            values.append(mentions)
            colors.append(cluster_color)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Mentions: %{value}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        autosize=True,
        margin=dict(l=0, b=0, t=50, r=0),
    )

    return fig


def create_similarity_heatmap(
    result: SemanticAnalysisResult,
    top_n: int = 30,
    title: str = "Cause Area Similarity Matrix",
) -> go.Figure:
    """Create heatmap of cause area semantic similarities.

    Args:
        result: SemanticAnalysisResult from SemanticAnalyzer
        top_n: Number of top cause areas to include
        title: Chart title
    Returns:
        Plotly Figure
    """
    if result.similarity_matrix is None or not result.cause_areas:
        return go.Figure().update_layout(title="No similarity data available")

    # Get top N by mentions
    sorted_indices = sorted(
        range(len(result.cause_areas)),
        key=lambda i: result.mention_counts.get(result.cause_areas[i], 0),
        reverse=True,
    )[:top_n]

    # Extract submatrix
    labels = [result.cause_areas[i] for i in sorted_indices]
    matrix = result.similarity_matrix[np.ix_(sorted_indices, sorted_indices)]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            hovertemplate=("<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.2f}<extra></extra>"),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_tickangle=-45,
        autosize=True,
        margin=dict(l=0, b=0, t=50, r=0),
    )

    return fig


def create_cluster_bar_chart(
    result: SemanticAnalysisResult,
    title: str = "Mentions by Cluster",
) -> go.Figure:
    """Create bar chart of mentions by cluster.

    Args:
        result: SemanticAnalysisResult
        title: Chart title
    Returns:
        Plotly Figure
    """
    if not result.clusters:
        return go.Figure().update_layout(title="No clusters available")

    # Sort clusters by total mentions
    clusters = sorted(result.clusters, key=lambda c: c.total_mentions, reverse=True)

    names = [c.name for c in clusters]
    mentions = [c.total_mentions for c in clusters]
    n_members = [len(c.members) for c in clusters]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=mentions,
            marker_color=colors,
            text=[f"{m} mentions<br>{n} items" for m, n in zip(mentions, n_members)],
            textposition="outside",
            hovertemplate=("<b>%{x}</b><br>Total mentions: %{y}<br><extra></extra>"),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="",
        yaxis_title="Total Mentions",
        xaxis_tickangle=-30,
        height=450,
        margin=dict(b=120),
    )

    return fig
