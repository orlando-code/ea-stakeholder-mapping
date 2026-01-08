"""Visualization module for maps, charts, and network diagrams.

Maps:
- Interactive choropleth showing country distribution
- Organization location markers

Charts:
- Cause area bar charts
- Method comparison charts
- Category breakdown pie charts

Network:
- Semantic network showing cause area similarity
- Cluster treemap
- Similarity heatmap
"""

from sm.viz.charts import (
    create_category_breakdown_pie,
    create_cause_area_bar_chart,
    create_comparison_scatter,
    create_expertise_vs_interest_chart,
    create_extraction_comparison_chart,
    create_undervalued_chart,
)
from sm.viz.maps import (
    assign_iso_codes,
    create_interactive_map,
    create_map_with_dropdown,
    get_country_iso3,
)
from sm.viz.network import (
    create_cluster_bar_chart,
    create_cluster_treemap,
    create_semantic_network,
    create_similarity_heatmap,
)

__all__ = [
    # Maps
    "create_interactive_map",
    "create_map_with_dropdown",
    "assign_iso_codes",
    "get_country_iso3",
    # Charts
    "create_cause_area_bar_chart",
    "create_extraction_comparison_chart",
    "create_comparison_scatter",
    "create_expertise_vs_interest_chart",
    "create_undervalued_chart",
    "create_category_breakdown_pie",
    # Network
    "create_semantic_network",
    "create_cluster_treemap",
    "create_cluster_bar_chart",
    "create_similarity_heatmap",
]
