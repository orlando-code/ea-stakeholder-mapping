"""Analysis module for geographic, semantic, and recommendation analysis.

Geographic Analysis:
- Geocode locations and organizations
- Aggregate mentions by country
- Prepare data for map visualization

Semantic Analysis:
- Embed cause areas using sentence transformers
- Cluster similar cause areas
- Generate 2D coordinates for network visualization

Person Recommendations:
- Embed person profiles (expertise, interests, biography)
- Recommend connections based on similarity, complementarity, skill-matching
"""

from sm.analysis.geographic import (
    aggregate_country_mentions,
    aggregate_organization_mentions,
    geocode_location,
    geocode_locations_batch,
    geocode_organization,
    geocode_organizations_batch,
    prepare_geographic_data,
)
from sm.analysis.recommender import (
    PersonRecommendations,
    PersonRecommender,
    Recommendation,
)
from sm.analysis.semantic import (
    CauseAreaCluster,
    SemanticAnalysisResult,
    SemanticAnalyzer,
    aggregate_cause_areas,
    normalize_cause_area,
    normalize_cause_areas_fuzzy,
)

__all__ = [
    # Geographic
    "geocode_location",
    "geocode_organization",
    "geocode_locations_batch",
    "geocode_organizations_batch",
    "aggregate_country_mentions",
    "aggregate_organization_mentions",
    "prepare_geographic_data",
    # Semantic
    "SemanticAnalyzer",
    "SemanticAnalysisResult",
    "CauseAreaCluster",
    "aggregate_cause_areas",
    "normalize_cause_area",
    "normalize_cause_areas_fuzzy",
    # Recommender
    "PersonRecommender",
    "PersonRecommendations",
    "Recommendation",
]
