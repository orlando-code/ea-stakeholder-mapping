"""EA Stakeholder Mapping - Analyze attendee metadata and visualize interests.

This package provides tools for analyzing EA conference attendee data:

1. **Extraction**: Extract locations, organizations, and cause areas using:
   - NLP: spaCy-based extraction (fast, deterministic)
   - LLM: Ollama-based extraction with majority voting (accurate, handles nuance)

2. **Analysis**: 
   - Geographic distribution by country and organization
   - Semantic clustering of cause areas by similarity

3. **Visualization**:
   - Interactive maps showing geographic distribution
   - Cause area frequency charts
   - Semantic network showing cause area relationships

Quick Start:
    from sm import Pipeline
    
    # Create and run pipeline
    pipe = Pipeline(methods=["nlp", "llm"])
    pipe.load_data("data/attendees.csv")
    pipe.extract(text_columns=["biography", "help_me"])
    
    # Compare extraction methods
    comparison = pipe.compare_methods()
    print(comparison.summary())
    
    # Analyze and visualize
    pipe.analyze_semantic()
    fig = pipe.create_semantic_network()
    fig.show()

Alternative (manual) usage:
    from sm.extractors import NLPExtractor, LLMExtractor
    
    # NLP extraction
    nlp = NLPExtractor()
    result = nlp.extract_all("I work on AI safety at Oxford University")
    
    # LLM extraction with voting
    llm = LLMExtractor(n_runs=3)
    result = llm.extract_all("I work on AI safety at Oxford University")
"""

__version__ = "0.3.0"

# Core modules
from sm import analysis, cache, config, data, extractors, viz
from sm.pipeline import Pipeline, PipelineResults

# Extractors
from sm.extractors import (
    ExtractionResult,
    ExtractorComparator,
    LLMExtractor,
    NLPExtractor,
    check_ollama_available,
    parse_semicolon_keywords,
)

# Analysis
from sm.analysis import (
    SemanticAnalyzer,
    aggregate_cause_areas,
    aggregate_country_mentions,
    aggregate_organization_mentions,
    prepare_geographic_data,
)

# Visualization
from sm.viz import (
    create_cause_area_bar_chart,
    create_interactive_map,
    create_semantic_network,
)

# Data loading
from sm.data import (
    combine_text_columns,
    get_text_columns,
    load_attendee_data,
)

# Cache utilities
from sm.cache import clear_cache, get_cache_stats

__all__ = [
    # Version
    "__version__",
    # Modules
    "config",
    "cache",
    "data",
    "extractors",
    "analysis",
    "viz",
    # Pipeline
    "Pipeline",
    "PipelineResults",
    # Extractors
    "NLPExtractor",
    "LLMExtractor",
    "ExtractionResult",
    "ExtractorComparator",
    "check_ollama_available",
    "parse_semicolon_keywords",
    # Analysis
    "SemanticAnalyzer",
    "aggregate_cause_areas",
    "aggregate_country_mentions",
    "aggregate_organization_mentions",
    "prepare_geographic_data",
    # Visualization
    "create_interactive_map",
    "create_cause_area_bar_chart",
    "create_semantic_network",
    # Data
    "load_attendee_data",
    "get_text_columns",
    "combine_text_columns",
    # Cache
    "clear_cache",
    "get_cache_stats",
]
