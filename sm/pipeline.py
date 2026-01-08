"""Unified analysis pipeline for EA stakeholder mapping.

This module provides a high-level Pipeline class that orchestrates the complete
analysis workflow: data loading → extraction → geographic analysis → semantic
analysis → visualization.

Example:
    from sm.pipeline import Pipeline

    # Create pipeline with both methods
    pipe = Pipeline(methods=["nlp", "llm"])

    # Load and process data
    pipe.load_data("data/attendees.csv")
    pipe.extract(text_columns=["biography", "help_me"])

    # Analyze
    pipe.analyze_geographic()
    pipe.analyze_semantic()

    # Compare methods
    pipe.compare_methods()

    # Visualize
    pipe.create_visualizations()
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from sm import cache
from sm.data import load_attendee_data
from sm.extractors import ExtractorComparator, LLMExtractor, NLPExtractor, parse_semicolon_keywords


@dataclass
class PipelineResults:
    """Container for all pipeline results."""

    # Raw data
    df: Optional[pd.DataFrame] = None

    # Extraction results
    nlp_extracted: bool = False
    llm_extracted: bool = False

    # Geographic analysis
    country_counts: Optional[pd.DataFrame] = None
    organization_geo: Optional[pd.DataFrame] = None

    # Semantic analysis
    cause_areas: Optional[list[str]] = None
    cause_area_counts: Optional[dict[str, int]] = None
    semantic_result: Optional[object] = None  # SemanticAnalysisResult

    # Method comparison
    comparison: Optional[object] = None  # AggregateComparison


class Pipeline:
    """High-level pipeline for EA stakeholder analysis.

    Orchestrates the complete workflow from raw data to visualizations.
    Supports both NLP and LLM extraction methods, with built-in comparison.

    Attributes:
        df: Current DataFrame being processed
        results: Container for all analysis results
        nlp: NLP extractor instance (if enabled)
        llm: LLM extractor instance (if enabled)
    """

    def __init__(
        self,
        methods: list[Literal["nlp", "llm"]] = ["nlp", "llm"],
        llm_model: Optional[str] = None,
        llm_n_runs: int = 3,
        use_cache: bool = True,
    ):
        """Initialize pipeline.

        Args:
            methods: Extraction methods to use ("nlp", "llm", or both)
            llm_model: Ollama model name (uses config default if not provided)
            llm_n_runs: Number of LLM runs for majority voting
            use_cache: Whether to cache extraction results
        """
        self.methods = methods
        self.use_cache = use_cache
        self.results = PipelineResults()

        # Initialize extractors
        self.nlp = NLPExtractor(use_cache=use_cache) if "nlp" in methods else None
        self.llm = (
            LLMExtractor(
                model_name=llm_model,
                n_runs=llm_n_runs,
                use_cache=use_cache,
            )
            if "llm" in methods
            else None
        )

        self.df: Optional[pd.DataFrame] = None

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_data(
        self,
        filepath: Optional[str] = None,
        skip_rows: int = 5,
        anonymize: bool = True,
    ) -> pd.DataFrame:
        """Load attendee data.

        Args:
            filepath: Path to CSV/Excel file. Uses default if not provided.
            skip_rows: Header rows to skip
            anonymize: Remove identifying columns
        Returns:
            Loaded DataFrame
        """
        self.df = load_attendee_data(filepath, skip_rows, anonymize)
        self.results.df = self.df
        print(f"Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")
        return self.df

    def set_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set DataFrame directly.

        Args:
            df: DataFrame to use
        Returns:
            The DataFrame
        """
        self.df = df.copy()
        self.results.df = self.df
        return self.df

    # =========================================================================
    # Extraction
    # =========================================================================

    def extract(
        self,
        text_columns: list[str],
        semicolon_columns: Optional[list[str]] = None,
        progress: bool = True,
        parallel: bool = True,
        max_workers: int = 2,
    ) -> pd.DataFrame:
        """Run extraction using configured methods.

        Args:
            text_columns: Columns containing freeform text (biography, etc.)
            semicolon_columns: Columns with semicolon-separated keywords
            progress: Show progress bar
            parallel: Use parallel processing for LLM extraction (requires
                     OLLAMA_NUM_PARALLEL to be set for full benefit)
            max_workers: Maximum parallel workers for batch processing.
                        Keep low (2-3) to avoid overwhelming Ollama queue.
        Returns:
            DataFrame with extraction columns added
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Handle semicolon columns first (these are already structured)
        if semicolon_columns:
            for col in semicolon_columns:
                if col in self.df.columns:
                    self.df[f"{col}_parsed"] = self.df[col].apply(parse_semicolon_keywords)

        # Run NLP extraction (parallel doesn't help much for NLP as it's CPU-bound
        # and spaCy already uses multi-threading internally)
        if self.nlp:
            self.df = self.nlp.extract_from_df(
                self.df, text_columns, combine_columns=True, progress=progress
            )
            self.results.nlp_extracted = True

        # Run LLM extraction with parallel processing
        if self.llm:
            self.df = self.llm.extract_from_df(
                self.df,
                text_columns,
                combine_columns=True,
                progress=progress,
                parallel=parallel,
                max_workers=max_workers,
            )
            self.results.llm_extracted = True

        self.results.df = self.df
        return self.df

    def extract_single(
        self,
        text: str,
        method: Literal["nlp", "llm"] = "nlp",
    ) -> dict:
        """Extract from a single text.

        Args:
            text: Text to extract from
            method: Extraction method to use
        Returns:
            Dictionary with extraction results
        """
        if method == "nlp" and self.nlp:
            result = self.nlp.extract_all(text)
        elif method == "llm" and self.llm:
            result = self.llm.extract_all(text)
        else:
            raise ValueError(f"Method '{method}' not available")

        return result.to_dict()

    # =========================================================================
    # Method Comparison
    # =========================================================================

    def compare_methods(
        self,
        text_columns: Optional[list[str]] = None,
        max_rows: Optional[int] = None,
        progress: bool = True,
    ):
        """Compare NLP and LLM extraction methods.

        Args:
            text_columns: Columns to compare on (uses biography if not provided)
            max_rows: Limit rows for comparison (for testing)
            progress: Show progress bar
        Returns:
            AggregateComparison with statistics
        """
        if not (self.nlp and self.llm):
            raise ValueError("Both NLP and LLM methods required for comparison")

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        text_columns = text_columns or ["biography"]
        valid_cols = [c for c in text_columns if c in self.df.columns]

        comparator = ExtractorComparator(use_cache=self.use_cache)
        comparison = comparator.compare_dataframe(
            self.df, valid_cols, progress=progress, max_rows=max_rows
        )

        self.results.comparison = comparison
        return comparison

    # =========================================================================
    # Geographic Analysis
    # =========================================================================

    def analyze_geographic(
        self,
        location_column: Optional[str] = None,
        organization_column: Optional[str] = None,
        progress: bool = True,
        force_reload: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze geographic distribution.

        Args:
            location_column: Column with extracted locations
            organization_column: Column with extracted organizations
            progress: Show progress bar
            force_reload: Bypass geocoding cache
        Returns:
            Tuple of (country_counts_df, organization_geo_df)
        """
        if self.df is None:
            raise ValueError("No data loaded")

        # Import here to avoid circular imports
        from sm.analysis.geographic import (
            aggregate_country_mentions,
            aggregate_organization_mentions,
        )

        # Determine which extraction column to use
        if location_column is None:
            if "llm_locations" in self.df.columns:
                location_column = "llm_locations"
            elif "nlp_locations" in self.df.columns:
                location_column = "nlp_locations"
            else:
                raise ValueError("No location column found. Run extract() first.")

        if organization_column is None:
            if "llm_organizations" in self.df.columns:
                organization_column = "llm_organizations"
            elif "nlp_organizations" in self.df.columns:
                organization_column = "nlp_organizations"

        # Aggregate countries
        # print("Aggregating country mentions...")
        self.results.country_counts = aggregate_country_mentions(
            self.df, [location_column], force_reload=force_reload, progress=progress
        )

        # Aggregate organizations if column available
        if organization_column and organization_column in self.df.columns:
            # print("Geocoding organizations...")
            self.results.organization_geo = aggregate_organization_mentions(
                self.df, [organization_column], force_reload=force_reload, progress=progress
            )

        return self.results.country_counts, self.results.organization_geo

    # =========================================================================
    # Semantic Analysis
    # =========================================================================

    def analyze_semantic(
        self,
        cause_area_column: Optional[str] = None,
        semicolon_columns: Optional[list[str]] = None,
        n_clusters: Optional[int] = None,
        min_mentions: int = 2,
        use_predefined_categories: bool = False,
        similarity_threshold: float = 0.3,
        min_category_size: int = 1,
    ):
        """Analyze semantic similarity of cause areas.

        Args:
            cause_area_column: Column with extracted cause areas
            semicolon_columns: Additional columns with parsed keywords
            n_clusters: Number of clusters (auto if not provided, ignored if
                use_predefined_categories=True)
            min_mentions: Minimum mentions to include
            use_predefined_categories: If True, assign cause areas to predefined EA
                categories (from config.EA_CAUSE_CATEGORIES) based on semantic similarity
                instead of using unsupervised clustering.
            similarity_threshold: When using predefined categories, minimum similarity
                score to assign an item to a category (items below threshold go to "Other").
            min_category_size: When using predefined categories, minimum number of members
                per category. Categories with fewer members will have their items
                reassigned to the next-closest category that meets the size threshold.
        Returns:
            SemanticAnalysisResult
        """
        if self.df is None:
            raise ValueError("No data loaded")

        # Import here to avoid circular imports
        from sm.analysis.semantic import SemanticAnalyzer

        # Collect all cause areas
        all_causes = []

        # From extraction columns
        if cause_area_column is None:
            for col in ["llm_cause_areas", "nlp_cause_areas"]:
                if col in self.df.columns:
                    cause_area_column = col
                    break

        if cause_area_column and cause_area_column in self.df.columns:
            for items in self.df[cause_area_column]:
                if isinstance(items, list):
                    all_causes.extend(items)

        # From semicolon-parsed columns
        if semicolon_columns:
            for col in semicolon_columns:
                parsed_col = f"{col}_parsed"
                if parsed_col in self.df.columns:
                    for items in self.df[parsed_col]:
                        if isinstance(items, list):
                            all_causes.extend(items)

        # Also check standard columns
        for col in ["expertise_parsed", "interests_parsed"]:
            if col in self.df.columns:
                for items in self.df[col]:
                    if isinstance(items, list):
                        all_causes.extend(items)

        if not all_causes:
            print("Warning: No cause areas found for semantic analysis")
            return None

        # Normalize and count
        all_causes = [c.lower().strip() for c in all_causes if c]
        cause_counts = dict(Counter(all_causes))
        unique_causes = [c for c, count in cause_counts.items() if count >= min_mentions]

        mention_str = "mentions" if min_mentions != 1 else "mention"
        print(
            f"Found {len(unique_causes)} unique cause areas (minimum {min_mentions} {mention_str})"
        )

        # Run semantic analysis
        analyzer = SemanticAnalyzer(n_clusters=n_clusters)
        self.results.semantic_result = analyzer.analyze(
            unique_causes,
            cause_counts,
            use_predefined_categories=use_predefined_categories,
            similarity_threshold=similarity_threshold,
            min_category_size=min_category_size,
        )
        print(
            f"Automatically organised these into {len(self.results.semantic_result.clusters)} clusters"
            + (" unsupervised" if not use_predefined_categories else " using predefined categories")
        )
        self.results.cause_areas = unique_causes
        self.results.cause_area_counts = cause_counts

        return self.results.semantic_result

    # =========================================================================
    # Visualization
    # =========================================================================

    def create_map(self, show_organizations: bool = True):
        """Create geographic distribution map.

        Args:
            show_organizations: Include organization markers
        Returns:
            Plotly Figure
        """
        from sm.viz.maps import create_interactive_map

        if self.results.country_counts is None:
            raise ValueError("Run analyze_geographic() first")

        org_df = self.results.organization_geo if show_organizations else pd.DataFrame()

        return create_interactive_map(
            self.results.country_counts,
            org_df if org_df is not None else pd.DataFrame(),
        )

    def create_cause_area_chart(self, top_n: int = 25):
        """Create cause area bar chart.

        Args:
            top_n: Number of top cause areas to show
        Returns:
            Plotly Figure
        """
        from sm.viz.charts import create_cause_area_bar_chart

        if self.results.cause_areas is None:
            raise ValueError("Run analyze_semantic() first")

        return create_cause_area_bar_chart(
            self.results.cause_areas,
            self.results.cause_area_counts,
            top_n=top_n,
        )

    def create_semantic_network(self, min_mentions: int = 2):
        """Create semantic network visualization.

        Args:
            min_mentions: Minimum mentions for inclusion
        Returns:
            Plotly Figure
        """
        from sm.viz.network import create_semantic_network

        if self.results.semantic_result is None:
            raise ValueError("Run analyze_semantic() first")

        return create_semantic_network(
            self.results.semantic_result,
            min_mentions=min_mentions,
        )

    def create_comparison_chart(self):
        """Create extraction method comparison chart.

        Returns:
            Plotly Figure
        """
        from sm.viz.charts import create_extraction_comparison_chart

        if self.results.comparison is None:
            raise ValueError("Run compare_methods() first")

        return create_extraction_comparison_chart(self.results.comparison.to_dataframe())

    def create_expertise_vs_interest_chart(self, top_n: int = 25):
        """Create expertise vs interest chart.

        Returns:
            Plotly Figure
        """
        from sm.viz.charts import create_expertise_vs_interest_chart

        return create_expertise_vs_interest_chart(self.df, top_n=top_n)

    def create_undervalued_chart(self, top_n: int = 25):
        """Create expertise interest comparison chart.

        Returns:
            Matplotlib Figure
        """
        from sm.viz.charts import create_undervalued_chart

        if self.results.semantic_result is None:
            raise ValueError("Run analyze_semantic() first")

        return create_undervalued_chart(self.df, top_n=top_n)

    # =========================================================================
    # Person Recommendations
    # =========================================================================

    def create_recommender(
        self,
        expertise_col: str = "expertise_parsed",
        interests_col: str = "interests_parsed",
        bio_col: str = "biography",
        name_col: Optional[str] = None,
        augment_with_extraction: Optional[str] = None,
    ):
        """Create a person recommender for finding connections.

        The recommender suggests people to connect with based on:
        - Similar: High profile similarity (for direct collaboration)
        - Complementary: Moderate similarity (for cross-pollination)
        - Skill match: Their expertise matches your interests (and vice versa)
        - Wildcard: Very different profiles (for unexpected connections)

        Args:
            expertise_col: Column with parsed expertise list
            interests_col: Column with parsed interests list
            bio_col: Column with biography text
            name_col: Optional column with person names
            augment_with_extraction: Optionally augment expertise/interests with
                extracted cause areas. Options: "llm" (use llm_cause_areas),
                "nlp" (use nlp_cause_areas), or None (don't augment)
        Returns:
            PersonRecommender instance

        Example:
            recommender = pipe.create_recommender()
            recs = recommender.recommend(person_idx=0, top_k=5)
            print(recs.summary())

            # Augment with LLM-extracted cause areas
            recommender = pipe.create_recommender(augment_with_extraction="llm")
        """
        from sm.analysis.recommender import PersonRecommender

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return PersonRecommender(
            self.df,
            expertise_col=expertise_col,
            interests_col=interests_col,
            bio_col=bio_col,
            name_col=name_col,
            augment_with_extraction=augment_with_extraction,
        )

    # =========================================================================
    # Quick Analysis
    # =========================================================================

    def run_full_analysis(
        self,
        filepath: Optional[str] = None,
        text_columns: list[str] = ["biography"],
        semicolon_columns: list[str] = ["expertise", "interests"],
        compare: bool = True,
    ) -> PipelineResults:
        """Run complete analysis pipeline.

        Convenience method that runs all analysis steps.

        Args:
            filepath: Data file path
            text_columns: Freeform text columns
            semicolon_columns: Structured keyword columns
            compare: Whether to compare methods
        Returns:
            PipelineResults with all results
        """
        # Load
        self.load_data(filepath)

        # Extract
        self.extract(text_columns, semicolon_columns)

        # Compare methods (if both available)
        if compare and self.nlp and self.llm:
            self.compare_methods(text_columns)
            print(self.results.comparison.summary())

        # Geographic analysis
        try:
            self.analyze_geographic()
        except Exception as e:
            print(f"Geographic analysis skipped: {e}")

        # Semantic analysis
        try:
            self.analyze_semantic(semicolon_columns=semicolon_columns)
        except Exception as e:
            print(f"Semantic analysis skipped: {e}")

        return self.results

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self, category: Optional[str] = None) -> int:
        """Clear extraction cache.

        Args:
            category: Specific category to clear ('nlp', 'llm', 'geocoding')
        Returns:
            Number of files deleted
        """
        return cache.clear_cache(category)

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return cache.get_cache_stats()
