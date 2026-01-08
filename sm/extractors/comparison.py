"""Comparison utilities for evaluating NLP vs LLM extraction performance.

This module provides tools to compare extraction results between methods,
measuring overlap, unique extractions, and overall performance metrics.

Parallel processing:
- When OLLAMA_NUM_PARALLEL is set, comparisons across multiple texts are
  processed concurrently using ThreadPoolExecutor for significant speedup.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from sm.extractors.base import ExtractionResult
from sm.extractors.llm import LLMExtractor
from sm.extractors.nlp import NLPExtractor


@dataclass
class ComparisonResult:
    """Results of comparing two extraction methods on a single text."""

    text_preview: str = ""  # First 100 chars of source text

    # Per-category statistics
    locations: dict = field(default_factory=dict)
    organizations: dict = field(default_factory=dict)
    cause_areas: dict = field(default_factory=dict)

    def jaccard_similarity(self, category: str) -> float:
        """Calculate Jaccard similarity for a category."""
        stats = getattr(self, category, {})
        union_size = stats.get("union_size", 0)
        if union_size == 0:
            return 1.0  # Both empty = identical
        return stats.get("overlap_size", 0) / union_size

    def summary_dict(self) -> dict:
        """Get summary statistics as a dictionary."""
        return {
            "locations_nlp": self.locations.get("nlp_count", 0),
            "locations_llm": self.locations.get("llm_count", 0),
            "locations_overlap": self.locations.get("overlap_size", 0),
            "locations_jaccard": self.jaccard_similarity("locations"),
            "organizations_nlp": self.organizations.get("nlp_count", 0),
            "organizations_llm": self.organizations.get("llm_count", 0),
            "organizations_overlap": self.organizations.get("overlap_size", 0),
            "organizations_jaccard": self.jaccard_similarity("organizations"),
            "cause_areas_nlp": self.cause_areas.get("nlp_count", 0),
            "cause_areas_llm": self.cause_areas.get("llm_count", 0),
            "cause_areas_overlap": self.cause_areas.get("overlap_size", 0),
            "cause_areas_jaccard": self.jaccard_similarity("cause_areas"),
        }


def compare_extractions(
    nlp_result: ExtractionResult,
    llm_result: ExtractionResult,
    source_text: str = "",
) -> ComparisonResult:
    """Compare extraction results from NLP and LLM methods.

    Args:
        nlp_result: Extraction result from NLP method
        llm_result: Extraction result from LLM method
        source_text: Original source text (for reference)
    Returns:
        ComparisonResult with detailed statistics
    """
    comparison = ComparisonResult(
        text_preview=source_text[:100] + "..." if len(source_text) > 100 else source_text
    )

    # Compare each category
    for category in ["locations", "organizations", "cause_areas"]:
        nlp_items = set(x.lower().strip() for x in getattr(nlp_result, category, []))
        llm_items = set(x.lower().strip() for x in getattr(llm_result, category, []))

        overlap = nlp_items & llm_items
        only_nlp = nlp_items - llm_items
        only_llm = llm_items - nlp_items
        union = nlp_items | llm_items

        stats = {
            "nlp_count": len(nlp_items),
            "llm_count": len(llm_items),
            "overlap_size": len(overlap),
            "union_size": len(union),
            "overlap": sorted(overlap),
            "only_nlp": sorted(only_nlp),
            "only_llm": sorted(only_llm),
        }

        setattr(comparison, category, stats)

    return comparison


@dataclass
class AggregateComparison:
    """Aggregate comparison statistics across multiple texts."""

    n_texts: int = 0

    # Totals
    total_nlp: int = 0
    total_llm: int = 0
    total_overlap: int = 0

    # Per-category totals
    locations_nlp: int = 0
    locations_llm: int = 0
    locations_overlap: int = 0

    organizations_nlp: int = 0
    organizations_llm: int = 0
    organizations_overlap: int = 0

    cause_areas_nlp: int = 0
    cause_areas_llm: int = 0
    cause_areas_overlap: int = 0

    # All individual comparisons
    comparisons: list = field(default_factory=list)

    @property
    def avg_nlp_per_text(self) -> float:
        return self.total_nlp / self.n_texts if self.n_texts > 0 else 0

    @property
    def avg_llm_per_text(self) -> float:
        return self.total_llm / self.n_texts if self.n_texts > 0 else 0

    @property
    def overall_jaccard(self) -> float:
        """Overall Jaccard similarity."""
        union = self.total_nlp + self.total_llm - self.total_overlap
        if union == 0:
            return 1.0
        return self.total_overlap / union

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EXTRACTION METHOD COMPARISON",
            "=" * 60,
            f"Texts analyzed: {self.n_texts}",
            "",
            "LOCATIONS:",
            f"  NLP: {self.locations_nlp} total ({self.locations_nlp / self.n_texts:.1f}/text)",
            f"  LLM: {self.locations_llm} total ({self.locations_llm / self.n_texts:.1f}/text)",
            f"  Overlap: {self.locations_overlap}",
            "",
            "ORGANIZATIONS:",
            f"  NLP: {self.organizations_nlp} total ({self.organizations_nlp / self.n_texts:.1f}/text)",
            f"  LLM: {self.organizations_llm} total ({self.organizations_llm / self.n_texts:.1f}/text)",
            f"  Overlap: {self.organizations_overlap}",
            "",
            "CAUSE AREAS:",
            f"  NLP: {self.cause_areas_nlp} total ({self.cause_areas_nlp / self.n_texts:.1f}/text)",
            f"  LLM: {self.cause_areas_llm} total ({self.cause_areas_llm / self.n_texts:.1f}/text)",
            f"  Overlap: {self.cause_areas_overlap}",
            "",
            "OVERALL:",
            f"  Total NLP extractions: {self.total_nlp}",
            f"  Total LLM extractions: {self.total_llm}",
            f"  Total overlap: {self.total_overlap}",
            f"  Overall Jaccard similarity: {self.overall_jaccard:.2%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to summary DataFrame."""
        rows = []
        for category in ["locations", "organizations", "cause_areas"]:
            nlp = getattr(self, f"{category}_nlp")
            llm = getattr(self, f"{category}_llm")
            overlap = getattr(self, f"{category}_overlap")
            union = nlp + llm - overlap

            rows.append(
                {
                    "category": category,
                    "nlp_total": nlp,
                    "llm_total": llm,
                    "overlap": overlap,
                    "jaccard": overlap / union if union > 0 else 1.0,
                    "nlp_avg": nlp / self.n_texts if self.n_texts > 0 else 0,
                    "llm_avg": llm / self.n_texts if self.n_texts > 0 else 0,
                }
            )
        return pd.DataFrame(rows)


class ExtractorComparator:
    """Compare NLP and LLM extraction methods.

    Example:
        comparator = ExtractorComparator()

        # Compare on single text
        result = comparator.compare_text("I work on AI safety at Oxford")
        print(result.cause_areas["overlap"])

        # Compare on DataFrame (uses parallel processing)
        agg = comparator.compare_dataframe(df, ["biography", "help_me"])
        print(agg.summary())
    """

    def __init__(
        self,
        nlp_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_n_runs: int = 3,
        use_cache: bool = True,
        max_workers: int = 2,
    ):
        """Initialize comparator with both extractors.

        Args:
            nlp_model: spaCy model name (uses default if not provided)
            llm_model: Ollama model name (uses default if not provided)
            llm_n_runs: Number of LLM runs for majority voting
            use_cache: Whether to use cached results
            max_workers: Max parallel workers for DataFrame comparison.
                        Keep low (2-3) to avoid overwhelming Ollama queue.
        """
        self.nlp_extractor = NLPExtractor(
            spacy_model=nlp_model,
            use_cache=use_cache,
        )
        self.llm_extractor = LLMExtractor(
            model_name=llm_model,
            n_runs=llm_n_runs,
            use_cache=use_cache,
        )
        self.max_workers = max_workers

    def compare_text(self, text: str) -> ComparisonResult:
        """Compare extraction methods on a single text.

        Args:
            text: Text to extract from
        Returns:
            ComparisonResult with detailed statistics
        """
        nlp_result = self.nlp_extractor.extract_all(text)
        llm_result = self.llm_extractor.extract_all(text)
        return compare_extractions(nlp_result, llm_result, text)

    def _compare_text_safe(self, text: str, idx: int) -> tuple[int, Optional[ComparisonResult]]:
        """Thread-safe comparison wrapper that returns index for ordering.

        Args:
            text: Text to extract from
            idx: Original index for result ordering
        Returns:
            Tuple of (index, ComparisonResult or None)
        """
        if not text.strip():
            return idx, None
        try:
            return idx, self.compare_text(text)
        except Exception as e:
            print(f"Comparison failed for text {idx}: {e}")
            return idx, None

    def compare_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: list[str],
        combine_columns: bool = True,
        progress: bool = True,
        max_rows: Optional[int] = None,
        parallel: bool = True,
    ) -> AggregateComparison:
        """Compare extraction methods on a DataFrame.

        Uses parallel processing by default to leverage OLLAMA_NUM_PARALLEL.

        Args:
            df: DataFrame with text columns
            text_columns: Columns to extract from
            combine_columns: Whether to combine columns into single text
            progress: Show progress bar
            max_rows: Limit number of rows (for testing)
            parallel: Use parallel processing (default True)
        Returns:
            AggregateComparison with statistics
        """
        # Prepare texts
        valid_cols = [c for c in text_columns if c in df.columns]
        if combine_columns:
            texts = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else:
            texts = df[valid_cols[0]].fillna("").astype(str)

        if max_rows:
            texts = texts.head(max_rows)

        text_list = texts.tolist()

        # Initialize aggregate
        agg = AggregateComparison()

        if parallel and len(text_list) > 1:
            # Parallel processing
            pbar = tqdm(total=len(text_list), desc="Comparing extractors") if progress else None

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._compare_text_safe, text, i): i
                    for i, text in enumerate(text_list)
                }

                results = []
                for future in as_completed(futures):
                    idx, comparison = future.result()
                    if comparison is not None:
                        results.append((idx, comparison))
                    if pbar:
                        pbar.update(1)

                if pbar:
                    pbar.close()

            # Sort by original index to maintain order
            results.sort(key=lambda x: x[0])

            for _, comparison in results:
                agg.comparisons.append(comparison)
                agg.n_texts += 1
                self._accumulate_stats(agg, comparison)

        else:
            # Sequential processing
            iterator = tqdm(text_list, desc="Comparing extractors") if progress else text_list

            for text in iterator:
                if not text.strip():
                    continue

                comparison = self.compare_text(text)
                agg.comparisons.append(comparison)
                agg.n_texts += 1
                self._accumulate_stats(agg, comparison)

        return agg

    def _accumulate_stats(self, agg: AggregateComparison, comparison: ComparisonResult) -> None:
        """Accumulate comparison statistics into aggregate."""
        for category in ["locations", "organizations", "cause_areas"]:
            stats = getattr(comparison, category)

            setattr(
                agg, f"{category}_nlp", getattr(agg, f"{category}_nlp") + stats["nlp_count"]
            )
            setattr(
                agg, f"{category}_llm", getattr(agg, f"{category}_llm") + stats["llm_count"]
            )
            setattr(
                agg,
                f"{category}_overlap",
                getattr(agg, f"{category}_overlap") + stats["overlap_size"],
            )

            agg.total_nlp += stats["nlp_count"]
            agg.total_llm += stats["llm_count"]
            agg.total_overlap += stats["overlap_size"]

    def compare_to_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: list[str],
        combine_columns: bool = True,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Compare extractions and return detailed DataFrame.

        Returns a DataFrame with per-row comparison statistics.
        """
        agg = self.compare_dataframe(df, text_columns, combine_columns, progress)

        rows = []
        for comp in agg.comparisons:
            rows.append(comp.summary_dict())

        return pd.DataFrame(rows)


# Convenience function
def compare_extraction_results(
    nlp_result: ExtractionResult,
    llm_result: ExtractionResult,
) -> ComparisonResult:
    """Compare two extraction results."""
    return compare_extractions(nlp_result, llm_result)
