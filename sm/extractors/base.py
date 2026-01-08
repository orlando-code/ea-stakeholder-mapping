"""Base extractor class and data structures.

This module defines the common interface for all extraction methods (NLP and LLM).
Supports parallel processing via ThreadPoolExecutor for batch DataFrame operations.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


@dataclass
class ExtractionResult:
    """Container for extraction results from a single text.

    Attributes:
        locations: Geographic locations mentioned
        organizations: Organizations, companies, institutions
        cause_areas: EA cause areas and focus topics
        method: Extraction method ('nlp' or 'llm')
        model: Model name (e.g., 'en_core_web_trf' or 'llama3.2')
    """

    locations: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    cause_areas: list[str] = field(default_factory=list)
    method: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "locations": self.locations,
            "organizations": self.organizations,
            "cause_areas": self.cause_areas,
            "_method": self.method,
            "_model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict, method: str = "", model: str = "") -> "ExtractionResult":
        """Create from dictionary."""
        return cls(
            locations=data.get("locations", []),
            organizations=data.get("organizations", []),
            cause_areas=data.get("cause_areas", []),
            method=method or data.get("_method", ""),
            model=model or data.get("_model", ""),
        )

    def total_extractions(self) -> int:
        """Count total extractions across all categories."""
        return len(self.locations) + len(self.organizations) + len(self.cause_areas)

    def is_empty(self) -> bool:
        """Check if all extraction lists are empty."""
        return self.total_extractions() == 0


class BaseExtractor(ABC):
    """Abstract base class for text extractors.

    Both NLP (spaCy) and LLM (Ollama) extractors inherit from this class,
    ensuring a consistent interface for extraction operations.
    """

    def __init__(self, use_cache: bool = True):
        """Initialize extractor.

        Args:
            use_cache: Whether to use cached results
        """
        self.use_cache = use_cache

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return method identifier ('nlp' or 'llm')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass

    @abstractmethod
    def extract_locations(self, text: str) -> list[str]:
        """Extract geographic locations from text.

        Args:
            text: Input text
        Returns:
            List of location names
        """
        pass

    @abstractmethod
    def extract_organizations(self, text: str) -> list[str]:
        """Extract organizations from text.

        Args:
            text: Input text
        Returns:
            List of organization names
        """
        pass

    @abstractmethod
    def extract_cause_areas(self, text: str) -> list[str]:
        """Extract EA cause areas and topics from text.

        Args:
            text: Input text
        Returns:
            List of cause areas/topics
        """
        pass

    def extract_all(self, text: str) -> ExtractionResult:
        """Extract all entity types from text.

        Args:
            text: Input text to process
        Returns:
            ExtractionResult containing all extractions
        """
        if not text or not str(text).strip():
            return ExtractionResult(method=self.method_name, model=self.model_name)

        text = str(text).strip()

        return ExtractionResult(
            locations=self.extract_locations(text),
            organizations=self.extract_organizations(text),
            cause_areas=self.extract_cause_areas(text),
            method=self.method_name,
            model=self.model_name,
        )

    # =========================================================================
    # DataFrame Methods
    # =========================================================================

    def _extract_single_safe(
        self, text: str, idx: int
    ) -> tuple[int, list[str], list[str], list[str]]:
        """Thread-safe extraction wrapper that returns index for ordering.

        Args:
            text: Text to extract from
            idx: Original index for result ordering
        Returns:
            Tuple of (index, locations, organizations, cause_areas)
        """
        if not text.strip():
            return idx, [], [], []
        try:
            result = self.extract_all(text)
            return idx, result.locations, result.organizations, result.cause_areas
        except Exception as e:
            print(f"Extraction failed for row {idx}: {e}")
            return idx, [], [], []

    def extract_from_df(
        self,
        df: pd.DataFrame,
        text_columns: list[str],
        combine_columns: bool = True,
        progress: bool = True,
        parallel: bool = False,
        max_workers: int = 2,
    ) -> pd.DataFrame:
        """Extract all entities from DataFrame text columns.

        Args:
            df: Input DataFrame
            text_columns: Columns containing text to extract from
            combine_columns: If True, combine all columns into single text per row
            progress: Show progress bar
            parallel: Use parallel processing (recommended for LLM extraction)
            max_workers: Maximum number of parallel workers.
                        Keep low (2-3) to avoid overwhelming Ollama queue.
        Returns:
            DataFrame with new columns: '{method}_locations', '{method}_organizations',
            '{method}_cause_areas'
        """
        result_df = df.copy()
        prefix = self.method_name

        # Prepare text series
        valid_cols = [c for c in text_columns if c in df.columns]
        if not valid_cols:
            raise ValueError(f"No valid columns found: {text_columns}")

        if combine_columns:
            texts = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)
        else:
            texts = df[valid_cols[0]].fillna("").astype(str)

        text_list = texts.tolist()

        if parallel and len(text_list) > 1:
            # Parallel processing
            pbar = (
                tqdm(total=len(text_list), desc=f"{prefix.upper()} extraction (parallel)")
                if progress
                else None
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._extract_single_safe, text, i): i
                    for i, text in enumerate(text_list)
                }

                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                    if pbar:
                        pbar.update(1)

                if pbar:
                    pbar.close()

            # Sort by original index to maintain order
            results.sort(key=lambda x: x[0])

            locations_list = [r[1] for r in results]
            organizations_list = [r[2] for r in results]
            cause_areas_list = [r[3] for r in results]

        else:
            # Sequential processing
            locations_list = []
            organizations_list = []
            cause_areas_list = []

            iterator = tqdm(text_list, desc=f"{prefix.upper()} extraction") if progress else text_list

            for text in iterator:
                if text.strip():
                    result = self.extract_all(text)
                    locations_list.append(result.locations)
                    organizations_list.append(result.organizations)
                    cause_areas_list.append(result.cause_areas)
                else:
                    locations_list.append([])
                    organizations_list.append([])
                    cause_areas_list.append([])

        result_df[f"{prefix}_locations"] = locations_list
        result_df[f"{prefix}_organizations"] = organizations_list
        result_df[f"{prefix}_cause_areas"] = cause_areas_list

        return result_df

    def extract_locations_from_df(
        self,
        df: pd.DataFrame,
        columns: list[str],
        output_col: Optional[str] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Extract only locations from DataFrame.

        Args:
            df: Input DataFrame
            columns: Text columns to extract from
            output_col: Output column name (default: '{method}_locations')
            progress: Show progress bar
        Returns:
            DataFrame with new location column
        """
        result_df = df.copy()
        output_col = output_col or f"{self.method_name}_locations"

        valid_cols = [c for c in columns if c in df.columns]
        texts = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)

        iterator = (
            tqdm(texts, desc=f"Extracting locations ({self.method_name})") if progress else texts
        )

        result_df[output_col] = [
            self.extract_locations(text) if text.strip() else [] for text in iterator
        ]

        return result_df

    def extract_organizations_from_df(
        self,
        df: pd.DataFrame,
        columns: list[str],
        output_col: Optional[str] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Extract only organizations from DataFrame."""
        result_df = df.copy()
        output_col = output_col or f"{self.method_name}_organizations"

        valid_cols = [c for c in columns if c in df.columns]
        texts = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)

        iterator = (
            tqdm(texts, desc=f"Extracting organizations ({self.method_name})")
            if progress
            else texts
        )

        result_df[output_col] = [
            self.extract_organizations(text) if text.strip() else [] for text in iterator
        ]

        return result_df

    def extract_cause_areas_from_df(
        self,
        df: pd.DataFrame,
        columns: list[str],
        output_col: Optional[str] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Extract only cause areas from DataFrame."""
        result_df = df.copy()
        output_col = output_col or f"{self.method_name}_cause_areas"

        valid_cols = [c for c in columns if c in df.columns]
        texts = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)

        iterator = (
            tqdm(texts, desc=f"Extracting cause areas ({self.method_name})") if progress else texts
        )

        result_df[output_col] = [
            self.extract_cause_areas(text) if text.strip() else [] for text in iterator
        ]

        return result_df
