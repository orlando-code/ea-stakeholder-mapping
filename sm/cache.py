"""Unified caching system for extraction and geocoding results.

Cache Structure:
    .cache/
    ├── nlp/                    # spaCy-based extractions
    │   ├── locations/
    │   ├── organizations/
    │   └── cause_areas/
    ├── llm/                    # LLM-based extractions
    │   └── {model_name}/       # e.g., llama3.2
    │       ├── locations/
    │       ├── organizations/
    │       └── cause_areas/
    └── geocoding/              # Geocoding API results
        ├── geonames/
        └── google_maps/

Usage:
    from sm.cache import Cache

    # Load/save with type-safe methods
    cache = Cache()
    result = cache.load_nlp("locations", text)
    cache.save_nlp("locations", text, result)

    # Or use the module-level singleton
    from sm import cache
    cache.load_nlp("locations", text)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sm import config


class Cache:
    """Unified cache manager for all extraction and geocoding operations."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager.

        Args:
            cache_dir: Cache directory path. Uses config default if not provided.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Core Methods
    # =========================================================================

    def _hash_key(self, *args: str) -> str:
        """Generate MD5 hash from input strings."""
        key = "_".join(str(arg).lower().strip() for arg in args if arg)
        return hashlib.md5(key.encode()).hexdigest()

    def _get_path(self, category: str, subcategory: str, key: str) -> Path:
        """Get cache file path for a given category/subcategory/key."""
        subdir = self.cache_dir / category / subcategory
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{self._hash_key(key)}.json"

    def load(
        self,
        category: str,
        subcategory: str,
        key: str,
    ) -> Optional[Any]:
        """Load data from cache.

        Args:
            category: Cache category (e.g., 'nlp', 'llm', 'geocoding')
            subcategory: Sub-category (e.g., 'locations', 'llama3.2/cause_areas')
            key: Cache key (typically the input text)
        Returns:
            Cached data or None if not found
        """
        path = self._get_path(category, subcategory, key)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
                # Handle None-cached results
                if isinstance(data, dict) and data.get("_cached_none"):
                    return {"_cached_none": True}
                return data
        except (json.JSONDecodeError, IOError):
            return None

    def save(
        self,
        category: str,
        subcategory: str,
        key: str,
        data: Any,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save data to cache.

        Args:
            category: Cache category
            subcategory: Sub-category
            key: Cache key
            data: Data to cache
            metadata: Optional metadata to store
        """
        path = self._get_path(category, subcategory, key)

        try:
            cache_data = data if isinstance(data, dict) else {"data": data}

            if metadata:
                cache_data["_metadata"] = metadata
            cache_data["_cached_at"] = datetime.now().isoformat()

            with open(path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)
        except (IOError, TypeError) as e:
            print(f"Cache write error: {e}")

    def save_none(self, category: str, subcategory: str, key: str) -> None:
        """Cache a None/empty result to avoid repeated failed lookups."""
        self.save(category, subcategory, key, {"_cached_none": True})

    @staticmethod
    def is_none_cached(result: Any) -> bool:
        """Check if a cached result represents a None value."""
        return isinstance(result, dict) and result.get("_cached_none", False)

    # =========================================================================
    # Type-Safe NLP Methods
    # =========================================================================

    def load_nlp(self, extraction_type: str, text: str) -> Optional[Any]:
        """Load NLP extraction result.

        Args:
            extraction_type: Type of extraction ('locations', 'organizations', 'cause_areas')
            text: Source text that was extracted from
        Returns:
            Extraction result or None
        """
        result = self.load("nlp", extraction_type, text)
        if result and not self.is_none_cached(result):
            return result.get("result") if "result" in result else result
        return None

    def save_nlp(self, extraction_type: str, text: str, result: Any) -> None:
        """Save NLP extraction result."""
        self.save("nlp", extraction_type, text, {"result": result})

    # =========================================================================
    # Type-Safe LLM Methods
    # =========================================================================

    def load_llm(
        self,
        model_name: str,
        extraction_type: str,
        text: str,
    ) -> Optional[dict]:
        """Load LLM extraction result.

        Args:
            model_name: LLM model name (e.g., 'llama3.2')
            extraction_type: Type of extraction
            text: Source text
        Returns:
            Dict with 'result' and optionally 'attempts'
        """
        result = self.load("llm", f"{model_name}/{extraction_type}", text)
        if result and not self.is_none_cached(result):
            return result
        return None

    def save_llm(
        self,
        model_name: str,
        extraction_type: str,
        text: str,
        result: dict,
        attempts: Optional[list] = None,
    ) -> None:
        """Save LLM extraction result with voting metadata."""
        data = {"result": result}
        if attempts:
            data["attempts"] = attempts
            data["n_attempts"] = len(attempts)
        self.save("llm", f"{model_name}/{extraction_type}", text, data)

    # =========================================================================
    # Type-Safe Geocoding Methods
    # =========================================================================

    def load_geocoding(self, service: str, query: str) -> Optional[dict]:
        """Load geocoding result.

        Args:
            service: Geocoding service ('geonames', 'google_maps')
            query: Location/organization query
        Returns:
            Geocoded result or None
        """
        result = self.load("geocoding", service, query)
        if result and not self.is_none_cached(result):
            return result
        return None

    def save_geocoding(self, service: str, query: str, result: Optional[dict]) -> None:
        """Save geocoding result."""
        if result is None:
            self.save_none("geocoding", service, query)
        else:
            self.save("geocoding", service, query, result)

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
    ) -> int:
        """Clear cache files.

        Args:
            category: If provided, only clear this category
            subcategory: If provided, only clear this subcategory
        Returns:
            Number of files deleted
        """
        if category and subcategory:
            target = self.cache_dir / category / subcategory
        elif category:
            target = self.cache_dir / category
        else:
            target = self.cache_dir

        if not target.exists():
            return 0

        count = 0
        for item in target.rglob("*.json"):
            item.unlink()
            count += 1

        # Clean up empty directories
        for item in sorted(target.rglob("*"), reverse=True):
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()

        return count

    def stats(self) -> dict:
        """Get cache statistics."""
        stats = {"total_files": 0, "total_size_mb": 0, "categories": {}}

        if not self.cache_dir.exists():
            return stats

        for category in self.cache_dir.iterdir():
            if not category.is_dir() or category.name.startswith("."):
                continue

            cat_stats = {"files": 0, "size_bytes": 0, "subcategories": {}}

            for subcategory in category.iterdir():
                if subcategory.is_dir():
                    files = list(subcategory.glob("*.json"))
                    size = sum(f.stat().st_size for f in files)

                    cat_stats["subcategories"][subcategory.name] = {
                        "files": len(files),
                        "size_kb": round(size / 1024, 2),
                    }
                    cat_stats["files"] += len(files)
                    cat_stats["size_bytes"] += size

            cat_stats["size_mb"] = round(cat_stats["size_bytes"] / (1024 * 1024), 3)
            del cat_stats["size_bytes"]
            stats["categories"][category.name] = cat_stats
            stats["total_files"] += cat_stats["files"]
            stats["total_size_mb"] += cat_stats["size_mb"]

        stats["total_size_mb"] = round(stats["total_size_mb"], 3)
        return stats


# =============================================================================
# Module-Level Singleton and Convenience Functions
# =============================================================================

_default_cache: Optional[Cache] = None


def _get_cache() -> Cache:
    """Get the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


# NLP cache
def load_nlp_extraction(extraction_type: str, text: str) -> Optional[Any]:
    """Load NLP extraction from cache."""
    return _get_cache().load_nlp(extraction_type, text)


def save_nlp_extraction(extraction_type: str, text: str, result: Any) -> None:
    """Save NLP extraction to cache."""
    _get_cache().save_nlp(extraction_type, text, result)


# LLM cache
def load_llm_extraction(model_name: str, extraction_type: str, text: str) -> Optional[dict]:
    """Load LLM extraction from cache."""
    return _get_cache().load_llm(model_name, extraction_type, text)


def save_llm_extraction(
    model_name: str,
    extraction_type: str,
    text: str,
    result: dict,
    attempts: Optional[list] = None,
) -> None:
    """Save LLM extraction to cache."""
    _get_cache().save_llm(model_name, extraction_type, text, result, attempts)


# Geocoding cache
def load_geocoding(service: str, query: str) -> Optional[dict]:
    """Load geocoding result from cache."""
    return _get_cache().load_geocoding(service, query)


def save_geocoding(service: str, query: str, result: Optional[dict]) -> None:
    """Save geocoding result to cache."""
    _get_cache().save_geocoding(service, query, result)


def is_none_cached(result: Any) -> bool:
    """Check if a cached result represents None."""
    return Cache.is_none_cached(result)


# Management
def clear_cache(category: Optional[str] = None, subcategory: Optional[str] = None) -> int:
    """Clear cache files."""
    return _get_cache().clear(category, subcategory)


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return _get_cache().stats()
