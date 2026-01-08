"""Main tests for the sm package."""


class TestPackageImports:
    """Test that all package imports work."""

    def test_import_package(self):
        """Test basic package import."""
        import sm

        assert hasattr(sm, "__version__")

    def test_import_pipeline(self):
        """Test pipeline import."""
        from sm import Pipeline

        assert Pipeline is not None

    def test_import_extractors(self):
        """Test extractor imports."""
        from sm import ExtractionResult, LLMExtractor, NLPExtractor

        assert NLPExtractor is not None
        assert LLMExtractor is not None
        assert ExtractionResult is not None

    def test_import_analysis(self):
        """Test analysis imports."""
        from sm import (
            aggregate_cause_areas,
            aggregate_country_mentions,
        )

        assert aggregate_cause_areas is not None
        assert aggregate_country_mentions is not None

    def test_import_viz(self):
        """Test visualization imports."""
        from sm import (
            create_cause_area_bar_chart,
            create_interactive_map,
            create_semantic_network,
        )

        assert create_cause_area_bar_chart is not None
        assert create_interactive_map is not None
        assert create_semantic_network is not None

    def test_import_data(self):
        """Test data utilities import."""
        from sm import get_text_columns, load_attendee_data

        assert load_attendee_data is not None
        assert get_text_columns is not None

    def test_import_cache(self):
        """Test cache utilities import."""
        from sm import clear_cache, get_cache_stats

        assert clear_cache is not None
        assert get_cache_stats is not None


class TestDataModule:
    """Test data loading utilities."""

    def test_get_sample_data(self):
        """Test sample data generation."""
        from sm.data import get_sample_data

        df = get_sample_data(5)
        assert len(df) == 5
        assert "biography" in df.columns
        assert "company" in df.columns
        assert "expertise" in df.columns

    def test_get_text_columns(self, sample_dataframe):
        """Test text column identification."""
        from sm.data import get_text_columns

        cols = get_text_columns(sample_dataframe)
        assert "freeform_cols" in cols
        assert "semicolon_cols" in cols
        assert "biography" in cols["freeform_cols"]
        assert "expertise" in cols["semicolon_cols"]

    def test_combine_text_columns(self, sample_dataframe):
        """Test text column combination."""
        from sm.data import combine_text_columns

        combined = combine_text_columns(sample_dataframe, ["biography", "expertise"])
        assert len(combined) == len(sample_dataframe)
        assert all(isinstance(text, str) for text in combined)


class TestCacheModule:
    """Test cache functionality."""

    def test_cache_stats(self):
        """Test cache statistics."""
        from sm.cache import get_cache_stats

        stats = get_cache_stats()
        assert "total_files" in stats
        assert "total_size_mb" in stats
        assert "categories" in stats

    def test_cache_class(self):
        """Test Cache class."""
        from sm.cache import Cache

        cache = Cache()
        assert cache.cache_dir.exists()


class TestConfigModule:
    """Test configuration."""

    def test_config_paths(self):
        """Test config path attributes."""
        from sm import config

        assert hasattr(config, "REPO_DIR")
        assert hasattr(config, "DATA_DIR")
        assert hasattr(config, "CACHE_DIR")

    def test_ea_categories(self):
        """Test EA cause categories."""
        from sm import config

        assert hasattr(config, "EA_CAUSE_CATEGORIES")
        assert "ai_safety" in config.EA_CAUSE_CATEGORIES
        assert "animal_welfare" in config.EA_CAUSE_CATEGORIES
        assert "global_health" in config.EA_CAUSE_CATEGORIES

    def test_ollama_config(self):
        """Test Ollama configuration."""
        from sm.config import OllamaConfig

        assert hasattr(OllamaConfig, "DEFAULT_MODEL")
        assert hasattr(OllamaConfig, "DEFAULT_N_RUNS")
        assert hasattr(OllamaConfig, "DEFAULT_VOTE_THRESHOLD")
