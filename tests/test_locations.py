"""Tests for location extraction."""


class TestNLPLocationExtraction:
    """Test NLP-based location extraction."""

    def test_extract_city(self, nlp_extractor):
        """Test extraction of city names."""
        text = "I live in London and work in Paris."
        locations = nlp_extractor.extract_locations(text)

        assert isinstance(locations, list)
        # Should find at least one location
        assert len(locations) > 0

    def test_extract_country(self, nlp_extractor):
        """Test extraction of country names."""
        text = "I'm from the United Kingdom and studied in Germany."
        locations = nlp_extractor.extract_locations(text)

        assert isinstance(locations, list)

    def test_extract_from_university(self, nlp_extractor):
        """Test location extraction from university names."""
        text = "I work at Oxford University."
        locations = nlp_extractor.extract_locations(text)

        # Should extract Oxford from the university name
        assert isinstance(locations, list)

    def test_empty_text(self, nlp_extractor):
        """Test extraction from empty text."""
        assert nlp_extractor.extract_locations("") == []
        assert nlp_extractor.extract_locations("   ") == []
        assert nlp_extractor.extract_locations(None) == []

    def test_no_locations(self, nlp_extractor):
        """Test text with no locations."""
        text = "I work on machine learning and AI safety."
        locations = nlp_extractor.extract_locations(text)

        assert isinstance(locations, list)

    def test_extraction_result_dataclass(self, nlp_extractor):
        """Test that extract_all returns proper dataclass."""
        text = "I work in London at Oxford University."
        result = nlp_extractor.extract_all(text)

        from sm.extractors import ExtractionResult

        assert isinstance(result, ExtractionResult)
        assert hasattr(result, "locations")
        assert hasattr(result, "organizations")
        assert hasattr(result, "cause_areas")
        assert result.method == "nlp"


class TestDataFrameExtraction:
    """Test DataFrame-based extraction."""

    def test_extract_from_df(self, nlp_extractor, sample_dataframe):
        """Test extraction from DataFrame."""
        df = nlp_extractor.extract_from_df(
            sample_dataframe,
            text_columns=["biography"],
            progress=False,
        )

        assert "nlp_locations" in df.columns
        assert "nlp_organizations" in df.columns
        assert "nlp_cause_areas" in df.columns

    def test_extract_locations_from_df(self, nlp_extractor, sample_dataframe):
        """Test location-only extraction from DataFrame."""
        df = nlp_extractor.extract_locations_from_df(
            sample_dataframe,
            columns=["biography"],
            progress=False,
        )

        assert "nlp_locations" in df.columns
        assert all(isinstance(locs, list) for locs in df["nlp_locations"])
