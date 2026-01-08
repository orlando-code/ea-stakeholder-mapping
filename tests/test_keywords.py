"""Tests for cause area and keyword extraction."""


class TestSemicolonParsing:
    """Test semicolon-separated keyword parsing."""

    def test_basic_parsing(self):
        """Test basic semicolon parsing."""
        from sm.extractors import parse_semicolon_keywords

        text = "AI safety; Machine learning; Research"
        keywords = parse_semicolon_keywords(text)

        assert isinstance(keywords, list)
        assert len(keywords) == 3
        assert "ai safety" in keywords
        assert "machine learning" in keywords
        assert "research" in keywords

    def test_slash_expansion(self):
        """Test slash-separated sub-items."""
        from sm.extractors import parse_semicolon_keywords

        text = "AI/ML; Policy research"
        keywords = parse_semicolon_keywords(text)

        assert "ai" in keywords or "ml" in keywords

    def test_deduplication(self):
        """Test duplicate removal."""
        from sm.extractors import parse_semicolon_keywords

        text = "AI Safety; AI safety; ai safety"
        keywords = parse_semicolon_keywords(text, normalize=True)

        # Should only have one entry
        assert keywords.count("ai safety") == 1

    def test_empty_input(self):
        """Test empty/invalid input."""
        from sm.extractors import parse_semicolon_keywords

        assert parse_semicolon_keywords("") == []
        assert parse_semicolon_keywords("nan") == []
        assert parse_semicolon_keywords(None) == []


class TestNLPCauseAreaExtraction:
    """Test NLP-based cause area extraction."""

    def test_extract_ea_cause_areas(self, nlp_extractor):
        """Test extraction of EA cause areas."""
        text = "I work on AI safety and biosecurity research."
        areas = nlp_extractor.extract_cause_areas(text)

        assert isinstance(areas, list)
        # Should find some cause areas
        assert len(areas) > 0

    def test_extract_multiple_areas(self, nlp_extractor):
        """Test extraction of multiple cause areas."""
        text = """
        My research focuses on existential risk from AI and pandemic preparedness.
        I'm also interested in animal welfare and alternative proteins.
        """
        areas = nlp_extractor.extract_cause_areas(text)

        assert isinstance(areas, list)

    def test_empty_text(self, nlp_extractor):
        """Test extraction from empty text."""
        assert nlp_extractor.extract_cause_areas("") == []

    def test_non_ea_text(self, nlp_extractor):
        """Test extraction from non-EA text."""
        text = "I enjoy hiking and playing guitar."
        areas = nlp_extractor.extract_cause_areas(text)

        # Should return list (possibly empty)
        assert isinstance(areas, list)


class TestCauseAreaDataFrame:
    """Test cause area extraction from DataFrames."""

    def test_extract_cause_areas_from_df(self, nlp_extractor, sample_dataframe):
        """Test cause area extraction from DataFrame."""
        df = nlp_extractor.extract_cause_areas_from_df(
            sample_dataframe,
            columns=["biography"],
            progress=False,
        )

        assert "nlp_cause_areas" in df.columns
        assert all(isinstance(areas, list) for areas in df["nlp_cause_areas"])


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_to_dict(self, nlp_extractor, sample_text):
        """Test conversion to dictionary."""
        result = nlp_extractor.extract_all(sample_text)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "locations" in result_dict
        assert "organizations" in result_dict
        assert "cause_areas" in result_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        from sm.extractors import ExtractionResult

        data = {
            "locations": ["London"],
            "organizations": ["Oxford University"],
            "cause_areas": ["ai safety"],
        }
        result = ExtractionResult.from_dict(data, method="test", model="test")

        assert result.locations == ["London"]
        assert result.organizations == ["Oxford University"]
        assert result.cause_areas == ["ai safety"]
        assert result.method == "test"

    def test_total_extractions(self):
        """Test total extraction count."""
        from sm.extractors import ExtractionResult

        result = ExtractionResult(
            locations=["London", "Paris"],
            organizations=["Google"],
            cause_areas=["ai safety", "biosecurity"],
        )

        assert result.total_extractions() == 5

    def test_is_empty(self):
        """Test empty check."""
        from sm.extractors import ExtractionResult

        empty = ExtractionResult()
        assert empty.is_empty()

        non_empty = ExtractionResult(locations=["London"])
        assert not non_empty.is_empty()
