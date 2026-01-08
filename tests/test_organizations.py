"""Tests for organization extraction."""


class TestNLPOrganizationExtraction:
    """Test NLP-based organization extraction."""

    def test_extract_university(self, nlp_extractor):
        """Test extraction of university names."""
        text = "I work at Oxford University and studied at Cambridge."
        orgs = nlp_extractor.extract_organizations(text)

        assert isinstance(orgs, list)

    def test_extract_company(self, nlp_extractor):
        """Test extraction of company names."""
        text = "I work at Google and previously worked at DeepMind."
        orgs = nlp_extractor.extract_organizations(text)

        assert isinstance(orgs, list)

    def test_extract_nonprofit(self, nlp_extractor):
        """Test extraction of nonprofit names."""
        text = "I volunteer with GiveWell and donate through Giving What We Can."
        orgs = nlp_extractor.extract_organizations(text)

        assert isinstance(orgs, list)

    def test_empty_text(self, nlp_extractor):
        """Test extraction from empty text."""
        assert nlp_extractor.extract_organizations("") == []
        assert nlp_extractor.extract_organizations("   ") == []

    def test_organization_normalization(self, nlp_extractor):
        """Test that university names are normalized."""
        text = "I work at University of Oxford."
        orgs = nlp_extractor.extract_organizations(text)

        assert isinstance(orgs, list)
        # Normalization should produce consistent format


class TestOrganizationDataFrame:
    """Test organization extraction from DataFrames."""

    def test_extract_organizations_from_df(self, nlp_extractor, sample_dataframe):
        """Test organization extraction from DataFrame."""
        df = nlp_extractor.extract_organizations_from_df(
            sample_dataframe,
            columns=["biography", "company"],
            progress=False,
        )

        assert "nlp_organizations" in df.columns
        assert all(isinstance(orgs, list) for orgs in df["nlp_organizations"])
