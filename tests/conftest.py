"""Pytest configuration and fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "biography": [
                "I work in London at Oxford University on AI safety.",
                "I live in Paris and work at Google on climate research.",
                "I am based in Amsterdam focusing on animal welfare.",
            ],
            "company": [
                "Oxford University",
                "Google",
                "Good Food Institute",
            ],
            "expertise": [
                "AI safety; Machine learning; Research",
                "Climate change; Policy; Data science",
                "Animal welfare; Alternative proteins; Advocacy",
            ],
            "interests": [
                "Biosecurity; Career advice; Governance",
                "Effective giving; Outreach; Education",
                "Factory farming; Cultivated meat; Impact",
            ],
        }
    )


@pytest.fixture
def sample_text():
    """Sample text for extraction testing."""
    return "I work at Oxford University on AI safety research and biosecurity."


@pytest.fixture
def sample_texts():
    """Multiple sample texts for batch testing."""
    return [
        "PhD student at Cambridge studying global health interventions.",
        "Policy researcher at Rethink Priorities working on existential risk.",
        "Engineer at DeepMind interested in AI alignment.",
    ]


@pytest.fixture
def nlp_extractor():
    """NLP extractor instance."""
    from sm.extractors import NLPExtractor

    return NLPExtractor(use_cache=False)


@pytest.fixture
def sample_cause_areas():
    """Sample cause areas for semantic analysis testing."""
    return [
        "ai safety",
        "ai alignment",
        "machine learning safety",
        "animal welfare",
        "factory farming",
        "alternative proteins",
        "global health",
        "malaria prevention",
        "climate change",
        "biosecurity",
    ]
