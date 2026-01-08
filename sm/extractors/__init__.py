"""Extractors module for entity and keyword extraction.

Two extraction methods are available:
- NLP: spaCy-based extraction (fast, deterministic, rule-based)
- LLM: Ollama-based extraction with majority voting (more accurate, handles nuance)

Both methods implement the same interface via BaseExtractor.

Example:
    from sm.extractors import NLPExtractor, LLMExtractor, ExtractorComparator

    # Use NLP extraction (fast)
    nlp = NLPExtractor()
    result = nlp.extract_all("I work on AI safety at Oxford University")

    # Use LLM extraction (accurate, handles stochasticity)
    llm = LLMExtractor(n_runs=3)
    result = llm.extract_all("I work on AI safety at Oxford University")

    # Compare methods
    comparator = ExtractorComparator()
    comparison = comparator.compare_text("I work on AI safety at Oxford")
"""

from sm.extractors.base import BaseExtractor, ExtractionResult
from sm.extractors.comparison import (
    AggregateComparison,
    ComparisonResult,
    ExtractorComparator,
    compare_extraction_results,
    compare_extractions,
)
from sm.extractors.llm import (
    LLMExtractor,
    OllamaError,
    call_ollama,
    check_ollama_available,
    majority_vote,
)
from sm.extractors.nlp import (
    NLPExtractor,
    get_nlp_model,
    parse_semicolon_keywords,
)

__all__ = [
    # Base
    "BaseExtractor",
    "ExtractionResult",
    # NLP extractor
    "NLPExtractor",
    "get_nlp_model",
    "parse_semicolon_keywords",
    # LLM extractor
    "LLMExtractor",
    "OllamaError",
    "call_ollama",
    "check_ollama_available",
    "majority_vote",
    # Comparison
    "ExtractorComparator",
    "ComparisonResult",
    "AggregateComparison",
    "compare_extractions",
    "compare_extraction_results",
]
