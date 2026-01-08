"""NLP-based extraction using spaCy.

This module provides rule-based extraction using spaCy's NER and pattern matching.
It's fast and deterministic but less accurate than LLM extraction for nuanced topics.
"""

from typing import Optional

import spacy

from sm import cache, config
from sm.extractors.base import BaseExtractor

# =============================================================================
# spaCy Model Loading
# =============================================================================

_nlp_model: Optional[spacy.language.Language] = None
_nlp_model_name: str = ""


def get_nlp_model(model_name: Optional[str] = None) -> spacy.language.Language:
    """Load spaCy NLP model with fallback.

    Args:
        model_name: Model to load. Uses config default if not provided.
    Returns:
        Loaded spaCy Language model
    """
    global _nlp_model, _nlp_model_name

    model_name = model_name or config.NLPConfig.PREFERRED_MODEL

    # Return cached model if same
    if _nlp_model is not None and _nlp_model_name == model_name:
        return _nlp_model

    # Try to load requested model
    try:
        _nlp_model = spacy.load(model_name)
        _nlp_model_name = model_name
        return _nlp_model
    except OSError:
        pass

    # Try fallback
    fallback = config.NLPConfig.FALLBACK_MODEL
    try:
        print(f"Warning: {model_name} not found, using {fallback}")
        _nlp_model = spacy.load(fallback)
        _nlp_model_name = fallback
        return _nlp_model
    except OSError:
        raise RuntimeError(
            f"No spaCy model available. Install with:\n  python -m spacy download {fallback}"
        )


class NLPExtractor(BaseExtractor):
    """spaCy-based NLP extractor for entities and keywords.

    Uses spaCy's named entity recognition for locations and organizations,
    and keyword matching for cause areas based on the EA cause category taxonomy.

    Example:
        extractor = NLPExtractor()
        result = extractor.extract_all("I work on AI safety at Oxford University")
        print(result.locations)      # ['Oxford']
        print(result.organizations)  # ['Oxford University']
        print(result.cause_areas)    # ['ai safety']
    """

    # Organization types that indicate a specific location
    LOCATION_ORG_KEYWORDS = {
        "university",
        "college",
        "institute",
        "school",
        "academy",
        "hospital",
        "foundation",
        "center",
        "centre",
        "lab",
        "laboratory",
    }

    # Well-known university cities for location extraction
    KNOWN_UNIVERSITY_LOCATIONS = {
        "oxford",
        "cambridge",
        "harvard",
        "stanford",
        "yale",
        "princeton",
        "columbia",
        "berkeley",
        "amsterdam",
        "london",
        "berlin",
        "paris",
        "edinburgh",
        "manchester",
        "melbourne",
        "sydney",
        "toronto",
        "zurich",
        "boston",
        "chicago",
        "munich",
        "vienna",
        "copenhagen",
        "stockholm",
    }

    def __init__(
        self,
        spacy_model: Optional[str] = None,
        use_cache: bool = True,
    ):
        """Initialize NLP extractor.

        Args:
            spacy_model: spaCy model name (uses config default if not provided)
            use_cache: Whether to use cached results
        """
        super().__init__(use_cache=use_cache)
        self._model_name = spacy_model or config.NLPConfig.PREFERRED_MODEL
        self._nlp = get_nlp_model(self._model_name)
        self._actual_model_name = _nlp_model_name  # May differ from requested

    @property
    def method_name(self) -> str:
        return "nlp"

    @property
    def model_name(self) -> str:
        return self._actual_model_name

    # =========================================================================
    # Location Extraction
    # =========================================================================

    def extract_locations(self, text: str) -> list[str]:
        """Extract locations using spaCy NER.

        Extracts GPE (countries, cities), LOC (mountains, rivers), and
        locations inferred from organization names (e.g., "Oxford" from
        "Oxford University").
        """
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_nlp_extraction("locations", text)
            if cached is not None:
                return cached

        doc = self._nlp(text)

        # Extract location entities
        locations = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                locations.append(ent.text.strip())

        # Extract locations from organization names
        org_locations = self._extract_locations_from_orgs(doc)
        locations.extend(org_locations)

        # Deduplicate preserving order
        locations = list(dict.fromkeys(locations))

        if self.use_cache:
            cache.save_nlp_extraction("locations", text, locations)

        return locations

    def _extract_locations_from_orgs(self, doc) -> list[str]:
        """Extract location names embedded in organization entities."""
        org_locations = []

        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue

            org_text = ent.text.lower()

            # Only process orgs that look like they have a location
            if not any(kw in org_text for kw in self.LOCATION_ORG_KEYWORDS):
                continue

            # Check for known university cities
            for location in self.KNOWN_UNIVERSITY_LOCATIONS:
                if location in org_text and location.title() not in org_locations:
                    org_locations.append(location.title())
                    break

            # Also try to find GPE entities within the org name
            org_doc = self._nlp(ent.text)
            for sub_ent in org_doc.ents:
                if sub_ent.label_ == "GPE" and sub_ent.text not in org_locations:
                    org_locations.append(sub_ent.text)

        return org_locations

    # =========================================================================
    # Organization Extraction
    # =========================================================================

    def extract_organizations(self, text: str) -> list[str]:
        """Extract organizations using spaCy NER."""
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_nlp_extraction("organizations", text)
            if cached is not None:
                return cached

        doc = self._nlp(text)

        organizations = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_name = self._normalize_organization(ent.text)
                if org_name and org_name not in organizations:
                    organizations.append(org_name)

        if self.use_cache:
            cache.save_nlp_extraction("organizations", text, organizations)

        return organizations

    def _normalize_organization(self, org: str) -> str:
        """Normalize organization name."""
        org = org.strip().replace("/", "").strip()

        # Normalize university naming convention
        if "university" in org.lower():
            # Extract the location part
            parts = org.lower().replace("university", "").replace("of", "").split()
            location = " ".join(p.strip() for p in parts if p.strip())
            if location:
                return f"University of {location.title()}"

        return org

    # =========================================================================
    # Cause Area Extraction
    # =========================================================================

    def extract_cause_areas(self, text: str) -> list[str]:
        """Extract EA cause areas using keyword matching.

        Uses the EA cause category taxonomy defined in config to identify
        mentions of cause areas. Also extracts relevant noun phrases that
        might be cause-related.
        """
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_nlp_extraction("cause_areas", text)
            if cached is not None:
                return cached

        text_lower = text.lower()
        cause_areas = []

        # Match against known EA cause keywords
        for category, info in config.EA_CAUSE_CATEGORIES.items():
            for keyword in info["keywords"]:
                if keyword.lower() in text_lower:
                    # Use the keyword as-is (normalized)
                    normalized = keyword.lower().strip()
                    if normalized not in cause_areas:
                        cause_areas.append(normalized)

        # Also extract relevant noun phrases for broader coverage
        doc = self._nlp(text)
        noun_phrase_causes = self._extract_cause_noun_phrases(doc)

        for phrase in noun_phrase_causes:
            if phrase not in cause_areas:
                cause_areas.append(phrase)

        if self.use_cache:
            cache.save_nlp_extraction("cause_areas", text, cause_areas)

        return cause_areas

    def _extract_cause_noun_phrases(self, doc) -> list[str]:
        """Extract noun phrases that might be cause-related."""
        # Words that suggest the phrase is about a cause/topic
        cause_indicators = {
            "safety",
            "risk",
            "welfare",
            "research",
            "policy",
            "governance",
            "development",
            "health",
            "poverty",
            "climate",
            "environment",
            "security",
            "alignment",
            "ethics",
            "impact",
            "prevention",
            "preparedness",
            "advocacy",
            "intervention",
            "effectiveness",
        }

        phrases = []
        for chunk in doc.noun_chunks:
            # Filter by length (2-5 words)
            words = chunk.text.split()
            if not (1 <= len(words) <= 5):
                continue

            phrase = chunk.text.lower().strip()

            # Check if it contains a cause indicator
            if any(ind in phrase for ind in cause_indicators):
                if not self._is_stopphrase(phrase) and phrase not in phrases:
                    phrases.append(phrase)

        return phrases[:15]  # Limit to avoid noise

    def _is_stopphrase(self, phrase: str) -> bool:
        """Check if phrase is uninformative."""
        stop_phrases = {
            "the",
            "a",
            "an",
            "i",
            "my",
            "work",
            "job",
            "role",
            "position",
            "year",
            "years",
            "time",
            "way",
            "thing",
            "things",
            "people",
            "person",
            "lot",
            "part",
            "good",
            "great",
            "new",
            "first",
            "their",
            "our",
            "your",
            "this",
            "that",
            "these",
            "those",
        }
        words = phrase.split()

        if len(words) == 1 and words[0] in stop_phrases:
            return True
        if all(w in stop_phrases for w in words):
            return True

        return False


# =============================================================================
# Utility Functions
# =============================================================================


def parse_semicolon_keywords(
    text: str,
    normalize: bool = True,
) -> list[str]:
    """Parse semicolon-separated keywords (for expertise/interests columns).

    Args:
        text: Semicolon-separated string like "AI safety; Machine learning"
        normalize: Whether to lowercase and deduplicate
    Returns:
        List of keywords

    Example:
        >>> parse_semicolon_keywords("AI safety; Machine learning; AI Safety")
        ['ai safety', 'machine learning']
    """
    if not text or str(text).lower() in ("nan", "none", ""):
        return []

    # Split by semicolon
    keywords = [kw.strip() for kw in str(text).split(";")]

    # Handle slash-separated sub-items
    expanded = []
    for kw in keywords:
        if "/" in kw and len(kw.split("/")) <= 3:
            expanded.extend(part.strip() for part in kw.split("/"))
        elif kw:
            expanded.append(kw)

    # Filter empty
    keywords = [kw for kw in expanded if kw]

    # Normalize
    if normalize:
        seen = set()
        normalized = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in seen:
                normalized.append(kw_lower)
                seen.add(kw_lower)
        return normalized

    return keywords
