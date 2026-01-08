"""Configuration settings for the EA Stakeholder Mapping package.

This module provides centralized configuration for:
- File paths (data, cache directories)
- API credentials (GeoNames, Google Maps)
- LLM settings (Ollama models, voting parameters)
- NLP settings (spaCy models)
"""

from pathlib import Path

import yaml


def get_repo_dir() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent


# =============================================================================
# Directory Paths
# =============================================================================

REPO_DIR = get_repo_dir()
DATA_DIR = REPO_DIR / "data"
CACHE_DIR = REPO_DIR / ".cache"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)


# =============================================================================
# Load YAML config if available
# =============================================================================


def _load_yaml_config() -> dict:
    """Load configuration from YAML file if it exists."""
    config_path = REPO_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_config = _load_yaml_config()


# =============================================================================
# API Credentials
# =============================================================================

GEONAMES_USERNAME: str = _yaml_config.get("geonames", {}).get("username", "orlandocode")
GOOGLE_MAPS_API_KEY: str = _yaml_config.get("google_maps", {}).get("api_key", "")


# =============================================================================
# LLM Settings (Ollama)
# =============================================================================


class OllamaConfig:
    """Configuration for Ollama LLM extraction."""

    # Connection
    BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "llama3.2"
    TIMEOUT: int = 300  # seconds (increased for parallel processing queues)

    # Majority voting
    DEFAULT_N_RUNS: int = 3
    DEFAULT_VOTE_THRESHOLD: float = 0.5  # Item must appear in 50%+ of runs

    # Temperature for extraction (lower = more deterministic)
    TEMPERATURE: float = 0.3


# =============================================================================
# NLP Settings (spaCy)
# =============================================================================


class NLPConfig:
    """Configuration for spaCy NLP extraction."""

    # Model preferences (will try in order)
    PREFERRED_MODEL: str = "en_core_web_trf"  # Transformer-based (most accurate)
    FALLBACK_MODEL: str = "en_core_web_sm"  # Small model (faster, less accurate)


# =============================================================================
# EA-Specific Cause Area Categories
# =============================================================================

# These categories help with semantic clustering and labeling
EA_CAUSE_CATEGORIES = {
    "ai_safety": {
        "keywords": [
            "ai safety",
            "ai alignment",
            "machine learning safety",
            "agi safety",
            "artificial intelligence",
            "ml safety",
            "ai governance",
            "ai policy",
            "superintelligence",
            "ai risk",
            "alignment research",
            "interpretability",
            "robustness",
            "ai ethics",
        ],
        "label": "AI Safety & Governance",
    },
    "animal_welfare": {
        "keywords": [
            "animal welfare",
            "factory farming",
            "animal rights",
            "farmed animals",
            "animal advocacy",
            "wild animal suffering",
            "cage-free",
            "animal protection",
            "speciesism",
            "sentience",
        ],
        "label": "Animal Welfare",
    },
    "alternative_proteins": {
        "keywords": [
            "alternative proteins",
            "cultivated meat",
            "plant-based",
            "cellular agriculture",
            "clean meat",
            "lab-grown meat",
            "meat alternatives",
            "food technology",
            "fermentation",
            "precision fermentation",
        ],
        "label": "Alternative Proteins",
    },
    "global_health": {
        "keywords": [
            "global health",
            "public health",
            "malaria",
            "disease",
            "vaccination",
            "healthcare",
            "medicine",
            "epidemiology",
            "health policy",
            "infectious disease",
            "neglected tropical diseases",
            "health systems",
        ],
        "label": "Global Health",
    },
    "pandemic_preparedness": {
        "keywords": [
            "pandemic",
            "biosecurity",
            "biodefense",
            "pandemic preparedness",
            "biological risk",
            "pathogen",
            "epidemic",
            "outbreak",
            "biorisk",
            "biological threats",
            "gain of function",
        ],
        "label": "Pandemic Preparedness & Biosecurity",
    },
    "global_poverty": {
        "keywords": [
            "global poverty",
            "economic development",
            "cash transfers",
            "microfinance",
            "extreme poverty",
            "international development",
            "aid effectiveness",
            "givewell",
            "givedirectly",
            "poverty alleviation",
        ],
        "label": "Global Poverty & Development",
    },
    "existential_risk": {
        "keywords": [
            "existential risk",
            "x-risk",
            "extinction risk",
            "catastrophic risk",
            "global catastrophic risk",
            "civilizational risk",
            "human extinction",
            "longtermism",
            "long-term future",
            "future generations",
        ],
        "label": "Existential Risk & Longtermism",
    },
    "climate_environment": {
        "keywords": [
            "climate change",
            "climate",
            "environment",
            "sustainability",
            "carbon",
            "emissions",
            "clean energy",
            "renewable energy",
            "decarbonization",
            "net zero",
            "environmental",
            "green",
        ],
        "label": "Climate & Environment",
    },
    "nuclear_risk": {
        "keywords": [
            "nuclear",
            "nuclear weapons",
            "nuclear war",
            "nuclear security",
            "arms control",
            "disarmament",
            "nuclear risk",
            "nuclear policy",
        ],
        "label": "Nuclear Risk",
    },
    "space_governance": {
        "keywords": [
            "space",
            "space governance",
            "space policy",
            "space settlement",
            "asteroid",
            "extraterrestrial",
            "cosmic",
            "space exploration",
            "interplanetary",
            "space law",
        ],
        "label": "Space Governance",
    },
    "policy_governance": {
        "keywords": [
            "policy",
            "governance",
            "government",
            "regulation",
            "legislation",
            "advocacy",
            "political",
            "democracy",
            "institutions",
            "international relations",
            "diplomacy",
            "think tank",
        ],
        "label": "Policy & Governance",
    },
    "research_academia": {
        "keywords": [
            "research",
            "academia",
            "academic",
            "science",
            "scientific",
            "university",
            "PhD",
            "professor",
            "publication",
            "peer review",
            "methodology",
            "empirical",
        ],
        "label": "Research & Academia",
    },
    "ea_community": {
        "keywords": [
            "effective altruism",
            "EA",
            "cause prioritization",
            "impact",
            "evidence-based",
            "cost-effectiveness",
            "charity evaluation",
            "doing good better",
            "high impact",
        ],
        "label": "EA Community & Meta",
    },
    "operations_management": {
        "keywords": [
            "operations",
            "management",
            "leadership",
            "strategy",
            "nonprofit",
            "organization building",
            "hiring",
            "team building",
            "executive",
            "operations management",
        ],
        "label": "Operations & Management",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_category_keywords() -> dict[str, list[str]]:
    """Get flat mapping of category to keywords."""
    return {cat: info["keywords"] for cat, info in EA_CAUSE_CATEGORIES.items()}


def get_category_labels() -> dict[str, str]:
    """Get mapping of category ID to display label."""
    return {cat: info["label"] for cat, info in EA_CAUSE_CATEGORIES.items()}


def get_all_cause_keywords() -> set[str]:
    """Get all cause area keywords as a flat set."""
    keywords = set()
    for info in EA_CAUSE_CATEGORIES.values():
        keywords.update(info["keywords"])
    return keywords


# Legacy compatibility aliases
repo_dir = REPO_DIR
data_dir = DATA_DIR
cache_dir = CACHE_DIR
