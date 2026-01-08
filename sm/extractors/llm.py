"""LLM-based extraction using local Ollama instance.

This module provides extraction using a local LLM (via Ollama) with majority
voting to handle stochasticity. It's more accurate for nuanced cause areas
but slower than NLP extraction.

The majority voting approach:
1. Run the same extraction prompt N times (default: 3)
2. Collect all extracted items across runs
3. Keep only items that appear in >= threshold proportion of runs

This filters out hallucinations and random variations while keeping
consistently extracted items.

Parallel processing:
- When OLLAMA_NUM_PARALLEL is set, extraction runs and categories are
  processed concurrently using ThreadPoolExecutor for significant speedup.
"""

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from sm import cache, config
from sm.extractors.base import BaseExtractor, ExtractionResult

# =============================================================================
# Prompts
# =============================================================================

LOCATION_PROMPT = """Extract all geographic locations mentioned in this text.
Include: cities, countries, regions, states/provinces, and specific places.
Do NOT include organization names or generic terms.

Text: {text}

Return ONLY a JSON object with a "locations" key containing a list of location names.
Example: {{"locations": ["London", "United Kingdom", "Bay Area", "New York City"]}}"""


ORGANIZATION_PROMPT = """Extract all organizations mentioned in this text.
Include: companies, universities, research institutions, nonprofits, government agencies.
Use full official names when possible.

Text: {text}

Return ONLY a JSON object with an "organizations" key containing a list of organization names.
Example: {{"organizations": ["OpenAI", "University of Oxford", "GiveWell", "DeepMind"]}}"""


CAUSE_AREA_PROMPT = """You are extracting cause areas and focus topics from an EA (Effective Altruism) conference attendee's biography.

Extract the key cause areas, research interests, and thematic focus areas. Be specific and use standard EA terminology when applicable.

Common EA cause areas include (but are not limited to):
- AI safety, AI alignment, AI governance, machine learning safety
- Animal welfare, factory farming, alternative proteins, cultivated meat
- Global health, malaria prevention, neglected tropical diseases
- Global poverty, economic development, cash transfers
- Pandemic preparedness, biosecurity, biorisk
- Existential risk, longtermism, future generations
- Climate change, clean energy, environmental policy
- Nuclear risk, arms control
- Space governance, space policy
- Policy research, governance, institutional reform
- EA community building, effective giving

Text to analyze:
{text}

Return ONLY a JSON object with a "cause_areas" key containing a list of specific cause areas (1-4 words each).
Be specific: prefer "AI alignment" over just "AI", prefer "factory farming" over just "animals".
Example: {{"cause_areas": ["AI safety", "technical alignment research", "governance", "biosecurity"]}}"""


# =============================================================================
# Ollama API
# =============================================================================


class OllamaError(Exception):
    """Error communicating with Ollama API."""

    pass


def check_ollama_available(model: Optional[str] = None) -> tuple[bool, str]:
    """Check if Ollama is running and model is available.

    Args:
        model: Model to check for (uses config default if not provided)
    Returns:
        Tuple of (is_available, message)
    """
    model = model or config.OllamaConfig.DEFAULT_MODEL

    try:
        response = requests.get(
            f"{config.OllamaConfig.BASE_URL}/api/tags",
            timeout=5,
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        if model in model_names:
            return True, f"Ollama available with model '{model}'"
        else:
            available = ", ".join(model_names[:5]) or "none"
            return False, f"Model '{model}' not found. Available: {available}"

    except requests.exceptions.ConnectionError:
        return False, "Ollama not running. Start with: ollama serve"
    except Exception as e:
        return False, f"Error: {e}"


def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call local Ollama API.

    Args:
        prompt: Prompt to send
        model: Model name (uses config default if not provided)
        timeout: Request timeout (uses config default if not provided)
        temperature: Sampling temperature (uses config default if not provided)
    Returns:
        Response text
    Raises:
        OllamaError: If API call fails
    """
    model = model or config.OllamaConfig.DEFAULT_MODEL
    timeout = timeout or config.OllamaConfig.TIMEOUT
    temperature = temperature if temperature is not None else config.OllamaConfig.TEMPERATURE

    try:
        response = requests.post(
            f"{config.OllamaConfig.BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    except requests.exceptions.ConnectionError:
        raise OllamaError("Could not connect to Ollama. Make sure it's running: ollama serve")
    except requests.exceptions.Timeout:
        raise OllamaError(f"Request timed out after {timeout}s")
    except Exception as e:
        raise OllamaError(f"API error: {e}")


def parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response, handling common issues."""
    response = response.strip()

    # Remove markdown code blocks if present
    if "```" in response:
        matches = re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
        if matches:
            response = matches[0].strip()

    # Try to extract JSON object
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r"\{[^{}]*\}", response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from: {response[:200]}...")


# =============================================================================
# Majority Voting
# =============================================================================


def majority_vote(
    attempts: list[list[str]],
    threshold: Optional[float] = None,
) -> list[str]:
    """Apply majority voting to multiple extraction attempts.

    Args:
        attempts: List of extraction results (each is a list of strings)
        threshold: Minimum proportion of attempts that must include an item
                  (uses config default if not provided)
    Returns:
        List of items meeting the threshold, sorted by frequency
    """
    if not attempts:
        return []

    threshold = threshold if threshold is not None else config.OllamaConfig.DEFAULT_VOTE_THRESHOLD
    n_attempts = len(attempts)
    min_votes = max(1, int(n_attempts * threshold))

    # Count occurrences of each item (normalized)
    item_counts = Counter()
    for attempt in attempts:
        for item in attempt:
            if isinstance(item, str) and item.strip():
                item_counts[item.lower().strip()] += 1

    # Filter by threshold and sort by frequency
    voted_items = [item for item, count in item_counts.most_common() if count >= min_votes]

    return voted_items


# =============================================================================
# LLM Extractor
# =============================================================================


class LLMExtractor(BaseExtractor):
    """LLM-based extractor using Ollama with majority voting.

    Handles the stochasticity of LLM outputs by running extractions multiple
    times and applying majority voting to keep only consistent results.

    Example:
        extractor = LLMExtractor(n_runs=3)
        result = extractor.extract_all("I work on AI safety at Oxford")
        print(result.cause_areas)  # ['ai safety'] - only if found in 2+ runs
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        n_runs: Optional[int] = None,
        vote_threshold: Optional[float] = None,
        use_cache: bool = True,
        timeout: Optional[int] = None,
    ):
        """Initialize LLM extractor.

        Args:
            model_name: Ollama model name (uses config default if not provided)
            n_runs: Number of extraction runs for voting (uses config default)
            vote_threshold: Threshold for item inclusion (uses config default)
            use_cache: Whether to cache results
            timeout: API timeout in seconds (uses config default)
        """
        super().__init__(use_cache=use_cache)
        self._model_name = model_name or config.OllamaConfig.DEFAULT_MODEL
        self.n_runs = n_runs if n_runs is not None else config.OllamaConfig.DEFAULT_N_RUNS
        self.vote_threshold = (
            vote_threshold
            if vote_threshold is not None
            else config.OllamaConfig.DEFAULT_VOTE_THRESHOLD
        )
        self.timeout = timeout or config.OllamaConfig.TIMEOUT

    @property
    def method_name(self) -> str:
        return "llm"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _single_extraction_call(
        self,
        text: str,
        prompt_template: str,
        result_key: str,
        run_idx: int,
    ) -> Optional[list[str]]:
        """Execute a single extraction call (used for parallel execution).

        Args:
            text: Text to extract from
            prompt_template: Prompt template with {text} placeholder
            result_key: Key to extract from JSON response
            run_idx: Run index for error reporting
        Returns:
            List of extracted items or None on failure
        """
        try:
            prompt = prompt_template.format(text=text)
            response = call_ollama(
                prompt,
                model=self._model_name,
                timeout=self.timeout,
            )
            parsed = parse_json_response(response)

            # Extract and normalize items
            items = parsed.get(result_key, [])
            if isinstance(items, list):
                return [
                    str(item).lower().strip() for item in items if item and str(item).strip()
                ]
            return []

        except (OllamaError, ValueError) as e:
            print(f"LLM extraction attempt {run_idx + 1}/{self.n_runs} failed: {e}")
            return None

    def _run_extraction(
        self,
        text: str,
        prompt_template: str,
        result_key: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Run extraction multiple times and apply voting.

        Uses parallel execution when n_runs > 1 to leverage OLLAMA_NUM_PARALLEL.

        Args:
            text: Text to extract from
            prompt_template: Prompt template with {text} placeholder
            result_key: Key to extract from JSON response
        Returns:
            Tuple of (voted_results, all_attempts)
        """
        all_attempts = []

        if self.n_runs == 1:
            # Single run - no parallelization needed
            result = self._single_extraction_call(text, prompt_template, result_key, 0)
            if result is not None:
                all_attempts.append(result)
        else:
            # Parallel execution for majority voting
            with ThreadPoolExecutor(max_workers=self.n_runs) as executor:
                futures = {
                    executor.submit(
                        self._single_extraction_call, text, prompt_template, result_key, i
                    ): i
                    for i in range(self.n_runs)
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_attempts.append(result)

        if not all_attempts:
            return [], []

        # Apply majority voting
        voted = majority_vote(all_attempts, self.vote_threshold)
        return voted, all_attempts

    # =========================================================================
    # Extraction Methods
    # =========================================================================

    def extract_locations(self, text: str) -> list[str]:
        """Extract locations using LLM with majority voting."""
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_llm_extraction(self._model_name, "locations", text)
            if cached is not None:
                # Handle both old format (dict) and new format (list in result)
                result = cached.get("result", cached)
                if isinstance(result, dict):
                    return result.get("locations", [])
                return result if isinstance(result, list) else []

        result, attempts = self._run_extraction(text, LOCATION_PROMPT, "locations")

        # Cache - store list directly
        if self.use_cache:
            cache.save_llm_extraction(
                self._model_name,
                "locations",
                text,
                result,
                attempts=attempts,
            )

        return result

    def extract_organizations(self, text: str) -> list[str]:
        """Extract organizations using LLM with majority voting."""
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_llm_extraction(self._model_name, "organizations", text)
            if cached is not None:
                # Handle both old format (dict) and new format (list in result)
                result = cached.get("result", cached)
                if isinstance(result, dict):
                    return result.get("organizations", [])
                return result if isinstance(result, list) else []

        result, attempts = self._run_extraction(text, ORGANIZATION_PROMPT, "organizations")

        # Cache - store list directly
        if self.use_cache:
            cache.save_llm_extraction(
                self._model_name,
                "organizations",
                text,
                result,
                attempts=attempts,
            )

        return result

    def extract_cause_areas(self, text: str) -> list[str]:
        """Extract cause areas using LLM with majority voting."""
        if not text or not text.strip():
            return []

        # Check cache
        if self.use_cache:
            cached = cache.load_llm_extraction(self._model_name, "cause_areas", text)
            if cached is not None:
                # Handle both old format (dict with cause_areas key) and new format (list)
                result = cached.get("result", cached)
                if isinstance(result, dict):
                    return result.get("cause_areas", [])
                return result if isinstance(result, list) else []

        result, attempts = self._run_extraction(text, CAUSE_AREA_PROMPT, "cause_areas")

        # Cache - store as simple list for consistency
        if self.use_cache:
            cache.save_llm_extraction(
                self._model_name,
                "cause_areas",
                text,
                result,  # Store list directly, not wrapped in dict
                attempts=attempts,
            )

        return result

    def extract_all(self, text: str) -> ExtractionResult:
        """Extract all entity types from text using parallel execution.

        Runs locations, organizations, and cause_areas extractions concurrently
        for faster processing when OLLAMA_NUM_PARALLEL is configured.

        Args:
            text: Input text to process
        Returns:
            ExtractionResult containing all extractions
        """
        if not text or not str(text).strip():
            return ExtractionResult(method=self.method_name, model=self.model_name)

        text = str(text).strip()

        # Run all three extractions in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            loc_future = executor.submit(self.extract_locations, text)
            org_future = executor.submit(self.extract_organizations, text)
            cause_future = executor.submit(self.extract_cause_areas, text)

            return ExtractionResult(
                locations=loc_future.result(),
                organizations=org_future.result(),
                cause_areas=cause_future.result(),
                method=self.method_name,
                model=self.model_name,
            )
