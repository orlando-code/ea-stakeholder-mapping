"""Geographic analysis: geocoding and location aggregation.

This module provides tools to:
- Geocode locations to countries using GeoNames API
- Geocode organizations to coordinates using Google Maps API
- Aggregate location/organization mentions by country

The results can be used for geographic visualization.
"""

import re
from collections import defaultdict
from typing import Optional

import pandas as pd
import requests
from tqdm.auto import tqdm

from sm import cache, config

# =============================================================================
# Text Formatting Utilities
# =============================================================================


def format_organization_name(name: str) -> str:
    """Format organization name with proper casing.

    - Title case by default
    - Preserve words with adjacent uppercase (acronyms like "MIT", "UNESCO", "EA")
    - Preserve camelCase words (like "GiveWell", "OpenAI")
    - Preserve all-caps short words as acronyms

    Args:
        name: Raw organization name
    Returns:
        Properly formatted name
    """
    if not name:
        return name

    words = name.split()
    formatted_words = []

    for word in words:
        # Check conditions to preserve original casing:
        # 1. Has adjacent uppercase letters (acronyms like MIT, UNESCO, EA)
        has_adjacent_upper = bool(re.search(r"[A-Z]{2,}", word))
        # 2. Has camelCase pattern (uppercase letter after lowercase)
        has_camel_case = bool(re.search(r"[a-z][A-Z]", word))
        # 3. All uppercase and short (likely acronym)
        is_short_caps = word.isupper() and len(word) <= 5

        if has_adjacent_upper or has_camel_case or is_short_caps:
            # Keep as-is (it's an acronym or has intentional casing)
            formatted_words.append(word)
        else:
            # Title case
            formatted_words.append(word.title())

    return " ".join(formatted_words)


# Google Maps result types that indicate a real organization/establishment
ORGANIZATION_TYPES = {
    "establishment",
    "university",
    "school",
    "point_of_interest",
    "premise",
    "organization",
    "corporate",
    "company",
    "institution",
    "hospital",
    "research",
    "foundation",
    "embassy",
    "local_government_office",
    "city_hall",
}

# Types that indicate a generic geographic location (not an organization)
GENERIC_LOCATION_TYPES = {
    "political",
    "locality",
    "administrative_area_level_1",
    "administrative_area_level_2",
    "administrative_area_level_3",
    "country",
    "continent",
    "natural_feature",
    "postal_code",
    "route",
    "street_address",
    "neighborhood",
    "sublocality",
}

# =============================================================================
# Geocoding Functions
# =============================================================================


def query_geonames(location: str) -> Optional[dict]:
    """Query GeoNames API for a location."""
    response = requests.get(
        "http://api.geonames.org/searchJSON",
        params={
            "q": location,
            "maxRows": 1,
            "username": config.GEONAMES_USERNAME,
            "featureClass": "P",
        },
        timeout=10,
    )
    return response.json()


def geocode_location(
    location: str,
    force_reload: bool = False,
) -> Optional[dict]:
    """Geocode a location to country and coordinates using GeoNames.

    Args:
        location: Location name (city, region, country)
        force_reload: Bypass cache
    Returns:
        Dict with 'location', 'country', 'country_code', 'lat', 'lng'
        or None if not found
    """
    if not location or not str(location).strip():
        return None

    location = str(location).strip()

    # Check cache
    if not force_reload:
        cached = cache.load_geocoding("geonames", location)
        if not cache.is_none_cached(cached):  # if cached is None, continue without querying the API
            return cached

    # Query GeoNames API
    try:
        data = query_geonames(location)

        if data.get("geonames"):
            result = data["geonames"][0]
            geocoded = {
                "location": location,
                "name": result["name"],
                "country": result.get("countryName"),
                "country_code": result.get("countryCode"),
                "lat": float(result["lat"]),
                "lng": float(result["lng"]),
            }
            cache.save_geocoding("geonames", location, geocoded)
            return geocoded
        else:
            cache.save_geocoding("geonames", location, None)
            return None

    except Exception as e:
        print(f"Geocoding error for '{location}': {e}")
        return None


def _is_valid_organization_result(result: dict, organization: str) -> bool:
    """Check if a Google Maps result represents a real organization.

    Filters out results that are just generic geographic locations
    (e.g., "Moral Ambition Circle" matching a random place in India).

    Args:
        result: Google Maps geocode result
        organization: Original organization name searched
    Returns:
        True if result appears to be a real organization
    """
    result_types = set(result.get("types", []))

    # manual overrides
    if organization.lower() in [
        "health progress hub",
        "moral ambition circle",
        "university of ea",
        "university of (no",
        "Ecolife",
        "catalyze",
        "brain bar",
        "Emergency Response Coordination Centre (Ercc)",
        "ocean oasis",
        "European Startup Space",
        "effective altruism foundation",
        "european union",
        "eu (european union)",
        "ea society",
        "albert schweitzer foundation",  # Germany, not Malaysia
    ]:  # N.B. latter two indicate issue with LLM extraction rather than geocoding
        return False
    if organization in ["Jci Nigeria (Buk)"]:
        return True

    # If result has organization-related types, it's valid
    if result_types & ORGANIZATION_TYPES:
        return True

    # If result is ONLY generic location types, reject it
    if result_types and result_types.issubset(GENERIC_LOCATION_TYPES):
        return False

    # Check if organization name appears in the result name/address
    formatted_address = result.get("formatted_address", "").lower()
    org_lower = organization.lower()

    # If the result address doesn't contain any significant words from org name,
    # it's probably a false positive
    org_words = [w for w in org_lower.split() if len(w) > 3]
    if org_words:
        matches = sum(1 for w in org_words if w in formatted_address)
        if matches < len(org_words) * 0.3:  # Less than 30% word match
            return False

    # Default: accept if we can't determine otherwise
    return True


def geocode_organization(
    organization: str,
    force_reload: bool = False,
) -> Optional[dict]:
    """Geocode an organization to coordinates using Google Maps API.

    Only returns results when confident the geocode represents the actual
    organization (not a random geographic location with a similar name).

    Args:
        organization: Organization name
        force_reload: Bypass cache
    Returns:
        Dict with 'organization', 'country', 'lat', 'lng' or None
    """
    if not organization or not str(organization).strip():
        return None

    organization = str(organization).strip()

    # Check cache if not forcing reload
    if not force_reload:
        cached = cache.load_geocoding("google_maps", organization)
        if not cache.is_none_cached(cached):  # if cached is None, continue without querying the API
            return cached

    # Require Google Maps API key
    if not config.GOOGLE_MAPS_API_KEY:
        return None

    try:
        import googlemaps

        gmaps = googlemaps.Client(key=config.GOOGLE_MAPS_API_KEY)
        results = gmaps.geocode(organization)

        if results:
            result = results[0]

            # Validate this is actually an organization, not a random location
            if not _is_valid_organization_result(result, organization):
                cache.save_geocoding("google_maps", organization, None)
                return None

            location = result.get("geometry", {}).get("location", {})

            # Extract country
            country = ""
            country_code = ""
            for component in result.get("address_components", []):
                if "country" in component.get("types", []):
                    country = component.get("long_name", "")
                    country_code = component.get("short_name", "").upper()
                    break

            geocoded = {
                "organization": format_organization_name(organization),
                "name": result.get("formatted_address", organization),
                "country": country,
                "country_code": country_code,
                "lat": float(location.get("lat", 0)),
                "lng": float(location.get("lng", 0)),
            }
            cache.save_geocoding("google_maps", organization, geocoded)
            return geocoded
        else:
            cache.save_geocoding("google_maps", organization, None)
            return None

    except ImportError:
        print("Warning: googlemaps package not installed")
        return None
    except Exception as e:
        print(f"Organization geocoding error for '{organization}': {e}")
        cache.save_geocoding("google_maps", organization, None)
        return None


# =============================================================================
# Batch Geocoding
# =============================================================================


def geocode_locations_batch(
    locations: list[str],
    force_reload: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Geocode multiple locations.

    Args:
        locations: List of location names
        force_reload: Bypass cache
        progress: Show progress bar
    Returns:
        List of successfully geocoded results
    """
    results = []
    unique_locations = list(set(loc for loc in locations if loc and str(loc).strip()))

    iterator = tqdm(unique_locations, desc="Geocoding locations") if progress else unique_locations

    for location in iterator:
        result = geocode_location(location, force_reload=force_reload)
        if result and result.get("country"):
            results.append(result)

    return results


def geocode_organizations_batch(
    organizations: list[str],
    force_reload: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Geocode multiple organizations.

    Args:
        organizations: List of organization names
        force_reload: Bypass cache
        progress: Show progress bar
    Returns:
        List of successfully geocoded results
    """
    results = []
    unique_orgs = list(set(org for org in organizations if org and str(org).strip()))

    iterator = tqdm(unique_orgs, desc="Geocoding organizations") if progress else unique_orgs

    for org in iterator:
        result = geocode_organization(org, force_reload=force_reload)
        if result and result.get("lat") and result.get("lng"):
            results.append(result)

    return results


# =============================================================================
# Aggregation Functions
# =============================================================================


def aggregate_country_mentions(
    df: pd.DataFrame,
    location_columns: list[str],
    force_reload: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Aggregate location mentions by country.

    Each attendee (row) contributes at most 1 count per country.

    Args:
        df: DataFrame with location columns (containing lists of locations)
        location_columns: Column names containing location lists
        force_reload: Bypass geocoding cache
        progress: Show progress bar
    Returns:
        DataFrame with 'country', 'count', 'attendee_ids'
    """
    # Validate columns
    valid_cols = [c for c in location_columns if c in df.columns]
    if not valid_cols:
        raise ValueError(f"No valid location columns found: {location_columns}")

    # Collect all unique locations
    all_locations = set()
    attendee_locations = defaultdict(list)

    for idx, row in df.iterrows():
        for col in valid_cols:
            if pd.isna(row[col]).any():
                continue

            locations = row[col]
            if isinstance(locations, list):
                for loc in locations:
                    if loc and str(loc).strip():
                        loc_str = str(loc).strip()
                        all_locations.add(loc_str)
                        attendee_locations[idx].append(loc_str)

    if not all_locations:
        return pd.DataFrame(columns=["country", "count", "attendee_ids"])

    # Geocode all unique locations
    location_to_country = {}

    iterator = tqdm(all_locations, desc="Geocoding locations") if progress else all_locations
    for location in iterator:
        result = geocode_location(location, force_reload=force_reload)
        if result and result.get("country"):
            location_to_country[location] = result["country"]

    # Map attendees to countries (each attendee counts once per country)
    attendee_countries = defaultdict(set)
    for attendee_id, locations in attendee_locations.items():
        for loc in locations:
            country = location_to_country.get(loc)
            if country:
                attendee_countries[attendee_id].add(country)

    # Count country mentions
    country_counts = defaultdict(lambda: {"count": 0, "attendee_ids": []})
    for attendee_id, countries in attendee_countries.items():
        for country in countries:
            country_counts[country]["count"] += 1
            country_counts[country]["attendee_ids"].append(attendee_id)

    # Build result DataFrame
    result_data = [
        {"country": country, "count": data["count"], "attendee_ids": data["attendee_ids"]}
        for country, data in country_counts.items()
    ]

    return pd.DataFrame(result_data).sort_values("count", ascending=False).reset_index(drop=True)


def aggregate_organization_mentions(
    df: pd.DataFrame,
    organization_columns: list[str],
    force_reload: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Aggregate organization mentions with geocoding.

    Args:
        df: DataFrame with organization columns (containing lists)
        organization_columns: Column names containing organization lists
        force_reload: Bypass geocoding cache
    Returns:
        DataFrame with 'organization', 'count', 'lat', 'lng', 'country', 'attendee_ids'
    """
    # Validate columns
    valid_cols = [c for c in organization_columns if c in df.columns]
    if not valid_cols:
        raise ValueError(f"No valid organization columns found: {organization_columns}")

    # Collect organizations per attendee
    attendee_orgs = defaultdict(set)

    for idx, row in df.iterrows():
        for col in valid_cols:
            if col not in row or pd.isna(row[col]).any():
                continue

            orgs = row[col]
            if isinstance(orgs, list):
                for org in orgs:
                    if org and str(org).strip():
                        org_name = _normalize_organization(str(org).strip())
                        attendee_orgs[idx].add(org_name)

    # Count organization mentions
    org_counts = defaultdict(lambda: {"count": 0, "attendee_ids": []})
    for attendee_id, orgs in attendee_orgs.items():
        for org in orgs:
            org_counts[org]["count"] += 1
            org_counts[org]["attendee_ids"].append(attendee_id)

    # Geocode organizations
    unique_orgs = list(org_counts.keys())
    geocoded = geocode_organizations_batch(
        unique_orgs, force_reload=force_reload, progress=progress
    )
    org_geo_map = {g["organization"]: g for g in geocoded}

    # Build result DataFrame
    result_data = []
    for org, data in org_counts.items():
        geo = org_geo_map.get(org.title(), {})
        result_data.append(
            {
                "organization": org,
                "count": data["count"],
                "lat": geo.get("lat"),
                "lng": geo.get("lng"),
                "country": geo.get("country"),
                "attendee_ids": data["attendee_ids"],
            }
        )

    return pd.DataFrame(result_data).sort_values("count", ascending=False).reset_index(drop=True)


def _normalize_organization(name: str) -> str:
    """Normalize organization names for consistency.

    - Standardizes university name formats
    - Applies proper title casing (preserving acronyms)
    """
    name = name.strip()
    name_lower = name.lower()

    # Standardize university names
    if "university" in name_lower:
        # cut off anything after a comma (and the comma) since this messes with the geocoding
        parts = name_lower.replace("university", "").replace("of", "").replace(",", "").split()
        if len(parts) > 1:
            location = " ".join(p.strip() for p in parts if p.strip())
            if location:
                return f"University of {location.title()}"

    # Apply organization name formatting
    return format_organization_name(name)


# =============================================================================
# Convenience Functions
# =============================================================================


def prepare_geographic_data(
    df: pd.DataFrame,
    location_columns: list[str],
    organization_columns: list[str],
    force_reload: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare all geographic data for visualization.

    Args:
        df: DataFrame with extracted locations and organizations
        location_columns: Columns containing location lists
        organization_columns: Columns containing organization lists
        force_reload: Bypass geocoding cache
    Returns:
        Tuple of (country_df, organization_df)
    """
    country_df = aggregate_country_mentions(df, location_columns, force_reload)
    org_df = aggregate_organization_mentions(df, organization_columns, force_reload)

    return country_df, org_df
