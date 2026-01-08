"""Geographic map visualizations.

This module provides interactive choropleth maps showing:
- Country-level attendee distribution
- Organization location markers
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

try:
    import pycountry

    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False


# Country name variations for ISO code lookup
COUNTRY_NAME_VARIANTS = {
    "UK": "United Kingdom",
    "USA": "United States",
    "US": "United States",
    "United States of America": "United States",
    "The Netherlands": "Netherlands",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Russia": "Russian Federation",
    "Czech Republic": "Czechia",
    "Burma": "Myanmar",
    "Ivory Coast": "Côte d'Ivoire",
}


def get_country_iso3(country_name: str) -> Optional[str]:
    """Get ISO-3 code for a country name.

    Args:
        country_name: Country name (common name, official name, or ISO code)
    Returns:
        ISO-3 code (e.g., "USA", "GBR") or None if not found
    """
    if not PYCOUNTRY_AVAILABLE:
        return None

    if not country_name or not str(country_name).strip():
        return None

    country_name = str(country_name).strip()

    # Handle common variations
    if country_name in COUNTRY_NAME_VARIANTS:
        country_name = COUNTRY_NAME_VARIANTS[country_name]

    # Try lookup
    try:
        country = pycountry.countries.lookup(country_name)
        return country.alpha_3
    except LookupError:
        pass

    # Try searching
    try:
        for country in pycountry.countries:
            if (
                country_name.lower() in country.name.lower()
                or country_name.lower() in getattr(country, "common_name", "").lower()
            ):
                return country.alpha_3
    except (LookupError, AttributeError):
        pass

    # Try if it's already an ISO code
    try:
        if len(country_name) == 2:
            country = pycountry.countries.get(alpha_2=country_name.upper())
            if country:
                return country.alpha_3
        elif len(country_name) == 3:
            country = pycountry.countries.get(alpha_3=country_name.upper())
            if country:
                return country.alpha_3
    except (LookupError, AttributeError):
        pass

    return None


def assign_iso_codes(country_df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO codes to country DataFrame.

    Args:
        country_df: DataFrame with 'country' column
    Returns:
        DataFrame with 'ISO' column added
    """
    country_df = country_df.copy()
    country_df["ISO"] = country_df["country"].apply(get_country_iso3)

    # Report missing codes
    missing = country_df[country_df["ISO"].isna()]
    if len(missing) > 0:
        print(f"Warning: {len(missing)} countries without ISO codes: {missing['country'].tolist()}")

    return country_df.dropna(subset=["ISO"])


def create_interactive_map(
    country_df: pd.DataFrame,
    organization_df: Optional[pd.DataFrame] = None,
    title: str = "Geographic Distribution of Attendees",
) -> go.Figure:
    """Create interactive map with choropleth and organization markers.

    Args:
        country_df: DataFrame with 'country', 'count' columns
        organization_df: Optional DataFrame with 'organization', 'count', 'lat', 'lng'
        title: Map title
        height: Figure height in pixels
    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Process country data
    if country_df is not None and len(country_df) > 0:
        # Ensure ISO codes
        if "ISO" not in country_df.columns:
            country_df = assign_iso_codes(country_df)

        if len(country_df) > 0:
            fig.add_trace(
                go.Choropleth(
                    locations=country_df["ISO"],
                    z=country_df["count"],
                    text=country_df["country"],
                    colorscale="Blues",
                    marker_line_color="darkgray",
                    marker_line_width=0.5,
                    colorbar_title="Mentions",
                    hovertemplate="<b>%{text}</b><br>Mentions: %{z}<extra></extra>",
                )
            )

    # Add organization markers
    if organization_df is not None and len(organization_df) > 0:
        org_with_coords = organization_df.dropna(subset=["lat", "lng"])

        if len(org_with_coords) > 0:
            # Scale marker size
            max_count = org_with_coords["count"].max()
            min_size, max_size = 5, 25
            sizes = org_with_coords["count"].apply(
                lambda x: min_size + (x / max_count) * (max_size - min_size)
            )

            fig.add_trace(
                go.Scattergeo(
                    lon=org_with_coords["lng"],
                    lat=org_with_coords["lat"],
                    text=org_with_coords.apply(
                        lambda r: f"{r['organization']}: {r['count']} {'mentions' if r['count'] > 1 else 'mention'}",
                        axis=1,
                    ),
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color="#E85D04",
                        line=dict(width=1, color="darkred"),
                        opacity=0.7,
                    ),
                    name="Organizations",
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    showlegend=False,
                )
            )

    # dummy point for organisations
    fig.add_trace(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            text="",
            mode="markers",
            marker=dict(
                size=5,
                color="#E85D04",
                line=dict(width=1, color="darkred"),
                opacity=0.7,
            ),
            name="Organizations",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
            showocean=True,
            oceancolor="rgb(230, 245, 255)",
            showframe=False,
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.92,
            y=-0.2,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig


def create_map_with_dropdown(
    country_df: pd.DataFrame,
    organization_df: pd.DataFrame,
    attendee_df: pd.DataFrame,
) -> go.Figure:
    """Create map with attendee selection dropdown.

    Note: For large datasets, consider using Plotly Dash for better interactivity.

    Args:
        country_df: DataFrame with country mentions and 'attendee_ids'
        organization_df: DataFrame with organization mentions
        attendee_df: Original DataFrame with attendee data
    Returns:
        Plotly Figure with dropdown
    """
    # Ensure ISO codes
    country_df = assign_iso_codes(country_df)

    # Create base figure
    fig = create_interactive_map(country_df, organization_df)

    # Get attendee IDs
    attendee_ids = list(attendee_df.index)[:50]  # Limit for performance

    # Create dropdown buttons
    buttons = [
        {
            "label": "All Attendees",
            "method": "restyle",
            "args": [
                {
                    "z": [country_df["count"].tolist()],
                    "locations": [country_df["ISO"].tolist()],
                    "text": [country_df["country"].tolist()],
                },
                [0],
            ],
        }
    ]

    for att_id in attendee_ids:
        filtered = country_df[
            country_df["attendee_ids"].apply(
                lambda ids: att_id in ids if isinstance(ids, list) else False
            )
        ]

        buttons.append(
            {
                "label": f"Attendee {att_id}",
                "method": "restyle",
                "args": [
                    {
                        "z": [filtered["count"].tolist()],
                        "locations": [filtered["ISO"].tolist()],
                        "text": [filtered["country"].tolist()],
                    },
                    [0],
                ],
            }
        )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "y": 1.02,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
            }
        ],
        annotations=[
            {
                "text": "Filter:",
                "x": 0.02,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            }
        ],
    )

    return fig
