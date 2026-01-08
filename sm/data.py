"""Data loading and preprocessing utilities.

This module handles loading attendee data from CSV/Excel files and provides
utilities for working with text columns.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from sm import config


def load_attendee_data(
    filepath: Optional[str] = None,
    skip_rows: int = 5,
    anonymize: bool = True,
) -> pd.DataFrame:
    """Load and preprocess attendee data from CSV/Excel.

    Args:
        filepath: Path to data file. Uses default EAGx data if not provided.
        skip_rows: Number of header rows to skip (default: 5 for EAGx export)
        anonymize: Whether to drop identifying columns (default: True)
    Returns:
        Preprocessed DataFrame

    Example:
        df = load_attendee_data()
        df = load_attendee_data("data/my_event.csv", skip_rows=0)
    """
    if filepath is None:
        filepath = config.DATA_DIR / "EAGx_Amsterdam_11_12_25.csv"

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load based on file type
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath, skiprows=skip_rows)
    else:
        df = pd.read_csv(filepath, skiprows=skip_rows)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Standard column renames for EAGx data
    rename_map = {
        "how others can help me": "help_me",
        "how i can help others": "help_others",
        "job title": "job",
        "career stage": "career",
        "areas of expertise": "expertise",
        "areas of interest": "interests",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns},
        inplace=True,
    )

    # Anonymize if requested
    if anonymize:
        drop_cols = ["first name", "last name", "swapcard", "linkedin", "email"]
        df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            inplace=True,
            errors="ignore",
        )

    return df


def get_text_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Identify text columns by type for extraction.

    Args:
        df: DataFrame to analyze
    Returns:
        Dictionary with column lists for each type:
        - freeform_cols: Biography and help columns
        - semicolon_cols: Expertise/interests (semicolon-separated)
        - organization_cols: Columns likely to contain organization names
    """
    all_cols = set(df.columns)

    return {
        "freeform_cols": [c for c in ["biography", "help_me", "help_others"] if c in all_cols],
        "semicolon_cols": [c for c in ["expertise", "interests"] if c in all_cols],
        "organization_cols": [c for c in ["company", "biography"] if c in all_cols],
    }


def combine_text_columns(
    df: pd.DataFrame,
    columns: list[str],
    separator: str = " ",
) -> pd.Series:
    """Combine multiple text columns into a single text Series.

    Args:
        df: DataFrame with text columns
        columns: List of column names to combine
        separator: Separator between column values
    Returns:
        Series with combined text

    Example:
        combined = combine_text_columns(df, ["biography", "help_me"])
    """
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        raise ValueError(f"None of the columns {columns} found in DataFrame")

    return df[valid_cols].fillna("").astype(str).agg(separator.join, axis=1)


def get_sample_data(n_rows: int = 5) -> pd.DataFrame:
    """Get sample data for testing.

    Args:
        n_rows: Number of sample rows
    Returns:
        Sample DataFrame
    """
    return pd.DataFrame(
        {
            "biography": [
                "I work on AI safety research at Oxford University.",
                "Based in San Francisco, focusing on animal welfare and alternative proteins.",
                "PhD student at Cambridge studying global health interventions.",
                "Policy researcher at Rethink Priorities, interested in existential risk.",
                "Software engineer at Google, passionate about effective giving.",
            ][:n_rows],
            "company": [
                "Oxford University",
                "Good Food Institute",
                "Cambridge University",
                "Rethink Priorities",
                "Google",
            ][:n_rows],
            "expertise": [
                "AI safety; Machine learning; Technical research",
                "Alternative proteins; Cellular agriculture; Food science",
                "Global health; Epidemiology; RCT design",
                "Policy analysis; Cause prioritization; EA strategy",
                "Software engineering; Data science; Impact measurement",
            ][:n_rows],
            "interests": [
                "AI governance; Biosecurity; Career advice",
                "Factory farming; Animal advocacy; Climate",
                "Malaria; Neglected diseases; Health policy",
                "Longtermism; Nuclear risk; Forecasting",
                "Effective giving; Career change; EA community",
            ][:n_rows],
        }
    )
