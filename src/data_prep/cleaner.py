"""
Data leakage filtering for clinical notes.

Notes from MIMIC-IV may contain phrases that trivially leak the mortality label
(e.g. "patient expired", "deceased", "DNR"). Training contrastive embeddings on
such notes lets the model shortcut on the phrase rather than learning genuine
clinical semantics. This module filters them out before pairing.
"""

from __future__ import annotations

import polars as pl

# Case-insensitive phrases that indicate post-hoc knowledge of mortality or
# end-of-life decisions made before / at time of death.
LEAKAGE_PHRASES: list[str] = [
    r"expired",
    r"deceased",
    r"autopsy",
    r"\bDNR\b",
    r"do not resuscitate",
    r"\bDNI\b",
    r"do not intubate",
    r"comfort care",
    r"comfort measures",
    r"withdraw(?:al)?\s+(?:of\s+)?care",
    r"withdrew\s+care",
    r"passed away",
    r"pronounced dead",
    r"time of death",
]

LEAKAGE_REGEX = "(?i)(" + "|".join(LEAKAGE_PHRASES) + ")"


def filter_leakage_phrases(notes: pl.DataFrame, text_col: str = "text") -> pl.DataFrame:
    """
    Removes notes whose `text_col` matches any LEAKAGE_PHRASES pattern.

    Returns a new DataFrame with offending rows dropped and prints how many
    were removed.
    """
    if len(notes) == 0:
        return notes

    mask = pl.col(text_col).str.contains(LEAKAGE_REGEX)
    kept = notes.filter(~mask)
    dropped = len(notes) - len(kept)
    if dropped:
        print(
            f"    [cleaner] dropped {dropped} notes matching leakage phrases "
            f"({dropped / len(notes):.1%})"
        )
    return kept
