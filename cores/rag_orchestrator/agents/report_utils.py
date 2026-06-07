"""Shared utilities for report generation."""
import re

# v6.4.2: Italian prepositions — longest-first to avoid partial matches (S6-01 fix)
# Articulated: preposition + article (sulla, dello, nell', etc.)
# Base: simple prepositions (su, di, per, etc.)
_IT_PREPS = (
    "sulla|sullo|sulle|sugli|sull|sui|sul"  # su + articles
    "|della|dello|delle|degli|dell|dei|del"  # di + articles
    "|nella|nello|nelle|negli|nell|nei|nel"  # in + articles
    "|alla|allo|alle|agli|all|ai|al"         # a + articles
    "|dalla|dallo|dalle|dagli|dall|dai|dal"  # da + articles
    "|su|di|per|in|a|da|con|tra|fra"         # base (LAST — shortest)
)
_APO = r"[''']?"  # optional apostrophe (straight + curly)


def extract_subject(query: str, fallback: str = "") -> str:
    """Extract subject/topic from query for report naming.

    Removes common report request patterns to isolate the subject.
    v6.4.2: Expanded Italian preposition support (S6-01 fix).
    """
    patterns = [
        rf"^(fammi|fai|genera|crea|scrivi)\s+(un\s+)?(report|analisi|documento|sintesi)\s+({_IT_PREPS}){_APO}\s*",
        rf"^(report|analisi|audit|confronto|ricerca)\s+({_IT_PREPS}){_APO}\s*",
        rf"^(voglio|vorrei|mi serve)\s+(un\s+)?(report|analisi)\s+({_IT_PREPS}){_APO}\s*",
    ]

    subject = query
    for pattern in patterns:
        subject = re.sub(pattern, "", subject, flags=re.IGNORECASE).strip()

    if subject:
        subject = subject[0].upper() + subject[1:]

    return subject or fallback or query
