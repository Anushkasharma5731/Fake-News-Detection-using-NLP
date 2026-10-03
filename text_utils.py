"""Text normalization shared by model training and the web app."""

import re

import pandas as pd


def clean_text(text: object) -> str:
    """Normalize an article before vectorization."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
