"""
social_features.py — Extract social-media–specific numeric features from raw text.

Features capture stylistic signals that correlate with emotion expression in
Vietnamese social media (e.g., emoji use for joy, repeated characters for anger).
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

# ── Compiled patterns ────────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_HASHTAG_RE     = re.compile(r"#\w+")
_MENTION_RE     = re.compile(r"@\w+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")   # 3+ consecutive identical chars
_UPPER_RE       = re.compile(
    r"[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]"
)

FEATURE_NAMES: list[str] = [
    "text_length",        # total character count
    "word_count",         # whitespace-delimited token count
    "emoji_count",        # number of emoji sequences
    "hashtag_count",      # number of #tags
    "mention_count",      # number of @mentions
    "exclamation_count",  # number of "!" characters
    "uppercase_count",    # number of uppercase letters
    "repeated_char_count",# tokens with 3+ repeated chars (e.g. "haaaaa")
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(text: str) -> list[float]:
    """Compute social features for a single text sample.

    Args:
        text : raw (un-cleaned) text to preserve emoji / hashtag information

    Returns:
        List of ``NUM_FEATURES`` float values.
    """
    if not isinstance(text, str):
        return [0.0] * NUM_FEATURES

    return [
        float(len(text)),
        float(len(text.split())),
        float(len(_EMOJI_RE.findall(text))),
        float(len(_HASHTAG_RE.findall(text))),
        float(len(_MENTION_RE.findall(text))),
        float(text.count("!")),
        float(len(_UPPER_RE.findall(text))),
        float(len(_REPEAT_CHAR_RE.findall(text))),
    ]


def extract_features_df(texts) -> pd.DataFrame:
    """Extract features for a pandas Series or list of texts.

    Returns:
        DataFrame with columns = FEATURE_NAMES.
    """
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def normalize_features(
    df_feat: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalise feature matrix to [0, 1] using training statistics.

    Args:
        df_feat : DataFrame of raw feature values (training set only)

    Returns:
        Tuple of (normalised_array, feat_mins, feat_maxs),
        where feat_mins / feat_maxs have shape ``(NUM_FEATURES,)``.
    """
    arr  = df_feat.values.astype(np.float32)
    mins = arr.min(axis=0)          # shape (NUM_FEATURES,)
    maxs = arr.max(axis=0)
    return apply_normalization(arr, mins, maxs), mins, maxs


def apply_normalization(
    arr: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed min-max normalisation.

    Args:
        arr  : raw feature array, shape (N, NUM_FEATURES)
        mins : per-feature minimums from training set
        maxs : per-feature maximums from training set

    Returns:
        Normalised array with the same shape.
    """
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (arr.astype(np.float32) - mins) / denom
