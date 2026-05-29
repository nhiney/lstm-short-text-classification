"""
clean_text.py — Basic Vietnamese social-media text cleaning.
"""
from __future__ import annotations

import re
import unicodedata

# ── Compiled regex patterns ──────────────────────────────────────────────────
_URL     = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_EMOJI   = re.compile(
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
# Keep Vietnamese characters and basic Latin; strip everything else
_SPECIAL = re.compile(
    r"[^\w\s"
    r"áàảãạăắằẳẵặâấầẩẫậ"
    r"éèẻẽẹêếềểễệ"
    r"íìỉĩị"
    r"óòỏõọôốồổỗộơớờởỡợ"
    r"úùủũụưứừửữự"
    r"ýỳỷỹỵ"
    r"đ"
    r"]",
    re.UNICODE,
)
_SPACES = re.compile(r"\s{2,}")


def clean_text(text: str, remove_emoji: bool = True) -> str:
    """
    Cleaning pipeline for Vietnamese social-media text.

    Steps:
        1. NFC unicode normalisation
        2. Lowercase
        3. Strip URLs, @mentions, #hashtags
        4. Optionally strip emojis (keep for social-feature extraction)
        5. Strip remaining special characters (preserves Vietnamese diacritics)
        6. Collapse consecutive whitespace

    Args:
        text         : raw input string
        remove_emoji : set False when extracting emoji-based features

    Returns:
        Cleaned string (empty string if input is not a str).
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    if remove_emoji:
        text = _EMOJI.sub(" ", text)
    text = _SPECIAL.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text
