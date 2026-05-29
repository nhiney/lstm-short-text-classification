"""
teencode_normalize.py — Vietnamese social-media slang → standard Vietnamese.

The dictionary covers common abbreviations, emoticons, English loanwords,
and internet-specific vocabulary found on Vietnamese social platforms.
"""
from __future__ import annotations

import re

# ── Slang → Standard Vietnamese dictionary ──────────────────────────────────
TEENCODE_DICT: dict[str, str] = {
    # ── Negation / quantity ──────────────────────────────────────────────────
    "ko": "không", "k": "không", "kg": "không", "hok": "không",
    "khong": "không", "kh": "không",
    # ── Pronouns ─────────────────────────────────────────────────────────────
    "mik": "mình", "mk": "mình", "t": "tôi", "tui": "tôi",
    "bn": "bạn", "bh": "bây giờ", "ng": "người",
    # ── Verbs / adjectives ───────────────────────────────────────────────────
    "dc": "được", "đc": "được", "dk": "được",
    "ok": "ổn", "oke": "ổn", "okey": "ổn", "okay": "ổn",
    "cx": "cũng", "cg": "cũng",
    "vs": "với", "voi": "với",
    "r": "rồi", "rui": "rồi",
    "j": "gì", "gi": "gì",
    "lm": "làm", "lam": "làm",
    # ── Emotion words ────────────────────────────────────────────────────────
    "thik": "thích",
    "yeu": "yêu", "iu": "yêu",
    "buon": "buồn", "sad": "buồn",
    "vui": "vui", "happy": "vui",
    "met": "mệt",
    "tuc": "tức",
    "gian": "giận",
    "so": "sợ",
    "love": "yêu", "hate": "ghét",
    # ── Internet / social media ──────────────────────────────────────────────
    "vl": "vãi lắm", "vcl": "vãi cả lắm",
    "wtf": "trời ơi", "omg": "trời ơi",
    "lol": "haha", "kkk": "haha", "hihi": "hehe",
    "huhu": "huhu",
    "af": "",          # filler, remove
    "btw": "nhân tiện", "imo": "theo tôi", "tbh": "thật ra",
    "ngl": "thật mà", "fr": "thật", "rly": "thật sự",
    "tks": "cảm ơn", "thx": "cảm ơn", "ty": "cảm ơn",
    "camon": "cảm ơn",
    "ntn": "như thế nào", "nma": "nhưng mà", "nhma": "nhưng mà",
    "đb": "đặc biệt", "ht": "hết",
    # ── Relationships ────────────────────────────────────────────────────────
    "ck": "chồng", "vk": "vợ",
    "gf": "bạn gái", "bf": "bạn trai",
    "crush": "người thích",
    # ── Social actions ───────────────────────────────────────────────────────
    "fl": "theo dõi", "unfl": "bỏ theo dõi",
    "dm": "tin nhắn", "sp": "ủng hộ",
    "rep": "trả lời", "cmt": "bình luận",
    "tag": "gắn thẻ", "share": "chia sẻ",
    "like": "thích", "tym": "tim",
    # ── Sentence-ending particles ────────────────────────────────────────────
    "nha": "nhé", "nh": "nhé", "nhe": "nhé",
    "nè": "này", "ne": "này",
    "thui": "thôi", "thoy": "thôi",
    "luon": "luôn",
    "z": "vậy", "v": "vậy",
}

# Sort by length (longest first) to avoid partial replacements
_SORTED_KEYS = sorted(TEENCODE_DICT.keys(), key=len, reverse=True)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SORTED_KEYS) + r")\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def normalize_teencode(text: str) -> str:
    """
    Replace teencode tokens with their standard Vietnamese equivalents.

    Args:
        text : cleaned Vietnamese text (after clean_text())

    Returns:
        Text with slang tokens replaced; extra whitespace collapsed.
    """
    if not isinstance(text, str):
        return ""

    def _replace(match: re.Match) -> str:
        return TEENCODE_DICT.get(match.group(0).lower(), match.group(0))

    normalised = _PATTERN.sub(_replace, text)
    # Collapse whitespace introduced by empty replacements
    return " ".join(normalised.split())
