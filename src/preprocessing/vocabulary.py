"""
vocabulary.py — Word-level vocabulary builder for the LSTM model.

Builds a mapping word → integer index from the training corpus.
Handles OOV tokens and padding automatically.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import List


class Vocabulary:
    """Word-level vocabulary with PAD and UNK special tokens.

    Example::

        vocab = Vocabulary()
        vocab.build(train_texts, min_freq=2, max_size=30_000)
        ids = vocab.encode("tôi rất vui", max_len=20)
        vocab.save("data/processed/vocabulary.pkl")

        vocab2 = Vocabulary.load("data/processed/vocabulary.pkl")
    """

    PAD_TOKEN: str = "<PAD>"
    UNK_TOKEN: str = "<UNK>"
    PAD_IDX:   int = 0
    UNK_IDX:   int = 1

    def __init__(self) -> None:
        self._word2idx: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self._idx2word: dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }

    # ── Construction ─────────────────────────────────────────────────────────

    def build(
        self,
        texts: List[str],
        min_freq: int = 2,
        max_size: int = 30_000,
    ) -> None:
        """Build vocabulary from a list of tokenised strings.

        Args:
            texts    : whitespace-tokenised strings (after cleaning)
            min_freq : discard words with fewer occurrences
            max_size : maximum vocabulary size (including PAD / UNK)
        """
        counter: Counter = Counter()
        for text in texts:
            if isinstance(text, str):
                counter.update(text.split())

        # Sort by frequency descending, then alphabetically for ties
        candidates = sorted(
            [(w, c) for w, c in counter.items() if c >= min_freq],
            key=lambda x: (-x[1], x[0]),
        )

        # Reserve slots for PAD (0) and UNK (1)
        candidates = candidates[: max_size - 2]

        for idx, (word, _) in enumerate(candidates, start=2):
            self._word2idx[word] = idx
            self._idx2word[idx]  = word

    # ── Encoding / Decoding ──────────────────────────────────────────────────

    def encode(self, text: str, max_len: int) -> List[int]:
        """Convert text to a fixed-length list of token indices.

        Truncates if len(tokens) > max_len; pads with PAD_IDX otherwise.

        Args:
            text    : whitespace-tokenised input string
            max_len : output sequence length

        Returns:
            List of ``max_len`` integers.
        """
        tokens = text.split()[:max_len] if isinstance(text, str) else []
        ids = [self._word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token indices back to a string.

        Args:
            ids          : list of integer indices
            skip_special : if True, omit PAD and UNK tokens

        Returns:
            Space-joined string.
        """
        special = {self.PAD_IDX, self.UNK_IDX} if skip_special else set()
        words = [
            self._idx2word.get(i, self.UNK_TOKEN)
            for i in ids
            if i not in special
        ]
        return " ".join(words)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise vocabulary to disk using pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a previously saved Vocabulary from disk."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Total vocabulary size (including PAD and UNK)."""
        return len(self._word2idx)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, word: str) -> bool:
        return word in self._word2idx

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size})"
