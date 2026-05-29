"""
preprocess.py — End-to-end preprocessing pipeline.

Pipeline:
    raw xlsx  →  clean  →  normalise teencode  →  extract social features
              →  encode labels  →  stratified split  →  normalise features
              →  build LSTM vocabulary  →  save CSVs + artefacts
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.clean_text import clean_text
from src.preprocessing.teencode_normalize import normalize_teencode
from src.preprocessing.social_features import (
    FEATURE_NAMES,
    apply_normalization,
    extract_features_df,
    normalize_features,
)
from src.preprocessing.vocabulary import Vocabulary
from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_preprocessing(save: bool = True) -> dict:
    """Execute the full preprocessing pipeline.

    Args:
        save : persist artefacts to ``CFG.data.processed_dir`` when True

    Returns:
        Dict with keys: train_df, val_df, test_df, label_encoder,
        feat_mins, feat_maxs, vocabulary.
    """
    set_seed(CFG.data.random_seed)
    processed_dir = Path(CFG.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw dataset ──────────────────────────────────────────────────
    logger.info("Loading dataset from %s", CFG.data.raw_path)
    df = pd.read_excel(CFG.data.raw_path)
    logger.info("Loaded %d rows | columns: %s", len(df), df.columns.tolist())

    df = df[[CFG.data.text_col, CFG.data.label_col]].dropna().reset_index(drop=True)
    df.columns = ["text_raw", "label"]

    # ── 2. Clean text ────────────────────────────────────────────────────────
    logger.info("Cleaning text ...")
    # Keep emojis so that social-feature counts are accurate
    df["text_clean"] = df["text_raw"].apply(lambda t: clean_text(t, remove_emoji=False))

    # ── 3. Normalise teencode ────────────────────────────────────────────────
    logger.info("Normalising teencode ...")
    df["text_norm"] = df["text_clean"].apply(normalize_teencode)

    # ── 4. Social features (computed on raw text) ────────────────────────────
    logger.info("Extracting social features ...")
    feat_df = extract_features_df(df["text_raw"])
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # ── 5. Label encoding ────────────────────────────────────────────────────
    le = LabelEncoder()
    # Fix class order to match CFG.labels.codes
    le.classes_ = np.array(sorted(df["label"].unique()))
    df["label_enc"] = le.transform(df["label"])
    logger.info("Classes: %s", le.classes_.tolist())

    # ── 6. Stratified train / val / test split ───────────────────────────────
    train_df, tmp_df = train_test_split(
        df,
        test_size=1.0 - CFG.data.train_ratio,
        stratify=df["label"],
        random_state=CFG.data.random_seed,
    )
    relative_val = CFG.data.val_ratio / (CFG.data.val_ratio + CFG.data.test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=1.0 - relative_val,
        stratify=tmp_df["label"],
        random_state=CFG.data.random_seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    logger.info(
        "Split → train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── 7. Normalise social features (fit on train only) ─────────────────────
    feat_arr, feat_mins, feat_maxs = normalize_features(train_df[FEATURE_NAMES])
    train_df[FEATURE_NAMES] = feat_arr
    val_df[FEATURE_NAMES]   = apply_normalization(val_df[FEATURE_NAMES].values,  feat_mins, feat_maxs)
    test_df[FEATURE_NAMES]  = apply_normalization(test_df[FEATURE_NAMES].values, feat_mins, feat_maxs)

    # ── 8. Build LSTM vocabulary (fit on train only) ─────────────────────────
    logger.info("Building vocabulary ...")
    vocab = Vocabulary()
    vocab.build(
        train_df["text_norm"].tolist(),
        min_freq=CFG.preprocessing.min_word_freq,
        max_size=CFG.preprocessing.max_vocab_size,
    )
    logger.info("Vocabulary size: %d", vocab.size)

    # ── 9. Save artefacts ────────────────────────────────────────────────────
    if save:
        train_df.to_csv(processed_dir / "train.csv", index=False)
        val_df.to_csv(processed_dir / "val.csv",     index=False)
        test_df.to_csv(processed_dir / "test.csv",   index=False)

        with open(processed_dir / "label_encoder.pkl", "wb") as fh:
            pickle.dump(le, fh)

        vocab.save(processed_dir / "vocabulary.pkl")

        meta = {
            "feat_mins":   feat_mins.tolist(),
            "feat_maxs":   feat_maxs.tolist(),
            "classes":     le.classes_.tolist(),
            "vocab_size":  vocab.size,
            "num_features": len(FEATURE_NAMES),
        }
        with open(processed_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        logger.info("Artefacts saved to %s", processed_dir)

    return dict(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_encoder=le,
        feat_mins=feat_mins,
        feat_maxs=feat_maxs,
        vocabulary=vocab,
    )


if __name__ == "__main__":
    run_preprocessing()
