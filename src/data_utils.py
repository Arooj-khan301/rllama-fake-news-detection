"""Data loading, label normalization, deduplication, and stratified splitting."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.config import DEFAULT_LABEL_MAP

try:
    from sklearn.model_selection import train_test_split as _sk_split
except ImportError:
    _sk_split = None


def map_label(value: object, label_map: Dict[str, int] = DEFAULT_LABEL_MAP) -> int:
    """Maps a raw dataset label (string or int) to binary integer {0: Fake, 1: Real}."""
    key = str(value).strip().lower()
    if key not in label_map:
        raise ValueError(
            f"Unmapped label {value!r}. Supported keys: {sorted(label_map.keys())}"
        )
    return label_map[key]


def prepare_dataframe(
    raw_df: pd.DataFrame,
    text_col: str = "title",
    label_col: str = "real",
    label_map: Dict[str, int] = DEFAULT_LABEL_MAP,
) -> pd.DataFrame:
    """Validates columns, strips whitespace, maps binary labels, and deduplicates."""
    if text_col not in raw_df.columns or label_col not in raw_df.columns:
        raise ValueError(
            f"Columns ({text_col!r}, {label_col!r}) not found. "
            f"Available columns: {raw_df.columns.tolist()}"
        )
    df = (
        raw_df[[text_col, label_col]]
        .rename(columns={text_col: "text", label_col: "raw_label"})
        .copy()
    )
    df = df.dropna(subset=["text", "raw_label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label"] = df["raw_label"].apply(lambda x: map_label(x, label_map)).astype(int)
    df = (
        df[["text", "label"]]
        .drop_duplicates(subset=["text", "label"])
        .reset_index(drop=True)
    )
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both classes 0 (Fake) and 1 (Real).")
    if df["label"].value_counts().min() < 3:
        raise ValueError("Need at least 3 examples per class for stratified splits.")
    return df


def _fallback_stratified_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified split fallback when scikit-learn is not installed."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in df.groupby("label"):
        indices = rng.permutation(len(group))
        n_test = max(1, int(round(len(group) * test_size)))
        test_parts.append(group.iloc[indices[:n_test]])
        train_parts.append(group.iloc[indices[n_test:]])
    return (
        pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def make_stratified_splits(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates ~70% train / ~15% validation / ~15% test stratified splits."""
    if _sk_split is not None:
        train_val_df, test_df = _sk_split(
            df, test_size=0.15, stratify=df["label"], random_state=seed
        )
        train_df, val_df = _sk_split(
            train_val_df, test_size=0.1765, stratify=train_val_df["label"], random_state=seed
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
    train_val_df, test_df = _fallback_stratified_split(df, 0.15, seed)
    train_df, val_df = _fallback_stratified_split(train_val_df, 0.1765, seed)
    return train_df, val_df, test_df
