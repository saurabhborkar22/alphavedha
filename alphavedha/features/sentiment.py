"""Sentiment features — 8 news sentiment features.

Uses FinBERT (transformers) for news scoring. Graceful degradation when
transformers is not installed or no news data available.
Column naming: sent_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

SENTIMENT_FEATURE_COUNT = 8

_finbert_pipeline = None


def _get_finbert():
    """Lazily load FinBERT pipeline. Returns None if transformers unavailable."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline

        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
        )
        logger.info("finbert_loaded")
        return _finbert_pipeline
    except (ImportError, OSError):
        logger.warning("finbert_unavailable", msg="transformers not installed or model not cached")
        return None


def score_articles(articles: list[str]) -> list[dict[str, float]]:
    """Score a list of article texts with FinBERT.

    Returns list of dicts with keys: positive, negative, neutral.
    Falls back to neutral scores if FinBERT unavailable.
    """
    if not articles:
        return []

    pipe = _get_finbert()
    if pipe is None:
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(articles)

    scores = []
    for text in articles:
        truncated = text[:512]
        try:
            result = pipe(truncated)[0]
            score_dict = {item["label"]: item["score"] for item in result}
            scores.append(
                {
                    "positive": score_dict.get("positive", 0.0),
                    "negative": score_dict.get("negative", 0.0),
                    "neutral": score_dict.get("neutral", 0.0),
                }
            )
        except (RuntimeError, ValueError, IndexError, KeyError):
            scores.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
    return scores


def compute_sentiment_features(
    df: pd.DataFrame,
    daily_articles: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Compute 8 sentiment features.

    Args:
        df: DataFrame with DatetimeIndex.
        daily_articles: Dict mapping date strings (YYYY-MM-DD) to lists of article texts.

    Returns:
        DataFrame with 8 sent_* columns.
    """
    result = pd.DataFrame(index=df.index)

    daily_scores: dict[str, float] = {}
    daily_counts: dict[str, int] = {}
    daily_pos_ratio: dict[str, float] = {}
    daily_neg_ratio: dict[str, float] = {}

    if daily_articles:
        for date_str, articles in daily_articles.items():
            if not articles:
                daily_scores[date_str] = 0.0
                daily_counts[date_str] = 0
                daily_pos_ratio[date_str] = 0.0
                daily_neg_ratio[date_str] = 0.0
                continue

            scores = score_articles(articles)
            net_scores = [s["positive"] - s["negative"] for s in scores]
            daily_scores[date_str] = float(np.mean(net_scores))
            daily_counts[date_str] = len(articles)
            daily_pos_ratio[date_str] = float(
                sum(1 for s in net_scores if s > 0.1) / len(net_scores)
            )
            daily_neg_ratio[date_str] = float(
                sum(1 for s in net_scores if s < -0.1) / len(net_scores)
            )

    date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in df.index]

    news_score = pd.Series(
        [daily_scores.get(d, 0.0) for d in date_strs],
        index=df.index,
        dtype=float,
    )
    article_count = pd.Series(
        [daily_counts.get(d, 0) for d in date_strs],
        index=df.index,
        dtype=int,
    )

    result["sent_news_score"] = news_score
    result["sent_news_score_5d"] = news_score.rolling(5, min_periods=1).mean()

    velocity = news_score.diff()
    result["sent_velocity"] = velocity
    vel_mean = velocity.rolling(20, min_periods=5).mean()
    vel_std = velocity.rolling(20, min_periods=5).std()
    result["sent_velocity_zscore"] = (velocity - vel_mean) / vel_std.replace(0, np.nan)

    result["sent_article_count"] = article_count
    result["sent_no_news_flag"] = (article_count == 0).astype(int)

    result["sent_pos_ratio"] = pd.Series(
        [daily_pos_ratio.get(d, 0.0) for d in date_strs],
        index=df.index,
        dtype=float,
    )
    result["sent_neg_ratio"] = pd.Series(
        [daily_neg_ratio.get(d, 0.0) for d in date_strs],
        index=df.index,
        dtype=float,
    )

    logger.info(
        "sentiment_features_computed",
        n_features=len(result.columns),
        days_with_news=sum(1 for c in daily_counts.values() if c > 0),
    )
    return result
