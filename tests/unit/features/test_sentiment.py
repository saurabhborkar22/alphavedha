"""Tests for sentiment feature computation."""

from __future__ import annotations

import pandas as pd

from alphavedha.features.sentiment import SENTIMENT_FEATURE_COUNT, compute_sentiment_features


class TestSentimentFeatures:
    def test_returns_correct_count(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sentiment_features(sample_ohlcv)
        assert len(result.columns) == SENTIMENT_FEATURE_COUNT

    def test_neutral_when_no_news(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sentiment_features(sample_ohlcv)
        assert (result["sent_news_score"] == 0.0).all()
        assert (result["sent_no_news_flag"] == 1).all()

    def test_article_count_zero(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sentiment_features(sample_ohlcv)
        assert (result["sent_article_count"] == 0).all()

    def test_with_articles(self, sample_ohlcv: pd.DataFrame) -> None:
        date_str = sample_ohlcv.index[5].strftime("%Y-%m-%d")
        articles = {date_str: ["Stock rallied on strong earnings", "Positive guidance"]}
        result = compute_sentiment_features(sample_ohlcv, daily_articles=articles)
        assert result.loc[sample_ohlcv.index[5], "sent_article_count"] == 2
        assert result.loc[sample_ohlcv.index[5], "sent_no_news_flag"] == 0

    def test_velocity_computed(self, sample_ohlcv: pd.DataFrame) -> None:
        dates = sample_ohlcv.index
        articles = {}
        for i in range(5, 10):
            d = dates[i].strftime("%Y-%m-%d")
            articles[d] = [f"News article for day {i}"]
        result = compute_sentiment_features(sample_ohlcv, daily_articles=articles)
        assert "sent_velocity" in result.columns

    def test_ratios_bounded(self, sample_ohlcv: pd.DataFrame) -> None:
        date_str = sample_ohlcv.index[0].strftime("%Y-%m-%d")
        articles = {date_str: ["Good news", "Bad news", "Neutral news"]}
        result = compute_sentiment_features(sample_ohlcv, daily_articles=articles)
        assert result["sent_pos_ratio"].between(0, 1).all()
        assert result["sent_neg_ratio"].between(0, 1).all()
