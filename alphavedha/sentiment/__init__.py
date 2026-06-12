"""Social media sentiment pipeline for Indian equities."""

from alphavedha.sentiment.aggregator import SentimentAggregator, SocialSentimentResult
from alphavedha.sentiment.sources import RSSSource, RedditSource, SentimentPost

__all__ = [
    "SentimentAggregator",
    "SocialSentimentResult",
    "RSSSource",
    "RedditSource",
    "SentimentPost",
]
