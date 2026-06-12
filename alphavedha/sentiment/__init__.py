"""Social media sentiment pipeline for Indian equities."""

from alphavedha.sentiment.aggregator import SentimentAggregator, SocialSentimentResult
from alphavedha.sentiment.sources import RedditSource, RSSSource, SentimentPost

__all__ = [
    "RSSSource",
    "RedditSource",
    "SentimentAggregator",
    "SentimentPost",
    "SocialSentimentResult",
]
