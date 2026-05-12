"""AlphaVedha custom exceptions."""

from __future__ import annotations


class AlphaVedhaError(Exception):
    """Base exception for all AlphaVedha errors."""


class DataProviderError(AlphaVedhaError):
    """Raised when a data provider fails to fetch data."""


class DataQualityError(AlphaVedhaError):
    """Raised when data fails quality checks."""


class FeatureComputationError(AlphaVedhaError):
    """Raised when feature computation fails."""


class ModelTrainingError(AlphaVedhaError):
    """Raised when model training fails."""


class ModelNotFoundError(AlphaVedhaError):
    """Raised when a required model artifact is not found."""


class PredictionError(AlphaVedhaError):
    """Raised when prediction pipeline fails."""


class ValidationError(AlphaVedhaError):
    """Raised when validation checks fail."""


class ConfigError(AlphaVedhaError):
    """Raised when configuration is invalid."""


class SymbolNotFoundError(AlphaVedhaError):
    """Raised when a stock symbol is not found in the universe."""


class InsufficientDataError(AlphaVedhaError):
    """Raised when there is not enough historical data for computation."""


class CircuitBreakerTriggeredError(AlphaVedhaError):
    """Raised when a risk circuit breaker is triggered."""
