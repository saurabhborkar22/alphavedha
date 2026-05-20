"""Tests for custom exception hierarchy."""

from __future__ import annotations

import pytest

from alphavedha.exceptions import (
    AlphaVedhaError,
    CircuitBreakerTriggeredError,
    ConfigError,
    DataProviderError,
    DataQualityError,
    FeatureComputationError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
    PredictionError,
    SymbolNotFoundError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self) -> None:
        exceptions = [
            DataProviderError,
            DataQualityError,
            FeatureComputationError,
            ModelTrainingError,
            ModelNotFoundError,
            PredictionError,
            ValidationError,
            ConfigError,
            SymbolNotFoundError,
            InsufficientDataError,
            CircuitBreakerTriggeredError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, AlphaVedhaError), f"{exc_class.__name__} should inherit AlphaVedhaError"

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(AlphaVedhaError, Exception)

    def test_can_raise_and_catch_by_base(self) -> None:
        with pytest.raises(AlphaVedhaError):
            raise DataProviderError("test")

    def test_message_preserved(self) -> None:
        msg = "Model artifacts missing"
        err = ModelNotFoundError(msg)
        assert str(err) == msg

    def test_specific_catch(self) -> None:
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("not found")

    def test_does_not_catch_unrelated(self) -> None:
        with pytest.raises(DataProviderError):
            raise DataProviderError("provider error")
        # Ensure it doesn't match a different subclass
        with pytest.raises(DataProviderError):
            try:
                raise DataProviderError("test")
            except ModelNotFoundError:
                pytest.fail("Should not catch DataProviderError as ModelNotFoundError")
            except DataProviderError:
                raise
