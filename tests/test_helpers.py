"""
tests/test_helpers.py

Unit tests for utils/helpers.py.

Verifies:
    1. setup_logger: idempotent handler attachment, sets custom log levels.
    2. load_environment: mock load_dotenv (never reads real .env), monkeypatch env vars,
       tests present, missing, multiple missing naming all keys, and empty-string cases.
    3. validate_driver_code: lower-to-upper conversion and HelperValidationError on invalid inputs.
    4. build_session_id: space-to-underscore sanitization and validation errors on bad inputs.
    5. kmh_to_ms & ms_to_kmh: scalar and numpy array inputs, exact known values and round-trip identity.
    6. safe_divide: normal division, division by zero returning default, custom default.
"""

from __future__ import annotations

import logging
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from utils.helpers import (
    EnvironmentConfigurationError,
    HelperValidationError,
    build_session_id,
    kmh_to_ms,
    load_environment,
    ms_to_kmh,
    safe_divide,
    setup_logger,
    validate_driver_code,
)


class TestSetupLogger:
    def test_setup_logger_returns_logger(self) -> None:
        name = f"test_logger_{uuid4().hex}"
        logger = setup_logger(name)
        assert isinstance(logger, logging.Logger)
        assert logger.name == name
        assert logger.level == logging.INFO

    def test_setup_logger_applies_level_argument(self) -> None:
        name = f"test_logger_level_{uuid4().hex}"
        logger = setup_logger(name, level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_idempotent_no_duplicate_handlers(self) -> None:
        name = f"test_idempotent_{uuid4().hex}"
        logger1 = setup_logger(name)
        assert len(logger1.handlers) == 1

        logger2 = setup_logger(name)
        assert logger2 is logger1
        assert len(logger2.handlers) == 1


class TestLoadEnvironment:
    def test_load_environment_present_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MOCK_KEY_A", "valid_secret_123")
        with patch("utils.helpers.load_dotenv") as mock_dotenv:
            result = load_environment(["MOCK_KEY_A"])
            mock_dotenv.assert_called_once()
            assert result == {"MOCK_KEY_A": "valid_secret_123"}

    def test_load_environment_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
        with patch("utils.helpers.load_dotenv"):
            with pytest.raises(EnvironmentConfigurationError, match="MISSING_KEY_XYZ"):
                load_environment(["MISSING_KEY_XYZ"])

    def test_load_environment_multiple_missing_keys_names_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_ALPHA", raising=False)
        monkeypatch.delenv("MISSING_BETA", raising=False)
        with patch("utils.helpers.load_dotenv"):
            with pytest.raises(EnvironmentConfigurationError) as exc_info:
                load_environment(["MISSING_ALPHA", "MISSING_BETA"])
            err_msg = str(exc_info.value)
            assert "MISSING_ALPHA" in err_msg
            assert "MISSING_BETA" in err_msg

    def test_load_environment_empty_or_whitespace_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMPTY_KEY", "   ")
        with patch("utils.helpers.load_dotenv"):
            with pytest.raises(EnvironmentConfigurationError, match="EMPTY_KEY"):
                load_environment(["EMPTY_KEY"])

    def test_load_environment_default_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        with patch("utils.helpers.load_dotenv"):
            result = load_environment()
            assert "GEMINI_API_KEY" in result
            assert result["GEMINI_API_KEY"] == "test_gemini_key"


class TestValidateDriverCode:
    @pytest.mark.parametrize(
        ("input_code", "expected"),
        [
            ("ver", "VER"),
            ("nor", "NOR"),
            ("ham", "HAM"),
            ("al", "AL"),
            ("  pia  ", "PIA"),
        ],
    )
    def test_validate_driver_code_normalizes_to_uppercase(self, input_code: str, expected: str) -> None:
        assert validate_driver_code(input_code) == expected

    @pytest.mark.parametrize(
        "invalid_code",
        [
            "",
            "   ",
            "V",
            "VERSTAPPEN",
            "V3R",
            "V-R",
            "123",
            None,
        ],
    )
    def test_validate_driver_code_invalid_raises_error(self, invalid_code: str | None) -> None:
        with pytest.raises(HelperValidationError):
            validate_driver_code(invalid_code)  # type: ignore[arg-type]


class TestBuildSessionId:
    def test_build_session_id_replaces_spaces_and_formats(self) -> None:
        assert build_session_id(2024, "Abu Dhabi", "FP1") == "2024_Abu_Dhabi_FP1"
        assert build_session_id(2025, "  Monaco  ", "  Q  ") == "2025_Monaco_Q"
        assert build_session_id(2024, "São Paulo", "Sprint Shootout") == "2024_São_Paulo_Sprint_Shootout"

    @pytest.mark.parametrize(
        ("year", "gp", "session"),
        [
            (1949, "Monaco", "Q"),
            (2101, "Monaco", "Q"),
            ("2024", "Monaco", "Q"),  # type: ignore[arg-type]
            (2024, "", "Q"),
            (2024, "   ", "Q"),
            (2024, "Monaco", ""),
            (2024, "Monaco", "   "),
        ],
    )
    def test_build_session_id_invalid_inputs_raise_error(self, year: int, gp: str, session: str) -> None:
        with pytest.raises(HelperValidationError):
            build_session_id(year, gp, session)


class TestUnitConversions:
    @pytest.mark.parametrize("speed_kmh", [0.0, 36.0, 72.0, 180.0, 325.5])
    def test_speed_conversion_round_trip_scalar(self, speed_kmh: float) -> None:
        speed_ms = kmh_to_ms(speed_kmh)
        recovered_kmh = ms_to_kmh(speed_ms)
        assert recovered_kmh == pytest.approx(speed_kmh)

    def test_speed_conversion_numpy_arrays(self) -> None:
        speeds_kmh = np.array([0.0, 50.0, 120.0, 250.0, 340.0])
        speeds_ms = kmh_to_ms(speeds_kmh)
        recovered_kmh = ms_to_kmh(speeds_ms)
        np.testing.assert_allclose(recovered_kmh, speeds_kmh, rtol=1e-7, atol=1e-7)

    def test_kmh_to_ms_known_value(self) -> None:
        assert kmh_to_ms(360.0) == pytest.approx(100.0)
        assert ms_to_kmh(100.0) == pytest.approx(360.0)


class TestSafeDivide:
    def test_safe_divide_normal_division(self) -> None:
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(0.0, 5.0) == 0.0

    def test_safe_divide_zero_denominator_returns_default(self) -> None:
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=99.0) == 99.0
