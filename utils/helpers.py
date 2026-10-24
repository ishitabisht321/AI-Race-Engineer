"""
utils/helpers.py

Generic, reusable helper functions shared across the entire AI Race
Engineer project.

This module is imported by analysis/, llm/, app/, and tests/. It must
therefore have NO dependency on any of those layers, and must contain no:
    - telemetry calculations
    - FastF1 logic
    - Streamlit code
    - Gemini API code
    - business/racing rules

Everything here is a pure, generic utility: logging setup, environment
loading, string/format validation, unit conversion, and safe arithmetic.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Conversion factor: kilometers-per-hour to meters-per-second.
_KMH_TO_MS_FACTOR: float = 1.0 / 3.6

#: Conversion factor: meters-per-second to kilometers-per-hour.
_MS_TO_KMH_FACTOR: float = 3.6

#: Default logging format used by every logger created via setup_logger().
_DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

#: Valid FIA driver code pattern: two or three uppercase letters.
#: Kept in sync with models.schemas.DriverSummary's driver_code pattern
#: so a code that passes this helper is guaranteed to also pass Pydantic
#: validation downstream.
_DRIVER_CODE_PATTERN: re.Pattern[str] = re.compile(r"^[A-Z]{2,3}$")

#: Default set of environment variable names load_environment() requires
#: when the caller does not specify its own list.
_DEFAULT_REQUIRED_ENV_KEYS: tuple[str, ...] = ("GEMINI_API_KEY",)


class HelperValidationError(ValueError):
    """
    Raised when a helper-level input validation check fails (e.g. an
    invalid driver code or malformed session-ID input).

    Subclasses ValueError so existing `except ValueError` handling
    continues to work, while still allowing callers to catch these
    failures specifically.
    """


class EnvironmentConfigurationError(RuntimeError):
    """
    Raised when one or more required environment variables are missing
    or empty after attempting to load them.
    """


# --------------------------------------------------------------------------
# 1. Logging
# --------------------------------------------------------------------------


def setup_logger(name: str = "ai_race_engineer", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a reusable, console-logging project logger.

    Idempotent by design: calling this repeatedly with the same `name`
    (e.g. across Streamlit reruns, which re-execute the whole script on
    every UI interaction) will not attach duplicate handlers or produce
    duplicated log lines.

    Args:
        name: Logger name. Use `__name__` in most modules, or a shared
            project-wide name (e.g. "ai_race_engineer") for a root-style
            logger.
        level: Logging level to apply to this logger (default: logging.INFO).

    Returns:
        A configured `logging.Logger` with exactly one stream handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


_logger = setup_logger(__name__)


# --------------------------------------------------------------------------
# 2. Environment loading
# --------------------------------------------------------------------------


def load_environment(
    required_keys: Sequence[str] | None = None,
) -> dict[str, str]:
    """Load environment variables (via python-dotenv) and return them as a dict.

    Reads a local `.env` file if present, then reads each key in
    `required_keys` from the process environment. This is the single
    entry point secrets like GEMINI_API_KEY should pass through, so
    `llm/feedback.py` never touches `os.environ` directly.

    Args:
        required_keys: Names of environment variables that must be
            present and non-empty. Defaults to `("GEMINI_API_KEY",)` if
            not provided.

    Returns:
        A dict mapping each requested key to its string value.

    Raises:
        EnvironmentConfigurationError: If any required key is missing or
            empty, listing all missing keys in a single informative
            message.
    """
    load_dotenv()

    keys = tuple(required_keys) if required_keys is not None else _DEFAULT_REQUIRED_ENV_KEYS

    values: dict[str, str] = {}
    missing_keys: list[str] = []

    for key in keys:
        value = os.getenv(key)
        if value is None or value.strip() == "":
            missing_keys.append(key)
        else:
            values[key] = value

    if missing_keys:
        _logger.error("Missing required environment variables: %s", ", ".join(missing_keys))
        raise EnvironmentConfigurationError(
            "Missing required environment variable(s): "
            f"{', '.join(missing_keys)}. "
            "Set them in your .env file or process environment."
        )

    _logger.debug("Loaded environment variables: %s", ", ".join(values.keys()))
    return values


# --------------------------------------------------------------------------
# 3. Driver code validation
# --------------------------------------------------------------------------


def validate_driver_code(driver_code: str) -> str:
    """Validate and normalize a FIA driver code (e.g. "VER", "NOR", "HAM", "ALO").

    Args:
        driver_code: Raw driver code, in any case (e.g. "ver", "Ver", "VER").

    Returns:
        The normalized, uppercase driver code.

    Raises:
        HelperValidationError: If `driver_code` is empty, not a string,
            or does not match the two-to-three-uppercase-letter pattern
            after normalization.
    """
    if not isinstance(driver_code, str) or not driver_code.strip():
        raise HelperValidationError(
            f"driver_code must be a non-empty string, got {driver_code!r}."
        )

    normalized = driver_code.strip().upper()

    if not _DRIVER_CODE_PATTERN.match(normalized):
        raise HelperValidationError(
            f"'{driver_code}' is not a valid FIA driver code. "
            "Expected two or three uppercase letters, e.g. 'VER', 'NOR', 'HAM', 'ALO'."
        )

    return normalized


# --------------------------------------------------------------------------
# 4. Session ID construction
# --------------------------------------------------------------------------


def build_session_id(year: int, grand_prix: str, session: str) -> str:
    """Build a unique, human-readable session identifier.

    Combines a year, Grand Prix name, and session code into a single
    identifier string, e.g. `build_session_id(2025, "Monaco", "Q")`
    returns `"2025_Monaco_Q"`. Used as the canonical `SessionAnalysis.session_id`
    and as a cache/lookup key throughout the project.

    Args:
        year: Four-digit season year (e.g. 2025).
        grand_prix: Grand Prix name (e.g. "Monaco", "Abu Dhabi").
            Internal whitespace is collapsed to single underscores.
        session: Session code (e.g. "FP1", "Q", "R").

    Returns:
        A session ID string in the form "{year}_{grand_prix}_{session}".

    Raises:
        HelperValidationError: If `year` is not a plausible four-digit
            year, or if `grand_prix` / `session` are empty after
            stripping whitespace.
    """
    if not isinstance(year, int) or not (1950 <= year <= 2100):
        raise HelperValidationError(
            f"year must be a plausible four-digit season year, got {year!r}."
        )

    if not isinstance(grand_prix, str) or not grand_prix.strip():
        raise HelperValidationError(
            f"grand_prix must be a non-empty string, got {grand_prix!r}."
        )

    if not isinstance(session, str) or not session.strip():
        raise HelperValidationError(f"session must be a non-empty string, got {session!r}.")

    sanitized_gp = "_".join(grand_prix.strip().split())
    sanitized_session = "_".join(session.strip().split())

    return f"{year}_{sanitized_gp}_{sanitized_session}"


# --------------------------------------------------------------------------
# 5-6. Unit conversions
# --------------------------------------------------------------------------


def kmh_to_ms(speed_kmh: float) -> float:
    """Convert a speed value from kilometers-per-hour to meters-per-second.

    Args:
        speed_kmh: Speed in km/h.

    Returns:
        Equivalent speed in m/s.
    """
    return speed_kmh * _KMH_TO_MS_FACTOR


def ms_to_kmh(speed_ms: float) -> float:
    """Convert a speed value from meters-per-second to kilometers-per-hour.

    Args:
        speed_ms: Speed in m/s.

    Returns:
        Equivalent speed in km/h.
    """
    return speed_ms * _MS_TO_KMH_FACTOR


# --------------------------------------------------------------------------
# 7. Safe arithmetic
# --------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning a default value instead of raising on
    division by zero.

    Used throughout analysis/ (e.g. the delta-time integral, the
    performance-score calculation) wherever a denominator could
    legitimately be zero without indicating a bug.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if `denominator` is zero (default: 0.0).

    Returns:
        `numerator / denominator`, or `default` if `denominator == 0`.
    """
    if denominator == 0:
        _logger.warning(
            "safe_divide: division by zero avoided (numerator=%s), returning default=%s.",
            numerator,
            default,
        )
        return default

    return numerator / denominator
