"""
llm/feedback.py

The AI explanation layer of the AI Race Engineer project.

This module converts a deterministic, already-computed SessionAnalysis
into natural-language race engineer coaching, using the Google Gemini
API. It performs NO calculation, NO classification, and NO mutation of
its input.

This module must NOT:
    - calculate telemetry metrics (that is analysis/corners.py)
    - classify driving mistakes (that is analysis/engine.py)
    - modify SessionAnalysis (it is read-only here)
    - contain Streamlit code (that is app/main.py)
    - load FastF1 (that is analysis/alignment.py)

The only input this module accepts is a SessionAnalysis object; the only
output it produces is a plain-text string. If Gemini is unavailable for
any reason (bad key, timeout, network failure, empty response), this
module returns a safe fallback message and never raises to the caller.
"""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from models.schemas import SessionAnalysis
from utils.helpers import setup_logger

_logger = setup_logger(__name__)

#: Default Gemini model used for coaching feedback generation. Flash tier
#: is deliberately chosen: this task is short, well-structured text
#: generation (JSON in, up to three bullets out) with low reasoning depth
#: requirements, not a task that benefits from a larger, slower model.
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"

#: Default per-request timeout, in seconds.
DEFAULT_TIMEOUT_SECONDS: float = 15.0

#: Upper bound on generated response length, to keep output to a concise,
#: radio-style message rather than a long-form report.
_MAX_OUTPUT_TOKENS: int = 400

#: Low temperature favors a consistent, professional race-engineer tone
#: over creative variation between runs of the same analysis.
_TEMPERATURE: float = 0.2

#: Returned whenever Gemini cannot produce feedback for any reason. The
#: rest of the dashboard (charts, score, corner breakdown) must remain
#: fully usable even when this message is shown.
_FALLBACK_MESSAGE: str = (
    "Race engineer feedback is temporarily unavailable. "
    "Please review the corner-by-corner analysis above."
)

_SYSTEM_INSTRUCTION: str = """You are a Formula One Race Engineer.

You are only allowed to use the supplied JSON. Never invent telemetry.
Never estimate numbers. Never assume information not present in the JSON.

For every weakness you mention, explain:
- What happened
- Why it matters
- What the driver should change next lap

Respond with a maximum of three concise bullet points, in a professional
race engineer radio tone."""


class FeedbackGenerationError(Exception):
    """
    Internal exception used to unify Gemini failure modes before they are
    converted into the safe fallback message. Never raised to callers of
    get_race_engineer_feedback().
    """


def _build_client(api_key: str, timeout_seconds: float) -> genai.Client:
    """Construct a configured Gemini API client.

    Args:
        api_key: Google Gemini API key.
        timeout_seconds: Per-request timeout, in seconds.

    Returns:
        A configured `genai.Client`.

    Raises:
        FeedbackGenerationError: If the client cannot be constructed
            (e.g. malformed API key).
    """
    try:
        return genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
    except Exception as exc:
        raise FeedbackGenerationError(f"Failed to construct Gemini client: {exc}") from exc


def _request_feedback(
    client: genai.Client,
    model_name: str,
    analysis_json: str,
) -> str:
    """Send the analysis JSON to Gemini and return the raw response text.

    Args:
        client: A configured Gemini API client.
        model_name: Gemini model identifier to use.
        analysis_json: The serialized SessionAnalysis JSON to analyze.

    Returns:
        The raw text of Gemini's response.

    Raises:
        FeedbackGenerationError: If the API call fails, times out, or
            returns an empty/missing response.
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=analysis_json,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=_TEMPERATURE,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
        )
    except genai_errors.APIError as exc:
        # Covers invalid/expired API keys, quota exhaustion, and other
        # Gemini-side API errors (4xx/5xx responses from the service).
        raise FeedbackGenerationError(f"Gemini API error: {exc}") from exc
    except TimeoutError as exc:
        raise FeedbackGenerationError(f"Gemini request timed out: {exc}") from exc
    except Exception as exc:
        # Network failures and any other unexpected error land here so
        # this module can guarantee it never raises to its caller.
        raise FeedbackGenerationError(f"Unexpected error calling Gemini: {exc}") from exc

    feedback_text = getattr(response, "text", None)
    if not feedback_text or not feedback_text.strip():
        raise FeedbackGenerationError("Gemini returned an empty response.")

    return feedback_text.strip()


def get_race_engineer_feedback(
    analysis: SessionAnalysis,
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Generate race-engineer-style coaching feedback for a session analysis.

    Serializes `analysis` to JSON via `model_dump_json()` and sends it to
    Gemini as the sole source of truth, instructing the model to explain
    only what is present in that JSON. On any failure (invalid API key,
    timeout, network failure, empty response), a safe fallback message is
    returned instead of raising -- this function never crashes the caller.

    Args:
        analysis: The deterministic SessionAnalysis to explain. No raw
            telemetry, DataFrames, or NumPy arrays are ever passed to
            Gemini -- only this object's JSON serialization.
        api_key: Google Gemini API key.
        model_name: Gemini model identifier (default: DEFAULT_GEMINI_MODEL).
        timeout_seconds: Per-request timeout, in seconds
            (default: DEFAULT_TIMEOUT_SECONDS).

    Returns:
        Plain-text coaching feedback (at most three bullet points) on
        success, or a safe fallback message if Gemini is unavailable for
        any reason.
    """
    if not api_key or not api_key.strip():
        _logger.error("No Gemini API key supplied; returning fallback feedback message.")
        return _FALLBACK_MESSAGE

    try:
        analysis_json = analysis.model_dump_json()
    except Exception as exc:
        _logger.error("Failed to serialize SessionAnalysis to JSON: %s", exc)
        return _FALLBACK_MESSAGE

    _logger.info(
        "Requesting race engineer feedback for session '%s' via model '%s'.",
        analysis.session_id,
        model_name,
    )

    try:
        client = _build_client(api_key, timeout_seconds)
        feedback_text = _request_feedback(client, model_name, analysis_json)
    except FeedbackGenerationError as exc:
        _logger.error("Race engineer feedback generation failed: %s", exc)
        return _FALLBACK_MESSAGE

    _logger.info(
        "Successfully generated race engineer feedback for session '%s'.",
        analysis.session_id,
    )
    return feedback_text
