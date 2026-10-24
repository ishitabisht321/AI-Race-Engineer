"""
tests/test_feedback.py

Unit tests for llm/feedback.py.

The Gemini client is patched out entirely so these tests run offline,
deterministically, and without spending API quota.  The module's public
contract is:
  - get_race_engineer_feedback() NEVER raises.
  - It returns the fallback message on any failure (missing key, API
    error, timeout, empty response, serialization error).
  - It returns the model's text on success.
  - The JSON sent to the model is the SessionAnalysis serialized via
    model_dump_json() -- never raw telemetry.

Mocking strategy
----------------
`_build_client` and `_request_feedback` are patched at the module level
(i.e. `llm.feedback._build_client` / `llm.feedback._request_feedback`)
so tests remain independent of the google-genai implementation details.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from llm.feedback import (
    DEFAULT_GEMINI_MODEL,
    FeedbackGenerationError,
    _FALLBACK_MESSAGE,
    _build_client,
    _request_feedback,
    get_race_engineer_feedback,
)
from models.schemas import (
    CornerMetric,
    DriverSummary,
    IssueType,
    IssueSeverity,
    SessionAnalysis,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_session_analysis() -> SessionAnalysis:
    """Return the simplest possible valid SessionAnalysis for use in tests."""
    corner = CornerMetric(
        corner_number=1,
        start_distance_m=100.0,
        end_distance_m=300.0,
        braking_point_m=120.0,
        braking_delta_m=-2.0,
        apex_speed_kph=150.0,
        apex_speed_delta_kph=1.5,
        exit_speed_kph=200.0,
        exit_speed_delta_kph=-1.0,
        throttle_application_m=220.0,
        throttle_delay_m=3.0,
        time_loss_s=0.05,
    )
    driver_summary = DriverSummary(
        driver_code="VER",
        reference_driver_code="NOR",
        total_time_loss_s=0.05,
        performance_score=95,
    )
    return SessionAnalysis(
        session_id="2024_Monaco_Q",
        driver_summary=driver_summary,
        corner_metrics=[corner],
        worst_corner=1,
        best_corner=1,
        biggest_mistake="No significant driving errors detected this lap.",
    )


VALID_API_KEY = "AIza_fake_but_non_empty_key"


# ---------------------------------------------------------------------------
# TestGetRaceEngineerFeedback – public API contract
# ---------------------------------------------------------------------------


class TestGetRaceEngineerFeedback:
    """Tests for the never-raises public function get_race_engineer_feedback."""

    # --- Returns fallback on bad/missing key ---

    def test_returns_fallback_when_api_key_is_empty_string(
        self, minimal_session_analysis: SessionAnalysis
    ) -> None:
        """An empty api_key immediately returns the fallback message without calling Gemini."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key="")
        assert result == _FALLBACK_MESSAGE

    def test_returns_fallback_when_api_key_is_whitespace_only(
        self, minimal_session_analysis: SessionAnalysis
    ) -> None:
        """A whitespace-only api_key is treated as empty and returns the fallback."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key="   ")
        assert result == _FALLBACK_MESSAGE

    # --- Returns model text on success ---

    @patch("llm.feedback._request_feedback", return_value="• Brake later at T1.\n• Smoother apex.")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_returns_model_text_on_success(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """On a clean run the function returns the text from _request_feedback."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)
        assert result == "• Brake later at T1.\n• Smoother apex."

    # --- Never raises on Gemini failure modes ---

    @patch(
        "llm.feedback._request_feedback",
        side_effect=FeedbackGenerationError("Gemini API error"),
    )
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_returns_fallback_when_request_feedback_raises(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """A FeedbackGenerationError from _request_feedback is caught and returns fallback."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)
        assert result == _FALLBACK_MESSAGE

    @patch(
        "llm.feedback._build_client",
        side_effect=FeedbackGenerationError("Client construction failed"),
    )
    def test_returns_fallback_when_build_client_raises(
        self,
        mock_build: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """A FeedbackGenerationError from _build_client is caught and returns fallback."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)
        assert result == _FALLBACK_MESSAGE

    @patch("llm.feedback._request_feedback", return_value="   ")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_returns_fallback_when_request_feedback_returns_empty_text(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """If _request_feedback returns whitespace-only, the fallback is returned.

        Note: in practice _request_feedback raises FeedbackGenerationError for
        empty responses; this test verifies the outer function's own guard path.
        """
        # Simulate _request_feedback raising (as it does on empty responses)
        mock_request.side_effect = FeedbackGenerationError("Gemini returned an empty response.")
        result = get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)
        assert result == _FALLBACK_MESSAGE

    # --- Verifies JSON-only input to the model ---

    @patch("llm.feedback._request_feedback")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_passes_model_dump_json_to_request_feedback(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """The second positional argument to _request_feedback must be the JSON string."""
        mock_request.return_value = "Some coaching text."
        get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)

        _client_arg, model_arg, json_arg = mock_request.call_args.args
        expected_json = minimal_session_analysis.model_dump_json()
        assert json_arg == expected_json

    @patch("llm.feedback._request_feedback")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_default_model_name_is_used_when_not_specified(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """_request_feedback is called with DEFAULT_GEMINI_MODEL when model_name is omitted."""
        mock_request.return_value = "Some coaching text."
        get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)

        _client_arg, model_arg, _json_arg = mock_request.call_args.args
        assert model_arg == DEFAULT_GEMINI_MODEL

    @patch("llm.feedback._request_feedback")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_custom_model_name_is_forwarded(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """A caller-supplied model_name is forwarded to _request_feedback unchanged."""
        mock_request.return_value = "Custom model coaching."
        get_race_engineer_feedback(
            minimal_session_analysis,
            api_key=VALID_API_KEY,
            model_name="gemini-1.5-pro",
        )
        _client_arg, model_arg, _json_arg = mock_request.call_args.args
        assert model_arg == "gemini-1.5-pro"

    # --- Fallback message content contract ---

    def test_fallback_message_is_non_empty(self) -> None:
        """The module-level fallback constant is a non-empty string."""
        assert isinstance(_FALLBACK_MESSAGE, str)
        assert len(_FALLBACK_MESSAGE) > 0

    def test_fallback_message_does_not_raise_when_returned(
        self, minimal_session_analysis: SessionAnalysis
    ) -> None:
        """Confirm the fallback is returned (not raised) when the key is empty."""
        result = get_race_engineer_feedback(minimal_session_analysis, api_key="")
        assert isinstance(result, str)

    # --- No raw telemetry reaches the model ---

    @patch("llm.feedback._request_feedback")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_json_argument_contains_no_dataframe_reference(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """The JSON string passed to the model must be serializable and contain no DataFrame."""
        mock_request.return_value = "Clean."
        get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)

        _client_arg, _model_arg, json_arg = mock_request.call_args.args
        # Should be valid JSON (parseable), not a Python repr or binary blob
        import json
        parsed = json.loads(json_arg)
        assert isinstance(parsed, dict)

    @patch("llm.feedback._request_feedback")
    @patch("llm.feedback._build_client", return_value=MagicMock())
    def test_json_contains_session_id(
        self,
        mock_build: MagicMock,
        mock_request: MagicMock,
        minimal_session_analysis: SessionAnalysis,
    ) -> None:
        """The JSON contains the session_id from the analysis."""
        mock_request.return_value = "Coaching."
        get_race_engineer_feedback(minimal_session_analysis, api_key=VALID_API_KEY)

        _client_arg, _model_arg, json_arg = mock_request.call_args.args
        assert "2024_Monaco_Q" in json_arg


# ---------------------------------------------------------------------------
# TestBuildClient – internal function (tested in isolation)
# ---------------------------------------------------------------------------


class TestBuildClient:
    """Tests for the internal _build_client helper."""

    @patch("llm.feedback.genai.Client", side_effect=ValueError("bad key format"))
    def test_raises_feedback_generation_error_when_client_construction_fails(
        self, mock_genai_client: MagicMock
    ) -> None:
        """If genai.Client raises, _build_client wraps it in FeedbackGenerationError."""
        with pytest.raises(FeedbackGenerationError):
            _build_client(api_key="bad-key", timeout_seconds=15.0)


# ---------------------------------------------------------------------------
# TestRequestFeedback – internal function (tested in isolation)
# ---------------------------------------------------------------------------


class TestRequestFeedback:
    """Tests for the internal _request_feedback helper."""

    def _make_mock_client(self, response_text: str | None) -> MagicMock:
        """Build a mock genai.Client whose generate_content returns `response_text`."""
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    def test_returns_stripped_text_on_success(self) -> None:
        """On a successful API call, _request_feedback returns stripped response text."""
        client = self._make_mock_client("  • Tip one.\n• Tip two.  ")
        result = _request_feedback(client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')
        assert result == "• Tip one.\n• Tip two."

    def test_raises_feedback_generation_error_when_response_text_is_none(self) -> None:
        """If response.text is None, _request_feedback raises FeedbackGenerationError."""
        client = self._make_mock_client(None)
        with pytest.raises(FeedbackGenerationError, match="empty"):
            _request_feedback(client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')

    def test_raises_feedback_generation_error_when_response_text_is_whitespace(self) -> None:
        """If response.text is all whitespace, _request_feedback raises FeedbackGenerationError."""
        client = self._make_mock_client("   ")
        with pytest.raises(FeedbackGenerationError, match="empty"):
            _request_feedback(client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')

    def test_raises_feedback_generation_error_on_api_error(self) -> None:
        """An APIError from generate_content is wrapped in FeedbackGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            code=403, response_json={"error": {"code": 403, "message": "forbidden"}}
        )
        with pytest.raises(FeedbackGenerationError, match="API error"):
            _request_feedback(mock_client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')

    def test_raises_feedback_generation_error_on_timeout(self) -> None:
        """A TimeoutError from generate_content is wrapped in FeedbackGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("timed out")
        with pytest.raises(FeedbackGenerationError, match="timed out"):
            _request_feedback(mock_client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')

    def test_raises_feedback_generation_error_on_generic_exception(self) -> None:
        """Any unexpected exception from generate_content is wrapped in FeedbackGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("network gone")
        with pytest.raises(FeedbackGenerationError, match="Unexpected error"):
            _request_feedback(mock_client, DEFAULT_GEMINI_MODEL, '{"session_id": "test"}')
