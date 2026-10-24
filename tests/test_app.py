"""
tests/test_app.py

Integration and UI testing of app/main.py using streamlit.testing.v1.AppTest.

Tests verify:
    1. Default Demo Mode executes without exception, extracts 6 corners,
       and displays the canned radio coaching bullets.
    2. Real session mode with missing Gemini API key executes successfully,
       displays the offline radio message, renders charts & tables,
       and verifies Gemini was never called.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import analysis.alignment
import llm.feedback
import utils.helpers
from streamlit.testing.v1 import AppTest

from utils.demo_data import (
    DEMO_CORNER_BOUNDARIES,
    generate_demo_telemetry,
)
from utils.helpers import EnvironmentConfigurationError


def test_app_demo_mode_path() -> None:
    """Verify demo mode loads cleanly, computes 6 corners, and renders canned coaching radio."""
    at = AppTest.from_file("app/main.py")
    at.run(timeout=15)

    assert not at.exception, f"App raised unexpected exception: {at.exception}"

    # Verify demo banner is visible
    banner_texts = [info.value for info in at.info]
    assert any("Demo Mode Active" in text for text in banner_texts)

    # Verify canned coaching feedback is displayed in markdown
    markdown_texts = [m.value for m in at.markdown]
    assert any("Corner 3 Early Braking" in text for text in markdown_texts)
    assert any("Corner 5 Late Throttle" in text for text in markdown_texts)

    # Verify corner breakdown dataframe is rendered with 6 corners
    assert len(at.dataframe) >= 1
    corners_df = at.dataframe[0].value
    assert len(corners_df) == 6

    # Verify Plotly charts are generated (Speed, DeltaTime, Throttle, Brake)
    assert len(at.get("plotly_chart")) >= 4


def test_app_real_mode_missing_api_key_path() -> None:
    """Verify real mode with missing API key renders analysis, shows offline radio, and avoids LLM call."""
    at = AppTest.from_file("app/main.py")
    # Initial run in demo mode
    at.run(timeout=15)

    # Toggle off Demo Mode to enter Real Session Mode
    at.sidebar.toggle[0].set_value(False)
    at.run(timeout=15)

    # Provide valid corner boundaries for real mode (C3 and C4 with distinct metrics)
    sample_boundaries = [DEMO_CORNER_BOUNDARIES[2], DEMO_CORNER_BOUNDARIES[3]]
    assert len(at.sidebar.text_area) > 0
    at.sidebar.text_area[0].set_value(json.dumps(sample_boundaries))

    # Click Run Telemetry Analysis button
    at.sidebar.button[0].click()

    # Stub telemetry loading and environment reading so no network / FastF1 / Gemini call occurs
    mock_telemetry = generate_demo_telemetry()
    with patch.object(analysis.alignment, "load_and_align_laps", return_value=mock_telemetry), \
         patch.object(utils.helpers, "load_environment", side_effect=EnvironmentConfigurationError("No key")), \
         patch.object(llm.feedback, "get_race_engineer_feedback") as mock_gemini:

        at.run(timeout=15)

        assert not at.exception, f"App raised unexpected exception: {at.exception}"

        # Gemini must NOT be called when no API key is provided
        mock_gemini.assert_not_called()

        # Offline message must be shown in the radio card
        markdown_texts = [m.value for m in at.markdown]
        assert any("AI coaching radio is offline" in text for text in markdown_texts)

        # Telemetry charts and corner table must still render normally
        assert len(at.get("plotly_chart")) >= 4
        assert len(at.dataframe) >= 1
        df_result = at.dataframe[0].value
        assert len(df_result) == len(sample_boundaries)


def test_app_toggle_mode_clears_stale_results() -> None:
    """Verify switching from Demo to Real mode clears old demo results from the screen."""
    at = AppTest.from_file("app/main.py")
    at.run(timeout=15)

    assert not at.exception
    # Demo results are present
    assert len(at.dataframe) >= 1
    assert any("Demo Mode Active" in info.value for info in at.info)

    # Toggle off Demo Mode to switch to Real Mode (without clicking Analyze)
    at.sidebar.toggle[0].set_value(False)
    at.run(timeout=15)

    assert not at.exception
    # Old demo results, charts, and dataframe must be cleared from the UI
    assert len(at.dataframe) == 0
    assert len(at.get("plotly_chart")) == 0
    # Must prompt user to configure and run analysis
    info_messages = [info.value for info in at.info]
    assert any("Configure session parameters in the sidebar" in msg for msg in info_messages)
    assert not any("Demo Mode Active" in msg for msg in info_messages)

    # Toggle back to Demo Mode: auto_demo_run re-executes cleanly
    at.sidebar.toggle[0].set_value(True)
    at.run(timeout=15)

    assert not at.exception
    assert len(at.dataframe) >= 1
    assert any("Demo Mode Active" in info.value for info in at.info)

