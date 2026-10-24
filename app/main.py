"""
app/main.py

Streamlit entry point and UI for the AI Race Engineer platform.

Clean Architecture boundary:
    - Pure orchestration and presentation layer
    - NO telemetry calculations (delegated to analysis/alignment and analysis/corners)
    - NO rule classification or scoring (delegated to analysis/engine)
    - NO prompt construction or LLM calls (delegated to llm/feedback)
    - NO custom Plotly traces (delegated to utils/plotting)
"""

from __future__ import annotations

import html
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Ensure project root is always importable when launched via `streamlit run app/main.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from analysis.alignment import AlignmentError, load_and_align_laps
from analysis.corners import CornerBoundary, CornerMetricsError, extract_corner_metrics
from analysis.engine import EngineError, build_session_analysis
from llm.feedback import get_race_engineer_feedback
from models.schemas import SessionAnalysis
from utils.demo_data import (
    DEMO_COACHING_FEEDBACK,
    DEMO_CORNER_BOUNDARIES,
    DEMO_DRIVER_CODE,
    DEMO_REFERENCE_DRIVER_CODE,
    DEMO_SESSION_ID,
    generate_demo_telemetry,
)
from utils.helpers import (
    EnvironmentConfigurationError,
    HelperValidationError,
    build_session_id,
    load_environment,
    setup_logger,
)
from utils.plotting import plot_brake, plot_delta_time, plot_speed, plot_throttle

_logger = setup_logger(__name__)

#: Standard F1 session type options
_SESSION_TYPE_OPTIONS: tuple[str, ...] = ("FP1", "FP2", "FP3", "Q", "S", "R")

#: Custom CSS styling for dark racing HUD theme
_CUSTOM_CSS: str = textwrap.dedent(
    """
    <style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E;
        font-size: 0.85rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #E6E6E6;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Radio box styling */
    .radio-card {
        background: linear-gradient(145deg, #161B22, #0E1117);
        border: 1px solid #00D9FF;
        border-left: 5px solid #00D9FF;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .radio-header {
        color: #00D9FF;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .radio-body {
        color: #E6E6E6;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    """
)


@st.cache_data(show_spinner=False)
def _cached_load_and_align_laps(
    year: int,
    grand_prix: str,
    session_type: str,
    driver: str,
    reference_driver: str,
) -> pd.DataFrame:
    """Cached wrapper around analysis.alignment.load_and_align_laps."""
    return load_and_align_laps(
        year=year,
        grand_prix=grand_prix,
        session_type=session_type,
        driver=driver,
        reference_driver=reference_driver,
    )


def _parse_corner_boundaries(raw_json: str) -> list[CornerBoundary]:
    """Parse and validate user-supplied corner boundary JSON."""
    if not raw_json or not raw_json.strip():
        raise ValueError("Corner boundaries JSON is empty. Please provide corner definitions.")

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corner boundaries are not valid JSON: {exc}") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Corner boundaries must be a non-empty JSON list.")

    boundaries: list[CornerBoundary] = []
    for entry in parsed:
        if not isinstance(entry, dict) or not {"corner", "start", "end"} <= entry.keys():
            raise ValueError("Each corner must contain 'corner', 'start', and 'end' keys.")
        boundaries.append(
            {
                "corner": int(entry["corner"]),
                "start": float(entry["start"]),
                "end": float(entry["end"]),
            }
        )
    return boundaries


def _resolve_api_key(sidebar_key: str) -> str:
    """Resolve Gemini API key: sidebar first, then environment fallback."""
    if sidebar_key and sidebar_key.strip():
        return sidebar_key.strip()

    try:
        env_vars = load_environment(["GEMINI_API_KEY"])
        return env_vars.get("GEMINI_API_KEY", "").strip()
    except EnvironmentConfigurationError:
        return ""


def _render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return the user configuration."""
    st.sidebar.title("🏎️ AI Race Engineer")
    st.sidebar.caption("F1 Telemetry Analytics & Driver Coaching")

    st.sidebar.markdown("---")
    demo_mode = st.sidebar.toggle(
        "Demo Mode (Offline)",
        value=True,
        help="Runs an offline 6-corner synthetic session without FastF1 or Gemini requirements.",
    )

    if demo_mode:
        st.sidebar.info("Using pre-calibrated demo telemetry (**DEM** vs **REF**).")
        year = 2024
        grand_prix = "Synthetic"
        session_type = "Q"
        driver = DEMO_DRIVER_CODE
        reference_driver = DEMO_REFERENCE_DRIVER_CODE
        corner_boundaries_raw = json.dumps(DEMO_CORNER_BOUNDARIES, indent=2)
    else:
        st.sidebar.subheader("Session Setup")
        year = st.sidebar.number_input("Season Year", min_value=1950, max_value=2100, value=2024, step=1)
        grand_prix = st.sidebar.text_input("Grand Prix", value="Monza")
        session_type = st.sidebar.selectbox("Session Type", options=_SESSION_TYPE_OPTIONS, index=3)
        driver = st.sidebar.text_input("Driver Code", value="VER", max_chars=3)
        reference_driver = st.sidebar.text_input("Reference Driver Code", value="NOR", max_chars=3)

        with st.sidebar.expander("Corner boundaries (JSON)", expanded=False):
            st.caption("Paste circuit corner definitions [{corner, start, end} in meters].")
            corner_boundaries_raw = st.text_area(
                "Corner boundaries",
                value="",
                placeholder='[\n  {"corner": 1, "start": 300, "end": 500}\n]',
                height=160,
                label_visibility="collapsed",
            )

    st.sidebar.markdown("---")
    sidebar_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Optional: leave blank to fall back to GEMINI_API_KEY in .env.",
    )

    analyze_clicked = st.sidebar.button("Run Telemetry Analysis", type="primary", width="stretch")

    return {
        "demo_mode": demo_mode,
        "year": int(year),
        "grand_prix": grand_prix,
        "session_type": session_type,
        "driver": driver.upper().strip(),
        "reference_driver": reference_driver.upper().strip(),
        "corner_boundaries_raw": corner_boundaries_raw,
        "api_key": sidebar_api_key,
        "analyze_clicked": analyze_clicked,
    }


def _execute_pipeline(
    selections: dict[str, Any],
) -> tuple[SessionAnalysis, pd.DataFrame, str, dict[str, float], bool] | None:
    """Execute the full analysis pipeline with real-time progress and duration tracking."""
    timings: dict[str, float] = {}
    is_demo = selections["demo_mode"]

    try:
        corner_boundaries = _parse_corner_boundaries(str(selections["corner_boundaries_raw"]))
    except ValueError as exc:
        st.error(f"Invalid corner boundaries: {exc}")
        return None

    status_container = st.status("Analyzing Session Telemetry...", expanded=True)

    with status_container:
        # Step 1 & 2: Load and align
        st.write("🛰️ **Stage 1 & 2:** Loading & aligning telemetry...")
        t0 = time.perf_counter()
        if is_demo:
            aligned_df = generate_demo_telemetry()
            session_id = DEMO_SESSION_ID
            timings["load_and_align"] = time.perf_counter() - t0
        else:
            try:
                aligned_df = _cached_load_and_align_laps(
                    year=selections["year"],
                    grand_prix=selections["grand_prix"],
                    session_type=selections["session_type"],
                    driver=selections["driver"],
                    reference_driver=selections["reference_driver"],
                )
                session_id = build_session_id(
                    selections["year"], selections["grand_prix"], selections["session_type"]
                )
                timings["load_and_align"] = time.perf_counter() - t0
            except (HelperValidationError, AlignmentError) as exc:
                st.error(f"Telemetry loading failed: {exc}")
                return None
            except Exception as exc:
                _logger.exception("Unexpected error loading session telemetry: %s", exc)
                st.error(f"Failed to load session: {exc}")
                return None

        # Step 3: Corner metrics
        st.write("📐 **Stage 3:** Slicing corner windows and extracting metrics...")
        t1 = time.perf_counter()
        try:
            corner_metrics = extract_corner_metrics(aligned_df, corner_boundaries)
            timings["corners"] = time.perf_counter() - t1
        except CornerMetricsError as exc:
            st.error(f"Corner metrics extraction failed: {exc}")
            return None
        except Exception as exc:
            _logger.exception("Unexpected error extracting corner metrics: %s", exc)
            st.error(f"Failed to extract corner metrics: {exc}")
            return None

        # Step 4: Rule Engine
        st.write("⚖️ **Stage 4:** Executing deterministic Rule Engine & scoring...")
        t2 = time.perf_counter()
        try:
            analysis = build_session_analysis(
                corner_metrics=corner_metrics,
                session_id=session_id,
                driver_code=selections["driver"],
                reference_driver_code=selections["reference_driver"],
            )
            timings["rule_engine"] = time.perf_counter() - t2
        except (HelperValidationError, EngineError) as exc:
            st.error(f"Rule Engine execution failed: {exc}")
            return None
        except Exception as exc:
            _logger.exception("Unexpected error running Rule Engine: %s", exc)
            st.error(f"Analysis engine failed: {exc}")
            return None

        # Step 5: Gemini Coaching Feedback
        st.write("📻 **Stage 5:** Preparing AI Race Engineer feedback...")
        t3 = time.perf_counter()

        if is_demo:
            # Demo mode is strictly self-contained and offline
            feedback_text = DEMO_COACHING_FEEDBACK
        else:
            resolved_key = _resolve_api_key(str(selections["api_key"]))
            if resolved_key:
                feedback_text = get_race_engineer_feedback(analysis, api_key=resolved_key)
            else:
                feedback_text = (
                    "AI coaching radio is offline (no Gemini API key supplied in sidebar or .env).\n"
                    "Please configure an API key to enable live race engineer radio coaching."
                )
        timings["llm_feedback"] = time.perf_counter() - t3

        total_elapsed = sum(timings.values())
        timings["total"] = total_elapsed
        status_container.update(
            label=f"Pipeline Complete ({total_elapsed:.2f}s)",
            state="complete",
            expanded=False,
        )

    return analysis, aligned_df, feedback_text, timings, is_demo


def _render_demo_banner() -> None:
    """Render a persistent synthetic demo mode notice."""
    st.info(
        "⚠️ **Demo Mode Active** — Displaying pre-calibrated synthetic telemetry (DEM vs REF). "
        "No real FastF1 session or external network calls are used."
    )


def _render_overview_tab(
    analysis: SessionAnalysis,
    aligned_df: pd.DataFrame,
    is_demo: bool,
) -> None:
    """Render the executive Overview tab."""
    if is_demo:
        _render_demo_banner()

    summary = analysis.driver_summary

    # KPI metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Driver Score", f"{summary.performance_score}/100")
    col2.metric("Total Time Lost", f"{summary.total_time_loss_s:.3f} s")
    col3.metric("Worst Corner", f"Corner {analysis.worst_corner}")
    col4.metric("Best Corner", f"Corner {analysis.best_corner}")

    st.markdown(f"**Primary Driving Fault:** `{analysis.biggest_mistake}`")
    st.markdown("---")

    # Overview key charts
    col_left, col_right = st.columns(2)
    with col_left:
        fig_speed = plot_speed(
            aligned_df["Distance"],
            aligned_df["DriverSpeed"],
            aligned_df["ReferenceSpeed"],
            summary.driver_code,
            summary.reference_driver_code,
        )
        st.plotly_chart(fig_speed, width="stretch", key="overview_speed_chart")

    with col_right:
        fig_delta = plot_delta_time(aligned_df["Distance"], aligned_df["DeltaTime"])
        st.plotly_chart(fig_delta, width="stretch", key="overview_delta_chart")


def _render_telemetry_tab(
    aligned_df: pd.DataFrame,
    driver: str,
    reference_driver: str,
    is_demo: bool,
) -> None:
    """Render full telemetry comparison traces."""
    if is_demo:
        _render_demo_banner()

    distance = aligned_df["Distance"]

    fig_speed = plot_speed(
        distance,
        aligned_df["DriverSpeed"],
        aligned_df["ReferenceSpeed"],
        driver,
        reference_driver,
    )
    st.plotly_chart(fig_speed, width="stretch", key="telemetry_speed_chart")

    fig_throttle = plot_throttle(
        distance,
        aligned_df["DriverThrottle"],
        aligned_df["ReferenceThrottle"],
        driver,
        reference_driver,
    )
    st.plotly_chart(fig_throttle, width="stretch", key="telemetry_throttle_chart")

    fig_brake = plot_brake(
        distance,
        aligned_df["DriverBrake"],
        aligned_df["ReferenceBrake"],
        driver,
        reference_driver,
    )
    st.plotly_chart(fig_brake, width="stretch", key="telemetry_brake_chart")

    fig_delta = plot_delta_time(distance, aligned_df["DeltaTime"])
    st.plotly_chart(fig_delta, width="stretch", key="telemetry_delta_chart")


def _render_corners_tab(analysis: SessionAnalysis, is_demo: bool) -> None:
    """Render the corner-by-corner tabular breakdown."""
    if is_demo:
        _render_demo_banner()

    rows = []
    for corner in analysis.corner_metrics:
        is_worst = corner.corner_number == analysis.worst_corner
        is_best = corner.corner_number == analysis.best_corner

        tag = ""
        if is_worst:
            tag = " 🔴 [Worst]"
        elif is_best:
            tag = " 🟢 [Best]"

        rows.append(
            {
                "Corner": f"C{corner.corner_number}{tag}",
                "Window (m)": f"{corner.start_distance_m:.0f} - {corner.end_distance_m:.0f}",
                "Brake Delta (m)": f"{corner.braking_delta_m:+.1f}",
                "Apex Speed Δ (km/h)": f"{corner.apex_speed_delta_kph:+.1f}",
                "Exit Speed Δ (km/h)": f"{corner.exit_speed_delta_kph:+.1f}",
                "Throttle Delay (m)": f"{corner.throttle_delay_m:+.1f}",
                "Time Loss (s)": f"{corner.time_loss_s:+.3f}",
                "Issue": corner.issue_type.value,
                "Severity": corner.severity.value.upper(),
            }
        )

    df_corners = pd.DataFrame(rows)
    st.dataframe(df_corners, width="stretch", hide_index=True)


def _render_radio_tab(analysis: SessionAnalysis, feedback_text: str, is_demo: bool) -> None:
    """Render the AI Race Engineer radio card and transparency payload viewer."""
    if is_demo:
        _render_demo_banner()

    safe_feedback = html.escape(feedback_text).replace("\n", "<br>")

    radio_html = textwrap.dedent(
        f"""
        <div class="radio-card">
            <div class="radio-header">
                🎧 TEAM RADIO &bull; PIT WALL TO {analysis.driver_summary.driver_code}
            </div>
            <div class="radio-body">
                {safe_feedback}
            </div>
        </div>
        """
    )
    st.markdown(radio_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔒 Architectural Transparency & Privacy")
    st.caption(
        "AI Race Engineer follows strict architectural isolation: **Gemini never receives raw telemetry, "
        "dataframes, or coordinates**. Only the pre-computed, deterministic JSON summary below is provided."
    )

    with st.expander("View exact JSON payload sent to Gemini", expanded=False):
        st.code(analysis.model_dump_json(indent=2), language="json")


def _render_diagnostics_tab(
    analysis: SessionAnalysis,
    aligned_df: pd.DataFrame,
    timings: dict[str, float],
    is_demo: bool,
) -> None:
    """Render diagnostic telemetry, timings, and runtime metadata."""
    if is_demo:
        _render_demo_banner()

    st.subheader("Session Diagnostics")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    d_col1.metric("Session ID", analysis.session_id)
    d_col2.metric("Telemetry Grid Samples", f"{len(aligned_df):,} pts")
    d_col3.metric("Corner Windows Analyzed", len(analysis.corner_metrics))
    d_col4.metric("Pipeline Duration", f"{timings.get('total', 0.0):.2f} s")

    st.markdown("#### Pipeline Stage Durations")
    timing_data = [
        {"Stage": "1 & 2: Load & Align Telemetry", "Duration": f"{timings.get('load_and_align', 0.0):.3f} s"},
        {"Stage": "3: Corner Slicing & Metrics", "Duration": f"{timings.get('corners', 0.0):.3f} s"},
        {"Stage": "4: Rule Engine Classification", "Duration": f"{timings.get('rule_engine', 0.0):.3f} s"},
        {"Stage": "5: AI Race Engineer Radio", "Duration": f"{timings.get('llm_feedback', 0.0):.3f} s"},
    ]
    st.table(timing_data)


def _render_how_it_works_tab() -> None:
    """Render the architecture and data-flow documentation."""
    st.subheader("🏎️ How AI Race Engineer Works")
    explanation = textwrap.dedent(
        """
        ```
        FastF1 Telemetry ──► 1m Spatial Grid ──► Corner Slicing ──► Rule Engine ──► Gemini AI
        (Time-series)         (Δt Integration)    (Braking, Apex)    (Score & Faults) (Coaching Radio)
                                                                            │
                                                                            ▼
                                                                Plotly Dashboard & Tables
        ```

        ### Core Architectural Principles
        1. **Deterministic Python Core**:
           All telemetry loading, distance resampling (1-meter grid), delta-time integration
           ($\\Delta t = \\int (1/v - 1/v_{ref}) ds$), apex finding, and penalty scoring occur strictly in pure Python.
           Identical telemetry always produces identical metrics.
        2. **LLM as Explainer Only**:
           Gemini never performs calculations, never estimates speeds, and never accesses raw telemetry.
           It turns the finished `SessionAnalysis` JSON into concise coaching bullets.
        3. **Safe Fallbacks & Zero Crashes**:
           If network connectivity fails or an API key is missing, all telemetry analytics, charts,
           and corner-by-corner tables remain fully operational.
        """
    )
    st.markdown(explanation)


def main() -> None:
    """Main Streamlit execution entry point."""
    st.set_page_config(
        page_title="AI Race Engineer",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    selections = _render_sidebar()

    # Reset result if user toggles between Demo Mode and Real Session Mode
    active_mode = selections["demo_mode"]
    if "active_mode" in st.session_state and st.session_state["active_mode"] != active_mode:
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("auto_demo_run", None)
    st.session_state["active_mode"] = active_mode

    # Automatically run on initial load in demo mode, or when Analyze button is clicked
    should_run = selections["analyze_clicked"] or (
        active_mode and "auto_demo_run" not in st.session_state
    )

    if should_run:
        st.session_state["auto_demo_run"] = True
        result = _execute_pipeline(selections)
        if result is not None:
            st.session_state["pipeline_result"] = result

    if "pipeline_result" in st.session_state:
        analysis, aligned_df, feedback_text, timings, result_is_demo = st.session_state["pipeline_result"]

        t_overview, t_telemetry, t_corners, t_radio, t_diag, t_how = st.tabs(
            [
                "🏁 Overview",
                "📈 Telemetry",
                "🔍 Corners",
                "📻 AI Engineer Radio",
                "🛠️ Diagnostics",
                "ℹ️ How It Works",
            ]
        )

        with t_overview:
            _render_overview_tab(analysis, aligned_df, result_is_demo)

        with t_telemetry:
            _render_telemetry_tab(
                aligned_df,
                analysis.driver_summary.driver_code,
                analysis.driver_summary.reference_driver_code,
                result_is_demo,
            )

        with t_corners:
            _render_corners_tab(analysis, result_is_demo)

        with t_radio:
            _render_radio_tab(analysis, feedback_text, result_is_demo)

        with t_diag:
            _render_diagnostics_tab(analysis, aligned_df, timings, result_is_demo)

        with t_how:
            _render_how_it_works_tab()
    else:
        st.info("Configure session parameters in the sidebar and click **Run Telemetry Analysis**.")


if __name__ == "__main__":
    main()
