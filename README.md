# AI-Race-Engineer
AI-Powered Telemetry Analytics &amp; Driver Coaching Platform
<div align="center">

# 🏎️ AI Race Engineer

**An AI-Powered Formula 1 Telemetry Analytics & Driver Coaching Platform**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastF1](https://img.shields.io/badge/Telemetry-FastF1-black)](https://github.com/theOehrly/Fast-F1)
[![Pydantic](https://img.shields.io/badge/Validated%20with-Pydantic%20v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

</div>

---

Deterministic Python analytics meet AI-generated coaching: **AI Race Engineer** loads real Formula 1
telemetry, spatially aligns two drivers' laps corner-by-corner, and turns the resulting metrics into
professional race-engineer-style feedback — with every number computed in Python and Gemini used
strictly as the explanation layer, never the calculator.

---

## Overview

**AI Race Engineer** compares a driver's fastest lap against a reference driver's, corner by corner,
and produces the kind of analysis a real F1 race engineer would deliver over the radio.

- **What it does** — Loads session telemetry via FastF1, aligns both drivers' Speed, Throttle, and
  Brake channels onto a shared distance axis, computes cumulative delta time, extracts deterministic
  per-corner metrics (braking point, apex speed, exit speed, throttle timing), classifies driving
  issues with a rule engine, and asks Gemini to translate the resulting structured analysis into
  concise coaching feedback.

- **Why distance alignment matters** — Telemetry is naturally indexed by time, but two drivers never
  reach the same corner at the same elapsed time. Comparing samples at equal timestamps compares
  *different physical locations* on track, producing meaningless deltas. Resampling both drivers onto
  a common **track-distance** grid guarantees that "Distance = 850m" means the same point on the
  circuit for both drivers — exactly how real motorsport telemetry tools compare driving performance.

- **Why AI is used only for coaching** — Every metric, threshold, and classification in this project
  is computed by deterministic Python (NumPy, Pandas, a rule engine with configurable thresholds).
  Gemini never sees raw telemetry and never performs a calculation — it receives a single structured
  JSON summary and its only job is turning already-computed facts into natural, professional language.
  This keeps every number in the app auditable, reproducible, and hallucination-free.

---

## Features

- 🏁 FastF1 telemetry loading with on-disk caching
- ⏱️ Fastest valid lap extraction per driver
- 📏 Distance-based spatial alignment (not timestamp-based)
- 📈 NumPy spatial interpolation across Speed, Throttle, and Brake channels
- ➕ Cumulative delta-time calculation via numerical integration
- 🌀 Corner-by-corner metric extraction (braking point, apex, exit, throttle delay)
- ⚙️ Deterministic, configurable rule-based driver coaching engine
- 🤖 AI Race Engineer powered by Google Gemini — explanation only, never calculation
- 📊 Interactive, themed Plotly dashboard with shared-axis telemetry charts
- 🖥️ Professional Streamlit UI
- 🧱 Clean, modular, fully testable architecture

---

## Architecture

```
                    FastF1 (raw session telemetry)
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Alignment Engine   │  distance-based interpolation
                 │  (analysis/alignment)│  + cumulative delta time
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Corner Metrics     │  braking / apex / exit / throttle
                 │  (analysis/corners)  │  per corner, per driver
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Rule Engine       │  threshold classification
                 │  (analysis/engine)   │  scoring · worst/best corner
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Session Analysis    │  validated Pydantic JSON
                 │  (models/schemas)    │  the ONLY thing sent to Gemini
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Google Gemini      │  JSON → natural-language
                 │  (llm/feedback)      │  race engineer coaching
                 └─────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Streamlit Dashboard │  charts · score · AI radio card
                 │     (app/main)       │  (orchestration only)
                 └─────────────────────┘
```

Domain logic (`analysis/`, `models/`) has zero dependency on FastF1's live API shape, Streamlit, or
Gemini — every layer is independently testable and the LLM boundary accepts exactly one input type:
a validated `SessionAnalysis` JSON payload.

---

## Folder Structure

```
AI-Race-Engineer/
├── analysis/
│   ├── alignment.py       # FastF1 loading, spatial interpolation, delta-time integral
│   ├── corners.py         # Per-corner deterministic metric extraction
│   └── engine.py          # Rule engine, scoring, SessionAnalysis assembly
│
├── llm/
│   └── feedback.py        # Gemini adapter — JSON in, coaching text out
│
├── app/
│   └── main.py             # Streamlit orchestration layer (no business logic)
│
├── models/
│   └── schemas.py          # Pydantic v2 domain models (CornerMetric, SessionAnalysis, ...)
│
├── utils/
│   ├── helpers.py          # Logging, env loading, validation, unit conversions
│   └── plotting.py         # Themed Plotly Graph Objects chart builders
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Telemetry | FastF1 |
| Numerical computing | NumPy, SciPy |
| Data manipulation | Pandas |
| Validation | Pydantic v2 |
| Visualization | Plotly (Graph Objects) |
| UI | Streamlit |
| AI coaching | Google Gemini API |

---

## Installation

```bash
git clone https://github.com/yourusername/AI-Race-Engineer.git
cd AI-Race-Engineer

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

streamlit run app/main.py
```

Set your Gemini API key in the sidebar, or via a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

## Screenshots

> _Screenshots to be added._

**Dashboard**

`[ placeholder ]`

**Telemetry Charts**

`[ placeholder ]`

**AI Race Engineer Radio**

`[ placeholder ]`

---

## Future Work

**Version 2**
- SQLite persistence layer (drivers, sessions, analysis results, driver scores)
- Historical performance tracking across sessions
- Repository interface isolated for a future PostgreSQL swap

**Version 3**
- Machine learning consistency prediction
- Anomaly detection in corner execution
- Driver clustering by driving style

---

## Learning Outcomes

This project was built to demonstrate production-grade software engineering, not just a working demo:

- **Clean Architecture** — strict dependency inversion between domain logic, adapters, and frameworks;
  the analysis core has zero knowledge of FastF1, Streamlit, or Gemini.
- **Scientific Computing** — NumPy-based spatial interpolation and cumulative numerical integration
  applied to real-world sensor data.
- **Signal Processing** — distance-domain resampling and noise-threshold-based event detection
  (braking point, throttle application).
- **Time-Series Analysis** — converting time-indexed telemetry into a spatially aligned, comparable
  representation.
- **LLM Integration** — a strict, auditable JSON boundary between deterministic computation and
  generative explanation, with graceful degradation when the LLM is unavailable.
- **Pydantic** — immutable, validated domain models as the single source of truth for data shape
  across every layer.
- **Modular Design** — a Strategy-pattern-ready rule engine designed so ML-based classification can be
  added later without rewriting existing modules.
- **Data Engineering** — a resilient pipeline from raw, noisy telemetry to a clean, structured,
  deterministic analytical output.

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## Author

**Your Name**
B.Tech CSE (AI & ML)

[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/yourusername)
