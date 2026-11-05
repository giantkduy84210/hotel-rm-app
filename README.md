# Hotel RM DSS -- application (CO4031)

Decision support system for hotel revenue management on the Antonio et al. (2019)
dataset (119,390 bookings, H1 resort / H2 city, Jul 2015 - Aug 2017), with the
**model base management subsystem (MBMS)** as the course topic.

## Structure
- `etl.py` -- extract / transform (5 steps) / clean (logged rules) / load DuckDB /
  6 quality gates (pipeline exits non-zero if any gate fails)
- `warehouse/hotel_rm.duckdb` -- star schema: `fact_room_night` (booking-night grain),
  `fact_daily_occupancy` (dense snapshot), 7 conformed dimensions, plus the MBMS
  tables `model_catalog` and `fact_scenario_run`
- `mbms.py` -- the MBMS: the model directory (`model_catalog`), the execution /
  integration / command processor (`run_model()` -- the single entry point for the
  UI, the experiments and the tests), and the run log (`fact_scenario_run`)
- `models/` -- M1 Holt-Winters forecasting + seasonal-naive baselines, M2 Littlewood
  group decision, M3 allocation MILP with bid prices (PuLP/CBC), M4 overbooking
  (newsvendor critical ratio + Monte Carlo), M5 pricing by goal seeking (brentq + tornado)
- `tests/` -- known-answer tests, run through the same registry path as the UI
- `experiments/` -- M1 rolling backtest, M4 replay evaluation (3 policies x 3 stress
  levels), the four demonstration scenarios S1-S4
- `app.py` -- Streamlit interface, four screens mapped to Simon's phases

## Run
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# data: place H1.csv and H2.csv (Antonio et al. 2019, Data in Brief) under data/raw/
.venv/bin/python etl.py                     # build the warehouse (6 quality gates)
.venv/bin/python -m pytest tests/ -q         # known-answer tests via run_model()
.venv/bin/python experiments/backtest_m1.py  # M1 backtest; winner stored in catalog
.venv/bin/python experiments/replay_m4.py    # overbooking replay evaluation
.venv/bin/python experiments/scenarios.py    # S1-S4 demonstration scenarios
.venv/bin/streamlit run app.py              # the interactive application
```

## Data
`H1.csv` / `H2.csv` from Antonio, Almeida & Nunes (2019), "Hotel booking demand
datasets", Data in Brief 26:104412 (doi:10.1016/j.dib.2018.11.126), CC BY 4.0.
