"""
Model Base Management Subsystem (MBMS) -- the course topic.

Implements the three relational organs declared in report Section 3.4:
  * model_catalog      -- the model directory (one row per model, carrying
                         the four-role variable declaration, the parameter
                         schema the registry validates against, assumptions,
                         limitations, method, version, metrics)
  * run_model()        -- the execution / integration / command processor
                         (single entry point for every caller: UI, experiments,
                         tests -- all validated against the catalog)
  * fact_scenario_run  -- the transaction log (parameters + result + headline
                         KPI per execution -- the memory of every what-if)
"""

import importlib
import json
from datetime import datetime, timezone

import duckdb

# ----------------------------------------------------------------------
# The model directory (design follows report Table 3.9: the theory fixes the
# directory's duties -- catalog, model definitions, availability & capability
# -- not the fields; this schema is our design, column by column).
# ----------------------------------------------------------------------
CATALOG = [
    {
        "model_id": "M1",
        "name": "Demand Forecasting",
        "purpose": "How many rooms will the hotel sell each night of the coming weeks?",
        "model_group": "predictive",
        "classification": "dynamic; risk; tactical",
        "variables": {
            "decision": "none (a predictive model informs decisions, it makes none)",
            "uncontrollable_random": "y_t = realised nightly rooms sold",
            "uncontrollable_parameter": "alpha,beta,gamma in (0,1) (estimated in fitting); s=7; calendar",
            "intermediate": "L_t level, b_t trend, S_t seasonal index",
            "result": "y_hat_{T+h} per future night; MAE/RMSE/WAPE from backtest",
        },
        "params_schema": {
            "hotel": {"type": "str", "values": ["H1", "H2"], "required": True},
            "horizon": {"type": "int", "min": 1, "max": 120, "default": 28},
            "method": {"type": "str", "values": ["auto", "holt_winters", "snaive7", "yoy_snaive364"],
                       "default": "auto"},
        },
        "assumptions": {
            "A1": "additive Holt-Winters with weekly (s=7) seasonality",
            "A2": "smoothing constants estimated by minimising SSE in the fitting step",
            "A3": "baselines compete on the same backtest; the winner is used, not assumed",
        },
        "limitations": "forecasts based on realised bookings (conservative: no unconstraining)",
        "method": "statsmodels ExponentialSmoothing (additive trend, additive weekly seasonal) + seasonal-naive baselines",
        "module_path": "models.m1_forecast",
        "version": "1.0",
    },
    {
        "model_id": "M2",
        "name": "Littlewood Group Decision",
        "purpose": "A group offers rooms at a discount -- accept or reject?",
        "model_group": "optimisation via analytic formula",
        "classification": "static; risk; tactical",
        "variables": {
            "decision": "y = protection level (equivalently accept/reject via G <= C - y*)",
            "uncontrollable_random": "D_H ~ N(mu, sigma^2) transient demand; y_hat = M1's forecast (observed before the decision)",
            "uncontrollable_parameter": "p_L, p_H rates; C capacity; G rooms, n nights",
            "intermediate": "X = min(D_H, y); E[X] via normal-loss",
            "result": "Delta = expected-revenue difference; accept/reject; breakeven group rate",
        },
        "params_schema": {
            "hotel": {"type": "str", "values": ["H1", "H2"], "required": True},
            "arrival_date": {"type": "date", "required": True},
            "nights": {"type": "int", "min": 1, "max": 30, "required": True},
            "rooms": {"type": "int", "min": 1, "max": 200, "required": True},
            "rate": {"type": "float", "min": 1, "max": 1000, "required": True},
            "mu": {"type": "float", "min": 0, "default": None},
            "sigma": {"type": "float", "min": 0.01, "default": None},
            "p_H": {"type": "float", "min": 1, "max": 1000, "default": None},
        },
        "assumptions": {
            "A1": "discounted (group) demand arrives before full-fare demand (checkable: group lead times are longer)",
            "A2": "D_H normal; E[min(D,x)] via the normal-loss function",
        },
        "limitations": "two-class, single-property; one request at a time (M3 handles the competing week)",
        "method": "Littlewood closed form y* = mu + sigma*Phi^-1(1 - p_L/p_H); scipy.stats.norm",
        "module_path": "models.m2_littlewood",
        "version": "1.0",
    },
    {
        "model_id": "M3",
        "name": "Allocation by Linear Programming",
        "purpose": "Several groups compete for a week of capacity -- which to accept, and what is a room-night worth?",
        "model_group": "optimisation via algorithm",
        "classification": "static; certainty; tactical",
        "variables": {
            "decision": "y_{d,t} >= 0 transient rooms per (night,type); z_g in {0,1} accept group g",
            "uncontrollable_random": "D_hat_{d,t}: M1's forecast, observed before the decisions",
            "uncontrollable_parameter": "K_{d,t} capacity; r_{d,t} transient rates; R_g group revenue; u_g room-nights used",
            "intermediate": "pi_{d,t} shadow prices of the capacity constraints (bid prices)",
            "result": "Z* optimal revenue; accepted set; bid prices",
        },
        "params_schema": {
            "hotel": {"type": "str", "values": ["H1", "H2"], "required": True},
            "start_date": {"type": "date", "required": True},
            "end_date": {"type": "date", "required": True},
            "offers": {"type": "list", "required": True},
        },
        "assumptions": {
            "A1": "point forecasts taken as given (certainty class; randomness handled upstream by M1/M2)",
            "A2": "per-type capacity estimated by the same max-rooms-sold method as the hotel level",
        },
        "limitations": "week horizon; transient rates approximated by season averages per type",
        "method": "mixed-integer LP (PuLP, CBC solver); dual values of capacity constraints extracted as bid prices",
        "module_path": "models.m3_allocation",
        "version": "1.0",
    },
    {
        "model_id": "M4",
        "name": "Overbooking",
        "purpose": "Given cancellations, how many bookings should we accept for a night?",
        "model_group": "optimisation via analytic formula, verified by simulation",
        "classification": "static; risk; tactical",
        "variables": {
            "decision": "B = booking limit (overbooking level B - C)",
            "uncontrollable_random": "S ~ Binomial(B, p) arrivals",
            "uncontrollable_parameter": "p show-up probability at cut-off h; C capacity; r room revenue; c_w walk cost (1.5x ADR default)",
            "intermediate": "O = min(S,C) occupied; W = (S-C)+ walked",
            "result": "pi(B) expected profit; B*; P(walk >= 1); E[walks]",
        },
        "params_schema": {
            "hotel": {"type": "str", "values": ["H1", "H2"], "required": True},
            "season": {"type": "str", "values": ["High", "Shoulder", "Low"], "required": True},
            "cutoff": {"type": "int", "min": 1, "max": 30, "default": 7},
            "showup_override": {"type": "float", "min": 0.05, "max": 1.0, "default": None},
            "walk_cost_multiple": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.5},
            "mc_reps": {"type": "int", "min": 1000, "max": 100000, "default": 20000},
            "seed": {"type": "int", "default": 42},
        },
        "assumptions": {
            "A1": "show-up probability estimated at a cut-off h days before arrival, from bookings still on the books",
            "A2": "walk cost is management-set (subjective per the literature); default 1.5x ADR",
            "A3": "S binomial: independent show-ups, common p",
        },
        "limitations": "single night; independence across nights and bookings; binomial is a simplification of the true mixture",
        "method": "newsvendor critical ratio: smallest B with P(S >= C) >= r/(r+c_w) (scipy.stats.binom.sf scan) + Monte Carlo verification",
        "module_path": "models.m4_overbooking",
        "version": "1.0",
    },
    {
        "model_id": "M5",
        "name": "Pricing by Goal Seeking",
        "purpose": "What nightly ADR is needed to reach the target RevPAR in a given season?",
        "model_group": "what-if by formula",
        "classification": "static; uncertainty; tactical",
        "variables": {
            "decision": "ADR = nightly average daily rate",
            "uncontrollable_random": "e = price elasticity of demand (value genuinely unknown)",
            "uncontrollable_parameter": "occ_0, ADR_0 season baselines; target RevPAR",
            "intermediate": "occ(ADR) = occ_0 (ADR/ADR_0)^e demand response",
            "result": "RevPAR(ADR) = ADR * occ(ADR); required ADR (goal seeking); tornado",
        },
        "params_schema": {
            "hotel": {"type": "str", "values": ["H1", "H2"], "required": True},
            "season": {"type": "str", "values": ["High", "Shoulder", "Low"], "required": True},
            "elasticity": {"type": "float", "min": -0.99, "max": -0.01, "default": -0.8},
            "target_uplift": {"type": "float", "min": 0.01, "max": 2.0, "default": 0.20},
        },
        "assumptions": {
            "A1": "constant-elasticity demand response occ(ADR) = occ_0 (ADR/ADR_0)^e",
            "A2": "elasticity NOT estimable from this data (no independent price variation) -- set by the manager, sensitivity shown by tornado",
            "A3": "goal seeking well posed for -1 < e < 0",
        },
        "limitations": "no macro factors, no competitor pricing (dataset has neither); elasticity assumed",
        "method": "Brent root-finding on RevPAR(ADR) - target; tornado varies each parameter +/- 20%",
        "module_path": "models.m5_pricing",
        "version": "1.0",
    },
]


# ----------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------
def init_mbms(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_catalog (
            model_id       VARCHAR PRIMARY KEY,
            name           VARCHAR,
            purpose        VARCHAR,
            model_group    VARCHAR,
            classification VARCHAR,
            variables      VARCHAR,   -- JSON: the four-role declaration
            params_schema  VARCHAR,   -- JSON: what run_model() validates against
            assumptions    VARCHAR,   -- JSON
            limitations    VARCHAR,
            method         VARCHAR,
            module_path    VARCHAR,
            version        VARCHAR,
            metrics        VARCHAR,   -- JSON: evaluation results (updated by experiments)
            created_at     TIMESTAMP
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS fact_scenario_run (
            run_id        BIGINT,
            model_id      VARCHAR,
            scenario_name VARCHAR,
            params        VARCHAR,   -- JSON: the full parameter set
            result        VARCHAR,   -- JSON: structured result
            headline_kpi  DOUBLE,
            run_at        TIMESTAMP
        )""")
    now = datetime.now(timezone.utc)
    # preserve stored evaluation metrics across re-initialisation: every script
    # calls init_mbms(), and INSERT OR REPLACE would otherwise wipe the winners
    # the backtest stored (M1's auto mode reads them)
    existing = dict(con.execute(
        "SELECT model_id, metrics FROM model_catalog").fetchall())
    for m in CATALOG:
        con.execute(
            "INSERT OR REPLACE INTO model_catalog VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [m["model_id"], m["name"], m["purpose"], m["model_group"],
             m["classification"], json.dumps(m["variables"]),
             json.dumps(m["params_schema"]), json.dumps(m["assumptions"]),
             m["limitations"], m["method"], m["module_path"], m["version"],
             existing.get(m["model_id"]), now])


# ----------------------------------------------------------------------
# run_model: the execution, integration, and command processor
# ----------------------------------------------------------------------
def _validate(model: dict, params: dict) -> None:
    schema = model["params_schema"]
    for key, spec in schema.items():
        if spec.get("required") and (params.get(key) is None):
            raise ValueError(f"[{model['model_id']}] missing required parameter: {key}")
    for key, val in params.items():
        if key not in schema:
            raise ValueError(f"[{model['model_id']}] unknown parameter: {key}")
        spec = schema[key]
        if val is None:
            continue
        t = spec["type"]
        if t == "str" and "values" in spec and val not in spec["values"]:
            raise ValueError(f"[{model['model_id']}] {key}={val!r} not in {spec['values']}")
        if t in ("int", "float"):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"[{model['model_id']}] {key} must be numeric")
            if "min" in spec and val < spec["min"]:
                raise ValueError(f"[{model['model_id']}] {key}={val} < min {spec['min']}")
            if "max" in spec and val > spec["max"]:
                raise ValueError(f"[{model['model_id']}] {key}={val} > max {spec['max']}")


def run_model(con: duckdb.DuckDBPyConnection, model_id: str, params: dict,
              scenario_name: str | None = None) -> dict:
    """The single entry point every caller goes through (UI, experiments, tests)."""
    row = con.execute(
        "SELECT model_id, module_path, params_schema FROM model_catalog WHERE model_id = ?",
        [model_id]).fetchone()
    if row is None:
        raise ValueError(f"unknown model: {model_id} (not in model_catalog)")
    _, module_path, schema_json = row
    _validate({"model_id": model_id, "params_schema": json.loads(schema_json)}, params)

    module = importlib.import_module(module_path)
    result = module.run(con, params)

    headline = result.get("headline_kpi")
    con.execute(
        "INSERT INTO fact_scenario_run VALUES (DEFAULT, ?, ?, ?, ?, ?, ?)",
        [model_id, scenario_name, json.dumps(params, default=str),
         json.dumps(result, default=str), headline, datetime.now(timezone.utc)])
    return result


def load_params(con: duckdb.DuckDBPyConnection, model_id: str, params: dict) -> dict:
    """Fill in the catalog's declared defaults for omitted parameters."""
    schema = json.loads(con.execute(
        "SELECT params_schema FROM model_catalog WHERE model_id = ?",
        [model_id]).fetchone()[0])
    out = dict(params)
    for key, spec in schema.items():
        if key not in out or out[key] is None:
            if "default" in spec and spec["default"] is not None:
                out[key] = spec["default"]
    return out
