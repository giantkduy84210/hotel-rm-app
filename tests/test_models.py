"""
Known-answer tests for M1-M5. All model runs go through run_model() -- the
same registry path the interactive interface uses -- so the tests exercise the
full validation chain (catalog lookup, parameter validation, logging).
"""

import numpy as np
import pytest
from scipy import stats

import mbms
from models.m2_littlewood import expected_min_classic
from models.m4_overbooking import expected_pi, optimal_booking_limit
from models.m5_pricing import required_adr
from models import m1_forecast as m1


# ----------------------------------------------------------------------
# M2 -- Littlewood closed form
# ----------------------------------------------------------------------
def test_m2_littlewood_symmetric_case():
    """p_L/p_H = 0.5 and D ~ N(100, 30): y* = mu + sigma * Phi^-1(0.5) = 100."""
    y_star = 100 + 30 * stats.norm.ppf(1 - 100 / 200)
    assert y_star == pytest.approx(100.0, abs=1e-9)


def test_m2_normal_loss_at_mean():
    """E[min(D, x)] at x = mu: mu - sigma * phi(0) = 100 - 30 * 0.39894."""
    assert expected_min_classic(100, 30, 100) == pytest.approx(100 - 30 * stats.norm.pdf(0), abs=1e-9)


def test_m2_run_via_registry(con):
    res = mbms.run_model(con, "M2", {
        "hotel": "H2", "arrival_date": "2017-08-11", "nights": 3, "rooms": 10,
        "rate": 70.0, "p_H": 100.0, "mu": 80.0, "sigma": 20.0}, scenario_name="test")
    # y* = 80 + 20 * Phi^-1(1 - 0.7) = 80 + 20 * Phi^-1(0.3)
    y_expected = 80 + 20 * stats.norm.ppf(0.3)
    assert res["y_star"] == pytest.approx(y_expected, abs=0.15)
    # reject: 10 > C - y* requires the warehouse capacity; check consistency
    C = res["capacity"]
    assert res["accept"] == (10 <= C - y_expected + 1e-9)
    # breakeven rate must reproduce Delta = 0 when used as p_L
    e_min_c = expected_min_classic(80, 20, C)
    e_min_cg = expected_min_classic(80, 20, C - 10)
    p_star = 100 * (e_min_c - e_min_cg) / 10
    assert res["breakeven_rate"] == pytest.approx(p_star, abs=0.05)


def test_m2_rejects_bad_params(con):
    with pytest.raises(ValueError):
        mbms.run_model(con, "M2", {"hotel": "H1", "arrival_date": "2017-08-01",
                                   "nights": 3, "rooms": 10, "rate": 150.0, "p_H": 100.0,
                                   "mu": 80, "sigma": 20})  # p_L >= p_H
    with pytest.raises(ValueError):
        mbms.run_model(con, "M2", {"hotel": "H1"})  # missing required params


# ----------------------------------------------------------------------
# M4 -- newsvendor critical ratio
# ----------------------------------------------------------------------
def test_m4_scan_matches_bruteforce():
    """The scan must return the smallest B with P(S >= C) >= r/(r+c_w),
    verified by an independent explicit-pmf loop."""
    C, p, r, c_w = 10, 0.80, 100.0, 50.0
    ratio = r / (r + c_w)
    expected = None
    for B in range(C, 4 * C):
        prob = sum(stats.binom.pmf(s, B, p) for s in range(C, B + 1))  # explicit P(S >= C)
        if prob >= ratio:
            expected = B
            break
    assert optimal_booking_limit(C, p, r, c_w) == expected


def test_m4_expected_pi_known():
    """pi(B) for B = C: no walks possible -> pi = r * E[S] = r * B * p."""
    C, p, r, c_w = 10, 0.9, 100.0, 150.0
    assert expected_pi(C, C, p, r, c_w) == pytest.approx(r * C * p, abs=1e-6)


def test_m4_run_via_registry(con):
    res = mbms.run_model(con, "M4", {"hotel": "H2", "season": "High",
                                     "cutoff": 7, "mc_reps": 5000}, scenario_name="test")
    assert res["B_star"] >= res["capacity"]
    assert 0 < res["showup_probability"] <= 1
    assert res["critical_ratio"] == pytest.approx(
        res["adr"] / (res["adr"] + res["walk_cost"]), abs=0.01)
    # Monte Carlo mean must be close to the analytic expected profit
    assert res["mc"]["profit_mean"] == pytest.approx(res["expected_profit"], rel=0.02)


# ----------------------------------------------------------------------
# M3 -- MILP allocation (structural known-answers)
# ----------------------------------------------------------------------
def test_m3_must_accept_and_must_reject(con):
    week = {"start_date": "2017-08-07", "end_date": "2017-08-13", "hotel": "H2"}
    tiny = dict(week, offers=[{"arrival": "2017-08-08", "nights": 2, "rooms": 1,
                               "room_type": "A", "rate": 500.0}])
    huge = dict(week, offers=[{"arrival": "2017-08-08", "nights": 2, "rooms": 5000,
                               "room_type": "A", "rate": 1.0}])
    r1 = mbms.run_model(con, "M3", tiny, scenario_name="test")
    assert r1["accepted_ids"] == [0]
    r2 = mbms.run_model(con, "M3", huge, scenario_name="test")
    assert r2["accepted_ids"] == []


def test_m3_bid_prices_nonnegative(con):
    res = mbms.run_model(con, "M3", {
        "hotel": "H1", "start_date": "2017-08-07", "end_date": "2017-08-13",
        "offers": [{"arrival": "2017-08-08", "nights": 3, "rooms": 15,
                    "room_type": "A", "rate": 60.0}]}, scenario_name="test")
    assert all(b["bid_price"] >= 0 for b in res["bid_prices"])


# ----------------------------------------------------------------------
# M5 -- goal seeking
# ----------------------------------------------------------------------
def test_m5_brentq_matches_closed_form():
    """Closed form: a* = (target / (occ0 * ADR0^-e))^(1/(1+e))."""
    occ0, adr0, e, target = 0.70, 100.0, -0.8, 77.0
    a_cf = (target / (occ0 * adr0 ** (-e))) ** (1 / (1 + e))
    assert required_adr(occ0, adr0, e, target) == pytest.approx(a_cf, rel=1e-6)


def test_m5_run_via_registry(con):
    res = mbms.run_model(con, "M5", {"hotel": "H1", "season": "Low",
                                     "elasticity": -0.8, "target_uplift": 0.2},
                         scenario_name="test")
    # the required ADR must achieve the target RevPAR (round-trip check)
    occ = res["baseline"]["occ_0"] * (res["required_adr"] / res["baseline"]["ADR_0"]) ** -0.8
    revpar = res["required_adr"] * occ
    assert revpar == pytest.approx(res["target_revpar"], rel=0.005)
    assert res["adr_increase_pct"] > 0


# ----------------------------------------------------------------------
# M1 -- forecasting baselines
# ----------------------------------------------------------------------
def test_m1_snaive7_repeats_last_week():
    import pandas as pd
    idx = pd.date_range("2020-01-01", periods=21)
    train = pd.Series(np.arange(21.0), index=idx)
    fc = m1.forecast_snaive7(train, 10)
    # seasonal-naive repeats the last 7 days: [14..20] then [14,15,16]
    assert list(fc) == [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 14.0, 15.0, 16.0]


def test_m1_run_via_registry(con):
    res = mbms.run_model(con, "M1", {"hotel": "H1", "horizon": 7}, scenario_name="test")
    assert len(res["forecast"]) == 7
    assert all(f["rooms"] > 0 for f in res["forecast"])


# ----------------------------------------------------------------------
# Registry itself
# ----------------------------------------------------------------------
def test_registry_logs_runs(con):
    n = con.execute("SELECT COUNT(*) FROM fact_scenario_run WHERE model_id = 'M1'").fetchone()[0]
    assert n >= 1  # the run above was logged


def test_registry_rejects_unknown_model(con):
    with pytest.raises(ValueError):
        mbms.run_model(con, "M9", {})
