"""
M2 -- Littlewood group decision (static, risk; optimisation via analytic formula).

The classical two-class rule: protect y* rooms for full-rate (transient) demand
where  P(D_H > y*) = p_L / p_H,  with D_H ~ N(mu, sigma^2) the transient demand
for the requested nights. Accept the group iff its size fits the unprotected
remainder:  G <= C - y*.  Expected revenues are compared through the normal-loss
function  E[min(D, x)] = mu - sigma * [phi(z) - z * (1 - Phi(z))],  z = (x-mu)/sigma.
"""

import numpy as np
from scipy import stats

import models.m1_forecast as m1

HOTEL_KEY = {"H1": 1, "H2": 2}


def _season_of(con, date) -> str:
    key = int(np.datetime64(date).astype("datetime64[D]").astype(str).replace("-", ""))
    row = con.execute("SELECT Season FROM dim_date WHERE DateKey = ?", [key]).fetchone()
    return row[0] if row else "Shoulder"


def transient_inputs(con, hotel: str, date: str) -> dict:
    """Default p_H (transient rate) and per-night mu, sigma from M1.

    mu  = M1's forecast for the night (hotel-level) x transient room share
    sigma = the pooled std of M1's forecast errors (rolling backtest)
    """
    season = _season_of(con, date)
    row = con.execute("""
        SELECT AVG(f.ADR),            -- transient rate p_H (season mean)
               SUM(CASE WHEN s.MarketSegment <> 'Groups' THEN 1 ELSE 0 END)::DOUBLE
                 / COUNT(*)            -- transient booking share
        FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey)
        JOIN dim_market_segment s USING (MarketSegmentKey)
        JOIN dim_date d USING (DateKey)
        WHERE h.HotelCode = ? AND f.IsCanceled = 0 AND d.Season = ?
          AND d.DateKey <= 20170831""",
        [hotel, season]).fetchone()
    p_H, transient_share = row[0], row[1]

    series = m1.read_series(con, hotel)
    cutoff = pd_cutoff = series.index[series.index < np.datetime64(date)][-1]
    horizon = 1
    mu = float(m1.forecast_hw(series[:pd_cutoff], horizon)[0]) * transient_share

    bt = m1.backtest(con, hotel, n_cutoffs=8)
    sigma = next(r["sigma_pooled"] for r in bt["metrics"] if r["method"] == "holt_winters")
    return {"p_H": float(p_H), "mu": mu, "sigma": float(sigma), "season": season,
            "transient_share": float(transient_share)}


def expected_min_classic(mu: float, sigma: float, x: float) -> float:
    """E[min(D, x)] for D ~ N(mu, sigma^2), via the normal-loss function:
    E[min(D,x)] = mu - sigma * [phi(z) - z * (1 - Phi(z))],  z = (x - mu)/sigma."""
    z = (x - mu) / sigma
    return mu - sigma * (stats.norm.pdf(z) - z * stats.norm.sf(z))


def run(con, params: dict) -> dict:
    hotel = params["hotel"]
    arrival = params["arrival_date"]
    nights = int(params["nights"])
    G = int(params["rooms"])
    p_L = float(params["rate"])

    defaults = transient_inputs(con, hotel, arrival)
    p_H = float(params.get("p_H") or defaults["p_H"])
    mu = float(params.get("mu") or defaults["mu"])
    sigma = float(params.get("sigma") or defaults["sigma"])
    if p_L >= p_H:
        raise ValueError("group rate p_L must be below the transient rate p_H")

    C = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                    [hotel]).fetchone()[0]

    # protection level from the Littlewood ratio (per night, same mu/sigma)
    y_star = mu + sigma * stats.norm.ppf(1 - p_L / p_H)
    unprotected = C - y_star
    accept = G <= unprotected

    # expected-revenue comparison: accept vs reject, per night D_H ~ N(mu, sigma^2)
    e_min_c = expected_min_classic(mu, sigma, C)          # reject: sell to transients up to C
    e_min_cg = expected_min_classic(mu, sigma, C - G)     # accept: transients get C - G
    delta_per_night = p_L * G + p_H * e_min_cg - p_H * e_min_c
    delta_total = delta_per_night * nights

    # breakeven group rate: p_L* with delta_total = 0
    p_L_star = p_H * (e_min_c - e_min_cg) / G

    return {
        "model_id": "M2",
        "hotel": hotel,
        "arrival_date": arrival,
        "nights": nights,
        "rooms": G,
        "p_L": p_L,
        "p_H": round(p_H, 2),
        "mu": round(mu, 1),
        "sigma": round(sigma, 1),
        "capacity": C,
        "y_star": round(y_star, 1),
        "unprotected_capacity": round(unprotected, 1),
        "accept": bool(accept),
        "delta_revenue": round(delta_total, 0),
        "breakeven_rate": round(p_L_star, 2),
        "transient_share": round(defaults["transient_share"], 3),
        "season": defaults["season"],
        "headline_kpi": round(delta_total, 0),
    }
