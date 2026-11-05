"""
M4 -- overbooking (static, risk; optimisation via analytic formula + simulation).

With S ~ Binomial(B, p) arrivals, the expected profit is
    pi(B) = r * E[min(S, C)] - c_w * E[(S - C)+]
and the optimal limit is the smallest B satisfying the newsvendor critical ratio
    P(S >= C) >= r / (r + c_w).
The show-up probability p is estimated at a cut-off h days before arrival, from
bookings still on the books (the dataset records values as of the day prior to
arrival; industry practice revises forecasts 7-30 days before arrival).
"""

import numpy as np
from scipy import stats

HOTEL_KEY = {"H1": 1, "H2": 2}


def showup_probability(con, hotel: str, season: str, cutoff: int) -> dict:
    """p(h): share of on-the-books bookings (cut-off h days before arrival)
    that finally showed up. Pooled over the season's arrival nights."""
    row = con.execute("""
        WITH arrivals AS (
            SELECT DISTINCT BookingId, ArrivalDateKey, CancelDateKey, IsCanceled, LeadTime
            FROM fact_room_night
        ),
        onbooks AS (
            SELECT a.* FROM arrivals a
            JOIN dim_date d ON d.DateKey = a.ArrivalDateKey
            JOIN dim_hotel h ON h.HotelKey = (CASE WHEN a.BookingId LIKE 'H1%' THEN 1 ELSE 2 END)
            WHERE h.HotelCode = ?
              AND d.Season = ?
              AND a.ArrivalDateKey - a.LeadTime <= a.ArrivalDateKey - ?
              AND (a.IsCanceled = 0
                   OR a.CancelDateKey > a.ArrivalDateKey - ?)
        )
        SELECT COUNT(*),
               SUM(CASE WHEN IsCanceled = 0 THEN 1 ELSE 0 END)
        FROM onbooks""", [hotel, season, cutoff, cutoff]).fetchone()
    on_books, kept = row[0], row[1]
    if not on_books:
        raise ValueError(f"no on-the-books bookings for {hotel} {season} at cut-off {cutoff}")
    return {"p": kept / on_books, "on_books": on_books, "kept": kept, "cutoff": cutoff}


def season_adr(con, hotel: str, season: str) -> float:
    return con.execute("""
        SELECT AVG(f.ADR) FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
        WHERE h.HotelCode = ? AND f.IsCanceled = 0 AND d.Season = ?""",
        [hotel, season]).fetchone()[0]


def expected_pi(B: int, C: int, p: float, r: float, c_w: float) -> float:
    """pi(B) = r E[min(S,C)] - c_w E[(S-C)+] with S ~ Binomial(B, p)."""
    s = np.arange(0, B + 1)
    pm = stats.binom.pmf(s, B, p)
    return float(r * np.minimum(s, C).dot(pm) - c_w * np.maximum(s - C, 0).dot(pm))


def optimal_booking_limit(C: int, p: float, r: float, c_w: float) -> int:
    """Smallest B >= C with P(S >= C) >= r / (r + c_w)  (S ~ Bin(B, p))."""
    ratio = r / (r + c_w)
    B = C
    while B < 4 * C:
        if stats.binom.sf(C - 1, B, p) >= ratio:
            break
        B += 1
    return B


def run(con, params: dict) -> dict:
    hotel = params["hotel"]
    season = params["season"]
    h = int(params.get("cutoff") or 7)
    cw_mult = float(params.get("walk_cost_multiple") or 1.5)
    reps = int(params.get("mc_reps") or 20000)
    seed = int(params.get("seed") or 42)

    est = showup_probability(con, hotel, season, h)
    p = est["p"]
    if params.get("showup_override") is not None:      # what-if: parameter override at run time
        p = float(params["showup_override"])
        est = dict(est, p_override=p)
    C = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                    [hotel]).fetchone()[0]
    r = season_adr(con, hotel, season)
    c_w = cw_mult * r

    # critical-ratio scan: smallest B >= C with P(S >= C) >= r / (r + c_w)
    ratio = r / (r + c_w)
    B = optimal_booking_limit(C, p, r, c_w)

    pi_star = expected_pi(B, C, p, r, c_w)
    p_walk = float(stats.binom.sf(C, B, p))
    e_walk = float(np.maximum(np.arange(0, B + 1) - C, 0).dot(stats.binom.pmf(np.arange(0, B + 1), B, p)))

    # Monte Carlo verification (fixed seed, 95% CI)
    rng = np.random.default_rng(seed)
    S = rng.binomial(B, p, size=reps)
    profit = r * np.minimum(S, C) - c_w * np.maximum(S - C, 0)
    ci = (float(np.percentile(profit, 2.5)), float(np.percentile(profit, 97.5)))
    mc_walk_prob = float((S > C).mean())
    mc_walk_mean = float(np.maximum(S - C, 0).mean())

    return {
        "model_id": "M4",
        "hotel": hotel,
        "season": season,
        "cutoff_days": h,
        "showup_probability": round(p, 4),
        "showup_override_used": params.get("showup_override") is not None,
        "on_books": est["on_books"],
        "kept": est["kept"],
        "capacity": C,
        "adr": round(float(r), 2),
        "walk_cost": round(c_w, 2),
        "critical_ratio": round(ratio, 4),
        "B_star": B,
        "overbooking_level": B - C,
        "expected_profit": round(pi_star, 0),
        "p_walk_at_least_one": round(p_walk, 4),
        "expected_walks": round(e_walk, 2),
        "mc": {"reps": reps, "profit_mean": round(float(profit.mean()), 0),
               "profit_ci95": [round(ci[0], 0), round(ci[1], 0)],
               "walk_prob": round(mc_walk_prob, 4),
               "walk_mean": round(mc_walk_mean, 2)},
        "headline_kpi": float(B),
    }
