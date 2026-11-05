"""
M5 -- pricing by goal seeking (static, uncertainty; what-if by formula).

The constant-elasticity demand response
    occ(ADR) = occ_0 * (ADR / ADR_0)^e
induces
    RevPAR(ADR) = ADR * occ(ADR).
Goal seeking inverts the curve: find the minimum ADR with RevPAR(ADR) = target
(Brent's method; well posed for -1 < e < 0). The tornado varies each parameter
by +/- 20% and reports the resulting required ADR.
"""

import numpy as np
from scipy import optimize

HOTEL_KEY = {"H1": 1, "H2": 2}


def season_baselines(con, hotel: str, season: str) -> dict:
    """occ_0, ADR_0, RevPAR_0 for one hotel and season, from the warehouse."""
    row = con.execute("""
        SELECT AVG(night_rooms)::DOUBLE, MAX(cap) FROM (
            SELECT o.DateKey, SUM(o.RoomsSold) AS night_rooms, ANY_VALUE(o.Capacity) AS cap
            FROM fact_daily_occupancy o
            JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
            WHERE h.HotelCode = ? AND d.Season = ?
            GROUP BY o.DateKey)""", [hotel, season]).fetchone()
    occ0 = row[0] / row[1]
    adr0 = con.execute("""
        SELECT AVG(f.ADR) FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
        WHERE h.HotelCode = ? AND f.IsCanceled = 0 AND d.Season = ?""",
        [hotel, season]).fetchone()[0]
    return {"occ_0": float(occ0), "ADR_0": float(adr0), "RevPAR_0": float(occ0 * adr0)}


def required_adr(occ0: float, adr0: float, e: float, target: float) -> float:
    def revpar_minus_target(a: float) -> float:
        return a * occ0 * (a / adr0) ** e - target

    lo, hi = 0.1 * adr0, 12.0 * adr0
    f_lo, f_hi = revpar_minus_target(lo), revpar_minus_target(hi)
    if f_lo * f_hi > 0:
        raise ValueError("no root in bracket; check elasticity range (-1 < e < 0)")
    return float(optimize.brentq(revpar_minus_target, lo, hi))


def run(con, params: dict) -> dict:
    hotel = params["hotel"]
    season = params["season"]
    e = float(params.get("elasticity") or -0.8)
    uplift = float(params.get("target_uplift") or 0.20)

    base = season_baselines(con, hotel, season)
    occ0, adr0 = base["occ_0"], base["ADR_0"]
    target = base["RevPAR_0"] * (1 + uplift)

    a_req = required_adr(occ0, adr0, e, target)
    occ_req = occ0 * (a_req / adr0) ** e

    # tornado: +/- 20% per parameter (each side independent; a side with no
    # finite root -- e.g. elasticity too close to -1 -- is reported as None)
    def _try(fn):
        try:
            return round(fn(), 2)
        except ValueError:
            return None

    cases = {
        "elasticity e": (lambda: required_adr(occ0, adr0, min(e * 0.8, -0.01), target),
                         lambda: required_adr(occ0, adr0, max(e * 1.2, -0.99), target)),
        "occupancy occ_0": (lambda: required_adr(occ0 * 1.2, adr0, e, target),
                            lambda: required_adr(occ0 * 0.8, adr0, e, target)),
        "baseline ADR_0": (lambda: required_adr(occ0, adr0 * 1.2, e, target),
                           lambda: required_adr(occ0, adr0 * 0.8, e, target)),
        "target": (lambda: required_adr(occ0, adr0, e, target * 1.2),
                   lambda: required_adr(occ0, adr0, e, target * 0.8)),
    }
    tornado = []
    for name, (fn_hi, fn_lo) in cases.items():
        tornado.append({"parameter": name, "low": _try(fn_lo), "high": _try(fn_hi),
                        "base": round(a_req, 2)})

    return {
        "model_id": "M5",
        "hotel": hotel,
        "season": season,
        "elasticity": e,
        "baseline": {k: round(v, 4) for k, v in base.items()},
        "target_revpar": round(target, 2),
        "target_uplift": uplift,
        "required_adr": round(a_req, 2),
        "adr_increase_pct": round(100 * (a_req / adr0 - 1), 1),
        "occupancy_at_required_adr": round(occ_req, 4),
        "tornado": tornado,
        "headline_kpi": round(a_req, 2),
    }
