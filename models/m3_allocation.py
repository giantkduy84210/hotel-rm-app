"""
M3 -- allocation by mixed-integer linear programming (static, certainty).

max  sum_{d,t} r_{d,t} y_{d,t} + sum_g R_g z_g
s.t. y_{d,t} + sum_{g on (d,t)} u_g z_g <= K_t          (per night x type)
     sum_t y_{d,t} + sum_{g on d} u_g z_g <= C           (per night, hotel level)
     0 <= y_{d,t} <= D_hat_{d,t}                          (forecast demand bound)
     z_g in {0,1}

Bid prices: the MILP is solved first for the accept/reject decision; the
accepted/rejected binaries are then fixed and the LP re-solved, whose capacity
duals are the bid prices (the marginal value of one more room-night).
"""

import numpy as np
import pandas as pd
import pulp

import models.m1_forecast as m1

HOTEL_KEY = {"H1": 1, "H2": 2}


def _per_type_capacity(con, hotel: str) -> dict:
    """K_t: max rooms sold in one night for room type t, after warm-up."""
    rows = con.execute("""
        SELECT r.ReservedRoomType, MAX(o.RoomsSold)
        FROM fact_daily_occupancy o
        JOIN dim_hotel h USING (HotelKey)
        JOIN dim_room_type r USING (RoomTypeKey)
        WHERE h.HotelCode = ? AND o.DateKey > 20150930
        GROUP BY 1""", [hotel]).fetchall()
    return dict(rows)


def _transient_rates(con, hotel: str, season: str) -> dict:
    """r_{d,t}: mean realised ADR of transient bookings per (type, season)."""
    rows = con.execute("""
        SELECT r.ReservedRoomType, AVG(f.ADR)
        FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey)
        JOIN dim_room_type r ON r.RoomTypeKey = f.ReservedRoomTypeKey
        JOIN dim_date d USING (DateKey)
        JOIN dim_market_segment s USING (MarketSegmentKey)
        WHERE h.HotelCode = ? AND f.IsCanceled = 0 AND s.MarketSegment <> 'Groups'
          AND d.Season = ?
        GROUP BY 1""", [hotel, season]).fetchall()
    return {k: float(v) for k, v in rows}


def _transient_shares(con, hotel: str) -> dict:
    """Share of each room type among kept transient booking-nights."""
    rows = con.execute("""
        SELECT r.ReservedRoomType,
               SUM(CASE WHEN s.MarketSegment <> 'Groups' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
        FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey)
        JOIN dim_room_type r ON r.RoomTypeKey = f.ReservedRoomTypeKey
        JOIN dim_market_segment s USING (MarketSegmentKey)
        WHERE h.HotelCode = ? AND f.IsCanceled = 0
        GROUP BY 1""", [hotel]).fetchall()
    return {k: float(v) for k, v in rows}


def _forecast_per_night(con, hotel: str, dates) -> dict:
    """D_hat_d: M1's forecast for each night of the window."""
    series = m1.read_series(con, hotel)
    cutoff = series.index[series.index < np.datetime64(dates[0])][-1]
    fc = m1.forecast_hw(series[:cutoff], len(dates))
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(dates, fc)}


def run(con, params: dict) -> dict:
    hotel = params["hotel"]
    start = pd.Timestamp(params["start_date"])
    end = pd.Timestamp(params["end_date"])
    offers = params["offers"]

    dates = pd.date_range(start, end, freq="D")
    K = _per_type_capacity(con, hotel)
    C = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                    [hotel]).fetchone()[0]
    shares = _transient_shares(con, hotel)
    D_hat = _forecast_per_night(con, hotel, dates)

    # rates by season of each night
    rate_cache: dict = {}
    types = sorted(K)

    def rate_for(t: str, d: pd.Timestamp) -> float:
        key = int(d.strftime("%Y%m%d"))
        season = con.execute("SELECT Season FROM dim_date WHERE DateKey = ?", [key]).fetchone()[0]
        if (t, season) not in rate_cache:
            rate_cache[(t, season)] = _transient_rates(con, hotel, season).get(t, 80.0)
        return rate_cache[(t, season)]

    prob = pulp.LpProblem("allocation", pulp.LpMaximize)
    y = {(d.strftime("%Y-%m-%d"), t): pulp.LpVariable(f"y_{d.strftime('%Y%m%d')}_{t}", lowBound=0)
         for d in dates for t in types}
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(len(offers))}

    # objective
    prob += pulp.lpSum(rate_for(t, d) * y[(d.strftime("%Y-%m-%d"), t)]
                       for d in dates for t in types
                       if rate_for(t, d) > 0) \
        + pulp.lpSum(float(o["rooms"]) * int(o["nights"]) * float(o["rate"]) * z[i]
                     for i, o in enumerate(offers))

    # constraints
    cap_type, cap_hotel = {}, {}
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        cap_hotel[d] = pulp.LpConstraint(
            pulp.lpSum(y[(ds, t)] for t in types)
            + pulp.lpSum(float(o["rooms"]) * z[i] for i, o in enumerate(offers)
                         if (pd.Timestamp(o["arrival"]).date() <= d.date()
                             and d < pd.Timestamp(o["arrival"]) + pd.Timedelta(days=int(o["nights"])))),
            sense=pulp.LpConstraintLE, name=f"capH_{d.strftime('%Y%m%d')}", rhs=C)
        prob += cap_hotel[d]
        for t in types:
            cap_type[(ds, t)] = pulp.LpConstraint(
                y[(ds, t)]
                + pulp.lpSum(float(o["rooms"]) * z[i] for i, o in enumerate(offers)
                              if o["room_type"] == t
                              and (pd.Timestamp(o["arrival"]).date() <= d.date()
                                   and d < pd.Timestamp(o["arrival"]) + pd.Timedelta(days=int(o["nights"])))),
                sense=pulp.LpConstraintLE, name=f"capT_{d.strftime('%Y%m%d')}_{t}",
                rhs=float(K[t]))
            prob += cap_type[(ds, t)]
            # forecast demand bound
            dh = D_hat[ds] * shares.get(t, 0.0)
            prob += y[(ds, t)] <= max(dh, 0.0)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"M3 LP status: {status}")

    accepted = [i for i in range(len(offers)) if z[i].value() > 0.5]
    Z_star = pulp.value(prob.objective)

    # ---- second solve: fix z, re-solve as LP -> capacity duals = bid prices ----
    prob2 = pulp.LpProblem("allocation_fixed", pulp.LpMaximize)
    y2 = {(d.strftime("%Y-%m-%d"), t): pulp.LpVariable(f"y2_{d.strftime('%Y%m%d')}_{t}", lowBound=0)
          for d in dates for t in types}
    z2 = {i: (1 if i in accepted else 0) for i in range(len(offers))}

    prob2 += pulp.lpSum(rate_for(t, d) * y2[(d.strftime("%Y-%m-%d"), t)]
                        for d in dates for t in types if rate_for(t, d) > 0) \
        + pulp.lpSum(float(o["rooms"]) * int(o["nights"]) * float(o["rate"]) * z2[i]
                     for i, o in enumerate(offers))
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        used_h = pulp.lpSum(float(o["rooms"]) * z2[i] for i, o in enumerate(offers)
                            if (pd.Timestamp(o["arrival"]).date() <= d.date()
                                and d < pd.Timestamp(o["arrival"]) + pd.Timedelta(days=int(o["nights"]))))
        con_h = pulp.LpConstraint(pulp.lpSum(y2[(ds, t)] for t in types) + used_h,
                                  sense=pulp.LpConstraintLE, name=f"capH_{d.strftime('%Y%m%d')}", rhs=C)
        prob2 += con_h
        for t in types:
            used_t = pulp.lpSum(float(o["rooms"]) * z2[i] for i, o in enumerate(offers)
                                 if o["room_type"] == t
                                 and (pd.Timestamp(o["arrival"]).date() <= d.date()
                                      and d < pd.Timestamp(o["arrival"]) + pd.Timedelta(days=int(o["nights"]))))
            con_t = pulp.LpConstraint(y2[(ds, t)] + used_t, sense=pulp.LpConstraintLE,
                                      name=f"capT_{d.strftime('%Y%m%d')}_{t}", rhs=float(K[t]))
            prob2 += con_t
            dh = D_hat[ds] * shares.get(t, 0.0)
            prob2 += y2[(ds, t)] <= max(dh, 0.0)
    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    bid_prices = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        for t in types:
            c = prob2.get_constraint_by_name(f"capT_{d.strftime('%Y%m%d')}_{t}")
            pi = c.pi if c.pi is not None else 0.0
            if pi and pi > 1e-6:
                bid_prices.append({"date": ds, "room_type": t,
                                   "bid_price": round(float(pi), 2)})

    offer_results = [
        {"offer": o, "accepted": i in accepted,
         "revenue": float(o["rooms"]) * int(o["nights"]) * float(o["rate"])}
        for i, o in enumerate(offers)]

    return {
        "model_id": "M3",
        "hotel": hotel,
        "window": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
        "optimal_revenue": round(float(Z_star), 0),
        "offers": offer_results,
        "accepted_ids": accepted,
        "bid_prices": bid_prices,
        "headline_kpi": round(float(Z_star), 0),
    }
