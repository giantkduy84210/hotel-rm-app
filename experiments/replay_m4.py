"""
Experiment 2 -- M4 replay evaluation (report Section 5.1), corrected design.

A booking-limit policy caps how many bookings may be ON THE BOOKS covering any
given night -- so the replay processes bookings in chronological booking order
and tracks, for every stay night, the accepted bookings covering it:

    * no overbooking:  L = C'
    * optimal:          L = B*(season of the night, capacity C')
    * accept-all:       L = infinity

Survivors are the finally-kept bookings among the accepted (actual outcomes
from the data); occupied = min(survivors, C'); walked = (survivors - C')+;
net = r(night) * occupied - c_w * walked, summed over the evaluation window
(arrivals 2016-06-01 .. 2017-08-31; earlier arrivals whose stays reach into
the window are included). Stress levels 100% / 90% / 80% of estimated capacity.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb

import models.m4_overbooking as m4
from mbms import init_mbms

OUT = ROOT / "experiments" / "results"
ARR_FROM, ARR_TO = 20160601, 20170831  # arrivals window (31-day lookback for spillover)


def load_bookings(con, hotel):
    """One row per booking: arrival, nights, kept, ADR, booking date."""
    return con.execute("""
        SELECT f.BookingId,
               MIN(f.DateKey) AS arr,
               COUNT(*) AS n_nights,
               ANY_VALUE(f.IsCanceled) AS canceled,
               ANY_VALUE(f.ADR) AS adr,
               MIN(f.DateKey) - ANY_VALUE(f.LeadTime) AS bdate
        FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
        WHERE h.HotelCode = ?
        GROUP BY f.BookingId HAVING MIN(f.DateKey) BETWEEN ? AND ?
        ORDER BY bdate""", [hotel, ARR_FROM, ARR_TO]).fetchall()


def replay(bookings, dates_idx, L_of_night, C, r_of_night, c_w, window):
    """Run one policy; returns totals. L_of_night: date -> limit (or None = inf)."""
    accepted_counts = defaultdict(int)      # date -> accepted bookings covering it
    surv_counts = defaultdict(int)          # date -> kept bookings covering it
    for bid, arr, n_nights, canceled, adr, bdate in bookings:
        nights = [arr + i for i in range(n_nights)]
        limit = L_of_night(arr)
        if limit is not None and any(accepted_counts[d] >= limit for d in nights):
            continue                          # rejected: some covered night is full
        for d in nights:
            accepted_counts[d] += 1
            if canceled == 0:
                surv_counts[d] += 1
    total_rev = total_wc = 0.0
    occupied = walked = 0
    for d in window:
        s = surv_counts.get(d, 0)
        occ, wlk = min(s, C), max(s - C, 0)
        r = r_of_night.get(d, 100.0)
        total_rev += r * occ
        total_wc += c_w * wlk
        occupied += occ
        walked += wlk
    return {"net_revenue": round(total_rev - total_wc, 0),
            "revenue": round(total_rev, 0), "walk_cost": round(total_wc, 0),
            "occupied_room_nights": occupied, "walked_guests": walked}


def main():
    con = duckdb.connect(str(ROOT / "warehouse" / "hotel_rm.duckdb"))
    init_mbms(con)
    results = {}
    for hotel in ("H1", "H2"):
        C = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                        [hotel]).fetchone()[0]
        adr = con.execute("""
            SELECT AVG(f.ADR) FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
            WHERE h.HotelCode = ? AND f.IsCanceled = 0""", [hotel]).fetchone()[0]
        c_w = 1.5 * adr
        bookings = load_bookings(con, hotel)
        window = [r[0] for r in con.execute(
            "SELECT DateKey FROM dim_date WHERE DateKey BETWEEN 20160701 AND 20170831").fetchall()]
        season_of = dict(con.execute("SELECT DateKey, Season FROM dim_date").fetchall())
        r_of_night = dict(con.execute("""
            SELECT f.DateKey, AVG(f.ADR) FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            WHERE h.HotelCode = ? AND f.IsCanceled = 0
            GROUP BY f.DateKey""", [hotel]).fetchall())

        res_hotel = {"capacity": C, "adr": round(adr, 2), "walk_cost": round(c_w, 2),
                     "n_bookings": len(bookings), "n_nights": len(window), "policies": {}}
        for stress in (1.0, 0.9, 0.8):
            Cs = int(round(C * stress))
            # optimal limits recomputed at the stressed capacity
            B_star = {s: m4.optimal_booking_limit(
                Cs, m4.showup_probability(con, hotel, s, 7)["p"], adr, c_w)
                for s in ("High", "Shoulder", "Low")}
            pol = {
                "no_overbooking": replay(bookings, None, lambda d: Cs, Cs, r_of_night, c_w, window),
                "optimal": replay(bookings, None,
                                  lambda d: B_star[season_of[d]], Cs, r_of_night, c_w, window),
                "accept_all": replay(bookings, None, lambda d: None, Cs, r_of_night, c_w, window),
            }
            pol["B_star"] = B_star
            res_hotel["policies"][f"stress_{int(stress * 100)}"] = pol
        results[hotel] = res_hotel
        print(f"\n=== {hotel} (C={C}, ADR={adr:.2f}, c_w={c_w:.2f}, "
              f"{len(bookings):,} bookings, {len(window)} nights) ===")
        for stress, pol in res_hotel["policies"].items():
            print(f"  {stress}: no-OB {pol['no_overbooking']['net_revenue']:>11,.0f} | "
                  f"optimal {pol['optimal']['net_revenue']:>11,.0f} "
                  f"({pol['optimal']['walked_guests']:>3} walks) | "
                  f"accept-all {pol['accept_all']['net_revenue']:>11,.0f} "
                  f"({pol['accept_all']['walked_guests']:>4} walks)")

    OUT.mkdir(exist_ok=True)
    (OUT / "replay_m4.json").write_text(json.dumps(results, indent=2))
    print("\n[ok] results/replay_m4.json")
    con.close()


if __name__ == "__main__":
    main()
