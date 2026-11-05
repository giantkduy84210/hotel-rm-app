"""
Experiment 3 -- the four demonstration scenarios (report Section 5.2).

S1  cancellation-rate increase (what-if, overbooking): p drops 5pp in the
    peak season -> new B*, new walk risks.
S2  group request (what-if, allocation): 10 rooms x 3 nights at 70% of the
    transient rate, run for BOTH hotels in peak and off-peak -- the same offer
    can be accepted at one property and rejected at the other.
S3  RevPAR target (goal seeking, pricing): +20% RevPAR in the low season ->
    required ADR + tornado sensitivity.
S4  weekly allocation with competing groups (what-if, LP): five offers compete
    for one August week; then one rejected offer's rate is raised to its
    breakeven level and the verdict is tested again.

All runs go through run_model() and are logged in fact_scenario_run.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb

from mbms import init_mbms, run_model

OUT = ROOT / "experiments" / "results"


def main():
    con = duckdb.connect(str(ROOT / "warehouse" / "hotel_rm.duckdb"))
    init_mbms(con)
    out = {}

    # ------------------------------------------------------------------ S1
    print("=== S1: cancellation rate +5pp (peak season, both hotels) ===")
    out["S1"] = {}
    for hotel in ("H1", "H2"):
        base = run_model(con, "M4", {"hotel": hotel, "season": "High", "cutoff": 7},
                         scenario_name=f"S1-base-{hotel}")
        p_new = round(base["showup_probability"] - 0.05, 4)
        shock = run_model(con, "M4", {"hotel": hotel, "season": "High", "cutoff": 7,
                                       "showup_override": p_new},
                          scenario_name=f"S1-shock-{hotel}")
        out["S1"][hotel] = {"base": base, "shock": shock}
        print(f"  {hotel}: p {base['showup_probability']} -> {p_new}; "
              f"B* {base['B_star']} -> {shock['B_star']} "
              f"(overbooking {base['overbooking_level']} -> {shock['overbooking_level']}); "
              f"profit {base['expected_profit']:,.0f} -> {shock['expected_profit']:,.0f}")

    # ------------------------------------------------------------------ S2
    print("=== S2: group request, 3 nights at 70% of transient rate (10 and 25 rooms) ===")
    out["S2"] = {}
    for hotel in ("H1", "H2"):
        for label, date in (("High", "2017-08-11"), ("Low", "2017-01-13")):
            for G in (10, 25):
                ph = run_model(con, "M2", {"hotel": hotel, "arrival_date": date,
                                           "nights": 3, "rooms": G, "rate": 1.0},
                               scenario_name=f"S2-probe-{hotel}-{label}-{G}")
                p_H = ph["p_H"]
                res = run_model(con, "M2", {"hotel": hotel, "arrival_date": date, "nights": 3,
                                             "rooms": G, "rate": round(0.70 * p_H, 2)},
                                scenario_name=f"S2-{hotel}-{label}-G{G}")
                out["S2"][f"{hotel}-{label}-G{G}"] = res
                print(f"  {hotel} {label} G={G}: p_H={p_H}, p_L={round(0.70 * p_H, 2)}, "
                      f"mu={res['mu']}, sigma={res['sigma']}, y*={res['y_star']}, "
                      f"unprotected={res['unprotected_capacity']}: "
                      f"{'ACCEPT' if res['accept'] else 'REJECT'} "
                      f"(Delta {res['delta_revenue']:,.0f}, breakeven {res['breakeven_rate']})")

    # ------------------------------------------------------------------ S3
    print("=== S3: +20% RevPAR target, low season (goal seeking) ===")
    out["S3"] = {}
    for hotel in ("H1", "H2"):
        res = run_model(con, "M5", {"hotel": hotel, "season": "Low",
                                    "elasticity": -0.8, "target_uplift": 0.20},
                        scenario_name=f"S3-{hotel}")
        out["S3"][hotel] = res
        b = res["baseline"]
        print(f"  {hotel}: occ_0={b['occ_0']}, ADR_0={b['ADR_0']}, RevPAR_0={b['RevPAR_0']}"
              f" -> required ADR {res['required_adr']} ({res['adr_increase_pct']}%), "
              f"occ {res['occupancy_at_required_adr']}")
        for t in res["tornado"]:
            print(f"      tornado {t['parameter']}: [{t['low']}, {t['high']}]")

    # ------------------------------------------------------------------ S4
    print("=== S4: weekly LP allocation, H2, 2017-08-07..13, five competing offers ===")
    week = {"hotel": "H2", "start_date": "2017-08-07", "end_date": "2017-08-13"}
    offers = [
        {"arrival": "2017-08-07", "nights": 5, "rooms": 12, "room_type": "A", "rate": 108.0},
        {"arrival": "2017-08-08", "nights": 4, "rooms": 20, "room_type": "A", "rate": 88.0},
        {"arrival": "2017-08-09", "nights": 3, "rooms": 15, "room_type": "D", "rate": 95.0},
        {"arrival": "2017-08-10", "nights": 2, "rooms": 25, "room_type": "E", "rate": 78.0},
        {"arrival": "2017-08-11", "nights": 3, "rooms": 18, "room_type": "F", "rate": 82.0},
    ]
    res = run_model(con, "M3", dict(week, offers=offers), scenario_name="S4-baseline")
    out["S4"] = {"baseline": res}
    print(f"  Z* = {res['optimal_revenue']:,.0f}; accepted: {res['accepted_ids']}")
    for o in res["offers"]:
        print(f"    offer {o['offer']['arrival']} {o['offer']['room_type']} "
              f"{o['offer']['rooms']}x{o['offer']['nights']} @ {o['offer']['rate']}: "
              f"{'ACCEPT' if o['accepted'] else 'REJECT'}")
    bp = res["bid_prices"]
    print(f"  bid prices: {len(bp)} positive entries, "
          f"range {min((b['bid_price'] for b in bp), default=0):.1f}"
          f"-{max((b['bid_price'] for b in bp), default=0):.1f}")

    # S4 follow-up: raise a rejected offer's rate to breakeven and re-run
    rejected = [o for o in res["offers"] if not o["accepted"]]
    if rejected:
        o0 = rejected[0]["offer"]
        # breakeven: the offer's nightly rate at which accepting becomes optimal --
        # probe upward until the verdict flips (the LP's own breakeven)
        lo, hi = o0["rate"], 200.0
        for _ in range(20):
            mid = round((lo + hi) / 2, 2)
            r2 = run_model(con, "M3", dict(week, offers=[dict(o0, rate=mid)]),
                           scenario_name="S4-breakeven-probe")
            if r2["accepted_ids"] == [0]:
                hi = mid
            else:
                lo = mid
        r_final = run_model(con, "M3", dict(week, offers=[dict(o0, rate=hi)]),
                             scenario_name="S4-breakeven-final")
        out["S4"]["breakeven_probe"] = {"offer": o0, "breakeven_rate": hi,
                                        "accepted_at_breakeven": r_final["accepted_ids"] == [0]}
        print(f"  rejected offer {o0['arrival']} {o0['room_type']}: breakeven rate = {hi} "
              f"(accepted there: {r_final['accepted_ids'] == [0]})")

    OUT.mkdir(exist_ok=True)
    (OUT / "scenarios.json").write_text(json.dumps(out, indent=2, default=str))
    n = con.execute("SELECT COUNT(*) FROM fact_scenario_run").fetchone()[0]
    print(f"\n[ok] results/scenarios.json; fact_scenario_run now has {n} logged runs")
    con.close()


if __name__ == "__main__":
    main()
