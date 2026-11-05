"""
Experiment 1 -- M1 rolling backtest (report Section 5.1).

Rolling monthly cut-offs; each method forecasts 28 days ahead using only data
up to the cut-off. MAE, RMSE, WAPE per method per hotel. The winner is stored
into model_catalog.metrics (M1's 'auto' mode then uses it) -- and if a simple
baseline beats Holt-Winters, that is reported, not hidden.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb

import models.m1_forecast as m1
from mbms import init_mbms

OUT = ROOT / "experiments" / "results"


def main():
    con = duckdb.connect(str(ROOT / "warehouse" / "hotel_rm.duckdb"))
    init_mbms(con)
    results = {}
    for hotel in ("H1", "H2"):
        bt = m1.backtest(con, hotel, horizon=28, n_cutoffs=14)
        results[hotel] = bt
        print(f"\n=== {hotel} (n_cutoffs={bt['n_cutoffs']}, horizon 28d) ===")
        print(f"{'method':<16}{'MAE':>8}{'RMSE':>8}{'WAPE':>8}{'sigma':>8}")
        for r in bt["metrics"]:
            print(f"{r['method']:<16}{r['MAE']:>8.2f}{r['RMSE']:>8.2f}"
                  f"{r['WAPE']:>8.4f}{r['sigma_pooled']:>8.2f}")
        print(f"winner: {bt['winner']}")

    # store the winners in the catalog (M1 'auto' uses them)
    con.execute("UPDATE model_catalog SET metrics = ? WHERE model_id = 'M1'",
                [json.dumps(results)])
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest_m1.json").write_text(json.dumps(results, indent=2))
    print("\n[ok] winners stored in model_catalog.metrics; results/backtest_m1.json")
    con.close()


if __name__ == "__main__":
    main()
