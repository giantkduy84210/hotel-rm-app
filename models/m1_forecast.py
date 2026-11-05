"""
M1 -- demand forecasting (dynamic, risk; predictive model).

Holt-Winters exponential smoothing with additive trend and weekly seasonality
(s = 7), competing against two seasonal-naive baselines on the same rolling
backtest. All methods forecast the same dense nightly rooms-sold series read
from fact_daily_occupancy.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HOTEL_KEY = {"H1": 1, "H2": 2}


DATA_END = 20170831  # canonical analysis window (paper arrival dates end 2017-08-31)


def read_series(con, hotel: str) -> pd.Series:
    """Dense nightly rooms-sold series (kept bookings) for one hotel,
    within the canonical analysis window."""
    df = con.execute("""
        SELECT o.DateKey, SUM(o.RoomsSold) AS rooms
        FROM fact_daily_occupancy o JOIN dim_hotel h USING (HotelKey)
        WHERE h.HotelCode = ? AND o.DateKey <= ?
        GROUP BY o.DateKey ORDER BY o.DateKey""",
        [hotel, DATA_END]).fetch_df()
    idx = pd.to_datetime(df["DateKey"].astype(str), format="%Y%m%d")
    return pd.Series(df["rooms"].to_numpy(dtype=float), index=idx)


def forecast_hw(train: pd.Series, horizon: int) -> np.ndarray:
    fit = ExponentialSmoothing(
        train.to_numpy(), trend="add", seasonal="add", seasonal_periods=7).fit(optimized=True)
    return np.asarray(fit.forecast(horizon))


def forecast_snaive7(train: pd.Series, horizon: int) -> np.ndarray:
    last7 = train.iloc[-7:].to_numpy()
    reps = int(np.ceil(horizon / 7))
    return np.tile(last7, reps)[:horizon]


def forecast_yoy(train: pd.Series, horizon: int) -> np.ndarray:
    """Same weekday one year (364 days) ago, scaled by recent growth."""
    n = len(train)
    out = np.empty(horizon)
    recent = train.iloc[-28:].mean()
    prev = train.iloc[-28 - 364: -364].mean() if n >= 392 else train.iloc[:28].mean()
    growth = recent / prev if prev > 0 else 1.0
    for h in range(horizon):
        lag = n + h - 364
        out[h] = (train.iloc[lag] if lag >= 0 else train.iloc[0]) * growth
    return out


METHODS = {
    "holt_winters": forecast_hw,
    "snaive7": forecast_snaive7,
    "yoy_snaive364": forecast_yoy,
}


def run(con, params: dict) -> dict:
    hotel = params["hotel"]
    horizon = int(params.get("horizon") or 28)
    method = params.get("method") or "auto"
    series = read_series(con, hotel)

    # 'auto' = the method that won the last full backtest for this hotel
    if method == "auto":
        method = best_method(con, hotel)
    fc = METHODS[method](series, horizon)
    future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)

    return {
        "model_id": "M1",
        "hotel": hotel,
        "method": method,
        "horizon": horizon,
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "rooms": float(round(v, 1))}
            for d, v in zip(future, fc)],
        "total_rooms": float(round(fc.sum(), 0)),
        "headline_kpi": float(round(fc.sum(), 0)),
    }


def best_method(con, hotel: str) -> str:
    """Cached winner from the rolling backtest (falls back to Holt-Winters)."""
    try:
        import json as _json
        metrics = con.execute(
            "SELECT metrics FROM model_catalog WHERE model_id = 'M1'").fetchone()[0]
        if metrics:
            m = _json.loads(metrics)
            return m.get(hotel, {}).get("winner", "holt_winters")
    except Exception:
        pass
    return "holt_winters"


def backtest(con, hotel: str, horizon: int = 28, n_cutoffs: int = 12,
             methods=("holt_winters", "snaive7", "yoy_snaive364")) -> dict:
    """Rolling monthly cut-offs: at each, train on data up to the cut-off,
    forecast the next `horizon` days, score against the realised demand."""
    series = read_series(con, hotel)
    cutoffs = pd.date_range(series.index[-1] - pd.Timedelta(days=horizon * n_cutoffs + 30),
                            series.index[-1] - pd.Timedelta(days=horizon), freq="30D")[-n_cutoffs:]
    cutoffs = [c for c in cutoffs if len(series[:c]) >= 400]

    rows = []
    for method in methods:
        errs = []
        for c in cutoffs:
            train, test = series[:c], series[c + pd.Timedelta(days=1): c + pd.Timedelta(days=horizon)]
            if len(test) < horizon:
                continue
            fc = METHODS[method](train, horizon)
            e = fc - test.to_numpy()
            for lead, (err, act) in enumerate(zip(e, test.to_numpy()), start=1):
                errs.append({"lead": lead, "err": err, "actual": act})
        df = pd.DataFrame(errs)
        ae = df["err"].abs()
        rows.append({
            "method": method,
            "MAE": float(round(ae.mean(), 2)),
            "RMSE": float(round(np.sqrt((df["err"] ** 2).mean()), 2)),
            "WAPE": float(round(ae.sum() / df["actual"].abs().sum(), 4)),
            "sigma_pooled": float(round(df["err"].std(), 2)),
        })
    winner = min(rows, key=lambda r: r["WAPE"])["method"]
    return {"hotel": hotel, "n_cutoffs": len(cutoffs), "metrics": rows, "winner": winner}
