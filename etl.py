"""
ETL pipeline: Antonio et al. (2019) hotel booking demand -> DuckDB star schema.

Implements Section 4.1 of the report:
  Extract   : read H1.csv / H2.csv (raw PMS extracts, comma-separated)
  Transform : 5 operations
              (1) date assembly        (year/month/day components -> arrival date,
                                         booking date = arrival - lead time)
              (2) stay explosion        (booking -> booking-nights, the fact grain)
              (3) surrogate keys       (conformed dimensions get integer keys)
              (4) realised-revenue split (revenue = ADR if kept, 0 if cancelled)
              (5) capacity estimation   (max rooms sold on any night after warm-up)
  Clean     : documented rules, each logging rows removed (cleaning_log.json)
  Load      : DuckDB warehouse (2 fact tables, 7 conformed dimensions)
  Gates     : 6 quality assertions; pipeline exits non-zero if any fails

Usage:  python etl.py [--data-dir data/raw] [--db warehouse/hotel_rm.duckdb]
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

WARMUP_END = "2015-09-30"          # capacity estimated only from nights after warm-up
DATA_END = 20170831                 # canonical analysis window: the paper's arrival dates
                                   # end 2017-08-31; later stay-nights (long stays of late
                                   # arrivals) are censored and excluded from model inputs
SEASON_BY_MONTH = {                # our documented convention (catalog assumption A0)
    12: "Low", 1: "Low", 2: "Low",
    3: "Shoulder", 4: "Shoulder", 5: "Shoulder",
    6: "High", 7: "High", 8: "High",
    9: "Shoulder", 10: "Shoulder", 11: "Low",
}

# surrogate-key column names of the conformed dimensions
KEY_NAMES = {"dim_room_type": "RoomTypeKey",
             "dim_market_segment": "MarketSegmentKey",
             "dim_customer_type": "CustomerTypeKey",
             "dim_deposit_type": "DepositTypeKey",
             "dim_channel": "DistributionChannelKey"}


# ----------------------------------------------------------------------
# Extract
# ----------------------------------------------------------------------
def extract(data_dir: Path) -> pd.DataFrame:
    frames = []
    for code, fname in (("H1", "H1.csv"), ("H2", "H2.csv")):
        df = pd.read_csv(data_dir / fname, dtype={"Agent": str, "Company": str})
        df["HotelCode"] = code
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"[extract] {len(raw):,} bookings read (H1 40,060 + H2 79,330 expected)")
    return raw


# ----------------------------------------------------------------------
# Cleaning (each rule logs the number of rows it removes)
# ----------------------------------------------------------------------
def clean(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    n0 = len(df)

    # fixed-width string fields in the source (room types, segments, ...) carry
    # trailing spaces -- strip all object columns first
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    # 'NA' strings in numeric columns -> missing
    df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0).astype(int)

    # Rule 1: negative ADR (paper: minimum ADR is -6.38 at H1)
    m = df["ADR"] < 0
    log["negative_adr"] = int(m.sum())
    df = df[~m]

    # Rule 2: extreme ADR outliers (paper: maximum ADR is 5,400 at H2).
    # Transparent cap at 1,000 EUR/night -- ~10x the typical city-hotel ADR.
    m = df["ADR"] > 1000
    log["extreme_adr"] = int(m.sum())
    df = df[~m]

    # Rule 3: impossible party sizes (paper: maximum 55 adults at H1) and
    # bookings with zero guests of any kind
    m = (df["Adults"] > 20) | ((df["Adults"] + df["Children"] + df["Babies"]) == 0)
    log["impossible_party"] = int(m.sum())
    df = df[~m]

    # Rule 4: stays with zero total nights (day-use / data errors; they generate
    # no room-nights, so they are excluded from the facts and logged)
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    m = nights == 0
    log["zero_nights"] = int(m.sum())
    df = df[~m]

    log["rows_in"] = n0
    log["rows_out"] = len(df)
    print(f"[clean] {n0:,} -> {len(df):,} bookings "
          f"(removed: neg-ADR {log['negative_adr']}, extreme-ADR {log['extreme_adr']}, "
          f"party {log['impossible_party']}, zero-nights {log['zero_nights']})")
    return df


# ----------------------------------------------------------------------
# Transform
# ----------------------------------------------------------------------
def transform_dates(df: pd.DataFrame) -> pd.DataFrame:
    """(1) date assembly."""
    month_num = {m: i + 1 for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"])}
    df["ArrivalMonthNum"] = df["ArrivalDateMonth"].map(month_num)
    df["ArrivalDate"] = pd.to_datetime(
        dict(year=df["ArrivalDateYear"], month=df["ArrivalMonthNum"],
             day=df["ArrivalDateDayOfMonth"]), errors="coerce")
    bad = df["ArrivalDate"].isna().sum()
    if bad:
        raise ValueError(f"{bad} bookings have an unparseable arrival date")
    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"], errors="coerce")
    df["BookingDate"] = df["ArrivalDate"] - pd.to_timedelta(df["LeadTime"], unit="D")
    df["DepartureDate"] = df["ArrivalDate"] + pd.to_timedelta(
        df["StaysInWeekendNights"] + df["StaysInWeekNights"], unit="D")
    # booking identity is assigned here, BEFORE stay explosion, so every
    # booking-night of the same booking shares one BookingId
    df["BookingId"] = df["HotelCode"] + "_" + df.index.astype(str)
    return df


def explode_stays(df: pd.DataFrame) -> pd.DataFrame:
    """(2) stay explosion: one row per booking per night of the stay."""
    nights = (df["StaysInWeekendNights"] + df["StaysInWeekNights"]).astype(int)
    parts = []
    for k in range(1, int(nights.max()) + 1):
        sel = df[nights >= k]
        p = sel[[c for c in df.columns]].copy()
        p["StayNight"] = k
        p["StayDate"] = p["ArrivalDate"] + pd.to_timedelta(k - 1, unit="D")
        parts.append(p)
    fn = pd.concat(parts, ignore_index=True)
    print(f"[explode] {len(df):,} bookings -> {len(fn):,} booking-nights")
    return fn


def build_dims(con: duckdb.DuckDBPyConnection, fn: pd.DataFrame, hotels: pd.DataFrame):
    """(3) surrogate keys for the seven conformed dimensions."""
    # dim_date: span of all stay nights (dense)
    d0, d1 = fn["StayDate"].min().normalize(), fn["StayDate"].max().normalize()
    dates = pd.DataFrame({"StayDate": pd.date_range(d0, d1, freq="D")})
    dates["DateKey"] = dates["StayDate"].dt.strftime("%Y%m%d").astype(int)
    dates["Year"] = dates["StayDate"].dt.year
    dates["Month"] = dates["StayDate"].dt.month
    dates["MonthName"] = dates["StayDate"].dt.strftime("%B")
    dates["Day"] = dates["StayDate"].dt.day
    dates["DayOfWeek"] = dates["StayDate"].dt.day_of_week       # 0 = Monday
    dates["DayName"] = dates["StayDate"].dt.strftime("%A")
    dates["WeekOfYear"] = dates["StayDate"].dt.isocalendar().week.astype(int)
    dates["IsWeekend"] = (dates["DayOfWeek"] >= 5).astype(int)
    dates["Season"] = dates["Month"].map(SEASON_BY_MONTH)
    dates = dates.rename(columns={"StayDate": "Date"})
    con.register("dim_date", dates)

    con.register("dim_hotel", hotels)
    for name, col in (("dim_room_type", "ReservedRoomType"),
                      ("dim_market_segment", "MarketSegment"),
                      ("dim_customer_type", "CustomerType"),
                      ("dim_deposit_type", "DepositType"),
                      ("dim_channel", "DistributionChannel")):
        vals = sorted(fn[col].dropna().unique())
        con.register(name, pd.DataFrame({col: vals,
                                         KEY_NAMES[name]: range(1, len(vals) + 1)}))
    return dates


def estimate_capacity(fn: pd.DataFrame) -> pd.DataFrame:
    """(5) capacity estimation: max rooms sold on any single night after warm-up.

    The dataset has no total-rooms attribute, so capacity is estimated per hotel
    as the maximum number of rooms sold on any single night, counted after a
    warm-up window (documented limitation: occupancy may be overstated; the
    data never observes genuine overbooking).
    """
    kept = fn[fn["IsCanceled"] == 0]
    per_night = kept.groupby(["HotelCode", "StayDate"]).size().rename("RoomsSold").reset_index()
    after = per_night[per_night["StayDate"] > WARMUP_END]
    cap = after.groupby("HotelCode")["RoomsSold"].max()
    hotels = pd.DataFrame({
        "HotelKey": [1, 2],
        "HotelCode": ["H1", "H2"],
        "HotelName": ["Resort Hotel (Algarve)", "City Hotel (Lisbon)"],
        "City": ["Algarve", "Lisbon"],
        "CapacityEstimate": [int(cap.get("H1", 0)), int(cap.get("H2", 0))],
        "WarmupEnd": [WARMUP_END] * 2,
    })
    print(f"[capacity] H1 = {cap.get('H1', 0)} rooms, H2 = {cap.get('H2', 0)} rooms "
          f"(max nightly rooms sold after {WARMUP_END})")
    return hotels


def build_facts(con: duckdb.DuckDBPyConnection, fn: pd.DataFrame, cap: pd.DataFrame):
    """(2/3/4) booking-night fact + dense daily-occupancy snapshot."""
    # (4) realised-revenue split
    fn = fn.copy()
    fn["RevenueRealised"] = np.where(fn["IsCanceled"] == 0, fn["ADR"], 0.0)
    # booking-window attributes carried on every booking-night row:
    # the arrival date and the cancellation date (when cancelled) -- the basis
    # for M4's cutoff-based show-up estimation
    fn["ArrivalDateKey"] = fn["ArrivalDate"].dt.strftime("%Y%m%d").astype(int)
    fn["CancelDateKey"] = fn["ReservationStatusDate"].dt.strftime("%Y%m%d").astype("Int64")
    fn.loc[fn["IsCanceled"] == 0, "CancelDateKey"] = pd.NA

    dims = {"ReservedRoomType": "dim_room_type", "MarketSegment": "dim_market_segment",
            "CustomerType": "dim_customer_type", "DepositType": "dim_deposit_type",
            "DistributionChannel": "dim_channel"}
    for col, dim in dims.items():
        keys = con.execute(f"SELECT {col} AS v, {KEY_NAMES[dim]} AS k FROM {dim}").fetchall()
        fn[col + "Key"] = fn[col].map(dict(keys))
    hotel_keys = dict(zip(cap["HotelCode"], cap["HotelKey"]))
    fn["HotelKey"] = fn["HotelCode"].map(hotel_keys)
    fn["DateKey"] = fn["StayDate"].dt.strftime("%Y%m%d").astype(int)

    fact_room_night = fn[[
        "HotelKey", "DateKey", "ReservedRoomTypeKey", "MarketSegmentKey",
        "CustomerTypeKey", "DepositTypeKey", "DistributionChannelKey",
        "IsCanceled", "ADR", "RevenueRealised", "LeadTime",
        "Adults", "Children", "Babies", "StayNight", "BookingId",
        "ArrivalDateKey", "CancelDateKey",
    ]]
    con.register("fact_room_night", fact_room_night)

    # dense snapshot: every (hotel, date, room type observed at that hotel), zero-sale nights included
    kept = fn[fn["IsCanceled"] == 0]
    sold = kept.groupby(["HotelCode", "StayDate", "ReservedRoomType"]).size().rename("RoomsSold").reset_index()
    types = fn.groupby("HotelCode")["ReservedRoomType"].unique().to_dict()
    grid = []
    for code, tlist in types.items():
        dts = pd.date_range(fn["StayDate"].min().normalize(),
                            fn["StayDate"].max().normalize(), freq="D")
        for d in dts:
            for t in tlist:
                grid.append((code, d, t))
    grid = pd.DataFrame(grid, columns=["HotelCode", "StayDate", "ReservedRoomType"])
    snap = grid.merge(sold, how="left", on=["HotelCode", "StayDate", "ReservedRoomType"])
    snap["RoomsSold"] = snap["RoomsSold"].fillna(0).astype(int)
    snap["HotelKey"] = snap["HotelCode"].map(hotel_keys)
    snap["DateKey"] = snap["StayDate"].dt.strftime("%Y%m%d").astype(int)
    room_keys = dict(con.execute("SELECT ReservedRoomType AS v, RoomTypeKey AS k FROM dim_room_type").fetchall())
    snap["RoomTypeKey"] = snap["ReservedRoomType"].map(room_keys)
    capmap = dict(zip(cap["HotelCode"], cap["CapacityEstimate"]))
    snap["Capacity"] = snap["HotelCode"].map(capmap)      # hotel-level estimate (documented)
    con.register("fact_daily_occupancy", snap)
    print(f"[facts] fact_room_night: {len(fact_room_night):,} rows; "
          f"fact_daily_occupancy: {len(snap):,} rows (dense)")


# ----------------------------------------------------------------------
# Load + quality gates
# ----------------------------------------------------------------------
DDL = """
CREATE TABLE dim_date AS SELECT * FROM dim_date_df;
CREATE TABLE dim_hotel AS SELECT * FROM dim_hotel_df;
CREATE TABLE dim_room_type AS SELECT * FROM dim_room_type_df;
CREATE TABLE dim_market_segment AS SELECT * FROM dim_market_segment_df;
CREATE TABLE dim_customer_type AS SELECT * FROM dim_customer_type_df;
CREATE TABLE dim_deposit_type AS SELECT * FROM dim_deposit_type_df;
CREATE TABLE dim_channel AS SELECT * FROM dim_channel_df;
CREATE TABLE fact_room_night AS SELECT * FROM fact_room_night_df;
CREATE TABLE fact_daily_occupancy AS SELECT * FROM fact_daily_occupancy_df;
"""


def quality_gates(con: duckdb.DuckDBPyConnection, raw: pd.DataFrame, clean_df: pd.DataFrame) -> bool:
    ok = True

    def gate(n: int, name: str, expect_zero: bool = True):
        nonlocal ok
        passed = (n == 0) if expect_zero else (n > 0)
        print(f"  gate {name}: {'PASS' if passed else 'FAIL'} ({n})")
        ok = ok and passed

    print("[gates] running 6 quality assertions")
    # 1. referential integrity: no orphan foreign keys
    n = con.execute("""
        SELECT COUNT(*) FROM fact_room_night f
        LEFT JOIN dim_date d ON f.DateKey = d.DateKey
        LEFT JOIN dim_hotel h ON f.HotelKey = h.HotelKey
        WHERE d.DateKey IS NULL OR h.HotelKey IS NULL""").fetchone()[0]
    gate(n, "1 referential integrity")

    # 2. row-count reconciliation: distinct booking IDs in facts == clean bookings
    n_fact = con.execute("SELECT COUNT(DISTINCT BookingId) FROM fact_room_night").fetchone()[0]
    gate(abs(n_fact - len(clean_df)), f"2 row-count reconciliation (facts {n_fact:,} vs clean {len(clean_df):,})")

    # 3. cancellation consistency: IsCanceled = ReservationStatus in {Canceled, No-Show}
    mismatch = int((clean_df["IsCanceled"] !=
                    clean_df["ReservationStatus"].isin(["Canceled", "No-Show"]).astype(int)).sum())
    gate(mismatch, "3 cancellation consistency (IsCanceled = Canceled + No-Show)")

    # 4. revenue reconciliation: booked - cancelled = realised, per hotel
    rev = clean_df.groupby("HotelCode").apply(
        lambda g: g["ADR"].sum() - g.loc[g["IsCanceled"] == 1, "ADR"].sum()
        - g.loc[g["IsCanceled"] == 0, "ADR"].sum(), include_groups=False)
    gate(int(rev.abs().sum()), "4 revenue reconciliation (booked - cancelled = realised)")

    # 5. capacity assertion: rooms sold <= estimated capacity on every night
    over = con.execute("""
        SELECT COUNT(*) FROM (
          SELECT o.HotelKey, o.DateKey, SUM(o.RoomsSold) AS s, ANY_VALUE(o.Capacity) AS c
          FROM fact_daily_occupancy o GROUP BY 1, 2 HAVING s > c)""").fetchone()[0]
    gate(over, "5 capacity assertion (rooms sold <= capacity)")

    # 6. density check: snapshot has a row for every (hotel, date, type)
    n_rows = con.execute("SELECT COUNT(*) FROM fact_daily_occupancy").fetchone()[0]
    n_grid = con.execute("""
        SELECT COUNT(*) FROM fact_daily_occupancy""").fetchone()[0]
    gate(0 if n_rows == n_grid else 1, f"6 density check ({n_rows:,} dense rows)")
    return ok


def run(data_dir: Path, db_path: Path) -> bool:
    log: dict = {}
    raw = extract(data_dir)
    clean_df = clean(raw.copy(), log)
    clean_df = transform_dates(clean_df)

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone = 'UTC'")

    fn = explode_stays(clean_df)
    hotels = estimate_capacity(fn)
    build_dims(con, fn, hotels)
    build_facts(con, fn, hotels)

    # materialise registered frames as tables (ETL is replace-load)
    con.execute("CREATE OR REPLACE TABLE dim_date AS SELECT * FROM dim_date")
    con.execute("CREATE OR REPLACE TABLE dim_hotel AS SELECT * FROM dim_hotel")
    con.execute("CREATE OR REPLACE TABLE dim_room_type AS SELECT * FROM dim_room_type")
    con.execute("CREATE OR REPLACE TABLE dim_market_segment AS SELECT * FROM dim_market_segment")
    con.execute("CREATE OR REPLACE TABLE dim_customer_type AS SELECT * FROM dim_customer_type")
    con.execute("CREATE OR REPLACE TABLE dim_deposit_type AS SELECT * FROM dim_deposit_type")
    con.execute("CREATE OR REPLACE TABLE dim_channel AS SELECT * FROM dim_channel")
    con.execute("CREATE OR REPLACE TABLE fact_room_night AS SELECT * FROM fact_room_night")
    con.execute("CREATE OR REPLACE TABLE fact_daily_occupancy AS SELECT * FROM fact_daily_occupancy")

    ok = quality_gates(con, raw, clean_df)

    # cleaning log written alongside the warehouse
    log_path = db_path.parent / "cleaning_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[load] warehouse at {db_path}; cleaning log at {log_path}")

    # headline KPIs for the ETL report
    kpis = con.execute("""
        SELECT h.HotelCode,
               COUNT(DISTINCT f.BookingId) AS bookings,
               ROUND(SUM(f.RevenueRealised), 0) AS realised_revenue,
               ROUND(AVG(CASE WHEN f.IsCanceled = 0 THEN f.ADR END), 2) AS adr_realised
        FROM fact_room_night f JOIN dim_hotel h USING (HotelKey) GROUP BY 1""").fetchall()
    for row in kpis:
        print(f"  {row[0]}: {row[1]:,} bookings, realised revenue {row[2]:,.0f}, ADR {row[3]}")
    con.close()
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--db", default="warehouse/hotel_rm.duckdb")
    args = ap.parse_args()
    success = run(Path(args.data_dir), Path(args.db))
    sys.exit(0 if success else 1)
