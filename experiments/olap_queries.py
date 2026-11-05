"""The 8 OLAP queries demonstrating the warehouse (report Section 4.1)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
con = duckdb.connect(str(ROOT / "warehouse" / "hotel_rm.duckdb"), read_only=True)

print("Q1 ROLL-UP: realised revenue by hotel -> year -> month (H2 head)")
print(con.execute("""
    SELECT h.HotelCode, d.Year, d.Month, SUM(f.RevenueRealised) AS rev
    FROM fact_room_night f
    JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
    GROUP BY GROUPING SETS ((h.HotelCode), (h.HotelCode, d.Year), (h.HotelCode, d.Year, d.Month))
    ORDER BY 1, 2 NULLS LAST, 3 NULLS LAST LIMIT 6""").fetchall())

print("Q2 DRILL-DOWN: H2 monthly revenue by market segment (2017-08)")
print(con.execute("""
    SELECT s.MarketSegment, SUM(f.RevenueRealised) AS rev
    FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
    JOIN dim_market_segment s USING (MarketSegmentKey) JOIN dim_date d USING (DateKey)
    WHERE h.HotelCode = 'H2' AND d.Year = 2017 AND d.Month = 8
    GROUP BY 1 ORDER BY 2 DESC LIMIT 4""").fetchall())

print("Q3 SLICE: the Groups segment, both hotels, whole window")
print(con.execute("""
    SELECT h.HotelCode, COUNT(DISTINCT f.BookingId) AS group_bookings,
           SUM(f.RevenueRealised) AS rev, AVG(f.ADR) AS avg_adr
    FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
    JOIN dim_market_segment s USING (MarketSegmentKey)
    WHERE s.MarketSegment = 'Groups'
    GROUP BY 1""").fetchall())

print("Q4 DICE: H2 x High season x 2017")
print(con.execute("""
    SELECT COUNT(DISTINCT f.BookingId) AS bookings, SUM(f.RevenueRealised) AS rev
    FROM fact_room_night f JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
    WHERE h.HotelCode = 'H2' AND d.Season = 'High' AND d.Year = 2017""").fetchall())

print("Q5 SHOW-UP RATE: by hotel and season at cut-off h=7 (M4's basis)")
print(con.execute("""
    WITH a AS (SELECT DISTINCT BookingId, ArrivalDateKey, CancelDateKey, IsCanceled, LeadTime,
                      CASE WHEN BookingId LIKE 'H1%' THEN 'H1' ELSE 'H2' END AS hc
               FROM fact_room_night)
    SELECT a.hc, d.Season,
           SUM(CASE WHEN a.IsCanceled = 0 THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS showup_rate
    FROM a JOIN dim_date d ON d.DateKey = a.ArrivalDateKey
    WHERE a.ArrivalDateKey - a.LeadTime <= a.ArrivalDateKey - 7
      AND (a.IsCanceled = 0 OR a.CancelDateKey > a.ArrivalDateKey - 7)
    GROUP BY 1, 2 ORDER BY 1, 2""").fetchall())

print("Q6 GROUP SHARE: group vs transient revenue by hotel and season")
print(con.execute("""
    SELECT h.HotelCode, d.Season,
           SUM(CASE WHEN s.MarketSegment = 'Groups' THEN f.RevenueRealised ELSE 0 END)
             / SUM(f.RevenueRealised) AS group_share
    FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
    JOIN dim_date d USING (DateKey) JOIN dim_market_segment s USING (MarketSegmentKey)
    WHERE f.IsCanceled = 0
    GROUP BY 1, 2 ORDER BY 1, 2""").fetchall())

print("Q7 DAY-OF-WEEK: average rooms sold (H1)")
print(con.execute("""
    SELECT d.DayName, AVG(o.rooms) AS avg_rooms
    FROM (SELECT HotelKey, DateKey, SUM(RoomsSold) AS rooms
          FROM fact_daily_occupancy GROUP BY 1, 2) o
    JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
    WHERE h.HotelCode = 'H1' AND d.DateKey <= 20170831
    GROUP BY 1, d.DayOfWeek ORDER BY d.DayOfWeek""").fetchall())

print("Q8 LEAD TIME: median by market segment (H1)")
print(con.execute("""
    SELECT s.MarketSegment, median(f.LeadTime) AS med_lead
    FROM (SELECT BookingId, ANY_VALUE(LeadTime) AS LeadTime,
                 ANY_VALUE(MarketSegmentKey) AS MarketSegmentKey, ANY_VALUE(HotelKey) AS HotelKey
          FROM fact_room_night GROUP BY 1) f
    JOIN dim_hotel h USING (HotelKey) JOIN dim_market_segment s USING (MarketSegmentKey)
    WHERE h.HotelCode = 'H1'
    GROUP BY 1 ORDER BY 2 DESC LIMIT 5""").fetchall())
