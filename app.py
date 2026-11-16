"""
Hotel RM DSS -- Streamlit application (report Section 4.3).

Four screens mapped to Simon's phases:
  1. Dashboard            (intelligence) -- KPIs and charts from the warehouse
  1b. OLAP                (intelligence) -- roll-up, drill-down, slice, dice on the star schema
  2. Forecasting          (design)      -- M1 with method comparison + backtest
  3. Optimisation & What-if (design/choice) -- M2-M5 with parameter sliders
  4. Scenario comparison  (review)      -- the fact_scenario_run log

Design principle: the interface owns NO model logic -- every model number
comes from run_model(); the dashboard and the OLAP screen use the same
analytical SQL the experiment scripts use.
"""

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mbms  # noqa: E402
import models.m1_forecast as m1  # noqa: E402
import models.m4_overbooking as m4  # noqa: E402

st.set_page_config(page_title="Hotel RM DSS", page_icon="🏨", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2.0rem; padding-bottom: 3rem; }
  [data-testid="stMetric"] {
      background: linear-gradient(180deg, #F8FBFE 0%, #E9F1F9 100%);
      border: 1px solid #D8E3EE;
      border-left: 4px solid #1F4E79;
      border-radius: 10px;
      padding: 0.85rem 1rem;
      box-shadow: 0 1px 3px rgba(16, 44, 68, 0.07);
  }
  [data-testid="stMetricLabel"] p {
      font-size: 0.78rem; color: #48627A;
      text-transform: uppercase; letter-spacing: 0.05em;
  }
  [data-testid="stMetricValue"] {
      font-size: 1.65rem; font-weight: 600; letter-spacing: -0.02em;
  }
  div[class*="st-key-kpi_card_"] button {
      min-height: 74px;
      justify-content: flex-start;
      padding: 0.3rem 0.95rem;
      border-left: 4px solid #1F4E79;
      border-radius: 10px;
      background: linear-gradient(180deg, #F8FBFE 0%, #E9F1F9 100%);
  }
  div[class*="st-key-kpi_card_"] button p {
      font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em; color: #14293D;
  }
  div[class*="st-key-kpi_card_"] button[kind="primary"] {
      background: #1F4E79; border-color: #1F4E79;
  }
  div[class*="st-key-kpi_card_"] button[kind="primary"] p { color: #FFFFFF; }
  .kpi-name {
      font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; color: #1B2733; padding: 4px 0 10px 4px;
  }
  h1 { letter-spacing: -0.015em; }
  h2, h3 { letter-spacing: -0.01em; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 5px 16px; }
  div[data-testid="stExpander"] summary p { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


DB_PATH = os.environ.get("HOTEL_RM_DB", str(ROOT / "warehouse" / "hotel_rm.duckdb"))


@st.cache_resource
def get_con():
    con = duckdb.connect(DB_PATH)
    mbms.init_mbms(con)
    return con


HOTELS = ("H1", "H2")
con = get_con()

st.title("🏨 Revenue Management Workspace")
st.caption("Two properties in Portugal — forecast demand, price rooms, and allocate "
           "capacity from one booking record.")

# -- screen functions below; navigation is created at the end of this file --


# ======================================================================
# Screen 1b -- OLAP (classical operations on the star schema)
# ======================================================================
def screen_analytics():
    st.header(f"Analytics — {hotel_name}")
    st.caption("Roll-up, drill-down, slice, and dice executed directly against the star "
               "schema. This is the data path of the architecture: analytical SQL, no "
               "model logic involved.")

    tab_roll, tab_drill, tab_slice, tab_dice = st.tabs(
        ["Roll-up", "Drill-down", "Slice", "Dice"])

    # -- Roll-up: aggregate the fact along the time hierarchy ------------
    with tab_roll:
        st.subheader("Roll-up -- time hierarchy")
        st.caption("The booking-night fact aggregated from months to quarters to years.")
        c1, c2 = st.columns([3, 2])
        level = c1.radio("Aggregate by", ("Month", "Quarter", "Year"), horizontal=True)
        measure = c2.selectbox("Measure",
                               ("Realised revenue (EUR)", "Room-nights sold", "ADR (EUR)"))
        period_expr = {
            "Month": "strftime(d.Date, '%Y-%m')",
            "Quarter": "d.Year::VARCHAR || '-Q' || CAST((d.Month - 1) / 3 + 1 AS INTEGER)::VARCHAR",
            "Year": "d.Year::VARCHAR",
        }[level]
        measure_expr = {
            "Realised revenue (EUR)": "ROUND(SUM(f.RevenueRealised), 0)",
            "Room-nights sold": "SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END)",
            "ADR (EUR)": "ROUND(SUM(f.RevenueRealised) / NULLIF(SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END), 0), 2)",
        }[measure]
        df = con.execute(f"""
            SELECT {period_expr} AS period, {measure_expr} AS value
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            JOIN dim_date d USING (DateKey)
            WHERE h.HotelCode = ?
            GROUP BY 1 ORDER BY 1""", [hotel]).fetch_df()
        fig = px.bar(df, x="period", y="value", height=340)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_title=measure,
                          xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Result table"):
            st.dataframe(df, width="stretch")

    # -- Drill-down: monthly total decomposed into segments --------------
    with tab_drill:
        st.subheader("Drill-down -- month to market segment")
        st.caption("A monthly total decomposed into the market segments that produced it.")
        months = con.execute("""
            SELECT DISTINCT strftime(d.Date, '%Y-%m') AS m
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            JOIN dim_date d USING (DateKey)
            WHERE h.HotelCode = ? ORDER BY 1 DESC""", [hotel]).fetch_df()["m"].tolist()
        msel = st.selectbox("Month", months, index=0)
        df = con.execute("""
            SELECT s.MarketSegment AS segment,
                   SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END) AS room_nights,
                   ROUND(SUM(f.RevenueRealised), 0) AS revenue,
                   ROUND(SUM(f.RevenueRealised) / NULLIF(SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END), 0), 2) AS adr
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            JOIN dim_date d USING (DateKey)
            JOIN dim_market_segment s USING (MarketSegmentKey)
            WHERE h.HotelCode = ? AND strftime(d.Date, '%Y-%m') = ?
            GROUP BY 1 ORDER BY revenue DESC""", [hotel, msel]).fetch_df()
        fig = px.bar(df, x="segment", y="revenue", height=320,
                     hover_data=["room_nights", "adr"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="revenue (EUR)", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Result table"):
            st.dataframe(df, width="stretch")

    # -- Slice: one segment across the calendar --------------------------
    with tab_slice:
        st.subheader("Slice -- one market segment across the calendar")
        st.caption("The fact restricted to a single segment member, against the whole property.")
        segs = con.execute(
            "SELECT MarketSegment FROM dim_market_segment ORDER BY 1").fetch_df()[
            "MarketSegment"].tolist()
        ssel = st.selectbox("Market segment", segs,
                            index=segs.index("Direct") if "Direct" in segs else 0)
        df = con.execute("""
            SELECT strftime(d.Date, '%Y-%m') AS month,
                   SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END) AS room_nights,
                   ROUND(SUM(f.RevenueRealised), 0) AS revenue
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            JOIN dim_date d USING (DateKey)
            JOIN dim_market_segment s USING (MarketSegmentKey)
            WHERE h.HotelCode = ? AND s.MarketSegment = ?
            GROUP BY 1 ORDER BY 1""", [hotel, ssel]).fetch_df()
        tot = con.execute("""
            SELECT SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END) AS nights
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            WHERE h.HotelCode = ?""", [hotel]).fetch_df().iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Room-nights (slice)", f"{int(df.room_nights.sum()):,}")
        k2.metric("Revenue (slice)", f"€{df.revenue.sum():,.0f}")
        k3.metric("Share of property room-nights",
                  f"{100 * df.room_nights.sum() / tot.nights:.1f}%")
        fig = px.bar(df, x="month", y="room_nights", height=300)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="room-nights", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    # -- Dice: sub-cube across three dimensions --------------------------
    with tab_dice:
        st.subheader("Dice -- hotel × season × year")
        st.caption("A sub-cube selected on three dimensions at once.")
        c1, c2, c3 = st.columns(3)
        hot_sel = c1.multiselect("Hotels", HOTELS, default=list(HOTELS))
        seasons = con.execute(
            "SELECT DISTINCT Season FROM dim_date ORDER BY 1").fetch_df()["Season"].tolist()
        sea_sel = c2.multiselect("Seasons", seasons, default=seasons)
        years = [int(y) for y in con.execute(
            "SELECT DISTINCT Year FROM dim_date ORDER BY 1").fetch_df()["Year"]]
        yr_sel = c3.multiselect("Years", years, default=years)
        if hot_sel and sea_sel and yr_sel:
            df = con.execute(f"""
                SELECT h.HotelCode AS hotel, d.Season AS season, d.Year AS year,
                       SUM(CASE WHEN f.IsCanceled = 0 THEN 1 ELSE 0 END) AS room_nights,
                       ROUND(SUM(f.RevenueRealised), 0) AS revenue
                FROM fact_room_night f
                JOIN dim_hotel h USING (HotelKey)
                JOIN dim_date d USING (DateKey)
                WHERE h.HotelCode IN ({','.join(['?'] * len(hot_sel))})
                  AND d.Season IN ({','.join(['?'] * len(sea_sel))})
                  AND d.Year IN ({','.join(['?'] * len(yr_sel))})
                GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",
                hot_sel + sea_sel + yr_sel).fetch_df()
            fig = px.bar(df, x="season", y="room_nights", color="hotel",
                         facet_col="year", barmode="group", height=360)
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="room-nights")
            st.plotly_chart(fig, width="stretch")
            pivot = df.pivot_table(index=["hotel", "season"], columns="year",
                                   values="room_nights", fill_value=0, aggfunc="sum")
            with st.expander("Result cube"):
                st.dataframe(pivot, width="stretch")
        else:
            st.info("Select at least one value on each dimension to build the sub-cube.")


# ======================================================================
# Screen 1 -- Dashboard (intelligence)
# ======================================================================
def screen_dashboard():
    st.header(f"Dashboard — {hotel_name}")
    st.caption("What has happened: revenue, rate, occupancy, and cancellations, straight from the warehouse.")
    kpis = con.execute("""
        SELECT SUM(f.RevenueRealised)::DOUBLE,
               SUM(CASE WHEN f.IsCanceled = 0 THEN f.ADR * 1 ELSE 0 END)
                 / NULLIF(SUM(f.IsCanceled = 0), 0),
               SUM(f.IsCanceled = 0)::DOUBLE
        FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
        WHERE h.HotelCode = ?""", [hotel]).fetchone()
    # cancellation rate at BOOKING level (not booking-night weighted)
    cancel_rate = con.execute("""
        SELECT AVG(c) FROM (
            SELECT BookingId, MAX(f.IsCanceled)::DOUBLE AS c
            FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
            WHERE h.HotelCode = ? GROUP BY 1)""", [hotel]).fetchone()[0]
    cap = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                      [hotel]).fetchone()[0]
    nights = con.execute("""
        SELECT COUNT(DISTINCT o.DateKey) FROM fact_daily_occupancy o
        JOIN dim_hotel h USING (HotelKey) WHERE h.HotelCode = ?""", [hotel]).fetchone()[0]
    n_bookings = con.execute("""
        SELECT COUNT(DISTINCT BookingId) FROM fact_room_night f
        JOIN dim_hotel h USING (HotelKey) WHERE h.HotelCode = ?""", [hotel]).fetchone()[0]

    occ_pct = 100 * kpis[2] / (cap * nights)
    kpi_items = [
        ("revenue", f"€{kpis[0]:,.0f}", "Realised revenue"),
        ("adr", f"€{kpis[1]:.2f}", "ADR (kept)"),
        ("occ", f"{occ_pct:.1f}%", "Occupancy (est.)"),
        ("cancel", f"{100 * cancel_rate:.1f}%", "Cancellation rate"),
    ]
    if "kpi_sel" not in st.session_state:
        st.session_state["kpi_sel"] = None

    def _kpi_card(key, value, name):
        with st.container(key=f"kpi_card_{key}"):
            if st.button(value, key=f"kpi_btn_{key}", width="stretch",
                         type="primary" if st.session_state.get("kpi_sel") == key
                         else "secondary"):
                st.session_state["kpi_sel"] = (
                    None if st.session_state.get("kpi_sel") == key else key)
                st.rerun()
            st.markdown(f'<div class="kpi-name">{name}</div>', unsafe_allow_html=True)

    guides = {
        "revenue": dict(
            title="Realised revenue",
            definition="What the hotel actually earned from room sales — only stays that "
                       "really happened. A cancelled booking earns nothing.",
            formulas=[r"R_{\text{real}} \;=\; \sum_{\text{kept}} \left(\text{rate} \times \text{nights}\right)"],
            legend="kept = bookings whose guests actually stayed · rate × nights = what "
                   "that booking paid in total",
            for_property=f"At H1: {int(kpis[2]):,} room-nights stayed, earning "
                         f"€{kpis[0]:,.0f}.",
            caveat="Room revenue only — restaurant, spa and other departments are "
                   "outside this dataset."),
        "adr": dict(
            title="ADR (kept)",
            definition="The average nightly rate actually achieved across the rooms that "
                       "were sold.",
            formulas=[r"\text{ADR} \;=\; \frac{\text{room revenue}}{\text{room-nights sold}}"],
            legend="A five-night booking weighs five times a one-night booking — a "
                   "revenue-weighted average, not the published rate.",
            for_property=f"At H1: €{kpis[0]:,.0f} ÷ {int(kpis[2]):,} room-nights = "
                         f"€{kpis[1]:.2f}.",
            caveat="Discounts and the guest mix move it: the same published rate can "
                   "yield different ADRs."),
        "occ": dict(
            title="Occupancy (est.)",
            definition="How full the hotel was, on average, across the period.",
            formulas=[r"\text{Occupancy} \;=\; \frac{\text{rooms sold}}{\text{rooms available}}"],
            legend="Rooms available counts every room on every night of the period — "
                   "sold or not.",
            for_property=f"At H1: {int(kpis[2]):,} rooms sold ÷ ({cap} × {nights}) = "
                         f"{occ_pct:.1f}%.",
            caveat="The room count is estimated from the busiest nights (the dataset "
                   "publishes no inventory), so treat the figure as a close estimate."),
        "cancel": dict(
            title="Cancellation rate",
            definition="The share of bookings that never became stays — cancelled or "
                       "no-show.",
            formulas=[r"\frac{\text{cancelled bookings}}{\text{all bookings}}"],
            legend="Counted per booking: one cancelled reservation is one cancellation, "
                   "however many nights it covered.",
            for_property=f"At H1: {100 * cancel_rate:.1f}% of {n_bookings:,} bookings.",
            caveat="Only bookings that were made are visible here — guests who never "
                   "booked, or were turned away, do not appear."),
    }

    sel = st.session_state.get("kpi_sel")
    if sel is None:
        for col, (key, value, name) in zip(st.columns(4, gap="medium"), kpi_items):
            with col:
                _kpi_card(key, value, name)
    else:
        col_kpi, col_guide = st.columns([1.22, 1], gap="large")
        with col_kpi:
            for r in range(2):
                row = st.columns(2)
                for c in range(2):
                    key, value, name = kpi_items[2 * r + c]
                    with row[c]:
                        _kpi_card(key, value, name)
        with col_guide:
            with st.container(border=True):
                guide = guides[sel]
                st.markdown(f"#### {guide['title']}")
                st.markdown(guide["definition"])
                for line in guide["formulas"]:
                    st.latex(line)
                st.caption(guide["legend"])
                st.info(guide["for_property"])
                st.caption(f"Note — {guide['caveat']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly revenue and ADR")
        df = con.execute("""
            SELECT strftime(d.Date, '%Y-%m') AS month,
                   SUM(f.RevenueRealised) AS revenue,
                   SUM(f.RevenueRealised) / NULLIF(SUM(f.IsCanceled = 0), 0) AS adr
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
            WHERE h.HotelCode = ?
            GROUP BY 1 ORDER BY 1""", [hotel]).fetch_df()
        fig = go.Figure()
        fig.add_bar(x=df["month"], y=df["revenue"], name="revenue", yaxis="y")
        fig.add_scatter(x=df["month"], y=df["adr"], name="ADR", yaxis="y2",
                        mode="lines+markers")
        fig.update_layout(yaxis=dict(title="revenue €"),
                          yaxis2=dict(title="ADR €", overlaying="y", side="right"),
                          height=340, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Market segment mix (revenue)")
        df = con.execute("""
            SELECT s.MarketSegment AS segment, SUM(f.RevenueRealised) AS revenue
            FROM fact_room_night f
            JOIN dim_hotel h USING (HotelKey)
            JOIN dim_market_segment s USING (MarketSegmentKey)
            WHERE h.HotelCode = ? AND f.IsCanceled = 0
            GROUP BY 1 ORDER BY 2 DESC""", [hotel]).fetch_df()
        st.plotly_chart(px.pie(df, values="revenue", names="segment", height=340),
                        width="stretch")

    st.subheader("Occupancy by season (show-up basis, estimated capacity)")
    df = con.execute("""
        SELECT d.Season AS season, d.Date AS night, SUM(o.RoomsSold) AS rooms,
               ANY_VALUE(o.Capacity) AS capacity
        FROM fact_daily_occupancy o
        JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
        WHERE h.HotelCode = ? AND d.DateKey <= 20170831
        GROUP BY 1, 2 ORDER BY 2""", [hotel]).fetch_df()
    fig = px.scatter(df, x="night", y="rooms", color="season",
                     hover_data=["capacity"], height=300)
    fig.add_hline(y=df["capacity"].max(), line_dash="dot",
                  annotation_text="estimated capacity")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("What each indicator means — and how it is computed"):
        st.markdown("""
**Realised revenue** — room revenue actually earned. Cancelled and no-show stays earn
nothing, so the figure counts only the stays that really happened.

**ADR (kept)** — the average rate actually achieved: room revenue ÷ room-nights sold. It is
*not* the published rate: discounts and the guest mix move it.

**Occupancy (est.)** — rooms sold ÷ rooms available. The dataset publishes no room count, so
the inventory is estimated from the busiest nights (187 rooms at H1, 226 at H2).

**Cancellation rate** — the share of bookings that were cancelled or no-show, counted per
booking rather than per night.

**Monthly revenue and ADR** — the two KPIs together: bars are realised revenue per month,
the line is that month's ADR — you can see occupancy-driven months (bar high, rate flat) and
rate-driven months (rate high, bar lower).

**Market segment mix** — which segments produced the month's revenue; groups and OTA
channels carry different rates, so the mix changes the average.

**Occupancy by season** — nightly rooms sold against the capacity line; points stacked on the
line are sold-out nights, low scatter is the trough.
""")


# ======================================================================
# Screen 2 -- Forecasting (design)
# ======================================================================
def screen_forecast():
    st.header(f"Demand forecast — {hotel_name}")
    st.caption("What will happen: how many rooms will sell each night of the coming month, with the method chosen by backtest rather than assumption.")
    col1, col2 = st.columns([1, 3])
    with col1:
        horizon = st.slider("Horizon (nights)", 7, 56, 28,
                            help="Forecast horizon in nights — one forecast per night of the window.")
        method = st.selectbox("Method", ("auto (backtest winner)",
                                         "holt_winters", "snaive7", "yoy_snaive364"))
        if st.button("Run forecast", type="primary", icon=":material/play_arrow:"):
            try:
                m = "auto" if method.startswith("auto") else method
                res = mbms.run_model(con, "M1", {"hotel": hotel, "horizon": horizon,
                                                 "method": m},
                                     scenario_name=f"ui-M1-{hotel}")
                st.session_state["m1"] = res
            except Exception as e:
                st.error(f"M1 failed: {e}")
    res = st.session_state.get("m1")
    if res:
        st.success(f"method **{res['method']}** · total forecast rooms over "
                   f"{res['horizon']} nights: **{res['total_rooms']:,.0f}**")
        df = pd.DataFrame(res["forecast"])
        hist = con.execute("""
            SELECT d.Date AS night, SUM(o.RoomsSold) AS rooms
            FROM fact_daily_occupancy o
            JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
            WHERE h.HotelCode = ? AND d.Date > DATE '2017-06-01' AND d.DateKey <= 20170831
            GROUP BY 1 ORDER BY 1""", [hotel]).fetch_df()
        fig = go.Figure()
        fig.add_scatter(x=hist["night"], y=hist["rooms"], name="realised (last 60 nights)")
        fig.add_scatter(x=pd.to_datetime(df["date"]), y=df["rooms"], name="forecast",
                        mode="lines+markers")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Backtest (rolling monthly cut-offs, 28-day horizon)")
    if st.button("Run backtest (all methods compete)"):
        with st.spinner("running rolling backtest..."):
            bt = m1.backtest(con, hotel, horizon=28, n_cutoffs=14)
        st.session_state["bt"] = bt
    bt = st.session_state.get("bt")
    if bt:
        bt_df = pd.DataFrame(bt["metrics"])
        st.dataframe(bt_df.set_index("method"))
        bt_df["result"] = np.where(bt_df["method"] == bt["winner"], "winner", "competitor")
        fig = px.bar(bt_df, x="method", y="WAPE", color="result", height=240,
                     color_discrete_map={"winner": "#1F4E79", "competitor": "#A8BFD4"},
                     labels={"WAPE": "WAPE — lower is better", "color": ""})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None,
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
        if bt["winner"] != "holt_winters":
            st.info(f"**Honest evaluation**: the seasonal-naive baseline "
                    f"`{bt['winner']}` beats Holt-Winters on this hotel -- it is used, "
                    f"not hidden (the field's own evidence: simple methods often win).")


# ======================================================================
# Screen 3 -- Optimisation & what-if (design/choice)
# ======================================================================
def screen_rate():
    st.header(f"Rate & allocation — {hotel_name}")
    st.caption("The four decisions — accept a group request, allocate the week, set the "
               "booking limit, find the required rate — each computed by a declared model "
               "with its parameters and its honest output.")
    tab1, tab2, tab3, tab4 = st.tabs(["Group request", "Weekly allocation",
                                      "Overbooking", "Pricing"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        arrival = c1.date_input("Arrival", pd.Timestamp("2017-08-11").date())
        nights = c2.number_input("Nights", 1, 21, 3)
        rooms = c3.number_input("Rooms requested", 1, 100, 25)
        rate = st.number_input("Group rate (€/night)", 1.0, 500.0, 82.0, 5.0)
        if st.button("Evaluate request", type="primary", icon=":material/play_arrow:"):
            try:
                res = mbms.run_model(con, "M2", {
                    "hotel": hotel, "arrival_date": str(arrival),
                    "nights": int(nights), "rooms": int(rooms), "rate": float(rate)},
                    scenario_name=f"ui-M2-{hotel}")
                st.session_state["m2"] = res
            except ValueError as e:
                ph = con.execute("""
                    SELECT AVG(f.ADR) FROM fact_room_night f
                    JOIN dim_hotel h USING (HotelKey) JOIN dim_date d USING (DateKey)
                    JOIN dim_market_segment s USING (MarketSegmentKey)
                    WHERE h.HotelCode = ? AND f.IsCanceled = 0
                      AND s.MarketSegment <> 'Groups'
                      AND d.Season = (SELECT Season FROM dim_date
                                      WHERE DateKey = ?)""",
                    [hotel, int(str(arrival).replace('-', ''))]).fetchone()[0]
                st.error(f"{e}. The transient rate p_H for this context is about "
                         f"€{ph:.2f} -- lower the group rate below it "
                         f"(a typical group discount is 70% of p_H).")
        res = st.session_state.get("m2")
        if res:
            v = res["accept"]
            st.success(f"**{'ACCEPT' if v else 'REJECT'}** -- expected revenue difference "
                       f"€{res['delta_revenue']:,.0f} over {res['nights']} nights")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Protection level y*", f"{res['y_star']:.1f}",
                      help="Rooms held back for late transient demand — the Littlewood "
                           "threshold: protect while the last room's expected transient "
                           "revenue exceeds the group rate.")
            c2.metric("Unprotected capacity", f"{res['unprotected_capacity']:.1f}",
                      help="Capacity − y*: the rooms still available for group business. "
                           "A request is accepted only if it fits inside this figure.")
            c3.metric("Breakeven group rate", f"€{res['breakeven_rate']:.2f}",
                      help="The group rate at which accept and reject are worth the same — "
                           "equal to the opportunity cost of the rooms.")
            c4.metric("Transient rate p_H", f"€{res['p_H']:.2f}",
                      help="The high fare: season-mean realised ADR of kept bookings in this season.")

            # capacity ruler: protection level vs unprotected vs the request
            figR = go.Figure()
            figR.add_bar(y=["capacity"], x=[res["y_star"]], orientation="h",
                         marker_color="#1F4E79", name="protected (y*)", showlegend=False)
            figR.add_bar(y=["capacity"], x=[res["unprotected_capacity"]], orientation="h",
                         marker_color="#A8BFD4", name="unprotected", showlegend=False)
            figR.add_vline(x=res["rooms"], line_dash="dash", line_color="#C62828",
                           annotation_text=f"requested {res['rooms']} rooms",
                           annotation_position="top")
            figR.update_layout(barmode="stack", height=150, showlegend=False,
                               margin=dict(l=10, r=10, t=35, b=10),
                               xaxis_title=f"rooms (capacity {res['capacity']})")
            st.plotly_chart(figR, width="stretch")

            # revenue bridge: group money minus displaced transient demand
            group_rev = res["p_L"] * res["rooms"] * res["nights"]
            displaced = group_rev - res["delta_revenue"]
            figW = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute", "relative", "total"],
                x=["Group revenue", "Displaced transient demand", "Net effect"],
                y=[group_rev, -displaced, None],
                text=[f"€{group_rev:,.0f}", f"−€{displaced:,.0f}",
                      f"€{res['delta_revenue']:,.0f}"],
                textposition="outside",
                connector={"line": {"color": "#94A3B8"}},
                increasing={"marker": {"color": "#2E7D32"}},
                decreasing={"marker": {"color": "#C62828"}},
                totals={"marker": {"color": "#1F4E79"}}))
            figW.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                               yaxis_title="EUR", showlegend=False)
            st.plotly_chart(figW, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        start = c1.date_input("Week start", pd.Timestamp("2017-08-07").date())
        end = c2.date_input("Week end", pd.Timestamp("2017-08-13").date())
        st.markdown("Competing group offers (edit freely):")
        if "offers" not in st.session_state:
            st.session_state["offers"] = pd.DataFrame([
                {"arrival": "2017-08-07", "nights": 5, "rooms": 12, "room_type": "A", "rate": 108.0},
                {"arrival": "2017-08-08", "nights": 4, "rooms": 20, "room_type": "A", "rate": 88.0},
                {"arrival": "2017-08-09", "nights": 3, "rooms": 15, "room_type": "D", "rate": 95.0},
                {"arrival": "2017-08-10", "nights": 2, "rooms": 25, "room_type": "E", "rate": 78.0},
                {"arrival": "2017-08-11", "nights": 3, "rooms": 18, "room_type": "F", "rate": 82.0},
            ])
        offers = st.data_editor(st.session_state["offers"], num_rows="dynamic")
        if st.button("Optimise allocation", type="primary", icon=":material/play_arrow:"):
            try:
                offers = offers.dropna(subset=["arrival", "nights", "rooms", "room_type", "rate"])
                payload = [{"arrival": str(o.arrival)[:10], "nights": int(o.nights),
                            "rooms": int(o.rooms), "room_type": str(o.room_type).strip(),
                            "rate": float(o.rate)} for o in offers.itertuples()]
                res = mbms.run_model(con, "M3", {
                    "hotel": hotel, "start_date": str(start), "end_date": str(end),
                    "offers": payload}, scenario_name=f"ui-M3-{hotel}")
                st.session_state["m3"] = res
            except Exception as e:
                st.error(f"M3 failed: {e}")
        res = st.session_state.get("m3")
        if res:
            st.success(f"Optimal revenue Z* = €{res['optimal_revenue']:,.0f}")
            df = pd.DataFrame(res["offers"])
            df["accepted"] = df["accepted"].map({True: "✅ ACCEPT", False: "❌ REJECT"})
            st.dataframe(df)
            # offer timeline (accepted vs rejected)
            rows = []
            for i, o in enumerate(res["offers"]):
                of = o["offer"]
                s0 = pd.to_datetime(of["arrival"])
                rows.append({"offer": f"#{i + 1} · type {of['room_type']} · {of['rooms']} rm · €{of['rate']:.0f}",
                             "start": s0, "end": s0 + pd.Timedelta(days=int(of["nights"])),
                             "status": "accepted" if o["accepted"] else "rejected"})
            gdf = pd.DataFrame(rows)
            figG = px.timeline(gdf, x_start="start", x_end="end", y="offer", color="status",
                               color_discrete_map={"accepted": "#2E7D32", "rejected": "#C7CDD4"},
                               height=280)
            figG.update_yaxes(autorange="reversed")
            figG.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None,
                               yaxis_title=None, legend_title="")
            st.plotly_chart(figG, width="stretch")

            bp = pd.DataFrame(res["bid_prices"])
            if len(bp):
                st.markdown("**Bid prices** (shadow prices of the capacity constraints):")
                st.dataframe(bp)
                hm = bp.pivot_table(index="room_type", columns="date", values="bid_price")
                figH = px.imshow(hm, aspect="auto", color_continuous_scale="Blues",
                                 height=260, labels={"color": "bid price (€)"})
                figH.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None,
                                   yaxis_title="room type")
                st.plotly_chart(figH, width="stretch")

    with tab3:
        c1, c2, c3 = st.columns(3)
        season = c1.selectbox("Season", ("High", "Shoulder", "Low"))
        cutoff = c2.slider("Cut-off h (days before arrival)", 1, 30, 7,
                           help="Booking snapshot taken h days before arrival. Smaller h sees more "
                                "information but leaves less time to act on it.")
        cw = c3.slider("Walk cost multiple (× ADR)", 0.5, 3.0, 1.5, 0.1,
                       help="Compensation plus reputation cost of relocating one guest, as a "
                            "multiple of ADR. Subjective by nature — set by management.")
        with st.popover("ⓘ How the show-up rate and the limit are derived"):
            st.markdown("At the cut-off, the on-the-books snapshot gives "
                        "**p̂ = kept bookings ÷ bookings on the books** — the share that actually "
                        "arrived. The booking limit is the smallest B at which the probability that "
                        "arrivals reach capacity clears the critical ratio:")
            st.latex(r"B^* :\; P(S \ge C) \ge \frac{r}{r + c_w},\quad S \sim \mathrm{Bin}(B, p)")
            st.markdown("Equivalently: accept one more booking while its expected revenue **r** "
                        "exceeds the expected walk cost it creates — the newsvendor marginal "
                        "condition, with r/(r + c_w) as the critical ratio.")
        if st.button("Set booking limit", type="primary", icon=":material/play_arrow:"):
            try:
                res = mbms.run_model(con, "M4", {
                    "hotel": hotel, "season": season, "cutoff": int(cutoff),
                    "walk_cost_multiple": float(cw)}, scenario_name=f"ui-M4-{hotel}")
                st.session_state["m4"] = res
            except Exception as e:
                st.error(f"M4 failed: {e}")
        res = st.session_state.get("m4")
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Booking limit B*", f"{res['B_star']}",
                      help="Maximum bookings accepted for the arrival date: capacity plus "
                           "the deliberate overbooking level.")
            c2.metric("Overbooking level", f"{res['overbooking_level']}",
                      help="B* − capacity: how many bookings are accepted beyond the "
                           "physical room count.")
            c3.metric("P(walk ≥ 1)", f"{100 * res['p_walk_at_least_one']:.1f}%",
                      help="Probability that at least one guest must be relocated, given "
                           "the show-up distribution at B*.")
            c4.metric("E[walked guests]", f"{res['expected_walks']:.2f}",
                      help="Expected number of relocations E[(S − C)+], S ~ Binomial(B*, p̂).")
            st.markdown(f"**Show-up probability at cut-off h = {res['cutoff_days']} days: "
                        f"{res['showup_probability']}** — estimated from the on-the-books snapshot "
                        f"({res['on_books']:,} bookings, {res['kept']:,} kept). "
                        f"Critical ratio {res['critical_ratio']} · expected profit "
                        f"€{res['expected_profit']:,.0f} "
                        f"(Monte Carlo, {res['mc']['reps']:,} replications: "
                        f"€{res['mc']['profit_mean']:,.0f}, "
                        f"95% CI [{res['mc']['profit_ci95'][0]:,.0f}, "
                        f"{res['mc']['profit_ci95'][1]:,.0f}]).")

            p_hat = res["showup_probability"]
            B = res["B_star"]
            C = res["capacity"]
            ks = np.arange(max(C - 40, 0), B + 35)
            figD = go.Figure()
            figD.add_bar(x=ks, y=scipy_stats.binom.pmf(ks, B, p_hat),
                         name="arrivals", marker_color="#1F4E79")
            figD.add_vline(x=C, line_dash="dash", line_color="#C62828",
                           annotation_text=f"capacity {C}", annotation_position="top")
            figD.add_vline(x=B, line_dash="dot", line_color="#2E7D32",
                           annotation_text=f"limit B* {B}", annotation_position="top")
            figD.update_layout(height=260, showlegend=False,
                               margin=dict(l=10, r=10, t=35, b=10),
                               xaxis_title="guests arriving, S ~ Binomial(B*, p̂)",
                               yaxis_title="probability")
            st.plotly_chart(figD, width="stretch")

            b_grid = np.arange(max(C - 20, 1), B + 26)
            pi_curve = [m4.expected_pi(int(b), C, p_hat, res["adr"], res["walk_cost"])
                        for b in b_grid]
            figP = go.Figure()
            figP.add_scatter(x=b_grid, y=pi_curve, name="expected profit",
                             line=dict(color="#1F4E79"))
            figP.add_scatter(x=[B], y=[res["expected_profit"]], mode="markers",
                             name="optimum", marker=dict(color="#2E7D32", size=10))
            figP.add_vline(x=C, line_dash="dash", line_color="#94A3B8",
                           annotation_text="capacity")
            figP.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="booking limit B", yaxis_title="expected profit (€)",
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(figP, width="stretch")

    with tab4:
        c1, c2 = st.columns(2)
        season = c1.selectbox("Season", ("Low", "Shoulder", "High"))
        uplift = c2.slider("RevPAR target uplift", 5, 100, 20, 5, format="%d%%")
        e = st.slider("Price elasticity e (assumed)", -0.95, -0.05, -0.8, 0.05,
                      help="The data contains no price variation, so e cannot be estimated — it "
                           "is an assumption. The problem is well posed only for −1 < e < 0; the "
                           "tornado shows how much the answer depends on it.")
        with st.popover("ⓘ How the elasticity shapes the answer"):
            st.markdown("Occupancy responds to price with a constant elasticity *e*, so revenue "
                        "per available room follows:")
            st.latex(r"\mathrm{RevPAR}(p) = \mathrm{occ}_0\, p^{\,1+e}\,\mathrm{ADR}_0^{-e}")
            st.markdown("A finite answer exists only for **−1 < e < 0**: a milder elasticity "
                        "means a small rate rise reaches the target; as e → −1 the required rate "
                        "diverges, and beyond some point no finite rate exists — which the tornado "
                        "reports instead of extrapolating.")
        if st.button("Find required rate", type="primary", icon=":material/play_arrow:"):
            try:
                res = mbms.run_model(con, "M5", {
                    "hotel": hotel, "season": season, "elasticity": float(e),
                    "target_uplift": uplift / 100.0}, scenario_name=f"ui-M5-{hotel}")
                st.session_state["m5"] = res
            except Exception as e:
                st.error(f"M5 failed: {e}")
        res = st.session_state.get("m5")
        if res:
            b = res["baseline"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline RevPAR", f"€{b['RevPAR_0']:.2f}",
                      help="Current season RevPAR from the warehouse: realised ADR × "
                           "occupancy.")
            c2.metric("Required ADR", f"€{res['required_adr']:.2f}",
                      f"{res['adr_increase_pct']:.1f}%",
                      help="Rate needed to reach the target under the assumed elasticity — "
                           "a goal-seeking result, not a forecast.")
            c3.metric("Occupancy there", f"{100 * res['occupancy_at_required_adr']:.1f}%",
                      help="The occupancy the demand curve implies at the required rate.")
            st.markdown(f"**Baseline:** occupancy {100 * b['occ_0']:.1f}% at ADR "
                        f"€{b['ADR_0']:.2f} (RevPAR €{b['RevPAR_0']:.2f}) → "
                        f"target RevPAR €{res['target_revpar']:.2f}.")

            occ0, adr0 = b["occ_0"], b["ADR_0"]
            grid = np.linspace(0.4 * adr0, 2.6 * adr0, 240)
            revpar = occ0 * np.power(grid, 1 + res["elasticity"]) * np.power(adr0, -res["elasticity"])
            figV = go.Figure()
            figV.add_scatter(x=grid, y=revpar, name="RevPAR curve",
                             line=dict(color="#1F4E79"))
            figV.add_hline(y=res["target_revpar"], line_dash="dash", line_color="#C62828",
                           annotation_text=f"target €{res['target_revpar']:.2f}",
                           annotation_position="bottom right")
            if res["required_adr"] is not None:
                figV.add_vline(x=res["required_adr"], line_dash="dot", line_color="#2E7D32",
                               annotation_text=f"required ADR €{res['required_adr']:.2f}",
                               annotation_position="top")
            figV.add_scatter(x=[adr0], y=[b["RevPAR_0"]], mode="markers", name="baseline",
                             marker=dict(color="#F59E0B", size=10))
            figV.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="ADR (€)", yaxis_title="RevPAR (€)",
                               legend=dict(orientation="h", y=1.15))
            st.plotly_chart(figV, width="stretch")

            tor = pd.DataFrame(res["tornado"])
            base_val = res["required_adr"]
            st.markdown("**Tornado** — bars grow outwards from the baseline answer; "
                        "a longer arm means the answer depends more on that parameter "
                        "(±20%, one parameter at a time):")
            fig = go.Figure()

            def _span(row):
                vals = [v for v in (row["low"], row["high"]) if v is not None]
                return max((abs(v - base_val) for v in vals), default=0.0)

            tor = (tor.assign(_span=tor.apply(_span, axis=1))
                      .sort_values("_span", ascending=True))  # nhỏ dưới → to trên: dáng phễu
            for _, row in tor.iterrows():
                label = row["parameter"]
                notes = []
                for v, color, tag in ((row["low"], "#EF553B", "−20%"),
                                      (row["high"], "#636EFA", "+20%")):
                    if v is None:
                        notes.append(f"{tag}: no finite solution")
                        continue
                    if v <= base_val:      # tay trái: từ v tới base
                        fig.add_bar(y=[label], x=[base_val - v], base=[v],
                                    orientation="h", marker_color=color,
                                    name=tag, showlegend=False)
                    else:                  # tay phải: từ base tới v
                        fig.add_bar(y=[label], x=[v - base_val], base=[base_val],
                                    orientation="h", marker_color=color,
                                    name=tag, showlegend=False)
                if notes:
                    fig.add_annotation(y=label, x=base_val, text="; ".join(notes),
                                       showarrow=False, xanchor="left",
                                       font=dict(size=9, color="#B91C1C"))
            fig.add_vline(x=base_val, line_dash="dot",
                          annotation_text=f"base = €{base_val:.2f}")
            fig.update_yaxes(categoryorder="array",
                             categoryarray=tor["parameter"].tolist())
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10),
                              xaxis_title="required ADR (€)")
            st.plotly_chart(fig, width="stretch")

            sens = res.get("sensitivity")
            if sens:
                # ---- B: per-parameter mini tornados, 3 nested levels each ----
                from plotly.subplots import make_subplots
                st.markdown("**Seven-level tornado — one panel per parameter** "
                            "(±5% light / ±10% medium / ±20% solid; each panel has its own "
                            "scale — hover an arm for its value):")
                figB = make_subplots(rows=2, cols=2,
                                     subplot_titles=("elasticity e", "occupancy occ₀",
                                                     "baseline ADR₀", "target uplift"))
                for key, r, c in [("elasticity", 1, 1), ("occupancy", 1, 2),
                                  ("baseline_adr", 2, 1), ("target", 2, 2)]:
                    pts = sens[key]
                    invals = [q["adr"] for q in pts if q["adr"] is not None and q["in_bracket"]]
                    lo = min(invals + [res["required_adr"]])
                    hi = max(invals + [res["required_adr"]])
                    pad = (hi - lo) * 0.15
                    lo, hi = lo - pad, hi + pad
                    base_v = res["required_adr"]
                    for d, op in ((5, 0.35), (10, 0.65), (20, 1.0)):
                        for sgn, color in ((-1, "#EF553B"), (1, "#636EFA")):
                            q = next(z for z in pts if z["pct"] == sgn * d)
                            if q["adr"] is None:
                                continue
                            if q["in_bracket"]:
                                v = max(lo, min(hi, q["adr"]))
                                anchor = min(base_v, v)
                                figB.add_bar(
                                    y=[""], orientation="h", base=[anchor],
                                    x=[abs(v - anchor)], marker_color=color,
                                    opacity=op, row=r, col=c, showlegend=False,
                                    hovertext=[f"{key} {'−' if sgn < 0 else '+'}{d}% → €{q['adr']:,.0f}"],
                                    hoverinfo="text")
                            else:  # beyond bracket: ghost arm + annotation
                                figB.add_bar(
                                    y=[""], orientation="h", base=[base_v],
                                    x=[hi - base_v], marker_color=color,
                                    opacity=0.15, row=r, col=c, showlegend=False,
                                    hovertext=[f"{key} {'−' if sgn < 0 else '+'}{d}% → €{q['adr']:,.0f} (outside bracket)"],
                                    hoverinfo="text")
                                figB.add_annotation(
                                    row=r, col=c, showarrow=False, x=hi, y=0,
                                    xanchor="right", text=f"→ €{q['adr']:,.0f} outside bracket",
                                    font=dict(size=9, color="#B91C1C"))
                    figB.add_vline(x=base_v, line_dash="dot", row=r, col=c)
                figB.update_layout(barmode="overlay", height=480, showlegend=False,
                                   margin=dict(l=10, r=10, t=28, b=10))
                figB.update_xaxes(title_text="required ADR (€)")
                figB.update_yaxes(showticklabels=False)
                st.plotly_chart(figB, width="stretch")

                # ---- C: spider — 4 curves through 7 levels, log scale ----
                st.markdown("**Spider — four curves across seven levels, log scale**: shows the shape "
                            "of the response; the elasticity curve runs steeply upward as "
                            "e → −1 (points outside the bracket are drawn and flagged on hover):")
                figC = go.Figure()
                lvls = [-20, -10, -5, 0, 5, 10, 20]
                labs = ["−20%", "−10%", "−5%", "0%", "+5%", "+10%", "+20%"]
                ccolors = {"elasticity": "#636EFA", "occupancy": "#EF553B",
                           "baseline_adr": "#00CC96", "target": "#AB63FA"}
                for key, title in [("elasticity", "elasticity e"),
                                   ("occupancy", "occupancy occ₀"),
                                   ("baseline_adr", "baseline ADR₀"),
                                   ("target", "target uplift")]:
                    pts = sens[key]
                    xs, ys, hts = [], [], []
                    for lv, lab in zip(lvls, labs):
                        if lv == 0:
                            v, note = res["required_adr"], ""
                        else:
                            q = next(z for z in pts if z["pct"] == lv)
                            v = q["adr"]
                            note = "" if (v is None or q["in_bracket"]) else " (outside bracket)"
                        xs.append(lab)
                        ys.append(v)
                        hts.append(f"{title} {lab} → " + (f"€{v:,.0f}{note}" if v is not None else "not computable"))
                    figC.add_scatter(x=xs, y=ys, mode="lines+markers",
                                     name=title, line=dict(color=ccolors[key], width=2),
                                     hovertext=hts, hoverinfo="text")
                figC.add_hline(y=res["required_adr"], line_dash="dot",
                               annotation_text=f"base = €{res['required_adr']:.2f}")
                figC.update_layout(yaxis_type="log", height=360,
                                   xaxis_title="parameter change",
                                   yaxis_title="required ADR (€, log scale)",
                                   margin=dict(l=10, r=10, t=10, b=10),
                                   legend=dict(orientation="h", y=1.12))
                st.plotly_chart(figC, width="stretch")


# ======================================================================
# Screen 4 -- Scenario comparison (review)
# ======================================================================
def screen_history():
    st.header("Run history")
    n = con.execute("SELECT COUNT(*) FROM fact_scenario_run").fetchone()[0]
    st.caption(f"{n} runs logged in fact_scenario_run — every run, from this workspace, the "
               "experiments, or the tests, is reproducible from its logged parameters.")
    df = con.execute("""
        SELECT run_id, model_id, scenario_name, headline_kpi, run_at
        FROM fact_scenario_run ORDER BY run_id DESC LIMIT 200""").fetch_df()
    st.dataframe(df, width="stretch", height=320)
    sel = st.selectbox("Compare runs of model", ["M1", "M2", "M3", "M4", "M5"])
    runs = con.execute("""
        SELECT run_id, scenario_name, params, result
        FROM fact_scenario_run WHERE model_id = ? ORDER BY run_id DESC LIMIT 8""",
        [sel]).fetchall()
    chosen = st.multiselect(
        "Pick 2+ runs to compare side by side",
        [f"#{r[0]} {r[1] or '(unnamed)'}" for r in runs]) or None
    if chosen:
        ids = [int(c.split("#")[1].split(" ")[0]) for c in chosen]
        rows = [r for r in runs if r[0] in ids]
        for run_id, name, params, result in rows:
            st.markdown(f"**#{run_id} {name or ''}**")
            c1, c2 = st.columns(2)
            c1.json(json.loads(params))
            c2.json({k: v for k, v in json.loads(result).items()
                     if not isinstance(v, (list, dict))})


# ----------------------------------------------------------------------
# Navigation — product-style sidebar nav with icons and URL paths
# ----------------------------------------------------------------------
pg = st.navigation([
    st.Page(screen_dashboard, title="Dashboard", icon=":material/dashboard:",
            url_path="dashboard"),
    st.Page(screen_analytics, title="Analytics", icon=":material/query_stats:",
            url_path="analytics"),
    st.Page(screen_forecast, title="Demand forecast", icon=":material/trending_up:",
            url_path="forecast"),
    st.Page(screen_rate, title="Rate & allocation", icon=":material/tune:",
            url_path="rate-allocation"),
    st.Page(screen_history, title="Run history", icon=":material/history:",
            url_path="history"),
])

hotel = st.sidebar.selectbox("Property", HOTELS)
hotel_name = con.execute(
    "SELECT HotelName FROM dim_hotel WHERE HotelCode = ?", [hotel]).fetchone()[0]
st.sidebar.caption(f"{hotel_name} · capacity "
                   f"{con.execute('SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?', [hotel]).fetchone()[0]} rooms (estimated)")
st.sidebar.divider()
st.sidebar.caption("Screens follow Simon's phases — intelligence · design · "
                   "choice · review.")

pg.run()
