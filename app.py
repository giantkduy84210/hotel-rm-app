"""
Hotel RM DSS -- Streamlit application (report Section 4.3).

Four screens mapped to Simon's phases:
  1. Dashboard            (intelligence) -- KPIs and charts from the warehouse
  2. Forecasting          (design)      -- M1 with method comparison + backtest
  3. Optimisation & What-if (design/choice) -- M2-M5 with parameter sliders
  4. Scenario comparison  (review)      -- the fact_scenario_run log

Design principle: the interface owns NO model logic -- every model number
comes from run_model(); the dashboard uses the same analytical SQL the
experiment scripts use.
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mbms  # noqa: E402
import models.m1_forecast as m1  # noqa: E402

st.set_page_config(page_title="Hotel RM DSS", page_icon="🏨", layout="wide")


@st.cache_resource
def get_con():
    con = duckdb.connect(str(ROOT / "warehouse" / "hotel_rm.duckdb"))
    mbms.init_mbms(con)
    return con


HOTELS = ("H1", "H2")
con = get_con()

st.title("🏨 Hotel Revenue Management DSS")
st.caption("CO4031 -- model base: M1-M5, managed by the MBMS (model_catalog + run_model + "
           "fact_scenario_run). Every number below comes from the registry.")

page = st.sidebar.radio("Screen (Simon's phases)", [
    "1 · Dashboard (intelligence)",
    "2 · Forecasting (design)",
    "3 · Optimisation & what-if (design/choice)",
    "4 · Scenario comparison (review)",
])
hotel = st.sidebar.selectbox("Property", HOTELS)
hotel_name = con.execute(
    "SELECT HotelName FROM dim_hotel WHERE HotelCode = ?", [hotel]).fetchone()[0]
st.sidebar.caption(f"{hotel_name} · capacity "
                   f"{con.execute('SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?', [hotel]).fetchone()[0]} rooms (estimated)")


# ======================================================================
# Screen 1 -- Dashboard (intelligence)
# ======================================================================
if page.startswith("1"):
    st.header(f"Dashboard -- {hotel_name}")
    c1, c2, c3, c4 = st.columns(4)
    kpis = con.execute("""
        SELECT SUM(f.RevenueRealised)::DOUBLE,
               SUM(CASE WHEN f.IsCanceled = 0 THEN f.ADR * 1 ELSE 0 END)
                 / NULLIF(SUM(f.IsCanceled = 0), 0),
               SUM(f.IsCanceled = 0)::DOUBLE,
               AVG(CASE WHEN f.IsCanceled = 1 THEN 1.0 ELSE 0.0 END)
        FROM fact_room_night f JOIN dim_hotel h USING (HotelKey)
        WHERE h.HotelCode = ?""", [hotel]).fetchone()
    cap = con.execute("SELECT CapacityEstimate FROM dim_hotel WHERE HotelCode = ?",
                      [hotel]).fetchone()[0]
    nights = con.execute("""
        SELECT COUNT(DISTINCT o.DateKey) FROM fact_daily_occupancy o
        JOIN dim_hotel h USING (HotelKey) WHERE h.HotelCode = ?""", [hotel]).fetchone()[0]
    c1.metric("Realised revenue", f"€{kpis[0]:,.0f}")
    c2.metric("ADR (kept)", f"€{kpis[1]:.2f}")
    c3.metric("Occupancy (est.)", f"{100 * kpis[2] / (cap * nights):.1f}%")
    c4.metric("Cancellation rate", f"{100 * kpis[3]:.1f}%")

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
        st.plotly_chart(fig, use_container_width=True)
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
                        use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Screen 2 -- Forecasting (design)
# ======================================================================
elif page.startswith("2"):
    st.header(f"M1 -- demand forecast, {hotel_name}")
    col1, col2 = st.columns([1, 3])
    with col1:
        horizon = st.slider("Horizon (nights)", 7, 56, 28)
        method = st.selectbox("Method", ("auto (backtest winner)",
                                         "holt_winters", "snaive7", "yoy_snaive364"))
        if st.button("Run M1", type="primary"):
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
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Backtest (rolling monthly cut-offs, 28-day horizon)")
    if st.button("Run backtest (all methods compete)"):
        with st.spinner("running rolling backtest..."):
            bt = m1.backtest(con, hotel, horizon=28, n_cutoffs=14)
        st.session_state["bt"] = bt
    bt = st.session_state.get("bt")
    if bt:
        st.dataframe(pd.DataFrame(bt["metrics"]).set_index("method"))
        if bt["winner"] != "holt_winters":
            st.info(f"**Honest evaluation**: the seasonal-naive baseline "
                    f"`{bt['winner']}` beats Holt-Winters on this hotel -- it is used, "
                    f"not hidden (the field's own evidence: simple methods often win).")


# ======================================================================
# Screen 3 -- Optimisation & what-if (design/choice)
# ======================================================================
elif page.startswith("3"):
    st.header(f"Optimisation & what-if -- {hotel_name}")
    tab1, tab2, tab3, tab4 = st.tabs(["M2 · group request", "M3 · weekly allocation",
                                      "M4 · overbooking", "M5 · pricing"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        arrival = c1.date_input("Arrival", pd.Timestamp("2017-08-11").date())
        nights = c2.number_input("Nights", 1, 21, 3)
        rooms = c3.number_input("Rooms requested", 1, 100, 25)
        rate = st.number_input("Group rate (€/night)", 1.0, 500.0, 82.0, 5.0)
        if st.button("Run M2", type="primary"):
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
            c1.metric("Protection level y*", f"{res['y_star']:.1f}")
            c2.metric("Unprotected capacity", f"{res['unprotected_capacity']:.1f}")
            c3.metric("Breakeven group rate", f"€{res['breakeven_rate']:.2f}")
            c4.metric("Transient rate p_H", f"€{res['p_H']:.2f}")

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
        if st.button("Run M3", type="primary"):
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
            bp = pd.DataFrame(res["bid_prices"])
            if len(bp):
                st.markdown("**Bid prices** (shadow prices of the capacity constraints):")
                st.dataframe(bp)

    with tab3:
        c1, c2, c3 = st.columns(3)
        season = c1.selectbox("Season", ("High", "Shoulder", "Low"))
        cutoff = c2.slider("Cut-off h (days before arrival)", 1, 30, 7)
        cw = c3.slider("Walk cost multiple (× ADR)", 0.5, 3.0, 1.5, 0.1)
        if st.button("Run M4", type="primary"):
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
            c1.metric("Booking limit B*", f"{res['B_star']}")
            c2.metric("Overbooking level", f"{res['overbooking_level']}")
            c3.metric("P(walk ≥ 1)", f"{100 * res['p_walk_at_least_one']:.1f}%")
            c4.metric("E[walked guests]", f"{res['expected_walks']:.2f}")
            st.markdown(f"show-up probability at cut-off {res['cutoff_days']}d: "
                        f"**{res['showup_probability']}** (from "
                        f"{res['on_books']:,} on-the-books bookings, {res['kept']:,} kept) · "
                        f"critical ratio {res['critical_ratio']} · expected profit "
                        f"€{res['expected_profit']:,.0f} "
                        f"(Monte Carlo {res['mc']['reps']:,} reps: "
                        f"€{res['mc']['profit_mean']:,.0f}, "
                        f"95% CI [{res['mc']['profit_ci95'][0]:,.0f}, "
                        f"{res['mc']['profit_ci95'][1]:,.0f}])")

    with tab4:
        c1, c2 = st.columns(2)
        season = c1.selectbox("Season ", ("Low", "Shoulder", "High"))
        uplift = c2.slider("RevPAR target uplift", 5, 100, 20, 5, format="%d%%")
        e = st.slider("Price elasticity e (assumed)", -0.95, -0.05, -0.8, 0.05)
        if st.button("Run M5", type="primary"):
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
            c1.metric("Baseline RevPAR", f"€{b['RevPAR_0']:.2f}")
            c2.metric("Required ADR", f"€{res['required_adr']:.2f}",
                      f"{res['adr_increase_pct']:.1f}%")
            c3.metric("Occupancy there", f"{100 * res['occupancy_at_required_adr']:.1f}%")
            st.markdown(f"baseline: occ₀ {b['occ_0']}, ADR₀ €{b['ADR_0']:.2f}, "
                        f"target RevPAR €{res['target_revpar']:.2f}")
            tor = pd.DataFrame(res["tornado"])
            base_val = res["required_adr"]
            st.markdown("**Tornado** — thanh mọc từ đáp án gốc, tay越长 = đáp án càng phụ thuộc "
                        "tham số đó (±20% độc lập từng tham số):")
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
            st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Screen 4 -- Scenario comparison (review)
# ======================================================================
else:
    st.header("Scenario comparison (fact_scenario_run)")
    n = con.execute("SELECT COUNT(*) FROM fact_scenario_run").fetchone()[0]
    st.caption(f"{n} logged runs. Every run -- from the UI, the experiments, or the "
               "tests -- went through the same registry and is reproducible from its "
               "logged parameters.")
    df = con.execute("""
        SELECT run_id, model_id, scenario_name, headline_kpi, run_at
        FROM fact_scenario_run ORDER BY run_id DESC LIMIT 200""").fetch_df()
    st.dataframe(df, use_container_width=True, height=320)
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
