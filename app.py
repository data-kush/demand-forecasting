"""Demand Forecasting & Inventory Optimiser - interactive dashboard.

Run:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (accuracy_by_segment, growth_table, match_products, outlook,
                          portfolio_context, product_wape, seasonal_profile, watchlist)
from src.inventory import stock_needed_for, z_for_service
from src.insights import (OLLAMA_MODEL, fallback_briefing, ollama_status, qa_messages, stream_briefing,
                          stream_chat, stream_exec_summary, stream_po_email)
from src.ui import (AXIS, BLUE, ORANGE, RED, SLATE, TEAL, chart, footer, forecast_art, hero, inject_css,
                    kpi, steps, style)

ART = Path(__file__).parent / "artifacts"
AI, MA, LY = "LightGBM (AI model)", "8-week moving average", "Last year, trend-adjusted"
LABEL = {AI: "AI forecast", MA: "Spreadsheet average (8 weeks)", LY: "Same week last year"}
COLORS = {AI: BLUE, MA: ORANGE, LY: TEAL, "actual": SLATE}
FEATURE_NAMES = {
    "lag_1": "Sales last week", "lag_2": "Sales 2 weeks ago", "lag_3": "Sales 3 weeks ago",
    "lag_4": "Sales 4 weeks ago", "lag_6": "Sales 6 weeks ago", "lag_8": "Sales 8 weeks ago",
    "lag_13": "Sales 13 weeks ago", "lag_26": "Sales 26 weeks ago", "lag_52": "Same week last year",
    "rmean_4": "Average, last 4 weeks", "rmean_8": "Average, last 8 weeks",
    "rmean_13": "Average, last 13 weeks", "rmean_26": "Average, last 26 weeks",
    "rstd_8": "Volatility, last 8 weeks", "zero_share_8": "Weeks with no sales",
    "yoy_ratio": "Growth vs last year", "week_of_year": "Week of the year", "month": "Month",
    "xmas_period": "Christmas wind-down/closure", "weeks_to_xmas": "Weeks until Christmas",
    "sku_code": "Which product",
}

st.set_page_config(page_title="Demand Forecasting & Inventory Optimiser", page_icon=":material/insights:",
                   layout="wide")
inject_css("""
.result { font-size: 1.45rem; line-height: 1.4; font-weight: 500; }
.result b { color: #3b82f6; }
.st-key-card_result { border-left: 5px solid #3b82f6 !important; }
.big-order { font-size: 2.4rem; font-weight: 750; line-height: 1.1; }
.big-order-sub { opacity: .72; margin-bottom: .6rem; }
.calc { width: 100%; border-collapse: collapse; font-size: .95rem; }
.calc, .calc tr, .calc td { border: none !important; background: transparent !important; }
.calc td { padding: .35rem 0 !important; border-bottom: 1px solid color-mix(in srgb, currentColor 10%, transparent) !important; }
.calc td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.calc tr.total td { border-bottom: none !important; font-weight: 700; padding-top: .55rem !important; }
.product-title { font-size: 1.45rem; font-weight: 700; margin-bottom: .3rem; }
""")


@st.cache_data(max_entries=2)
def load(results_version: float):
    """results_version = when the pipeline last ran, so a new run refreshes the cache."""
    return {
        "panel": pd.read_pickle(ART / "panel.pkl"),
        "info": pd.read_pickle(ART / "sku_info.pkl").set_index("sku"),
        "bt": pd.read_pickle(ART / "backtest_holdout.pkl"),
        "acc": pd.read_csv(ART / "accuracy.csv", index_col=0),
        "curve": pd.read_pickle(ART / "service_curve.pkl"),
        "sigma": pd.read_pickle(ART / "sigma_holdout.pkl"),
        "future": pd.read_pickle(ART / "future_forecast.pkl"),
        "importance": pd.read_pickle(ART / "feature_importance.pkl"),
        "summary": json.loads((ART / "summary.json").read_text()),
    }


def gbp(x: float) -> str:
    return f"£{x:,.0f}"


if not (ART / "summary.json").exists():
    st.error("No results yet. Run the pipeline first:  `.venv\\Scripts\\python.exe -m src.pipeline`")
    st.stop()
d = load((ART / "summary.json").stat().st_mtime)
panel, info, bt, s, future = d["panel"], d["info"], d["bt"], d["summary"], d["future"]

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### :material/tune: Planning settings")
    lead_time = st.segmented_control("Supplier lead time", [1, 2, 3], default=2,
                                     format_func=lambda w: f"{w} week{'s' if w > 1 else ''}",
                                     help="How long after you place an order does the stock arrive?") or 2
    service_pct = st.slider("Customers served straight from stock", 80, 99, 95, format="%d%%",
                            help="Higher means fewer stock-outs, but more money tied up in stock.")
    service = service_pct / 100
    st.caption("Every number in the dashboard updates when you change these.")
    st.divider()
    st.markdown("**How to use this**")
    st.markdown("1. **Overview:** what better forecasting is worth\n"
                "2. **Insights:** trends, risks and an AI summary\n"
                "3. **Product planner:** forecast and order, product by product\n"
                "4. **Order plan:** enter your stock, get the purchase order\n"
                "5. **Ask AI:** questions in plain English")
    st.divider()
    ok, status = ollama_status()
    st.caption(f"{':green[●]' if ok else ':gray[●]'} Local AI: {status}")
    st.caption(f"Data: UCI Online Retail II, {s['data']['transactions']:,} sales lines, "
               f"{s['data']['products']} products, {s['data']['first_week']} to {s['data']['last_week']}.")
    st.caption(":material/dark_mode: Light or dark look: ⋮ menu (top right) → Settings → Theme.")

z = z_for_service(service)
P = lead_time + 1
sigma = d["sigma"][lead_time][AI]
fc_p = future.loc[future["h"] <= P].groupby("sku")["forecast"].sum()
safety = (z * sigma).reindex(fc_p.index).fillna(0)
target = fc_p + safety
curve = d["curve"].loc[d["curve"]["lead_time"] == lead_time]
needed = stock_needed_for(curve, service)
have_both = np.isfinite(needed.get(AI, np.nan)) and np.isfinite(needed.get(MA, np.nan))
sku_error = product_wape(bt)
acc = d["acc"]

for k, v in {"stock": pd.Series(0, index=info.index, dtype="int64"), "messages": [],
             "exec_summary": None, "po_email": None}.items():
    st.session_state.setdefault(k, v)


def build_plan(stock: pd.Series) -> pd.DataFrame:
    plan = pd.DataFrame({
        "Product": info["description"].str.title(),
        "Class": info["abc_class"] + info["xyz_class"],
        "Next week": future.loc[future["h"] == 1].set_index("sku")["forecast"].round(0),
        "Stock to have": target.round(0),
        "Your stock": stock,
        "Unit price": info["avg_price"],
    })
    plan["Order now"] = (plan["Stock to have"] - plan["Your stock"]).clip(lower=0)
    plan["Order value"] = (plan["Order now"] * plan["Unit price"]).round(0)
    plan.index.name = "SKU"
    return plan.sort_values("Order value", ascending=False)


def product_facts(sku: str, have: float) -> dict:
    row = info.loc[sku]
    hist = panel.loc[panel["sku"] == sku].set_index("week")["qty"]
    fut = future.loc[future["sku"] == sku].set_index("week")
    ly = hist.loc[hist.index > hist.index.max() - pd.Timedelta(weeks=65)].iloc[:13].sum()
    last13 = hist.iloc[-13:].sum()
    order = max(0.0, target[sku] - have)
    peak = fut["forecast"].idxmax()
    return {
        "Product": f"{row['description'].title()} ({sku})",
        "Revenue class": f"{row['abc_class']} (A = top sellers)",
        "Demand pattern": {"X": "steady", "Y": "variable", "Z": "erratic"}[row["xyz_class"]],
        "Units sold, last 13 weeks": f"{last13:,.0f}",
        "Change vs same 13 weeks last year": f"{last13 / ly - 1:+.0%}" if ly else "n/a",
        "Forecast units, next 13 weeks": f"{fut['forecast'].sum():,.0f}",
        "Peak forecast week": f"{peak:%d %b %Y} ({fut.loc[peak, 'forecast']:,.0f} units)",
        "Share of last year's units bought by its top 3 buyers": f"{row['top3_buyer_share']:.0%}",
        "Supplier lead time": f"{lead_time} week(s)",
        "Units in stock + on order": f"{have:,.0f}",
        "Recommended order now": (f"order {order:,.0f} units (about {gbp(order * row['avg_price'])})"
                                  if order > 0 else "no order needed this week"),
        "Safety stock included": f"{safety[sku]:,.0f} units for {service:.0%} service",
        "Typical forecast error for this product": f"{sku_error[sku]:.0%}",
    }


plan_all = build_plan(st.session_state.stock)
context = portfolio_context(s, {"lead_time": lead_time, "service": service}, needed, panel, future,
                            info, bt, plan_all)

# ---------------- hero ----------------
saving = needed[MA] - needed[AI] if have_both else np.nan
hero("Retail demand planning · AI forecasting",
     "Demand Forecasting &amp; Inventory Optimiser",
     "Weekly demand forecasts and stock recommendations for 100 products, tested on a quarter of real sales "
     "the model never saw, with plain-English briefings from a private, offline AI.",
     [(gbp(saving) if have_both else "–", "working capital freed"),
      (f"{saving / needed[MA]:.0%}" if have_both else "–", "less stock, same service"),
      (f"{s['data']['transactions'] / 1e6:.1f}M", "real transactions"),
      (str(s["data"]["products"]), "products planned")],
     forecast_art())

tab_over, tab_insights, tab_product, tab_plan, tab_ask, tab_method = st.tabs([
    ":material/insights: Overview", ":material/query_stats: Insights", ":material/inventory_2: Product planner",
    ":material/shopping_cart: Order plan", ":material/forum: Ask AI", ":material/menu_book: How it works"])

# ---------------- overview ----------------
with tab_over:
    with st.container(key="card_result"):
        st.markdown("<div class='eyebrow'>The result</div>", unsafe_allow_html=True)
        if have_both:
            st.markdown(
                f"<div class='result'>To serve <b>{service:.0%}</b> of customers straight from stock, the AI "
                f"forecast needs <b>{gbp(needed[AI])}</b> of stock on average, which is "
                f"<b>{gbp(saving)} ({saving / needed[MA]:.0%}) less</b> than planning with a "
                f"spreadsheet average.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result'>None of the methods can reliably serve {service:.0%} of customers "
                        f"with a {lead_time}-week lead time in this test. Try a lower service level.</div>",
                        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Working capital freed up", gbp(saving) if have_both else "–",
            "Less cash tied up in stock, same service", "cash")
    with c2:
        kpi("Stock needed (AI forecast)", gbp(needed[AI]) if have_both else "–",
            f"vs {gbp(needed[MA])} with the spreadsheet" if have_both else "", "stock")
    with c3:
        kpi("Forecast error", f"{acc.loc[AI, 'WAPE (1-4 weeks ahead)']:.0%}",
            f"vs {acc.loc[MA, 'WAPE (1-4 weeks ahead)']:.0%} for the spreadsheet · "
            f"<span class='good'>{s['error_reduction_vs_best_baseline']:.0%} lower</span>", "err")
    with c4:
        kpi("Under-forecasting", f"{-acc.loc[AI, 'Bias']:.0%}",
            f"vs {-acc.loc[MA, 'Bias']:.0%} for the spreadsheet · <span class='good'>fewer stock-outs</span>",
            "bias")

    left, right = st.columns([3, 2])
    with left, st.container(key="card_bars", height="stretch"):
        st.markdown(f"**Stock needed to serve {service:.0%} of customers**")
        st.caption(f"Average value of stock held, 100 products, {lead_time}-week lead time. Lower is better.")
        order = [LY, MA, AI]
        vals = [needed.get(m, np.nan) for m in order]
        fig = go.Figure(go.Bar(
            x=vals, y=[LABEL[m] for m in order], orientation="h", width=0.55, cliponaxis=False,
            marker=dict(color=[COLORS[m] for m in order], cornerradius=6),
            text=[gbp(v) if np.isfinite(v) else "not reached" for v in vals], textposition="outside",
            hovertemplate="%{y}: £%{x:,.0f}<extra></extra>"))
        style(fig, 270)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=70, t=10, b=10))
        top = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1
        fig.update_xaxes(tickprefix="£", range=[0, top * 1.22])
        fig.update_yaxes(showgrid=False)
        chart(fig)
    with right, st.container(key="card_why", height="stretch"):
        st.markdown("**Why it saves money**")
        st.markdown(
            "Every shop keeps a **safety buffer** of stock in case demand beats the forecast. The less you "
            "can trust the forecast, the bigger that buffer has to be.\n\n"
            "The AI forecast is a little more accurate and much less likely to **under-estimate** demand, so "
            "the same service needs a smaller buffer. That money stays in the bank instead of on the shelf.")
        st.markdown(":blue-badge[:material/verified: Tested on unseen data] "
                    ":green-badge[:material/savings: Measured in £, not just accuracy] "
                    ":violet-badge[:material/lock: Runs offline]")

    with st.container(key="card_curve"):
        st.markdown("**The full trade-off: stock held vs customers served**")
        st.caption(f"Replay of {s['data']['holdout'][0]} to {s['data']['holdout'][1]}: weekly orders, "
                   f"{lead_time}-week lead time. Each dot is one size of safety buffer. Up and to the left is "
                   "better: more customers served with less stock.")
        fig = go.Figure()
        for m in (AI, MA, LY):
            c = curve.loc[curve["model"] == m].sort_values("z")
            fig.add_trace(go.Scatter(x=c["avg_stock_value"], y=c["fill_rate"], name=LABEL[m],
                                     mode="lines+markers", line=dict(color=COLORS[m], width=2.5),
                                     marker=dict(size=7),
                                     hovertemplate="stock £%{x:,.0f} · served %{y:.1%}<extra>" + LABEL[m] + "</extra>"))
        fig.add_hline(y=service, line=dict(color=AXIS, dash="dot", width=1), annotation_text=f"target {service:.0%}")
        style(fig, 400, "Customers served from stock")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Average stock value held (£)", tickprefix="£")
        fig.update_yaxes(tickformat=".0%", rangemode="normal")
        chart(fig)

    with st.expander("Accuracy details and product mix"):
        a, b = st.columns(2)
        with a:
            tbl = acc.loc[[AI, MA, LY]].rename(index=LABEL)
            tbl.columns = ["Error, 1 week ahead", "Error, 1-4 weeks ahead", "Bias"]
            st.dataframe(tbl.style.format("{:.1%}"), width="stretch")
            st.caption("Error = WAPE (total absolute error ÷ total units). Negative bias = under-forecasting.")
        with b:
            mat = pd.crosstab(info["abc_class"], info["xyz_class"]).reindex(
                index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
            mat.index = ["A · top 70% of revenue", "B · next 20%", "C · last 10%"]
            mat.columns = ["Steady", "Variable", "Erratic"]
            st.dataframe(mat, width="stretch")
            st.caption("Number of products by revenue importance and how predictable their weekly demand is.")

# ---------------- insights ----------------
with tab_insights:
    with st.container(key="card_exec"):
        st.markdown("**:material/auto_awesome: AI analyst summary**")
        st.caption("Your local AI reads the analysis on this page plus this week's order plan and writes a "
                   "one-minute summary. Nothing leaves your computer.")
        if ok:
            if st.button("Write the executive summary", icon=":material/auto_awesome:", type="primary",
                         key="btn_exec"):
                with st.spinner("The local AI is reading the numbers. This takes about a minute..."):
                    st.session_state.exec_summary = st.write_stream(stream_exec_summary(context))
            elif st.session_state.exec_summary:
                st.markdown(st.session_state.exec_summary)
        else:
            st.caption(f"Local AI is off ({status}).")
        with st.expander("The facts the AI is given"):
            st.markdown(context)

    o = outlook(panel, future)
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Next 13 weeks, all products", f"{o['forecast_units']:,.0f} units",
            f"{o['change']:+.0%} vs the same weeks last year", "out")
    with k2:
        kpi("Busiest week ahead", f"{o['peak_week']:%d %b %Y}", f"{o['peak_units']:,.0f} units forecast", "peak")
    with k3:
        kpi("Sales from each product's top 3 buyers", f"{info['top3_buyer_share'].median():.0%}",
            "median product, last 52 weeks", "conc")

    with st.container(key="card_season"):
        st.markdown("**Seasonality: this year vs last year**")
        st.caption("Total weekly units across the 100 products. Sales build from September to a November peak, "
                   "then stop over Christmas.")
        prof = seasonal_profile(panel)
        fig = go.Figure()
        for yr, col in zip([c for c in prof.columns if c in (2010, 2011)], [ORANGE, BLUE]):
            fig.add_trace(go.Scatter(x=prof.index, y=prof[yr], name=str(yr), mode="lines",
                                     line=dict(color=col, width=2.5), hovertemplate="%{y:,.0f} units"))
        style(fig, 320, "Units per week")
        fig.update_xaxes(title="Week of the year", range=[1, 52])
        chart(fig)

    left, right = st.columns(2)
    with left, st.container(key="card_growth", height="stretch"):
        st.markdown("**Biggest movers vs last year**")
        st.caption("Units in the last 13 weeks compared with the same 13 weeks a year earlier.")
        g = growth_table(panel, info)
        movers = pd.concat([g.head(7), g.tail(7)]).sort_values("growth")
        fig = go.Figure(go.Bar(
            x=movers["growth"], y=movers["product"], orientation="h", cliponaxis=False,
            marker=dict(color=np.where(movers["growth"] >= 0, BLUE, RED), cornerradius=4),
            text=[f"{v:+.0%}" for v in movers["growth"]], textposition="outside",
            customdata=movers[["units_recent", "units_last_year"]],
            hovertemplate="%{y}<br>%{customdata[0]:,.0f} units vs %{customdata[1]:,.0f} last year<extra></extra>"))
        style(fig, 470)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=60, t=10, b=10))
        lo, hi = movers["growth"].min(), movers["growth"].max()
        fig.update_xaxes(tickformat="+.0%", zeroline=True, zerolinecolor=AXIS,
                         range=[min(lo, 0) - 0.15 * (hi - lo), hi + 0.15 * (hi - lo)])
        fig.update_yaxes(showgrid=False)
        chart(fig)
    with right, st.container(key="card_watch", height="stretch"):
        st.markdown("**Watchlist: unusual next 4 weeks**")
        st.caption("Products forecast 40% or more above or below the same 4 weeks last year. Comparing with "
                   "last year strips out the normal Christmas slowdown, so these are the real surprises.")
        wl = watchlist(panel, future, info).assign(change=lambda x: x["change"] * 100)
        st.dataframe(
            wl[["product", "signal", "last_year_units", "forecast_units", "change"]], height=430, width="stretch",
            column_config={
                "product": st.column_config.TextColumn("Product", width="medium"),
                "signal": st.column_config.TextColumn("Signal"),
                "last_year_units": st.column_config.NumberColumn("Same weeks last year", format="localized"),
                "forecast_units": st.column_config.NumberColumn("Forecast", format="%.0f"),
                "change": st.column_config.NumberColumn("Change", format="%+.0f%%"),
            })

    left, right = st.columns(2)
    with left, st.container(key="card_seg", height="stretch"):
        st.markdown("**Where the forecast is most reliable**")
        seg = accuracy_by_segment(bt, info).reset_index()
        seg["segment"] = seg["abc_class"] + seg["xyz_class"]
        seg["gain"] = (1 - seg["ai_error"] / seg["spreadsheet_error"]) * 100
        seg["ai_pct"], seg["ss_pct"] = seg["ai_error"] * 100, seg["spreadsheet_error"] * 100
        best, worst = seg.loc[seg["ai_error"].idxmin()], seg.loc[seg["ai_error"].idxmax()]
        top_gain = seg.loc[seg["gain"].idxmax()]
        st.caption(f"Forecast error by product group (A/B/C = revenue importance, X/Y/Z = steady/variable/"
                   f"erratic). Most reliable: **{best['segment']}** ({best['ai_error']:.0%} error), least: "
                   f"**{worst['segment']}** ({worst['ai_error']:.0%}). The AI's biggest edge is on "
                   f"**{top_gain['segment']}** products ({top_gain['gain']:.0f}% lower error); elsewhere it is "
                   "close to the spreadsheet, and its main win is less under-forecasting.")
        st.dataframe(
            seg[["segment", "products", "ai_pct", "ss_pct", "gain"]], hide_index=True, width="stretch",
            column_config={
                "segment": st.column_config.TextColumn("Group"),
                "products": st.column_config.NumberColumn("Products"),
                "ai_pct": st.column_config.ProgressColumn("AI error", format="%.0f%%", min_value=0, max_value=100),
                "ss_pct": st.column_config.NumberColumn("Spreadsheet error", format="%.0f%%"),
                "gain": st.column_config.NumberColumn("AI gain", format="%+.0f%%"),
            })
    with right, st.container(key="card_conc", height="stretch"):
        st.markdown("**Few big buyers make demand hard to predict**")
        cdf = info.assign(product=info["description"].str.title(), error=sku_error.reindex(info.index))
        corr = cdf["top3_buyer_share"].corr(cdf["error"])
        st.caption(f"Each dot is a product. The more of its sales come from its top 3 buyers, the bigger the "
                   f"forecast error (correlation {corr:.2f}). For these products, talking to the big buyers "
                   "beats any model.")
        fig = go.Figure(go.Scatter(
            x=cdf["top3_buyer_share"], y=cdf["error"], mode="markers",
            marker=dict(size=10, color=BLUE, opacity=0.75),
            text=cdf["product"], hovertemplate="%{text}<br>top 3 buyers %{x:.0%} · error %{y:.0%}<extra></extra>"))
        style(fig, 330, "Forecast error")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Share of units from top 3 buyers", tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        chart(fig)

# ---------------- product planner ----------------
with tab_product:
    sku = st.selectbox("Choose a product (type to search)", list(info.index),
                       format_func=lambda k: f"{info.loc[k, 'description'].title()}  ·  {k}")
    row = info.loc[sku]
    hist = panel.loc[panel["sku"] == sku].set_index("week")["qty"]
    fut = future.loc[future["sku"] == sku].set_index("week")
    back = bt.loc[(bt["sku"] == sku) & (bt["model"] == AI) & (bt["h"] == 1)].set_index("week")
    pattern = {"X": "Steady demand", "Y": "Variable demand", "Z": "Erratic demand"}[row["xyz_class"]]
    tier = {"A": "Top seller (A)", "B": "Mid seller (B)", "C": "Low seller (C)"}[row["abc_class"]]
    price = gbp(row["avg_price"]) if row["avg_price"] >= 10 else f"£{row['avg_price']:.2f}"

    with st.container(key="card_product"):
        st.markdown(f"<div class='product-title'>{row['description'].title()}</div>"
                    f"<div class='chips'><span>SKU {sku}</span><span>{tier}</span><span>{pattern}</span>"
                    f"<span>Unit price {price}</span><span>2-year revenue {gbp(row['total_revenue'])}</span>"
                    f"<span>Top 3 buyers: {row['top3_buyer_share']:.0%} of units</span></div>",
                    unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left, st.container(key="card_chart", height="stretch"):
        st.markdown("**Weekly sales and forecast**")
        st.caption("Grey: actual sales · dotted blue: what the model predicted a week ahead (test period) · "
                   "solid blue: next 13 weeks, with the shaded likely range (80%).")
        shown = hist.loc[hist.index > hist.index.max() - pd.Timedelta(weeks=52)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name="Actual sales", mode="lines",
                                 line=dict(color=COLORS["actual"], width=2), hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=back.index, y=back["forecast"], name="Model's test predictions",
                                 mode="lines", line=dict(color=BLUE, width=2, dash="dot"), hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=fut.index, y=fut["p90"], mode="lines", line=dict(width=0),
                                 showlegend=False, name="High (90%)", hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=fut.index, y=fut["p10"], mode="lines", fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(59,130,246,0.18)", name="Likely range", hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=fut.index, y=fut["forecast"], name="Forecast", mode="lines",
                                 line=dict(color=BLUE, width=2.5), hovertemplate="%{y:,.0f}"))
        today = hist.index.max()
        fig.add_shape(type="line", x0=today, x1=today, y0=0, y1=1, yref="paper",
                      line=dict(color=AXIS, width=1, dash="dot"))
        fig.add_annotation(x=today, y=1, yref="paper", text="Today", showarrow=False, yshift=8)
        chart(style(fig, 410, "Units per week"))

    with right, st.container(key="card_reco", height="stretch"):
        st.markdown("**What to order this week**")
        have = st.number_input("Units you already have (in stock + on order)", min_value=0,
                               value=int(round(fut["forecast"].iloc[0])), step=10, key=f"have_{sku}",
                               help="Pre-filled with an example. Replace it with your real number.")
        order_qty = max(0.0, target[sku] - have)
        st.markdown(
            f"<div class='big-order'>{order_qty:,.0f} units</div>"
            f"<div class='big-order-sub'>about {gbp(order_qty * row['avg_price'])} at the usual price</div>"
            "<table class='calc'>"
            f"<tr><td>Expected demand, next {P} weeks</td><td>{fc_p[sku]:,.0f}</td></tr>"
            f"<tr><td>+ Safety buffer for {service:.0%} service</td><td>{safety[sku]:,.0f}</td></tr>"
            f"<tr><td>= Stock you should have</td><td>{target[sku]:,.0f}</td></tr>"
            f"<tr><td>− Units you already have</td><td>{have:,.0f}</td></tr>"
            f"<tr class='total'><td>Order now</td><td>{order_qty:,.0f}</td></tr></table>",
            unsafe_allow_html=True)
        st.caption(f"Why {P} weeks? You order weekly and deliveries take {lead_time} week(s), "
                   f"so this order has to last {P} weeks.")

    facts = product_facts(sku, have)
    left, right = st.columns([2, 1])
    with left, st.container(key="card_brief", height="stretch"):
        st.markdown("**:material/auto_awesome: Briefing for the store manager**")
        if ok:
            st.caption(f"Written on your computer by a free local AI model ({OLLAMA_MODEL}) from the numbers "
                       "on this page. It can take a minute.")
            if st.button("Write the briefing", icon=":material/auto_awesome:", type="primary"):
                with st.spinner("The local AI is writing..."):
                    st.write_stream(stream_briefing(facts))
        else:
            st.caption(f"Local AI is off ({status}), so this is the standard summary.")
            st.markdown(fallback_briefing(facts))
    with right, st.container(key="card_facts", height="stretch"):
        st.markdown("**The numbers behind it**")
        st.dataframe(pd.DataFrame({"Item": list(facts), "Value": list(facts.values())}), hide_index=True,
                     width="stretch", height=300,
                     column_config={"Item": st.column_config.TextColumn(width="medium"),
                                    "Value": st.column_config.TextColumn(width="medium")})

    with st.expander("Week-by-week forecast table"):
        t = fut[["forecast", "p10", "p90"]].round(0).astype(int)
        t.index = t.index.strftime("%d %b %Y")
        t.columns = ["Forecast", "Low (10%)", "High (90%)"]
        st.dataframe(t.T, width="stretch")

# ---------------- order plan ----------------
with tab_plan:
    stock = st.session_state.stock
    plan = plan_all
    to_order = plan.loc[plan["Order now"] > 0]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Products to order", f"{len(to_order)} of {len(plan)}", "this week", "n")
    with k2:
        kpi("Units to order", f"{to_order['Order now'].sum():,.0f}", "across all products", "units")
    with k3:
        kpi("Purchase order value", gbp(to_order["Order value"].sum()), "at usual unit prices", "value")
    with k4:
        kpi("Planning settings", f"{lead_time} wk · {service:.0%}", "lead time · service level", "set")

    with st.container(key="card_plan"):
        st.markdown("**This week's purchase order**")
        st.caption("Type your current stock (on the shelf + already ordered) into the **Your stock ✎** column. "
                   "Change the lead time and service level in the sidebar.")
        only = st.toggle("Show only products that need an order", value=False)
        view = to_order if only else plan
        edited = st.data_editor(
            view, key="plan_editor", width="stretch", height=480,
            disabled=[c for c in plan.columns if c != "Your stock"],
            column_config={
                "Product": st.column_config.TextColumn(width="large"),
                "Class": st.column_config.TextColumn(help="A/B/C = revenue importance · X/Y/Z = steady/variable/erratic"),
                "Next week": st.column_config.NumberColumn("Forecast next week", format="localized"),
                "Stock to have": st.column_config.NumberColumn(
                    "Stock to have", format="localized", help="Expected demand until the next delivery + safety buffer"),
                "Your stock": st.column_config.NumberColumn("Your stock ✎", min_value=0, step=1, format="localized",
                                                            help="Type your stock on hand + on order"),
                "Unit price": st.column_config.NumberColumn("Unit price (£)", format="%.2f"),
                "Order now": st.column_config.NumberColumn(format="localized"),
                "Order value": st.column_config.NumberColumn("Order value (£)", format="localized"),
            })
        changed = edited["Your stock"].fillna(0).astype("int64")
        if not changed.equals(stock.reindex(changed.index)):
            st.session_state.stock.loc[changed.index] = changed
            st.rerun()
        st.download_button("Download purchase order (CSV)", plan.to_csv().encode(), "purchase_order.csv",
                           "text/csv", icon=":material/download:")

    with st.container(key="card_email"):
        st.markdown("**:material/mail: Draft the supplier email**")
        c1, c2 = st.columns([1, 2])
        with c1:
            n_lines = st.slider("Order lines to include (largest first)", 3, 20, 8, key="email_lines")
            top = to_order.head(n_lines)
            st.caption(f"The local AI writes the email around these {len(top)} lines. Quantities come straight "
                       "from the table above, and you can edit the draft before sending.")
            lines = [f"{r['Product']} (SKU {sku_}): {r['Order now']:,.0f} units" for sku_, r in top.iterrows()]
            if not ok:
                st.caption(f"Local AI is off ({status}).")
            elif not lines:
                st.caption("Nothing to order with the stock you entered.")
            elif st.button("Draft email", icon=":material/mail:", type="primary", key="btn_email"):
                with st.spinner("The local AI is drafting. This can take a minute..."):
                    st.session_state.po_email = "".join(stream_po_email(
                        lines, lead_time, top["Order now"].sum(), top["Order value"].sum()))
        with c2:
            if st.session_state.po_email:
                draft = st.text_area("Email draft (edit freely)", st.session_state.po_email, height=320)
                st.download_button("Download email (.txt)", draft.encode(), "supplier_email.txt", "text/plain",
                                   icon=":material/download:")
            else:
                st.info("Your draft will appear here.", icon=":material/edit_note:")

# ---------------- ask AI ----------------
SUGGESTIONS = {
    ":material/trending_up: Which products should I stock up on?": "Which products should I stock up on in the next 4 weeks, and why?",
    ":material/warning: What are the biggest risks?": "What are the biggest risks for my stock over the next month?",
    ":material/shopping_cart: Explain this week's order": "Explain this week's purchase order in simple terms.",
    ":material/help: How much Regency Cakestand should I order?": "How much should I order of the Regency Cakestand, and why?",
}
with tab_ask:
    chat_col, side_col = st.columns([3, 1.3])
    with side_col:
        with st.container(key="card_ask_about", height="stretch"):
            st.markdown("**:material/forum: Ask your data**")
            st.caption(f"A free AI model ({OLLAMA_MODEL}) running on your computer answers from the dashboard's "
                       "own numbers. Name a product to get its details. Answers take 20 to 60 seconds on a CPU.")
            st.markdown(":green-badge[:material/lock: Private] :blue-badge[:material/fact_check: Grounded in your data] "
                        ":gray-badge[:material/money_off: Free]")
            with st.expander("What the AI knows"):
                st.markdown(context)
            if st.session_state.messages and st.button("Clear conversation", icon=":material/delete:",
                                                       key="btn_clear", width="stretch"):
                st.session_state.messages = []
                st.rerun()
    with chat_col:
        if not ok:
            st.info(f"Start Ollama to use the assistant ({status}).", icon=":material/info:")
        else:
            chat = st.container(key="card_chatbox", height=520)
            prompt = st.chat_input("Ask about demand, stock or a product", submit_mode="disable", key="chat_q")
            with chat:
                if not st.session_state.messages:
                    picks = st.empty()
                    chosen = picks.pills("Try asking", list(SUGGESTIONS), key="chat_pills")
                    if chosen and not prompt:
                        prompt = SUGGESTIONS[chosen]
                    if prompt:
                        picks.empty()
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
                if prompt:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        skus = match_products(prompt, info)
                        pf = [product_facts(k, 0) for k in skus]
                        with st.status("Data used for this answer", type="compact", state="complete"):
                            st.markdown("Business facts (as on the Insights tab)"
                                        + (", plus product details for: " + ", ".join(f["Product"] for f in pf)
                                           if pf else "") + ". Product orders assume zero stock on hand.")
                        answer = st.write_stream(stream_chat(
                            qa_messages(context, pf, st.session_state.messages, prompt), max_words=140))
                    st.session_state.messages += [{"role": "user", "content": prompt},
                                                  {"role": "assistant", "content": answer}]

# ---------------- method ----------------
with tab_method:
    steps([
        (":material/database:", "Clean the data",
         f"{s['data']['transactions']:,} real sales lines. Cancellations, returns, postage and fees removed."),
        (":material/model_training:", "Forecast",
         "One LightGBM model learns from all 100 products: recent sales, seasonality, Christmas closure."),
        (":material/verified:", "Test fairly",
         "The last 13 weeks are hidden. Every week, each method forecasts using only earlier data."),
        (":material/inventory:", "Turn into stock",
         "Order up to: demand until the next delivery + a safety buffer sized on past errors."),
        (":material/auto_awesome:", "Explain",
         "A private local AI turns the numbers into briefings, summaries and emails."),
    ])
    a, b = st.columns([3, 2])
    with a, st.container(key="card_method", height="stretch"):
        st.markdown(f"""
**The data.** UCI Online Retail II: a UK online gift retailer, Dec 2009 to Dec 2011. The 100 highest-revenue
products that sell in at least 70% of weeks are planned week by week.

**A fair test.** The last 13 weeks ({s['data']['holdout'][0]} to {s['data']['holdout'][1]}) are hidden from the
models. The safety buffer is sized on the 13 weeks *before* that, so the money results aren't tuned on the
test period either.

**Three ways to forecast.**
- *Spreadsheet average:* the average of the last 8 weeks, which many small shops use.
- *Same week last year:* last year's sales, adjusted for recent growth.
- *AI forecast:* one machine-learning model (LightGBM) across all products, with quantile models for the likely
  range.

**From forecast to money.** Replaying real customer demand shows how many customers each method served and
how much stock it held. Comparing at the same service level shows the working capital freed.
""")
    with b, st.container(key="card_imp", height="stretch"):
        st.markdown("**What the AI looks at most**")
        share = d["importance"] / d["importance"].sum()
        imp = share.head(10)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_NAMES.get(i, i) for i in imp.index],
                               orientation="h", marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        style(fig, 400)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(tickformat=".0%")
        chart(fig)

footer("Built by quant_kush · demand forecasting & inventory optimisation demo")
