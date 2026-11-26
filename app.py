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

from src.inventory import stock_needed_for, z_for_service
from src.insights import OLLAMA_MODEL, fallback_briefing, ollama_status, stream_briefing

ART = Path(__file__).parent / "artifacts"
AI, MA, LY = "LightGBM (AI model)", "8-week moving average", "Last year, trend-adjusted"
COLORS = {AI: "#257de0", MA: "#f47230", LY: "#00b894", "actual": "#191919"}
INK2, GRID, AXIS = "#666666", "#e7e7e7", "#c2c2c2"

st.set_page_config(page_title="Demand Forecasting & Inventory Optimiser", layout="wide")


@st.cache_data
def load():
    if not (ART / "summary.json").exists():
        return None
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


def style(fig: go.Figure, height=380, y_title="Units per week") -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", font=dict(family="system-ui, Segoe UI, sans-serif"),
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=INK2), title=y_title,
                     title_font=dict(color=INK2), rangemode="tozero")
    return fig


def gbp(x: float) -> str:
    return f"£{x:,.0f}"


d = load()
if d is None:
    st.error("No results yet. Run the pipeline first:  `.venv\\Scripts\\python.exe -m src.pipeline`")
    st.stop()

panel, info, bt, s = d["panel"], d["info"], d["bt"], d["summary"]
future = d["future"]

# ---------------- sidebar ----------------
st.sidebar.header("Planning settings")
lead_time = st.sidebar.select_slider("Supplier lead time (weeks)", options=[1, 2, 3], value=2)
service = st.sidebar.slider("Target service level", 0.80, 0.99, 0.95, 0.01,
                            help="Probability of not running out before the next delivery.")
z = z_for_service(service)
P = lead_time + 1
sigma = d["sigma"][lead_time][AI]
fc_p = future.loc[future["h"] <= P].groupby("sku")["forecast"].sum()
safety = (z * sigma).reindex(fc_p.index).fillna(0)
target = fc_p + safety

st.sidebar.divider()
ok, status = ollama_status()
st.sidebar.caption(f"AI briefings: {'🟢' if ok else '⚪'} {status}")
st.sidebar.caption(f"Data: UCI Online Retail II · {s['data']['transactions']:,} sales lines · "
                   f"{s['data']['products']} products · weeks {s['data']['first_week']} to "
                   f"{s['data']['last_week']}")

st.title("Demand Forecasting & Inventory Optimiser")
st.caption("Weekly product demand forecasts and stock recommendations for an online retailer, "
           "tested on a quarter of data the models never saw.")

tab_impact, tab_product, tab_all, tab_method = st.tabs(
    ["Business impact", "Product planner", "Order plan (all products)", "How it works"])

# ---------------- business impact ----------------
with tab_impact:
    curve = d["curve"].loc[d["curve"]["lead_time"] == lead_time]
    needed = stock_needed_for(curve, service)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast error, AI model", f"{s['wape'][AI]:.1%}",
              f"{-s['error_reduction_vs_best_baseline']:.1%} vs best simple method",
              delta_color="inverse", help="WAPE: total absolute error ÷ total units, 1-4 weeks ahead.")
    if np.isfinite(needed.get(AI, np.nan)) and np.isfinite(needed.get(MA, np.nan)):
        saving = needed[MA] - needed[AI]
        c2.metric(f"Average stock needed for {service:.0%} fill", gbp(needed[AI]),
                  f"{-saving / needed[MA]:.1%} vs moving average", delta_color="inverse")
        c3.metric("Working capital released", gbp(saving),
                  help="Lower average stock value at the same fill rate, over the 100 products.")
    else:
        c2.metric(f"Stock needed for {service:.0%} fill", "not reached",
                  help="Some methods cannot reach this fill rate within the safety-stock range tested.")
        c3.metric("Working capital released", "–")
    c4.metric("Products planned", s["data"]["products"])

    st.subheader("Same service, less stock")
    st.caption(f"Holdout replay ({s['data']['holdout'][0]} to {s['data']['holdout'][1]}): order every "
               f"week, deliveries arrive after {lead_time} week(s). Each point is one safety-stock "
               "level. Further left and higher is better.")
    fig = go.Figure()
    for m in (AI, MA, LY):
        c = curve.loc[curve["model"] == m].sort_values("z")
        fig.add_trace(go.Scatter(x=c["avg_stock_value"], y=c["fill_rate"], name=m,
                                 mode="lines+markers", line=dict(color=COLORS[m], width=2),
                                 marker=dict(size=8, line=dict(color="#ffffff", width=2)),
                                 hovertemplate="stock £%{x:,.0f} · fill %{y:.1%}<extra>" + m + "</extra>"))
    fig.add_hline(y=service, line=dict(color=AXIS, dash="dot", width=1),
                  annotation_text=f"target {service:.0%}", annotation_font_color=INK2)
    style(fig, 420, "Fill rate (share of demand served)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Average stock value held (£)", tickprefix="£", title_font=dict(color=INK2))
    fig.update_yaxes(tickformat=".0%", rangemode="normal")
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Forecast accuracy on unseen data")
        acc = d["acc"].loc[[AI, MA, LY]]
        st.dataframe(acc.style.format("{:.1%}"), width="stretch")
        st.caption("WAPE: lower is better. Bias: negative means under-forecasting.")
    with b:
        st.subheader("Product mix (ABC × XYZ)")
        mat = pd.crosstab(info["abc_class"], info["xyz_class"]).reindex(
            index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        st.dataframe(mat, width="stretch")
        st.caption("A/B/C = top 70% / next 20% / last 10% of revenue. "
                   "X/Y/Z = steady / variable / erratic weekly demand.")

# ---------------- product planner ----------------
with tab_product:
    options = list(info.index)
    sku = st.selectbox("Product", options, format_func=lambda k: f"{k} · {info.loc[k, 'description']}")
    row = info.loc[sku]
    hist = panel.loc[panel["sku"] == sku].set_index("week")["qty"]
    fut = future.loc[future["sku"] == sku].set_index("week")
    back = bt.loc[(bt["sku"] == sku) & (bt["model"] == AI) & (bt["h"] == 1)].set_index("week")

    fig = go.Figure()
    shown = hist.loc[hist.index > hist.index.max() - pd.Timedelta(weeks=60)]
    fig.add_trace(go.Scatter(x=fut.index, y=fut["p90"], line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut.index, y=fut["p10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(37,125,224,0.15)", name="80% range",
                             hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name="Actual sales",
                             line=dict(color=COLORS["actual"], width=2), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=back.index, y=back["forecast"], name="Backtest (1 week ahead)",
                             line=dict(color=COLORS[AI], width=2, dash="dot"), hovertemplate="%{y:,.0f}"))
    fig.add_trace(go.Scatter(x=fut.index, y=fut["forecast"], name="Forecast",
                             line=dict(color=COLORS[AI], width=2), hovertemplate="%{y:,.0f}"))
    st.plotly_chart(style(fig, 400), width="stretch")

    ly_weeks = hist.loc[hist.index > hist.index.max() - pd.Timedelta(weeks=65)].iloc[:13]
    last13 = hist.iloc[-13:].sum()
    sku_bt = bt.loc[(bt["sku"] == sku) & (bt["model"] == AI)].dropna(subset=["qty"])
    sku_wape = (sku_bt["forecast"] - sku_bt["qty"]).abs().sum() / max(sku_bt["qty"].sum(), 1)

    st.subheader("Stock recommendation")
    k1, k2, k3, k4 = st.columns(4)
    stock_now = k1.number_input("Units in stock + on order", min_value=0,
                                value=int(round(fut["forecast"].iloc[0])), step=10)
    order = max(0.0, target[sku] - stock_now)
    k2.metric(f"Demand over next {P} weeks", f"{fc_p[sku]:,.0f}")
    k3.metric("Safety stock", f"{safety[sku]:,.0f}", help=f"Covers forecast error at {service:.0%} service.")
    k4.metric("Order now", f"{order:,.0f} units", f"≈ {gbp(order * row['avg_price'])}", delta_color="off")

    peak = fut["forecast"].idxmax()
    facts = {
        "Product": f"{row['description'].title()} ({sku})",
        "Revenue class": f"{row['abc_class']} (A = top sellers)",
        "Demand pattern": {"X": "steady", "Y": "variable", "Z": "erratic"}[row["xyz_class"]],
        "Units sold, last 13 weeks": f"{last13:,.0f}",
        "Change vs same 13 weeks last year": (f"{last13 / ly_weeks.sum() - 1:+.0%}"
                                              if ly_weeks.sum() else "n/a"),
        "Forecast units, next 13 weeks": f"{fut['forecast'].sum():,.0f}",
        "Peak forecast week": f"{peak:%d %b %Y} ({fut.loc[peak, 'forecast']:,.0f} units)",
        "Supplier lead time": f"{lead_time} week(s)",
        "Units in stock + on order": f"{stock_now:,}",
        "Recommended order now": (f"order {order:,.0f} units (about {gbp(order * row['avg_price'])})"
                                  if order > 0 else "no order needed this week"),
        "Safety stock included": f"{safety[sku]:,.0f} units for {service:.0%} service",
        "Typical forecast error for this product": f"{sku_wape:.0%}",
    }
    st.subheader("AI briefing")
    if st.button(f"Write briefing with local AI ({OLLAMA_MODEL})", disabled=not ok,
                 help=None if ok else status):
        with st.spinner("The local model is writing. This can take a minute on a CPU..."):
            st.write_stream(stream_briefing(facts))
    elif not ok:
        st.caption(f"Local AI unavailable ({status}). Showing the rule-based summary instead.")
        st.markdown(fallback_briefing(facts))
    with st.expander("Facts given to the AI"):
        st.json(facts)

# ---------------- order plan ----------------
with tab_all:
    st.caption(f"Order-up-to levels for the coming week, at {lead_time}-week lead time and "
               f"{service:.0%} service. Subtract what you hold and have on order to get the order quantity.")
    plan = pd.DataFrame({
        "Product": info["description"],
        "Class": info["abc_class"] + info["xyz_class"],
        "Next week forecast": future.loc[future["h"] == 1].set_index("sku")["forecast"],
        f"Demand next {P} weeks": fc_p,
        "Safety stock": safety,
        "Order-up-to level": target,
        "Unit price (£)": info["avg_price"],
    })
    plan["Order-up-to value (£)"] = plan["Order-up-to level"] * plan["Unit price (£)"]
    plan = plan.round(0).sort_values("Order-up-to value (£)", ascending=False)
    plan.index.name = "SKU"
    st.dataframe(plan, width="stretch", height=520)
    st.download_button("Download order plan (CSV)", plan.to_csv().encode(), "order_plan.csv", "text/csv")

# ---------------- method ----------------
with tab_method:
    st.markdown(f"""
**Data.** {s['data']['transactions']:,} cleaned sales lines from a UK online gift retailer
(UCI Online Retail II, Dec 2009 to Dec 2011). Cancelled orders, returns, postage and fees are removed.
The 100 highest-revenue products that sell in at least 70% of weeks are modelled as weekly demand.

**Honest testing.** The last 13 weeks ({s['data']['holdout'][0]} to {s['data']['holdout'][1]}) are held
out. Every week in that period, each method forecasts 1 to 4 weeks ahead using only earlier data,
just as it would be used in real weekly ordering. Safety stock is sized from errors on the 13 weeks
*before* the holdout, so the inventory results are not tuned on the test period either.

**Methods compared.**
- *8-week moving average*: what many small retailers do in a spreadsheet.
- *Last year, trend-adjusted*: same week last year, scaled by recent growth.
- *LightGBM (AI model)*: one gradient-boosted model across all products, using recent sales,
  rolling averages, same-week-last-year, seasonality and Christmas closures. 80% ranges come from
  quantile models.

**From forecast to money.** Each week the policy orders up to: forecast demand over lead time + 1 week,
plus safety stock (z × forecast error). Replaying real demand gives the fill rate and the average stock
value. Comparing methods at the same fill rate shows how much working capital a better forecast releases.
""")
    share = d["importance"] / d["importance"].sum()
    imp = share.head(10)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=COLORS[AI], cornerradius=4),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    style(fig, 360, "")
    fig.update_layout(title=dict(text="What drives the forecast (top 10 features)", font=dict(size=14)),
                      hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
