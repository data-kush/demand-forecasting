"""Business analysis on top of the forecasts, plus compact fact sheets for the local AI."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

AI, MA = "LightGBM (AI model)", "8-week moving average"


def growth_table(panel: pd.DataFrame, info: pd.DataFrame, weeks: int = 13) -> pd.DataFrame:
    """Units in the last `weeks` weeks vs the same weeks a year earlier, per product."""
    last = panel["week"].max()
    cur = panel.loc[panel["week"] > last - pd.Timedelta(weeks=weeks)].groupby("sku")["qty"].sum()
    ly = panel.loc[(panel["week"] > last - pd.Timedelta(weeks=52 + weeks))
                   & (panel["week"] <= last - pd.Timedelta(weeks=52))].groupby("sku")["qty"].sum()
    out = pd.DataFrame({"product": info["description"].str.title(), "units_recent": cur, "units_last_year": ly})
    out = out.loc[out["units_last_year"] > 0]
    out["growth"] = out["units_recent"] / out["units_last_year"] - 1
    return out.sort_values("growth", ascending=False)


def watchlist(panel: pd.DataFrame, future: pd.DataFrame, info: pd.DataFrame,
              threshold: float = 0.4, min_units: float = 50) -> pd.DataFrame:
    """Products whose next 4 weeks are forecast well above or below the SAME weeks last year.
    Comparing with last year removes the normal seasonal swing (e.g. the Christmas closure), so
    what is left is genuinely unusual."""
    nxt_rows = future.loc[future["h"] <= 4]
    nxt = nxt_rows.groupby("sku")["forecast"].sum()
    ly_weeks = pd.Index(nxt_rows["week"].unique()) - pd.Timedelta(weeks=52)
    ly = panel.loc[panel["week"].isin(ly_weeks)].groupby("sku")["qty"].sum()
    out = pd.DataFrame({"product": info["description"].str.title(), "last_year_units": ly,
                        "forecast_units": nxt})
    out = out.loc[out["last_year_units"] >= min_units]
    out["change"] = out["forecast_units"] / out["last_year_units"] - 1
    out = out.loc[out["change"].abs() >= threshold]
    out["signal"] = np.where(out["change"] > 0, "▲ Stock more", "▼ Stock less")
    out.index.name = "SKU"
    return out.sort_values("change", key=np.abs, ascending=False)


def product_wape(bt: pd.DataFrame, model: str = AI) -> pd.Series:
    d = bt.loc[bt["model"] == model].dropna(subset=["qty"])
    g = d.assign(ae=(d["forecast"] - d["qty"]).abs()).groupby("sku")
    return g["ae"].sum() / g["qty"].sum().clip(lower=1)


def accuracy_by_segment(bt: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Forecast error (WAPE) by revenue class x demand pattern, for the AI model and the spreadsheet."""
    d = bt.dropna(subset=["qty"]).merge(info[["abc_class", "xyz_class"]], left_on="sku", right_index=True)
    d["ae"] = (d["forecast"] - d["qty"]).abs()
    g = d.groupby(["abc_class", "xyz_class", "model"])
    w = (g["ae"].sum() / g["qty"].sum()).unstack("model")
    w["products"] = d.loc[d["model"] == AI].groupby(["abc_class", "xyz_class"])["sku"].nunique()
    return w.rename(columns={AI: "ai_error", MA: "spreadsheet_error"})[["products", "ai_error", "spreadsheet_error"]]


def seasonal_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Total weekly units by ISO week, one column per year."""
    tot = panel.groupby("week")["qty"].sum().reset_index()
    iso = tot["week"].dt.isocalendar()
    tot["year"], tot["week_of_year"] = iso["year"].astype(int), iso["week"].astype(int)
    return tot.pivot_table(index="week_of_year", columns="year", values="qty", aggfunc="sum")


def outlook(panel: pd.DataFrame, future: pd.DataFrame) -> dict:
    """Next 13 weeks forecast vs the same weeks last year, across all products."""
    nxt = future.groupby("week")["forecast"].sum()
    ly_weeks = nxt.index - pd.Timedelta(weeks=52)
    ly = panel.loc[panel["week"].isin(ly_weeks)].groupby("week")["qty"].sum()
    return {"forecast_units": float(nxt.sum()), "last_year_units": float(ly.sum()),
            "change": float(nxt.sum() / ly.sum() - 1) if ly.sum() else float("nan"),
            "peak_week": nxt.idxmax(), "peak_units": float(nxt.max())}


def _names(df: pd.DataFrame, col: str, fmt: str, n: int = 5) -> str:
    return "; ".join(f"{r['product']} ({format(r[col], fmt)})" for _, r in df.head(n).iterrows()) or "none"


def portfolio_context(s: dict, settings: dict, needed: pd.Series, panel, future, info, bt,
                      plan: pd.DataFrame) -> str:
    """A compact, numbers-only fact sheet about the whole business for the local AI (~400 words)."""
    g = growth_table(panel, info)
    wl = watchlist(panel, future, info)
    seg = accuracy_by_segment(bt, info)
    o = outlook(panel, future)
    acc_abc = seg.groupby(level=0).apply(lambda x: np.average(x["ai_error"], weights=x["products"]))
    conc = info.assign(product=info["description"].str.title()).sort_values("top3_buyer_share", ascending=False)
    to_order = plan.loc[plan["Order now"] > 0]
    saving = (f"£{needed[MA] - needed[AI]:,.0f} ({1 - needed[AI] / needed[MA]:.0%}) less average stock than "
              "the spreadsheet method" if np.isfinite(needed.get(AI, np.nan)) and np.isfinite(needed.get(MA, np.nan))
              else "target service level not reachable in the test")
    lines = [
        f"Business: UK online gift retailer, {s['data']['products']} top products, weekly data "
        f"{s['data']['first_week']} to {s['data']['last_week']}. Many customers are wholesale buyers.",
        f"Planning settings: supplier lead time {settings['lead_time']} weeks, target "
        f"{settings['service']:.0%} of customers served from stock.",
        f"Test results (13 unseen weeks): AI forecast error {s['wape'][AI]:.0%} vs spreadsheet "
        f"{s['wape'][MA]:.0%}. At the target service level the AI needs {saving}.",
        f"Next 13 weeks: forecast {o['forecast_units']:,.0f} units vs {o['last_year_units']:,.0f} in the same "
        f"weeks last year ({o['change']:+.0%}). Busiest week: {o['peak_week']:%d %b %Y} "
        f"({o['peak_units']:,.0f} units).",
        f"GROWING products (sales UP in the last 13 weeks vs last year): {_names(g, 'growth', '+.0%')}.",
        f"DECLINING products (sales DOWN vs last year): {_names(g.iloc[::-1], 'growth', '+.0%')}.",
        "Products forecast ABOVE the same 4 weeks last year, so STOCK MORE: "
        f"{_names(wl.loc[wl['change'] > 0], 'change', '+.0%')}.",
        "Products forecast BELOW the same 4 weeks last year, so STOCK LESS: "
        f"{_names(wl.loc[wl['change'] < 0], 'change', '+.0%')}.",
        "AI forecast error by revenue class: " + ", ".join(f"{k} {v:.0%}" for k, v in acc_abc.items())
        + " (A = top sellers).",
        f"Products most dependent on their top 3 buyers: {_names(conc, 'top3_buyer_share', '.0%', 4)}.",
        f"This week's purchase order: {len(to_order)} products, {to_order['Order now'].sum():,.0f} units, "
        f"£{to_order['Order value'].sum():,.0f}. Largest lines: "
        + "; ".join(f"{r['Product']} {r['Order now']:,.0f} units" for _, r in to_order.head(4).iterrows()) + ".",
    ]
    return "\n".join(f"- {l}" for l in lines)


def match_products(question: str, info: pd.DataFrame, limit: int = 2) -> list[str]:
    """Find products a question mentions, by SKU code or by words in the product name."""
    q = question.lower()
    hits = [sku for sku in info.index if sku.lower() in q]
    if hits:
        return hits[:limit]
    words = {w for w in re.findall(r"[a-z]{4,}", q)} - {"what", "which", "should", "order", "much", "many",
                                                       "stock", "week", "weeks", "will", "next", "sell",
                                                       "sales", "demand", "product", "products", "forecast"}
    if not words:
        return []
    scores = info["description"].str.lower().apply(lambda d: len(words & set(re.findall(r"[a-z]{4,}", d))))
    best = scores[scores > 0].sort_values(ascending=False)
    return list(best.index[:limit])
