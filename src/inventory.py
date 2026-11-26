"""Turn forecasts into stock decisions, and measure what better forecasts are worth in money."""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


def z_for_service(level: float) -> float:
    return NormalDist().inv_cdf(level)


def leadtime_error_sigma(bt: pd.DataFrame, periods: int) -> pd.Series:
    """Per-product RMSE of the forecast of total demand over the next `periods` weeks.
    Measured out-of-sample, so the safety stock reflects how wrong the model really is."""
    d = bt.loc[bt["h"] <= periods]
    agg = d.groupby(["sku", "origin"]).agg(fc=("forecast", "sum"), act=("qty", "sum"),
                                           n=("qty", "count"))
    agg = agg.loc[agg["n"] == periods]
    return np.sqrt(((agg["fc"] - agg["act"]) ** 2).groupby("sku").mean())


def order_up_to_targets(bt: pd.DataFrame, sigma: pd.Series, lead_time: int, z: float) -> pd.DataFrame:
    """Weekly review: target stock = forecast demand over lead time + 1 week, plus safety stock."""
    p = lead_time + 1
    fc = bt.loc[bt["h"] <= p].groupby(["sku", "origin"])["forecast"].sum().unstack("origin")
    return fc.add(z * sigma.reindex(fc.index).fillna(0), axis=0)


def simulate(bt: pd.DataFrame, sigma: pd.Series, lead_time: int, z: float,
             prices: pd.Series) -> dict:
    """Replay the holdout period week by week: order up to target, receive orders after the lead
    time, serve real demand, and count lost sales and stock held."""
    targets = order_up_to_targets(bt, sigma, lead_time, z)
    origins = targets.columns
    demand = (bt.loc[bt["h"] == 1].pivot(index="sku", columns="origin", values="qty")
                .reindex(index=targets.index, columns=origins).to_numpy())
    S = targets.to_numpy()
    n, T = S.shape
    on_hand = S[:, 0].copy()
    pipeline = np.zeros((n, T + lead_time + 1))
    end_stock = np.zeros((n, T))
    sold = np.zeros((n, T))
    for t in range(T):
        on_hand += pipeline[:, t]
        position = on_hand + pipeline[:, t + 1:].sum(axis=1)
        pipeline[:, t + lead_time] += np.maximum(0, S[:, t] - position)
        sold[:, t] = np.minimum(on_hand, demand[:, t])
        on_hand -= sold[:, t]
        end_stock[:, t] = on_hand
    price = prices.reindex(targets.index).to_numpy()
    demand_value = (demand * price[:, None]).sum()
    sold_value = (sold * price[:, None]).sum()
    return {
        "fill_rate": sold.sum() / demand.sum(),
        "avg_stock_value": (end_stock.mean(axis=1) * price).sum(),
        "lost_sales_value": demand_value - sold_value,
        "per_sku": pd.DataFrame({"fill_rate": sold.sum(1) / np.maximum(demand.sum(1), 1e-9),
                                 "avg_stock_units": end_stock.mean(1)}, index=targets.index),
    }


def service_curve(bt_by_model: dict[str, pd.DataFrame], sigma_by_model: dict[str, pd.Series],
                  lead_time: int, prices: pd.Series, z_grid=None) -> pd.DataFrame:
    """Fill rate vs average stock value for each forecasting method across safety-stock levels."""
    z_grid = np.round(np.linspace(0, 3, 31), 2) if z_grid is None else z_grid
    rows = []
    for model, bt in bt_by_model.items():
        for z in z_grid:
            r = simulate(bt, sigma_by_model[model], lead_time, z, prices)
            rows.append({"model": model, "lead_time": lead_time, "z": z,
                         "fill_rate": r["fill_rate"], "avg_stock_value": r["avg_stock_value"],
                         "lost_sales_value": r["lost_sales_value"]})
    return pd.DataFrame(rows)


def stock_needed_for(curve: pd.DataFrame, target_fill: float) -> pd.Series:
    """Smallest average stock value that reaches the target fill rate, per model (NaN if never)."""
    out = {}
    for model, c in curve.groupby("model"):
        ok = c.loc[c["fill_rate"] >= target_fill]
        out[model] = ok["avg_stock_value"].min() if len(ok) else np.nan
    return pd.Series(out)
