"""Load the UCI Online Retail II transactions and turn them into a clean weekly demand panel."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_CACHE = ROOT / "data" / "processed" / "raw_combined.pkl"

# Real products are 5 digits plus an optional colour/size suffix (e.g. 85123A).
# Everything else is postage, fees, manual adjustments, gift vouchers, etc.
PRODUCT_CODE = re.compile(r"^\d{5}[A-Z]{0,2}$")


def load_raw() -> pd.DataFrame:
    """Read both Excel sheets once, then reuse a fast pickle cache."""
    if RAW_CACHE.exists():
        return pd.read_pickle(RAW_CACHE)
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(RAW_CACHE)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Customer ID": "customer_id", "InvoiceDate": "date",
                            "StockCode": "sku", "Quantity": "qty", "Price": "price",
                            "Invoice": "invoice", "Description": "description",
                            "Country": "country"})
    df["invoice"] = df["invoice"].astype(str)
    df["sku"] = df["sku"].astype(str).str.upper().str.strip()
    df["description"] = df["description"].astype(str).str.strip()

    is_cancel = df["invoice"].str.startswith("C")

    # Orders that were later cancelled are not real demand: drop a sale when the same
    # customer cancelled exactly the same quantity of the same product.
    cancels = df.loc[is_cancel & df["customer_id"].notna(), ["customer_id", "sku", "qty"]].copy()
    cancels["qty"] = -cancels["qty"]
    cancel_keys = set(map(tuple, cancels.drop_duplicates().to_numpy()))

    sales = df.loc[~is_cancel & (df["qty"] > 0) & (df["price"] > 0)
                   & df["sku"].str.match(PRODUCT_CODE)].copy()
    keys = zip(sales["customer_id"], sales["sku"], sales["qty"])
    sales = sales.loc[[k not in cancel_keys for k in keys]]

    sales["revenue"] = sales["qty"] * sales["price"]
    sales["week"] = sales["date"].dt.to_period("W-SUN").dt.start_time  # weeks start Monday
    return sales


def build_weekly_panel(sales: pd.DataFrame, n_skus: int = 100,
                       min_active_share: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly units per product for the top-revenue products that sell regularly.

    Returns (panel, sku_info). The panel has one row per (sku, week) with zeros filled in.
    """
    # The final week in the file is only partly recorded, so drop it.
    last_full_week = sales["week"].max() - pd.Timedelta(weeks=1)
    sales = sales.loc[sales["week"] <= last_full_week]

    weekly = (sales.groupby(["sku", "week"], as_index=False)
                   .agg(qty=("qty", "sum"), revenue=("revenue", "sum"),
                        orders=("invoice", "nunique")))

    weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
    active_share = weekly.groupby("sku")["week"].nunique() / len(weeks)
    revenue = weekly.groupby("sku")["revenue"].sum()
    eligible = revenue[active_share >= min_active_share].sort_values(ascending=False)
    chosen = eligible.index[:n_skus]

    grid = pd.MultiIndex.from_product([chosen, weeks], names=["sku", "week"]).to_frame(index=False)
    panel = grid.merge(weekly, on=["sku", "week"], how="left")
    panel[["qty", "revenue", "orders"]] = panel[["qty", "revenue", "orders"]].fillna(0)

    sub = sales.loc[sales["sku"].isin(chosen)]
    info = (sub.groupby("sku")
               .agg(description=("description", lambda s: s.mode().iat[0]),
                    avg_price=("price", "median"),
                    total_revenue=("revenue", "sum"),
                    total_units=("qty", "sum"),
                    customers=("customer_id", "nunique"))
               .reindex(chosen))
    info["revenue_share"] = info["total_revenue"] / info["total_revenue"].sum()
    cum = info["revenue_share"].cumsum()
    info["abc_class"] = np.select([cum <= 0.7, cum <= 0.9], ["A", "B"], default="C")
    stats = panel.groupby("sku")["qty"].agg(["mean", "std"])
    info["demand_cv"] = (stats["std"] / stats["mean"]).reindex(chosen)
    info["xyz_class"] = pd.cut(info["demand_cv"], [-np.inf, 0.5, 1.0, np.inf],
                               labels=["X", "Y", "Z"]).astype(str)
    return panel.sort_values(["sku", "week"]).reset_index(drop=True), info.reset_index()


if __name__ == "__main__":
    raw = load_raw()
    sales = clean_transactions(raw)
    panel, info = build_weekly_panel(sales)
    print(f"clean sale lines: {len(sales):,}  |  products modelled: {panel.sku.nunique()}  "
          f"|  weeks: {panel.week.nunique()} ({panel.week.min():%Y-%m-%d} to {panel.week.max():%Y-%m-%d})")
    total = panel.groupby("week")["qty"].sum()
    print("\nLowest-volume weeks (holiday closures?):")
    print(total.nsmallest(6).to_string())
    print("\nTop products:")
    print(info[["sku", "description", "total_revenue", "abc_class", "xyz_class"]].head(8).to_string())
