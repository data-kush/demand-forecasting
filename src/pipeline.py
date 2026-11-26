"""End-to-end run: clean data -> backtest 3 forecasting methods -> inventory simulation ->
final 13-week forecast. Saves everything the dashboard needs into artifacts/.

Run:  .venv\\Scripts\\python.exe -m src.pipeline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.data_prep import ROOT, build_weekly_panel, clean_transactions, load_raw
from src.forecasting import (LGBMForecaster, MovingAverageForecaster, SeasonalNaiveForecaster,
                             accuracy_table, rolling_origin)
from src.inventory import leadtime_error_sigma, service_curve, stock_needed_for

ART = ROOT / "artifacts"
HOLDOUT_WEEKS = 13      # the last quarter is never seen while building the models
VALIDATION_WEEKS = 13   # the quarter before it is used to size safety stock
MAX_HORIZON = 4         # supports lead times of 1-3 weeks with weekly ordering
FUTURE_WEEKS = 13
TARGET_FILL = 0.95


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def main():
    ART.mkdir(exist_ok=True)
    log("loading and cleaning transactions")
    sales = clean_transactions(load_raw())
    panel, info = build_weekly_panel(sales)
    sku_codes = {s: i for i, s in enumerate(info["sku"])}
    prices = info.set_index("sku")["avg_price"]

    weeks = sorted(panel["week"].unique())
    holdout_start, holdout_end = weeks[-HOLDOUT_WEEKS], weeks[-1]
    val_start, val_end = weeks[-HOLDOUT_WEEKS - VALIDATION_WEEKS], weeks[-HOLDOUT_WEEKS - 1]
    log(f"validation {val_start:%Y-%m-%d}..{val_end:%Y-%m-%d}, "
        f"holdout {holdout_start:%Y-%m-%d}..{holdout_end:%Y-%m-%d}")

    makers = {
        "LightGBM (AI model)": lambda: LGBMForecaster(sku_codes),
        "8-week moving average": MovingAverageForecaster,
        "Last year, trend-adjusted": SeasonalNaiveForecaster,
    }
    bt = {"validation": {}, "holdout": {}}
    for window, (start, end) in {"validation": (val_start, val_end),
                                 "holdout": (holdout_start, holdout_end)}.items():
        train = panel.loc[panel["week"] < start]
        for name, make in makers.items():
            log(f"{window}: {name}")
            model = make().fit(train)
            res = rolling_origin(model, panel, start, end, MAX_HORIZON)
            res["model"] = name
            bt[window][name] = res

    holdout_all = pd.concat(bt["holdout"].values(), ignore_index=True)
    acc = accuracy_table(holdout_all)
    log("holdout accuracy:\n" + acc.round(3).to_string())

    log("inventory simulation (lead times 1-3 weeks)")
    curves, sigmas_hold = [], {}
    for lt in (1, 2, 3):
        sig_val = {m: leadtime_error_sigma(b, lt + 1) for m, b in bt["validation"].items()}
        curves.append(service_curve(bt["holdout"], sig_val, lt, prices))
        sigmas_hold[lt] = {m: leadtime_error_sigma(b, lt + 1) for m, b in bt["holdout"].items()}
    curve = pd.concat(curves, ignore_index=True)
    needed = stock_needed_for(curve.loc[curve["lead_time"] == 2], TARGET_FILL)
    log(f"avg stock value needed for {TARGET_FILL:.0%} fill (2-week lead time):\n{needed.round(0).to_string()}")

    log("training final model on all data and forecasting the next 13 weeks")
    final = LGBMForecaster(sku_codes, with_intervals=True).fit(panel)
    future = final.forecast(panel, FUTURE_WEEKS)

    ai, ma = "LightGBM (AI model)", "8-week moving average"
    best_baseline = acc.drop(index=ai)["WAPE (1-4 weeks ahead)"].idxmin()
    summary = {
        "data": {"transactions": int(len(sales)), "products": int(panel["sku"].nunique()),
                 "weeks": len(weeks), "first_week": f"{weeks[0]:%Y-%m-%d}",
                 "last_week": f"{weeks[-1]:%Y-%m-%d}",
                 "holdout": [f"{holdout_start:%Y-%m-%d}", f"{holdout_end:%Y-%m-%d}"]},
        "wape": acc["WAPE (1-4 weeks ahead)"].to_dict(),
        "best_baseline": best_baseline,
        "error_reduction_vs_best_baseline": float(1 - acc.loc[ai, "WAPE (1-4 weeks ahead)"]
                                                  / acc.loc[best_baseline, "WAPE (1-4 weeks ahead)"]),
        "target_fill": TARGET_FILL,
        "stock_needed": needed.to_dict(),
        "stock_saving_vs_moving_average": float(needed[ma] - needed[ai]),
        "stock_saving_pct_vs_moving_average": float(1 - needed[ai] / needed[ma]),
    }

    panel.to_pickle(ART / "panel.pkl")
    info.to_pickle(ART / "sku_info.pkl")
    holdout_all.to_pickle(ART / "backtest_holdout.pkl")
    acc.to_csv(ART / "accuracy.csv")
    curve.to_pickle(ART / "service_curve.pkl")
    pd.to_pickle(sigmas_hold, ART / "sigma_holdout.pkl")
    future.to_pickle(ART / "future_forecast.pkl")
    final.feature_importance.to_pickle(ART / "feature_importance.pkl")
    (ART / "summary.json").write_text(json.dumps(summary, indent=2))
    log("done:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
