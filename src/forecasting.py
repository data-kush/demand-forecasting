"""Forecasting models: a global LightGBM model plus two baselines a business would use today."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

LAGS = [1, 2, 3, 4, 6, 8, 13, 26, 52]
WINDOWS = [4, 8, 13, 26]
HISTORY_NEEDED = 56  # weeks of history the features look back over (lag 52 + margin)
QUANTILES = {"p10": 0.1, "p90": 0.9}


def calendar_features(weeks: pd.Series) -> pd.DataFrame:
    weeks = pd.to_datetime(weeks)
    iso = weeks.dt.isocalendar()
    md = weeks.dt.month * 100 + weeks.dt.day
    # This retailer winds down the week before Christmas and closes the week after it.
    xmas = np.select([(md >= 1225) & (md <= 1231), (md >= 1218) & (md <= 1224)], [2, 1], 0)
    return pd.DataFrame({
        "week_of_year": iso["week"].astype(int).to_numpy(),
        "month": weeks.dt.month.to_numpy(),
        "xmas_period": xmas,
        "weeks_to_xmas": ((pd.to_datetime(weeks.dt.year.astype(str) + "-12-25") - weeks).dt.days // 7).to_numpy(),
    }, index=weeks.index)


def make_features(panel: pd.DataFrame, sku_codes: dict[str, int]) -> pd.DataFrame:
    """Lag, rolling and calendar features. Only uses demand from earlier weeks, so rows with
    unknown (NaN) qty can be scored for forecasting."""
    df = panel[["sku", "week", "qty"]].sort_values(["sku", "week"]).reset_index(drop=True)
    df["y"] = np.log1p(df["qty"])
    g = df.groupby("sku", sort=False)["y"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    prev = g.shift(1)
    gp = prev.groupby(df["sku"], sort=False)
    for w in WINDOWS:
        df[f"rmean_{w}"] = gp.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["rstd_8"] = gp.transform(lambda s: s.rolling(8, min_periods=2).std())
    df["zero_share_8"] = (df.groupby("sku", sort=False)["qty"].shift(1).eq(0)
                          .groupby(df["sku"], sort=False)
                          .transform(lambda s: s.rolling(8, min_periods=1).mean()))
    df["yoy_ratio"] = df["rmean_4"] - df["lag_52"]
    df = pd.concat([df, calendar_features(df["week"])], axis=1)
    df["sku_code"] = df["sku"].map(sku_codes).astype("int32")
    return df


FEATURES = ([f"lag_{l}" for l in LAGS] + [f"rmean_{w}" for w in WINDOWS]
            + ["rstd_8", "zero_share_8", "yoy_ratio", "week_of_year", "month",
               "xmas_period", "weeks_to_xmas", "sku_code"])

BASE_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   verbose=-1, random_state=42)


class LGBMForecaster:
    """One model across all products, forecasting recursively week by week."""

    name = "LightGBM (AI model)"

    def __init__(self, sku_codes: dict[str, int], with_intervals: bool = False,
                 objective: str = "log"):
        self.sku_codes = sku_codes
        self.with_intervals = with_intervals
        # "tweedie" models raw units (suits lumpy, zero-heavy demand and is unbiased in units);
        # "log" fits log1p(units) with squared error.
        self.objective = objective
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def _to_units(self, key: str, pred: np.ndarray) -> np.ndarray:
        if key == "mean" and self.objective == "tweedie":
            return pred
        return np.expm1(pred)

    def fit(self, panel: pd.DataFrame) -> "LGBMForecaster":
        feats = make_features(panel, self.sku_codes).dropna(subset=["lag_1", "y"])
        X, y = feats[FEATURES], feats["y"]
        if self.objective == "tweedie":
            mean_model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3,
                                           **BASE_PARAMS).fit(X, feats["qty"],
                                                              categorical_feature=["sku_code"])
        else:
            mean_model = lgb.LGBMRegressor(objective="regression", **BASE_PARAMS).fit(
                X, y, categorical_feature=["sku_code"])
        self.models["mean"] = mean_model
        if self.with_intervals:
            for key, alpha in QUANTILES.items():
                self.models[key] = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                                                     **BASE_PARAMS).fit(
                    X, y, categorical_feature=["sku_code"])
        return self

    @property
    def feature_importance(self) -> pd.Series:
        m = self.models["mean"]
        return pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        hist = history[["sku", "week", "qty"]].copy()
        last = hist["week"].max()
        hist = hist.loc[hist["week"] > last - pd.Timedelta(weeks=HISTORY_NEEDED)]
        skus = hist["sku"].unique()
        out = []
        for h in range(1, horizon + 1):
            wk = last + pd.Timedelta(weeks=h)
            hist = pd.concat([hist, pd.DataFrame({"sku": skus, "week": wk, "qty": np.nan})],
                             ignore_index=True)
            feats = make_features(hist, self.sku_codes)
            row = feats.loc[feats["week"] == wk].set_index("sku").reindex(skus)
            res = pd.DataFrame({"sku": skus, "week": wk, "h": h})
            for key, model in self.models.items():
                res[key] = np.clip(self._to_units(key, model.predict(row[FEATURES])), 0, None)
            if self.with_intervals:  # keep the band ordered and around the point forecast
                res["p10"] = np.minimum(res["p10"], res["mean"])
                res["p90"] = np.maximum(res["p90"], res["mean"])
            out.append(res)
            # feed the point forecast back in as next week's "history"
            hist.loc[hist["week"] == wk, "qty"] = res.set_index("sku")["mean"].reindex(
                hist.loc[hist["week"] == wk, "sku"]).to_numpy()
        return pd.concat(out, ignore_index=True).rename(columns={"mean": "forecast"})


class MovingAverageForecaster:
    """What many small retailers do: average of the last 8 weeks."""

    name = "8-week moving average"

    def __init__(self, window: int = 8):
        self.window = window

    def fit(self, panel):
        return self

    def forecast(self, history, horizon):
        last = history["week"].max()
        recent = history.loc[history["week"] > last - pd.Timedelta(weeks=self.window)]
        avg = recent.groupby("sku")["qty"].mean()
        return pd.DataFrame([{"sku": s, "week": last + pd.Timedelta(weeks=h), "h": h, "forecast": v}
                             for h in range(1, horizon + 1) for s, v in avg.items()])


class SeasonalNaiveForecaster:
    """'Same week last year', scaled by how this year's last 8 weeks compare to last year's."""

    name = "Last year, trend-adjusted"

    def fit(self, panel):
        return self

    def forecast(self, history, horizon):
        h = history.set_index(["sku", "week"])["qty"]
        last = history["week"].max()
        weeks = history["week"]
        recent = history.loc[weeks > last - pd.Timedelta(weeks=8)].groupby("sku")["qty"].sum()
        ly = history.loc[(weeks > last - pd.Timedelta(weeks=60)) & (weeks <= last - pd.Timedelta(weeks=52))]
        ly = ly.groupby("sku")["qty"].sum()
        ratio = (recent / ly.replace(0, np.nan)).clip(0.25, 4).fillna(1.0)
        rows = []
        for step in range(1, horizon + 1):
            wk = last + pd.Timedelta(weeks=step)
            src = wk - pd.Timedelta(weeks=52)
            for s in ratio.index:
                base = h.get((s, src), np.nan)
                if np.isnan(base):
                    base = recent.get(s, 0) / 8
                rows.append({"sku": s, "week": wk, "h": step, "forecast": base * ratio[s]})
        return pd.DataFrame(rows)


def rolling_origin(forecaster, panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                   horizon: int) -> pd.DataFrame:
    """Every week from start to end, forecast the next `horizon` weeks using only earlier data.
    This mirrors how the forecast would actually be used in a weekly replenishment cycle."""
    results = []
    for origin in pd.date_range(start, end, freq="W-MON"):
        hist = panel.loc[panel["week"] < origin]
        fc = forecaster.forecast(hist, horizon)
        fc["origin"] = origin
        results.append(fc)
    out = pd.concat(results, ignore_index=True)
    return out.merge(panel[["sku", "week", "qty"]], on=["sku", "week"], how="left")


def accuracy_table(bt: pd.DataFrame) -> pd.DataFrame:
    """WAPE = total absolute error / total actual units (lower is better)."""
    d = bt.dropna(subset=["qty"]).copy()
    d["abs_err"] = (d["forecast"] - d["qty"]).abs()
    d["err"] = d["forecast"] - d["qty"]
    g = d.groupby("model")
    return pd.DataFrame({
        "WAPE (1 week ahead)": d[d.h == 1].groupby("model").apply(
            lambda x: x.abs_err.sum() / x.qty.sum(), include_groups=False),
        "WAPE (1-4 weeks ahead)": g.apply(lambda x: x.abs_err.sum() / x.qty.sum(), include_groups=False),
        "Bias": g.apply(lambda x: x.err.sum() / x.qty.sum(), include_groups=False),
    })
