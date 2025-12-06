from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import requests

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


# ========== 新闻结构体 ==========

@dataclass
class NewsItem:
    title: str
    description: str
    published_at: datetime
    url: str
    source: str

def dedup_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 里 'date' 列最多只出现一次：
      - 如果没有 'date'，原样返回
      - 如果有多个 'date'，只保留第一个，其余全部丢掉
    """
    if "date" not in df.columns:
        return df

    cols = list(df.columns)
    keep_mask = []
    seen_date = False
    for c in cols:
        if c != "date":
            keep_mask.append(True)
        else:
            if not seen_date:
                keep_mask.append(True)
                seen_date = True
            else:
                # 后面的所有 'date' 都扔掉
                keep_mask.append(False)

    return df.loc[:, keep_mask]

# ========== NewsAPI ==========

def _get_news_api_key() -> Optional[str]:
    if st is not None:
        try:
            key = st.secrets.get("NEWS_API_KEY", None)  # type: ignore
            if key:
                return key
        except Exception:
            pass
    return os.getenv("NEWS_API_KEY")


def search_stock_news(symbol: str, days: int = 7, max_results: int = 30) -> List[NewsItem]:
    api_key = _get_news_api_key()
    if not api_key:
        return []

    base_url = "https://newsapi.org/v2/everything"
    query = f"{symbol} stock OR 股票"

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    out: List[NewsItem] = []
    for art in data.get("articles", []):
        try:
            published_at = datetime.fromisoformat(
                art.get("publishedAt", "").replace("Z", "+00:00")
            )
        except Exception:
            published_at = datetime.utcnow()

        out.append(
            NewsItem(
                title=art.get("title", "") or "",
                description=art.get("description", "") or "",
                published_at=published_at,
                url=art.get("url", "") or "",
                source=(art.get("source") or {}).get("name", "") or "",
            )
        )
    return out


# ========== OHLCV → 技术特征 ==========

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data = dedup_date_column(data)

    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]

    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)

    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()]

    if "date" not in data.columns:
        data.insert(0, "date", pd.to_datetime(data.iloc[:, 0]))
    else:
        data["date"] = pd.to_datetime(data["date"])

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            data[col] = np.nan

    data = data.sort_values("date").reset_index(drop=True)

    # 收益率
    data["ret_1d"] = data["close"].pct_change()
    data["ret_5d"] = data["close"].pct_change(5)
    data["ret_10d"] = data["close"].pct_change(10)

    # 均线 + 波动
    for w in [5, 10, 20, 60]:
        data[f"ma_{w}"] = data["close"].rolling(w).mean()
        data[f"std_{w}"] = data["close"].rolling(w).std()
        data[f"vol_ma_{w}"] = data["volume"].rolling(w).mean()

    data["price_ma5_ratio"] = data["close"] / data["ma_5"]
    data["price_ma20_ratio"] = data["close"] / data["ma_20"]
    data["hl_range"] = (data["high"] - data["low"]) / data["close"]

    data = data.dropna().reset_index(drop=True)

    cols = ["date", "close"] + [
        c
        for c in data.columns
        if c
        not in [
            "open",
            "high",
            "low",
            "adj_close",
            "volume",
        ]
        and c not in ["ret_1d", "ret_5d", "ret_10d"]
    ]
    return data[cols]


# ========== RF 基线模型 ==========

@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    test_mape: float


class StockForecaster:
    def __init__(self, horizon: int = 5, random_state: int = 42) -> None:
        self.horizon = horizon
        self.model = RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, hist_df: pd.DataFrame) -> float:
        hist_df = dedup_date_column(hist_df)
        feat = make_features(hist_df).sort_values("date").reset_index(drop=True)
        X = feat.drop(columns=["date", "close"]).values
        y = feat["close"].values

        if len(y) < 30:
            self.fitted = False
            return float("nan")

        split = int(len(y) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        mape = float(np.mean(np.abs((y_pred - y_val) / y_val)))

        self.fitted = True
        return mape

    def forecast(self, hist_df: pd.DataFrame) -> ForecastResult:
        hist_df = dedup_date_column(hist_df)
        mape = self.fit(hist_df)
        if not self.fitted:
            empty = pd.DataFrame(columns=["date", "pred_close"])
            return ForecastResult(forecast_df=empty, test_mape=float("nan"))

        hist_df = hist_df.sort_values("date").reset_index(drop=True)
        last_date = hist_df["date"].iloc[-1]
        cur_df = hist_df.copy()

        dates: List[datetime] = []
        preds: List[float] = []

        for step in range(self.horizon):
            next_date = last_date + timedelta(days=1 + step)
            feat = make_features(cur_df)
            X_last = feat.drop(columns=["date", "close"]).values[-1:]

            pred = float(self.model.predict(X_last)[0])
            dates.append(next_date)
            preds.append(pred)

            new_row = {
                "date": next_date,
                "open": pred,
                "high": pred,
                "low": pred,
                "close": pred,
                "volume": cur_df["volume"].iloc[-1],
            }
            cur_df = pd.concat([cur_df, pd.DataFrame([new_row])], ignore_index=True)

        forecast_df = pd.DataFrame({"date": dates, "pred_close": preds})
        return ForecastResult(forecast_df=forecast_df, test_mape=mape)
