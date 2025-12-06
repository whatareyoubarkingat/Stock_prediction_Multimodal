# rag_engine_stock_1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import requests

# 可选导入 streamlit，用于在 Cloud 上读取 st.secrets
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


# ==========================
# 数据结构：新闻
# ==========================

@dataclass
class NewsItem:
    title: str
    description: str
    published_at: datetime
    url: str
    source: str


# ==========================
# 新闻搜索（示例：NewsAPI）
# ==========================

def _get_news_api_key() -> Optional[str]:
    # 优先 st.secrets，其次环境变量
    if st is not None:
        try:
            key = st.secrets.get("NEWS_API_KEY", None)  # type: ignore
            if key:
                return key
        except Exception:
            pass
    return os.getenv("NEWS_API_KEY")


def search_stock_news(
    symbol: str,
    days: int = 7,
    max_results: int = 30,
) -> List[NewsItem]:
    """
    用 NewsAPI 搜股票相关新闻（你也可以改成别的源）。
    """
    api_key = _get_news_api_key()
    if not api_key:
        return []

    base_url = "https://newsapi.org/v2/everything"
    # 用 ticker + 股票 关键字做个简单 query
    q = f"{symbol} stock OR 股票"

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "q": q,
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
    except Exception as e:
        print("[search_stock_news] failed:", repr(e))
        return []

    articles = data.get("articles", [])
    out: List[NewsItem] = []
    for art in articles:
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
                source=(art.get("source", {}) or {}).get("name", "") or "",
            )
        )

    return out


# ==========================
# OHLCV -> 技术特征
# ==========================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: 包含列 [date, open, high, low, close, volume]
    输出: 按日期排序后的特征 DataFrame，包含:
      - date
      - close
      - 一系列技术指标（MA, 波动率, 成交量均线, 高频 return 等）
    """
    data = df.copy()

    # 统一列名
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

    if "date" not in data.columns:
        data.insert(0, "date", pd.to_datetime(data.iloc[:, 0]))
    else:
        data["date"] = pd.to_datetime(data["date"])

    # 填补必需列
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            data[col] = np.nan

    data = data.sort_values("date").reset_index(drop=True)

    # 收益率
    data["ret_1d"] = data["close"].pct_change()
    data["ret_5d"] = data["close"].pct_change(5)
    data["ret_10d"] = data["close"].pct_change(10)

    # 均线
    for w in [5, 10, 20, 60]:
        data[f"ma_{w}"] = data["close"].rolling(w).mean()
        data[f"std_{w}"] = data["close"].rolling(w).std()
        data[f"vol_ma_{w}"] = data["volume"].rolling(w).mean()

    # 价格/均线比
    data["price_ma5_ratio"] = data["close"] / data["ma_5"]
    data["price_ma20_ratio"] = data["close"] / data["ma_20"]

    # 波动率 proxy
    data["hl_range"] = (data["high"] - data["low"]) / data["close"]

    # 丢掉前期 NaN
    data = data.dropna().reset_index(drop=True)

    # 保留关心列
    feature_cols = ["date", "close"] + [
        c for c in data.columns if c not in ["open", "high", "low", "adj_close", "volume"]
        and c not in ["ret_1d", "ret_5d", "ret_10d"]  # 你也可以保留这些 return
    ]
    return data[feature_cols]


# ==========================
# 简单 RF 基线预测器
# ==========================

@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    test_mape: float


class StockForecaster:
    """
    只用数值特征的随机森林基线，用于对比。
    """

    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
    ) -> None:
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
        feat = make_features(hist_df)
        feat = feat.sort_values("date").reset_index(drop=True)

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
        mape = self.fit(hist_df)
        if not self.fitted:
            forecast_df = pd.DataFrame(columns=["date", "pred_close"], data=[])
            return ForecastResult(forecast_df=forecast_df, test_mape=float("nan"))

        hist_df = hist_df.sort_values("date").reset_index(drop=True)
        last_date = hist_df["date"].iloc[-1]
        cur_df = hist_df.copy()

        dates: List[datetime] = []
        preds: List[float] = []

        for step in range(self.horizon):
            next_date = last_date + timedelta(days=1 + step)
            feat = make_features(cur_df)
            X_last = feat.drop(columns=["date", "close"]).values[-1:].copy()
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
