# rag_engine_stock_1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List

import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import requests   # 新增：用于调第三方新闻 API


REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 统一列名为小写，去空格
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # 常见别名兼容
    rename_map = {
        "datetime": "date",
        "time": "date",
        "adj close": "close",
        "adj_close": "close",
        "vol": "volume",
    }
    df = df.rename(columns=rename_map)
    return df


def load_ohlcv(file) -> pd.DataFrame:
    """
    支持 CSV / XLSX.
    需要包含 Date/Open/High/Low/Close/Volume.
    """
    if isinstance(file, str):
        path = file
        if path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    else:
        # streamlit UploadedFile
        name = file.name.lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

    df = _standardize_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 需要列 {REQUIRED_COLS}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 如果有整行都是 NaN 的，可以顺手丢掉
    df = df.dropna(subset=["close"])

    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    用过去信息构造特征（不泄露未来）
    """
    x = df.copy()
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["ret1"] = x["close"].pct_change()
    x["ret5"] = x["close"].pct_change(5)
    x["sma5"] = x["close"].rolling(5).mean()
    x["sma10"] = x["close"].rolling(10).mean()
    x["sma20"] = x["close"].rolling(20).mean()
    x["ema10"] = x["close"].ewm(span=10, adjust=False).mean()
    x["volatility10"] = x["ret1"].rolling(10).std()
    x["rsi14"] = rsi(x["close"], 14)
    x["vol_chg"] = x["volume"].pct_change()

    # 滞后特征（昨天、前天的收盘/回报）
    for lag in [1, 2, 3, 5]:
        x[f"close_lag{lag}"] = x["close"].shift(lag)
        x[f"ret1_lag{lag}"] = x["ret1"].shift(lag)

    return x


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    test_mape: Optional[float]


# ============ 新增：新闻数据结构 & 搜索函数 ============

@dataclass
class NewsItem:
    title: str
    url: str
    published_at: datetime
    source: str
    description: str


def _get_news_api_key() -> str:
    """
    从环境变量读取 NewsAPI 的 key.
    你可以在 shell 里 export:
        export NEWSAPI_API_KEY="xxxxx"
    """
    key = os.getenv("NEWSAPI_API_KEY")
    if not key:
        raise RuntimeError(
            "未配置新闻 API Key。请设置环境变量 NEWSAPI_API_KEY，例如：\n"
            'export NEWSAPI_API_KEY="你的 NewsAPI key"'
        )
    return key


def search_stock_news(query: str, max_results: int = 10) -> List[NewsItem]:
    """
    使用 NewsAPI 搜索与股票相关的最新新闻。

    query: 股票代码或公司名，如 "600519" 或 "贵州茅台" 或 "AAPL"
    max_results: 返回的最大新闻数
    """
    api_key = _get_news_api_key()

    url = "https://newsapi.org/v2/everything"
    # 这里可以根据需要调节时间区间 / 语言等
    params = {
        "q": query,
        "language": "zh",      # 主要中文新闻；可以改成 "en" 或不填
        "sortBy": "publishedAt",
        "pageSize": max_results,
    }
    headers = {"X-Api-Key": api_key}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", [])
    results: List[NewsItem] = []

    for art in articles:
        title = art.get("title") or ""
        url_ = art.get("url") or ""
        desc = art.get("description") or ""
        source = (art.get("source") or {}).get("name") or ""
        published_str = art.get("publishedAt") or ""

        try:
            published_dt = (
                datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if published_str
                else datetime.utcnow()
            )
        except Exception:
            published_dt = datetime.utcnow()

        results.append(
            NewsItem(
                title=title,
                url=url_,
                published_at=published_dt,
                source=source,
                description=desc,
            )
        )

    return results

# ================== 原有预测类不变 ==================

class StockForecaster:
    """
    一个离线、轻量的“短期趋势预测”Demo：
    - 特征工程
    - 随机森林回归预测下一日 Close
    - 递归多步预测未来 horizon 天
    """

    def __init__(self, n_estimators: int = 400, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> Optional[float]:
        feat = make_features(df)
        feat = feat.dropna().reset_index(drop=True)

        # 预测目标：下一天 close
        feat["target"] = feat["close"].shift(-1)
        feat = feat.dropna()

        y = feat["target"].values
        # 去掉非特征列
        drop_cols = ["date", "target"]
        self.feature_cols = [c for c in feat.columns if c not in drop_cols]

        X = feat[self.feature_cols].values

        # 简单时间序列切分：最后 20% 当测试集
        split = int(len(feat) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        self.fitted = True

        # 测试集 MAPE（仅做参考）
        if len(X_test) > 0:
            pred = self.model.predict(X_test)
            mape = float(np.mean(np.abs((y_test - pred) / (y_test + 1e-9))) * 100)
            return mape
        return None

    def _next_day_features(self, hist_df: pd.DataFrame) -> np.ndarray:
        feat = make_features(hist_df).dropna()
        last_row = feat.iloc[-1]
        return last_row[self.feature_cols].values.reshape(1, -1)

    def predict_future(self, df: pd.DataFrame, horizon: int = 5) -> ForecastResult:
        if not self.fitted:
            mape = self.fit(df)
        else:
            mape = None

        hist = df.copy()
        preds = []
        dates = []

        last_date = hist["date"].iloc[-1]

        for i in range(horizon):
            X_next = self._next_day_features(hist)
            next_close = float(self.model.predict(X_next)[0])

            next_date = last_date + pd.Timedelta(days=1)
            last_date = next_date

            preds.append(next_close)
            dates.append(next_date)

            # 为了递归预测，构造“下一天的伪数据”
            # open/high/low 这里用 close 近似（Demo 简化）
            new_row = {
                "date": next_date,
                "open": next_close,
                "high": next_close,
                "low": next_close,
                "close": next_close,
                "volume": hist["volume"].iloc[-1],  # volume 用最近值
            }
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

        forecast_df = pd.DataFrame({"date": dates, "pred_close": preds})
        return ForecastResult(forecast_df=forecast_df, test_mape=mape)
