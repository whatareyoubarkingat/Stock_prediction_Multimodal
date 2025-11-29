# rag_engine_stock_1.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
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


# ============================================================
# 基本配置
# ============================================================

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


# ============================================================
# 新闻相关的数据结构 & 工具
# ============================================================

@dataclass
class NewsItem:
    title: str
    description: Optional[str]
    url: str
    source: str
    published_at: datetime
    content: Optional[str] = None


def _get_news_api_key() -> str:
    """
    先从 Streamlit Cloud 的 secrets 里拿 NEWS_API_KEY，
    如果没有，再退回到环境变量。
    """
    # 1) 先看 st.secrets
    if st is not None:
        for key in ("NEWS_API_KEY", "NEWSAPI_KEY", "NEWS_API_TOKEN"):
            try:
                if key in st.secrets:
                    return str(st.secrets[key])
            except Exception:
                # 本地没有 st.secrets 之类的情况直接忽略
                pass

    # 2) 再看环境变量
    for key in ("NEWS_API_KEY", "NEWSAPI_KEY", "NEWS_API_TOKEN"):
        val = os.getenv(key)
        if val:
            return val

    # 3) 都没有就抛错（外层会 catch，不会让整个 app 崩掉）
    raise RuntimeError(
        "未找到 NewsAPI API Key，请在环境变量或 Streamlit secrets 中设置 "
        "NEWS_API_KEY / NEWSAPI_KEY / NEWS_API_TOKEN。"
    )


def search_stock_news(query: str, max_results: int = 10) -> List[NewsItem]:
    """
    使用 NewsAPI 搜索与股票相关的最新新闻。

    query: 股票代码或公司名，如 "600519" / "贵州茅台" / "AAPL"
    max_results: 返回的最大新闻数
    """
    api_key = _get_news_api_key()
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "zh",      # 主要中文新闻；必要时改成 "en" 或去掉
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", []) or []
    items: List[NewsItem] = []

    for art in articles:
        title = art.get("title") or ""
        if not title:
            continue

        description = art.get("description")
        url_ = art.get("url") or ""
        source_name = (art.get("source") or {}).get("name") or ""
        published_at_str = art.get("publishedAt") or ""
        try:
            published_at = datetime.fromisoformat(
                published_at_str.replace("Z", "+00:00")
            )
        except Exception:
            published_at = datetime.utcnow()

        content = art.get("content")
        items.append(
            NewsItem(
                title=title,
                description=description,
                url=url_,
                source=source_name,
                published_at=published_at,
                content=content,
            )
        )

    return items


# ============================================================
# K 线加载 & 特征工程
# ============================================================

def load_ohlcv(csv_path: str) -> pd.DataFrame:
    """
    从本地 CSV 加载 OHLCV，确保有：
        date, open, high, low, close, volume
    现在主流程用的是 yfinance，这个函数主要是为了兼容“上传 CSV”的老逻辑。
    """
    df = pd.read_csv(csv_path)

    # 处理 date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # 如果没有 date，就用第一列当作日期
        df.insert(0, "date", pd.to_datetime(df.iloc[:, 0]))

    # 兼容 yfinance 默认列名
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 如果没有 close，就用 adj_close 顶上
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 中缺少必要列: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入原始 OHLCV，输出包含各种技术指标的特征 DataFrame。
    """
    if df is None or df.empty:
        raise ValueError("输入 df 为空。")

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    x = df.copy()

    # 统一列名
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    x = x.rename(columns=rename_map)

    # 日期处理
    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").reset_index(drop=True)

    # 必要列检查
    missing = [c for c in REQUIRED_COLS if c not in x.columns]
    if missing:
        raise ValueError(f"make_features: 缺少必要列: {missing}")

    # ⭐⭐ 核心修复：避免任何情况 fallback 到 pandas 的 to_numeric ⭐⭐
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        x[c] = x[c].astype(float)

    # 技术指标
    x["ret_1"] = x["close"].pct_change()

    for w in (3, 5, 10, 20):
        x[f"ma_{w}"] = x["close"].rolling(w).mean()
        x[f"ret_std_{w}"] = x["ret_1"].rolling(w).std()
        x[f"vol_ma_{w}"] = x["volume"].rolling(w).mean()

    # 收盘价相对 MA20 的比值
    ma20 = x["ma_20"].astype(float)
    close = x["close"].astype(float)

    # 原始比值
    ratio = close / (ma20 + 1e-8)

    # 强制转成 1D
    ratio_1d = np.asarray(ratio, dtype=float).reshape(-1)

    # 🔥 强制对齐长度：补齐 / 截断，让结果长度与 x 完全一致
    if len(ratio_1d) != len(x):
        fixed = np.full(len(x), np.nan, dtype=float)
        L = min(len(ratio_1d), len(fixed))
        fixed[:L] = ratio_1d[:L]
        ratio_1d = fixed  # 覆盖为修复后的版本

    # 写入列
    x["close_over_ma20"] = ratio_1d

    return x



# ============================================================
# 随机森林模型：仅基于价格特征
# ============================================================

@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame
    test_mape: Optional[float]


class StockForecaster:
    """
    仅使用价格特征（make_features）做回归预测的 RandomForest 封装。
    """

    def __init__(
        self,
        n_estimators: int = 400,
        random_state: int = 42,
        min_train_size: int = 10,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.min_train_size = min_train_size
        self.fitted = False

    # ------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> float:
        """
        用历史数据训练随机森林，并返回一个简单的 Test MAPE 作为参考。
        """
        feat = make_features(df).dropna().reset_index(drop=True)

        if len(feat) < self.min_train_size:
            # 数据太少，直接全量训练，不做 train/test 拆分，也不算 MAPE
            X = feat.drop(columns=["date", "close"]).values
            y = feat["close"].values
            self.model.fit(X, y)
            self.fitted = True
            return float("nan")


        X = feat.drop(columns=["date", "close"]).values
        y = feat["close"].values

        # 简单划分：前 80% 训练，后 20% 测试
        split_idx = int(len(feat) * 0.8)
        if split_idx <= 0 or split_idx >= len(feat):
            # 极端情况：直接全量训练，不算 MAPE
            self.model.fit(X, y)
            self.fitted = True
            return float("nan")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model.fit(X_train, y_train)
        self.fitted = True

        y_pred = self.model.predict(X_test)
        eps = 1e-8
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + eps))) * 100.0)
        return mape

    # ------------------------------------------------------------

    def predict_future(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
    ) -> ForecastResult:
        """
        递归方式预测未来 horizon 天的 close，并返回 ForecastResult。
        """
        if not self.fitted:
            mape = self.fit(df)
        else:
            mape = float("nan")

        # 用一份可变副本递推未来数据
        hist = df.copy()
        if "date" not in hist.columns:
            raise ValueError("predict_future: df 中缺少 'date' 列。")

        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").reset_index(drop=True)

        preds: List[float] = []
        dates: List[datetime] = []

        last_date = hist["date"].max()

        for _ in range(horizon):
            feat_all = make_features(hist).dropna().reset_index(drop=True)
            if feat_all.empty:
                raise RuntimeError("预测时特征为空，请检查输入数据。")

            last_row = feat_all.iloc[-1]
            X_last = last_row.drop(labels=["date", "close"]).values.reshape(1, -1)

            next_close = float(self.model.predict(X_last)[0])

            # 这里简单地 +1 天；实际项目可以改成“只加交易日”的逻辑
            next_date = last_date + pd.Timedelta(days=1)
            last_date = next_date

            preds.append(next_close)
            dates.append(next_date)

            # 为了递归预测，构造“下一天的伪数据”
            new_row = {
                "date": next_date,
                "open": next_close,
                "high": next_close,
                "low": next_close,
                "close": next_close,
                "volume": hist["volume"].iloc[-1],  # 沿用上一日成交量
            }
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

        forecast_df = pd.DataFrame({"date": dates, "pred_close": preds})
        return ForecastResult(forecast_df=forecast_df, test_mape=mape)
