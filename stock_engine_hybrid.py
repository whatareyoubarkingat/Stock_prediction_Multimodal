from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta
import io
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

from rag_engine_stock_1 import NewsItem, make_features, dedup_date_column


# ========== Qwen-VL 客户端（可选） ==========

class QwenVLClient:
    def __init__(
        self,
        model: str = "qwen3-vl-plus",
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._enabled = False
        self._client = None

        api_key = os.getenv(api_key_env)
        if not api_key:
            return

        try:
            import openai  # type: ignore

            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract_kline_factors(self, image_bytes: bytes) -> Dict[str, float]:
        """
        调 Qwen3-VL，看一张K线图，返回:
          - short_trend [-1,1]
          - volatility [0,1]
          - pattern_strength [0,1]
        """
        if not self._enabled or self._client is None:
            return {}

        import base64
        import json

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "你是技术分析师，请根据这张K线图，总结近期走势特征，"
            "并输出一个 JSON，对象包含："
            "short_trend(-1到1之间浮点数),"
            "volatility(0到1之间浮点数),"
            "pattern_strength(0到1之间浮点数)。"
            "只输出JSON，不要加解释。"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image": b64,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                temperature=0.2,
            )
            text = resp.choices[0].message.content
            # 兼容 list / str
            if isinstance(text, list):
                text = "".join(getattr(part, "text", "") for part in text)  # type: ignore
            data = json.loads(text)  # type: ignore
            out = {}
            for k in ["short_trend", "volatility", "pattern_strength"]:
                try:
                    out[k] = float(data.get(k, 0.0))
                except Exception:
                    out[k] = 0.0
            return out
        except Exception:
            return {}


# ========== 画窗口K线图，转成 bytes ==========

def render_kline_image(df_window: pd.DataFrame) -> bytes:
    df = df_window.copy().reset_index(drop=True)
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(6, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    x = np.arange(len(df))
    ax_price.plot(x, df["close"].values, linewidth=1.0)
    ax_price.set_ylabel("Price")

    up = df["close"] >= df["open"]
    down = ~up
    ax_price.bar(
        x[up],
        df["close"][up] - df["open"][up],
        bottom=df["open"][up],
        width=0.6,
    )
    ax_price.bar(
        x[down],
        df["open"][down] - df["close"][down],
        bottom=df["close"][down],
        width=0.6,
    )

    ax_vol.bar(x, df["volume"].values, width=0.6)
    ax_vol.set_ylabel("Vol")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ========== 结果结构体 ==========

@dataclass
class HybridForecastResult:
    forecast_df: pd.DataFrame
    test_mae: float
    model_info: str


# ========== Hybrid 模型本体 ==========

class HybridForecaster:
    """
    多模态：
      - 数值：价格 + 技术指标（make_features）
      - 文本：新闻 embedding（SentenceTransformer）
      - 图像：最近 window 天 K 线 → Qwen3-VL 因子（可关掉）
    """

    def __init__(
        self,
        window: int = 30,
        horizon: int = 5,
        use_qwen_vl: bool = True,
        qwen_model: str = "qwen3-vl-plus",
        random_state: int = 42,
    ) -> None:
        self.window = window
        self.horizon = horizon

        self.text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.use_qwen_vl = use_qwen_vl
        self.qwen_client = QwenVLClient(model=qwen_model) if use_qwen_vl else None

        self.regressor = RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fitted = False

    # ----- 文本：按日聚合新闻 embedding -----

    def _build_daily_news_embedding(self, news_list: List[NewsItem]) -> pd.DataFrame:
        if not news_list:
            return pd.DataFrame(columns=["date"])

        texts = []
        dates = []
        for item in news_list:
            t = (item.title or "") + " " + (item.description or "")
            texts.append(t.strip())
            dates.append(item.published_at.date())

        emb = self.text_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        df = pd.DataFrame(emb)
        df["date"] = dates
        agg = df.groupby("date").mean().reset_index()

        agg["date"] = pd.to_datetime(agg["date"])

        rename_map = {i: f"news_emb_{i}" for i in range(agg.shape[1] - 1)}
        agg = agg.rename(columns=rename_map)
        return agg

    # ----- 构建训练 Panel（特征 + 标签） -----

    def _build_panel(
        self, hist_df: pd.DataFrame, news_list: List[NewsItem]
    ) -> (pd.DataFrame, np.ndarray):
        hist_df = dedup_date_column(hist_df)
        feat_df = make_features(hist_df)
        news_daily = self._build_daily_news_embedding(news_list)

        X = feat_df.merge(news_daily, on="date", how="left")
        X = X.sort_values("date").reset_index(drop=True)
        X = X.fillna(0.0)

        closes = hist_df.sort_values("date")["close"].values
        y = closes[1:]
        X = X.iloc[:-1, :].reset_index(drop=True)

        # 图像因子（可选）
        if self.use_qwen_vl and self.qwen_client and self.qwen_client.enabled:
            kline_short = []
            kline_vol = []
            kline_pattern = []

            df_sorted = hist_df.sort_values("date").reset_index(drop=True)
            for idx in range(len(X)):
                cur_date = X.loc[idx, "date"]
                pos_arr = df_sorted.index[df_sorted["date"] == cur_date].tolist()
                if not pos_arr:
                    kline_short.append(0.0)
                    kline_vol.append(0.0)
                    kline_pattern.append(0.0)
                    continue
                pos = pos_arr[0]
                start = max(0, pos - self.window + 1)
                window_df = df_sorted.iloc[start : pos + 1]

                img_bytes = render_kline_image(window_df)
                factors = self.qwen_client.extract_kline_factors(img_bytes)

                kline_short.append(float(factors.get("short_trend", 0.0)))
                kline_vol.append(float(factors.get("volatility", 0.0)))
                kline_pattern.append(float(factors.get("pattern_strength", 0.0)))

            X["kline_short_trend"] = kline_short
            X["kline_volatility"] = kline_vol
            X["kline_pattern_strength"] = kline_pattern
        else:
            X["kline_short_trend"] = 0.0
            X["kline_volatility"] = 0.0
            X["kline_pattern_strength"] = 0.0

        return X, y

    # ----- 训练 + 验证 -----

    def fit(self, hist_df: pd.DataFrame, news_list: List[NewsItem]) -> float:
        X_all, y_all = self._build_panel(hist_df, news_list)
        feat_cols = [c for c in X_all.columns if c != "date"]
        X = X_all[feat_cols].values
        y = y_all

        if len(y) < 30:
            self.fitted = False
            return float("nan")

        split = int(len(y) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.regressor.fit(X_train, y_train)
        y_pred = self.regressor.predict(X_val)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        self.fitted = True
        return mae

    # ----- 滚动预测 -----

    def forecast(
        self, hist_df: pd.DataFrame, news_list: List[NewsItem]
    ) -> HybridForecastResult:
        val_mae = self.fit(hist_df, news_list)
        if not self.fitted:
            empty = pd.DataFrame(columns=["date", "pred_close"])
            return HybridForecastResult(
                forecast_df=empty,
                test_mae=float("nan"),
                model_info="Hybrid model not fitted.",
            )

        hist_df = hist_df.sort_values("date").reset_index(drop=True)
        last_date = hist_df["date"].iloc[-1]
        cur_df = hist_df.copy()

        pred_dates: List[datetime] = []
        pred_vals: List[float] = []

        for step in range(self.horizon):
            next_date = last_date + timedelta(days=1 + step)

            X_all, _ = self._build_panel(cur_df, news_list)
            feat_cols = [c for c in X_all.columns if c != "date"]
            X_last = X_all.iloc[-1:][feat_cols].values

            pred = float(self.regressor.predict(X_last)[0])
            pred_dates.append(next_date)
            pred_vals.append(pred)

            new_row = {
                "date": next_date,
                "open": pred,
                "high": pred,
                "low": pred,
                "close": pred,
                "volume": cur_df["volume"].iloc[-1],
            }
            cur_df = pd.concat([cur_df, pd.DataFrame([new_row])], ignore_index=True)

        forecast_df = pd.DataFrame({"date": pred_dates, "pred_close": pred_vals})

        info = (
            "Hybrid(RandomForest + text embedding"
            + (
                " + Qwen3-VL K-line"
                if self.use_qwen_vl and self.qwen_client and self.qwen_client.enabled
                else ""
            )
            + f"), val MAE={val_mae:.4f}"
        )

        return HybridForecastResult(
            forecast_df=forecast_df,
            test_mae=val_mae,
            model_info=info,
        )