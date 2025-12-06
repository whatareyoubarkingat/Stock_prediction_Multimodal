# stock_engine_hybrid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from datetime import datetime, timedelta
import io
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

# 这里复用后端里的 NewsItem 和 make_features
from rag_engine_stock_1 import NewsItem, make_features

# ==========================
# Qwen3-VL 客户端（可选）
# ==========================

class QwenVLClient:
    """
    通过阿里云 Model Studio 的 OpenAI-Compatible 接口调用 Qwen3-VL。
    你需要在环境中：
      pip install openai
      export DASHSCOPE_API_KEY="你的key"
    并确保 base_url 指向 DashScope 兼容地址。
    """

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
            # 没有 key，直接关闭
            return

        try:
            import openai  # type: ignore

            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )
            self._enabled = True
        except Exception as e:
            print("[QwenVLClient] init failed:", repr(e))
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def extract_kline_factors(self, image_bytes: bytes) -> Dict[str, float]:
        """
        调 Qwen3-VL 看 K 线图，要求它返回 JSON 格式的几个数值因子：
          - short_trend: [-1, 1] 近期趋势（向上/向下）
          - volatility: [0, 1] 波动率
          - pattern_strength: [0, 1] 形态信号强度（头肩顶、三角形等）
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
            # 兼容 list / str 两种
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text)
            data = json.loads(text)
            out = {}
            for k in ["short_trend", "volatility", "pattern_strength"]:
                v = float(data.get(k, 0.0))
                out[k] = v
            return out
        except Exception as e:
            print("[QwenVLClient] call failed:", repr(e))
            return {}


# ==========================
# 工具函数：K 线图渲染
# ==========================

def render_kline_image(df_window: pd.DataFrame) -> bytes:
    """
    输入最近 N 天的 OHLCV DataFrame，输出一张 PNG 格式的 K 线图 bytes。
    为了避免引入 mplfinance，这里画简化版：蜡烛 + 成交量条。
    """
    df = df_window.copy().reset_index(drop=True)
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    x = np.arange(len(df))
    ax_price.plot(x, df["close"].values, linewidth=1.0)
    ax_price.set_ylabel("Price")

    # 简单的红绿柱代替蜡烛
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
    return buf.getvalue()


# ==========================
# Hybrid 结果结构体
# ==========================

@dataclass
class HybridForecastResult:
    forecast_df: pd.DataFrame
    test_mae: float
    model_info: str


# ==========================
# HybridForecaster
# ==========================

class HybridForecaster:
    """
    多模态混合预测器：

    - 数值模态：价格 + 技术指标（make_features）
    - 文本模态：新闻 title + description -> SentenceTransformer embedding
    - 图像模态：最近 window 天 K 线 -> Qwen3-VL -> 3 个数值因子

    模型：RandomForest 回归（你可以后续切到 XGBoost）。
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

    # ---------- 文本特征：按日期聚合新闻 embedding ----------

    def _build_daily_news_embedding(
        self, news_list: List[NewsItem]
    ) -> pd.DataFrame:
        if not news_list:
            return pd.DataFrame(columns=["date", "news_emb_" + str(i) for i in range(384)])

        texts = []
        dates = []
        for item in news_list:
            # title + desc 拼在一起
            t = (item.title or "") + " " + (item.description or "")
            texts.append(t.strip())
            dates.append(item.published_at.date())

        # 句向量
        emb = self.text_model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # 按日期求均值
        df = pd.DataFrame(emb)
        df["date"] = dates
        agg = df.groupby("date").mean().reset_index()
        # 重命名列
        rename_map = {i: f"news_emb_{i}" for i in range(agg.shape[1] - 1)}
        agg = agg.rename(columns=rename_map)
        return agg

    # ---------- 构建训练面板数据 ----------

    def _build_panel(
        self,
        hist_df: pd.DataFrame,
        news_list: List[NewsItem],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        返回:
          X: 特征 DataFrame（包含 date）
          y: 目标向量（预测下一天 close）
        """
        # 数值特征
        feat_df = make_features(hist_df)  # 包含 date, close, 以及各种技术指标

        # 文本特征（日级）
        news_daily = self._build_daily_news_embedding(news_list)

        # 先合并数值 + 文本
        X = feat_df.merge(news_daily, on="date", how="left")

        # 按日期排序
        X = X.sort_values("date").reset_index(drop=True)

        # 补 NaN（没新闻的天）
        X = X.fillna(0.0)

        # 为了方便和 K 线窗口对齐，索引 +1 当下一天标签
        closes = hist_df.sort_values("date")["close"].values
        # y[i] 对应 X[i] 那天之后的下一天 close
        # 因此可用 len(closes)-1 个样本
        y = closes[1:]
        X = X.iloc[:-1, :].reset_index(drop=True)

        # 加入 Qwen-VL K线因子（可选）
        if self.use_qwen_vl and self.qwen_client and self.qwen_client.enabled:
            kline_features = {
                "short_trend": [],
                "volatility": [],
                "pattern_strength": [],
            }

            df_sorted = hist_df.sort_values("date").reset_index(drop=True)

            for idx in range(len(X)):
                # X[idx] 对应的日期
                cur_date = X.loc[idx, "date"]
                # 找到这个日期在原 DataFrame 中的 index
                pos = df_sorted.index[df_sorted["date"] == cur_date]
                if len(pos) == 0:
                    # 没找到，填 0
                    for k in kline_features.keys():
                        kline_features[k].append(0.0)
                    continue

                pos = int(pos[0])
                start = max(0, pos - self.window + 1)
                window_df = df_sorted.iloc[start : pos + 1]

                img_bytes = render_kline_image(window_df)
                factors = self.qwen_client.extract_kline_factors(img_bytes)

                kline_features["short_trend"].append(factors.get("short_trend", 0.0))
                kline_features["volatility"].append(factors.get("volatility", 0.0))
                kline_features["pattern_strength"].append(
                    factors.get("pattern_strength", 0.0)
                )

            for k, v in kline_features.items():
                X[f"kline_{k}"] = v
        else:
            # 没开 Qwen-VL，就先全部 0
            X["kline_short_trend"] = 0.0
            X["kline_volatility"] = 0.0
            X["kline_pattern_strength"] = 0.0

        return X, y

    # ---------- 训练 + 验证 ----------

    def fit(self, hist_df: pd.DataFrame, news_list: List[NewsItem]) -> float:
        """
        训练模型，返回验证集 MAE（最后 20% 作为验证集）。
        """
        X_all, y_all = self._build_panel(hist_df, news_list)

        # 拆 date 列出来，其余为真正特征
        dates = X_all["date"]
        feat_cols = [c for c in X_all.columns if c != "date"]
        X = X_all[feat_cols].values
        y = y_all

        if len(y) < 30:
            # 样本太少，没法训练
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

    # ---------- 滚动预测 ----------

    def forecast(
        self, hist_df: pd.DataFrame, news_list: List[NewsItem]
    ) -> HybridForecastResult:
        """
        先用历史数据训练，再从最后一天开始向后滚动预测 horizon 天。
        """
        val_mae = self.fit(hist_df, news_list)
        if not self.fitted:
            # 回退：直接返回空
            forecast_df = pd.DataFrame(
                columns=["date", "pred_close"], data=[]
            )
            return HybridForecastResult(
                forecast_df=forecast_df,
                test_mae=float("nan"),
                model_info="Hybrid model not fitted (too few samples).",
            )

        # 准备滚动预测：逐天往后推
        hist_df = hist_df.sort_values("date").reset_index(drop=True)
        last_date = hist_df["date"].iloc[-1]
        cur_df = hist_df.copy()

        pred_dates: List[datetime] = []
        pred_vals: List[float] = []

        for step in range(self.horizon):
            next_date = last_date + timedelta(days=1 + step)
            # 这里只是简单+1天，实际可以跳过周末 / 节假日（TODO）

            # 用当前数据重新构造 panel（只用数值+文本+K线特征，对应当前最后一天）
            X_all, _ = self._build_panel(cur_df, news_list)
            feat_cols = [c for c in X_all.columns if c != "date"]
            X_last = X_all.iloc[-1:][feat_cols].values

            pred = float(self.regressor.predict(X_last)[0])
            pred_dates.append(next_date)
            pred_vals.append(pred)

            # 把预测值接到 cur_df，方便后续窗口计算
            new_row = {
                "date": next_date,
                "open": pred,
                "high": pred,
                "low": pred,
                "close": pred,
                "volume": cur_df["volume"].iloc[-1],  # 直接复用上一天成交量
            }
            cur_df = pd.concat([cur_df, pd.DataFrame([new_row])], ignore_index=True)

        forecast_df = pd.DataFrame({"date": pred_dates, "pred_close": pred_vals})
        info = (
            f"Hybrid(RandomForest + text embedding"
            f"{' + Qwen3-VL K-line' if self.use_qwen_vl and self.qwen_client and self.qwen_client.enabled else ''}), "
            f"val MAE={val_mae:.4f}"
        )
        return HybridForecastResult(
            forecast_df=forecast_df,
            test_mae=val_mae,
            model_info=info,
        )
