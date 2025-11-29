# stock_engine_hybrid.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# 如果你原来的 load_ohlcv 在 stock_engine_local 里，可以这样：
from rag_engine_stock_1 import load_ohlcv, make_features, NewsItem


# ====== 一些超参数，可以根据需要自己调 ======
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NEWS_LOOKBACK_DAYS = 3      # 每个交易日向前看 N 天新闻做平均
SEQ_WINDOW = 30             # 序列长度：用过去 30 天预测下一天
GRU_HIDDEN = 64
GRU_LAYERS = 1
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====== 文本编码器，把 NewsItem 列表转成向量 ======
class NewsTextEncoder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        # 多语言小模型，能处理中文/英文
        self.model = SentenceTransformer(model_name)

    def encode_news(self, news_list: List[NewsItem]) -> pd.DataFrame:
        """
        输入 NewsItem 列表，输出一个 DataFrame:
        columns: [published_date (date), embedding (np.ndarray)]
        """
        if not news_list:
            # 空表
            return pd.DataFrame(columns=["published_date", "embedding"])

        texts = []
        dates = []
        for item in news_list:
            text = (item.title or "") + " " + (item.description or "")
            text = text.strip()
            if not text:
                continue
            texts.append(text)
            dates.append(item.published_at.date())

        if not texts:
            return pd.DataFrame(columns=["published_date", "embedding"])

        embeddings = self.model.encode(texts, show_progress_bar=True)
        rows = []
        for d, emb in zip(dates, embeddings):
            rows.append({"published_date": d, "embedding": emb.astype(np.float32)})

        df = pd.DataFrame(rows)
        return df


def build_daily_news_embedding(
    trade_dates: pd.Series,
    news_emb_df: pd.DataFrame,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> np.ndarray:
    """
    对齐到交易日：对每个交易日 date_t，
    收集 [date_t - lookback_days + 1, date_t] 内的所有新闻向量取平均。
    返回形状：(len(trade_dates), emb_dim)
    """
    if news_emb_df.empty:
        # 没新闻，全 0
        return np.zeros((len(trade_dates), 1), dtype=np.float32)

    # group by 日期，先把同一天的新闻平均一下
    grouped = (
        news_emb_df
        .groupby("published_date")["embedding"]
        .apply(lambda xs: np.mean(np.stack(xs, axis=0), axis=0))
    )

    all_dates = trade_dates.dt.date.tolist()
    emb_dim = len(next(iter(grouped.values)))
    result = np.zeros((len(all_dates), emb_dim), dtype=np.float32)

    for idx, d in enumerate(all_dates):
        start = d - timedelta(days=lookback_days - 1)
        # 找到窗口内所有日期的 embedding
        window_vecs = []
        cur = start
        while cur <= d:
            if cur in grouped.index:
                window_vecs.append(grouped[cur])
            cur += timedelta(days=1)

        if window_vecs:
            result[idx] = np.mean(np.stack(window_vecs, axis=0), axis=0)
        else:
            # 没新闻就保持 0，或者你可以用最近非零值填充
            result[idx] = np.zeros(emb_dim, dtype=np.float32)

    return result


# ====== PyTorch 序列模型：价格 + 新闻特征 → 下一天 close ======
class PriceNewsGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = GRU_HIDDEN, num_layers: int = GRU_LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, h_n = self.gru(x)
        # 取最后一个时间步的 hidden
        last_hidden = out[:, -1, :]
        y = self.fc(last_hidden).squeeze(-1)
        return y


# ====== Dataset：把 (价格+新闻特征序列) 组织成 (seq, target) ======
class PriceNewsDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, window: int):
        """
        features: (T, D_total)
        targets: (T,)  # 实际只有从 window 到 T-1 才能配齐
        """
        self.features = features
        self.targets = targets
        self.window = window
        self.T = len(features)

    def __len__(self):
        # 最后一个可用 index: T-2 (因为 target 是 t+1)
        return self.T - self.window

    def __getitem__(self, idx):
        # 序列: [idx, idx+window-1], target: idx+window 的 close
        x_seq = self.features[idx:idx + self.window]
        y = self.targets[idx + self.window]  # 下一个时间步的 close
        return torch.from_numpy(x_seq).float(), torch.tensor(y, dtype=torch.float32)


@dataclass
class HybridForecastResult:
    forecast_df: pd.DataFrame
    test_mae: Optional[float]


# ====== 训练 + 预测的封装类 ======
class HybridForecaster:
    """
    价格特征 + 新闻文本嵌入 → 序列模型预测 future close。
    """

    def __init__(self, window: int = SEQ_WINDOW, device: str = DEVICE):
        self.window = window
        self.device = device
        self.model: Optional[PriceNewsGRU] = None
        self.fitted = False
        self.input_dim: Optional[int] = None

    def _prepare_data(
        self,
        df: pd.DataFrame,
        news_list: List[NewsItem],
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """
        返回：
        - features_all: (T, D_total)
        - close_all: (T,)
        - dates: pd.Series of datetime
        """
        # 1) 价格特征
        feat_df = make_features(df).dropna().reset_index(drop=True)
        dates = feat_df["date"].copy().reset_index(drop=True)
        close_all = feat_df["close"].values.astype(np.float32)

        price_cols = [
            c for c in feat_df.columns
            if c not in ["date"]
        ]
        X_price = feat_df[price_cols].values.astype(np.float32)

        # 2) 新闻编码 & 日对齐
        encoder = NewsTextEncoder()
        news_emb_df = encoder.encode_news(news_list)
        if news_emb_df.empty:
            # 没新闻就只有价格特征
            news_emb = np.zeros((len(feat_df), 1), dtype=np.float32)
        else:
            news_emb = build_daily_news_embedding(
                trade_dates=dates,
                news_emb_df=news_emb_df,
                lookback_days=NEWS_LOOKBACK_DAYS,
            )

        # 3) 拼接
        features_all = np.concatenate([X_price, news_emb], axis=1)
        self.input_dim = features_all.shape[1]
        return features_all, close_all, dates

    def fit(self, df: pd.DataFrame, news_list: List[NewsItem]) -> Optional[float]:
        """
        训练模型。返回测试集 MAE（仅参考）。
        """
        features_all, close_all, dates = self._prepare_data(df, news_list)
        T = len(features_all)
        if T <= self.window + 10:
            raise ValueError("数据太短，不足以训练序列模型，请准备更长的历史 K 线。")

        # 按时间划分：前 80% 训练，后 20% 测试
        split = int(T * 0.8)
        train_feats = features_all[:split]
        train_close = close_all[:split]
        test_feats = features_all[split:]
        test_close = close_all[split:]

        train_ds = PriceNewsDataset(train_feats, train_close, self.window)
        test_ds = PriceNewsDataset(test_feats, test_close, self.window)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = PriceNewsGRU(input_dim=self.input_dim).to(self.device)
        self.model = model

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()  # MAE 比较直观

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_batch)

            avg_loss = total_loss / len(train_loader.dataset)
            print(f"[Epoch {epoch+1}/{EPOCHS}] train MAE: {avg_loss:.4f}")

        # 测试集 MAE
        model.eval()
        if len(test_ds) > 0:
            preds = []
            reals = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = model(x_batch)
                    preds.append(y_pred.cpu().numpy())
                    reals.append(y_batch.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            reals = np.concatenate(reals, axis=0)
            test_mae = float(np.mean(np.abs(preds - reals)))
        else:
            test_mae = None

        self.fitted = True
        return test_mae

    def predict_future(
        self,
        df: pd.DataFrame,
        news_list: List[NewsItem],
        horizon: int = 5,
    ) -> HybridForecastResult:
        """
        使用已训练好的 GRU 序列模型预测未来 horizon 天的收盘价。

        为了避免递推时伪造未来 K 线导致数值完全漂移，这里采用「单步预测 + 平铺」策略：
        - 利用真实历史数据构造最后一个长度为 window 的特征序列；
        - 模型只预测下一天的收盘价 next_close；
        - 将 next_close 在时间轴上平铺 horizon 天，用于前端展示。
        """
        # 1) 如有必要先训练
        if not self.fitted:
            test_mae = self.fit(df, news_list)
        else:
            test_mae = None

        # 2) 使用完整历史数据构造特征
        features_all, close_all, dates = self._prepare_data(df, news_list)
        T = len(features_all)
        if T < self.window:
            raise ValueError("历史序列长度不足以构造一个窗口。")

        # 3) 取最后一个 window 的特征序列作为当前状态
        seq_feats = features_all[-self.window :]  # (window, input_dim)
        x_seq = torch.from_numpy(seq_feats).float().unsqueeze(0).to(self.device)  # (1, window, D)

        # 4) 预测下一天的 close
        self.model.eval()
        with torch.no_grad():
            next_close = float(self.model(x_seq).cpu().item())

        # 5) 构造未来 horizon 天日期，并将 next_close 平铺
        last_date = pd.to_datetime(dates.iloc[-1])
        pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
        preds = [next_close] * horizon

        forecast_df = pd.DataFrame({
            "date": pred_dates,
            "pred_close": preds,
        })
        return HybridForecastResult(forecast_df=forecast_df, test_mae=test_mae)
