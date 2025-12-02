# app_stock_1.py / app_stock_yf.py

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

from rag_engine_stock_1 import (
    StockForecaster,
    search_stock_news,
    NewsItem,
)

from stock_engine_hybrid import HybridForecaster


# ========== 页面配置 ==========
///st.set_page_config(page_title="Stock K-line Forecast (yfinance)", layout="wide")
st.set_page_config(page_title="Stock K-line Forecast (yfinance)", layout="wide")


# ========== 免责声明文本 ==========
DISCLAIMER_TEXT = """
**【重要声明：使用本系统即视为您已完全理解并同意以下条款】**

本系统仅为技术演示、学术研究和个人学习目的而开发，不构成任何形式的投资建议、财务建议、证券交易建议或风险提示。
系统输出由算法自动生成，可能存在错误或偏差，不保证准确性与可靠性。

**您应自行承担使用本系统进行任何投资决策所产生的全部风险与后果。**
开发者不对因使用或无法使用本系统造成的任何直接或间接损失承担责任。

如您不同意上述条款，请立即停止使用本系统。

**【IMPORTANT NOTICE: By using this system, you acknowledge and agree to all terms below】**

This system is developed solely for technical demonstration, academic research, and personal learning purposes.  
It does **not** constitute any form of investment advice, financial advice, securities trading recommendation, or risk warning.  
All outputs are generated automatically by algorithms and may contain errors or inaccuracies.  
Accuracy and reliability are **not guaranteed**.

**You assume full responsibility for any investment decisions made based on the use of this system.**  
The developer shall not be liable for any direct or indirect losses arising from the use of, or inability to use, this system.

If you do not agree with the above terms, please discontinue using this system immediately.

**【重要なお知らせ：本システムを使用することにより、以下の条項を完全に理解し、同意したものとみなされます】**

本システムは、技術デモ、学術研究、および個人的な学習目的のみを目的として開発されたものです。  
投資アドバイス、財務アドバイス、証券取引の推奨、またはリスク警告を提供するものではありません。  
本システムの出力はすべてアルゴリズムによって自動生成されており、誤りや不正確な内容が含まれる可能性があります。  
正確性や信頼性は**一切保証されません**。

**本システムを利用して行った投資判断によって生じるすべてのリスクおよび結果は、利用者自身の責任となります。**  
本システムの利用または利用不能によって発生した直接的または間接的な損害について、開発者は一切の責任を負いません。

上記の条款に同意できない場合は、直ちに本システムの利用を中止してください。
"""

# ========== 是否同意免责声明 ==========
if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False


@st.dialog("免责声明 / Disclaimer")
def disclaimer_dialog():
    html_text = DISCLAIMER_TEXT.replace("\n", "<br>")

    st.markdown(
        f"""
        <div style="
            height: 260px;
            overflow-y: auto;
            padding: 14px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
            line-height: 1.6;
            font-size: 0.95rem;
        ">
            {html_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    agree = st.checkbox("我已阅读并同意上述免责声明")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("同意并继续", disabled=not agree):
            st.session_state.accepted_disclaimer = True
            st.rerun()
    with col2:
        if st.button("不同意并退出"):
            st.session_state.accepted_disclaimer = False
            st.stop()


# 先弹免责声明
if not st.session_state.accepted_disclaimer:
    disclaimer_dialog()

# ========== 标题 ==========
st.title("📈 K线预测")
st.caption("⚠️ 仅用于学习 / 演示，不构成任何投资建议。")


# ========== 使用 yfinance 下载 OHLCV ==========
def fetch_ohlcv_from_yf(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    从 yfinance 下载日线 K 线数据，并转换为统一的
    [date, open, high, low, close, volume] 格式。
    自动处理 yfinance 返回的 MultiIndex 列情况。
    """
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if data is None or data.empty:
        raise ValueError("yfinance 未返回数据，请检查股票代码或网络连接。")

    # ⭐ 关键：如果是 MultiIndex 列（比如 ('Open','AAPL')），拍平成一层
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]) for col in data.columns]

    df = data.reset_index()

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
    df = df.rename(columns=rename_map)

    # 如果没有 close，就用 adj_close 顶上
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"下载的数据缺少必要列：{missing}")

    df = df[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"])

    # ⭐ 把 OHLCV 强制转成 1 维数值型
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ========== 侧边栏：参数设置 ==========
with st.sidebar:
    st.header("参数设置")

    ticker = st.text_input(
        "股票代码（yfinance 格式）",
        value="AAPL",  # 默认 AAPL
        help="例如：AAPL、MSFT、600519.SS 等",
    )

    period = st.selectbox(
        "历史数据区间",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        index=2,  # 默认 1y
    )

    horizon = st.slider("预测未来天数", 1, 30, 5)

    # ⭐ 新增：模型选择（自动 / 仅价格 / 多模态）
    model_choice = st.radio(
        "模型选择",
        options=[
            "自动选择（推荐）",
            "仅价格模型（随机森林）",
            "多模态模型（价格 + 新闻）",
        ],
        index=0,
        help=(
            "自动选择：如果新闻和历史数据足够，则优先使用多模态模型；否则回退到随机森林。\n"
            "仅价格模型：只使用历史价格（随机森林）。\n"
            "多模态模型：强制尝试价格 + 新闻的 GRU 模型，失败会自动回退到随机森林。"
        ),
    )

    train_btn = st.button("✅ 一键：下载数据 + 搜索新闻 + 训练并预测")


# 没点按钮时的提示
if not train_btn:
    st.info("在左侧输入股票代码，选择模型，然后点击「一键：下载数据 + 搜索新闻 + 训练并预测」。")
    st.stop()

ticker = ticker.strip()
if not ticker:
    st.warning("请先输入股票代码。")
    st.stop()

# ========== 1. 用 yfinance 下载历史 K 线 ==========
try:
    with st.spinner(f"正在从 yfinance 下载 {ticker} 的历史 K 线数据（{period}）..."):
        df = fetch_ohlcv_from_yf(ticker, period=period)
except Exception as e:
    st.error(f"下载 K 线数据失败：{e}")
    st.stop()

if df.empty:
    st.error("历史 K 线数据为空，请尝试调整股票代码或时间区间。")
    st.stop()

st.write(f"当前获取到的历史 K 线数据条数：**{len(df)}**")

# ========== 画历史 K 线 ==========
st.subheader(f"历史 K 线（{ticker}）")

# ⭐ 只保留 OHLC 都是数值的行，用于画 K 线
df_ohlc = df.dropna(subset=["open", "high", "low", "close"]).copy()

st.write("用于绘制 K 线的有效数据条数：", len(df_ohlc))
st.write("数据列类型：")
st.write(df_ohlc.dtypes)

if df_ohlc.empty:
    st.warning("虽然成功拉取到了数据，但 OHLC 列均为非数值或 NaN，无法绘制 K 线。")
else:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_ohlc["date"],
                open=df_ohlc["open"],
                high=df_ohlc["high"],
                low=df_ohlc["low"],
                close=df_ohlc["close"],
                name="K-line",
            )
        ]
    )
    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="日期",
        yaxis_title="价格",
    )
    fig.update_yaxes(autorange=True)

    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# ========== 2. 使用股票代码自动搜索相关新闻 ==========
with st.spinner("正在搜索相关新闻（第三方数据源，仅供参考）..."):
    try:
        # 这里直接用 ticker 作为关键词；
        news_list = search_stock_news(ticker, max_results=100)
    except Exception as e:
        st.error(f"新闻搜索失败（不会中断预测，只是无法用到新闻特征）：{e}")
        news_list = []

if news_list:
    st.subheader("📰 近期相关新闻（仅供参考）")
    st.caption(
        "新闻由第三方数据源提供，可能存在延迟、错误或不完整；"
        "请勿将其视为任何形式的投资建议。"
    )

    # 展示前 8 条
    for item in news_list[:8]:
        with st.expander(
            f"{item.title} —— {item.source}｜{item.published_at.strftime('%Y-%m-%d %H:%M')}"
        ):
            if item.description:
                st.write(item.description)
            st.markdown(f"[🔗 前往原文]({item.url})")
else:
    st.info("暂未找到相关新闻，若选择多模态模型可能会自动回退为仅使用价格特征的模型。")

st.markdown("---")

# ========== 3. 训练 + 预测：根据用户选择决定模型 ==========
result = None
use_hybrid = False

MIN_SEQ_LEN_FOR_HYBRID = 120  # 多模态模型的最小 K 线长度
can_use_hybrid = (len(df) >= MIN_SEQ_LEN_FOR_HYBRID) and (len(news_list) >= 2)

# —— 根据前端选择分支 ——
if model_choice == "仅价格模型（随机森林）":
    # 完全不尝试多模态，直接随机森林
    with st.spinner("正在训练随机森林模型（仅使用价格特征）..."):
        rf = StockForecaster()
        rf_result = rf.predict_future(df, horizon=horizon)
        result = rf_result
        use_hybrid = False

elif model_choice == "多模态模型（价格 + 新闻）":
    # 用户强制选择多模态；如果条件不足就提示并回退到随机森林
    if not can_use_hybrid:
        if len(df) < MIN_SEQ_LEN_FOR_HYBRID:
            st.warning(
                f"历史 K 线数据不足 {MIN_SEQ_LEN_FOR_HYBRID} 条，"
                "无法使用多模态模型，将自动回退到仅使用价格特征的随机森林模型。"
            )
        elif len(news_list) < 2:
            st.warning(
                "相关新闻条数过少（少于 2 条），"
                "无法使用多模态模型，将自动回退到仅使用价格特征的随机森林模型。"
            )
        with st.spinner("正在训练随机森林模型（仅使用价格特征）..."):
            rf = StockForecaster()
            rf_result = rf.predict_future(df, horizon=horizon)
            result = rf_result
            use_hybrid = False
    else:
        # 条件满足，尝试多模态；失败则回退
        try:
            with st.spinner("正在训练『价格 + 新闻』多模态模型 (GRU)..."):
                hf = HybridForecaster()
                hybrid_result = hf.predict_future(
                    df,
                    news_list=news_list,
                    horizon=horizon,
                )
                result = hybrid_result
                use_hybrid = True
        except Exception as e:
            st.error(f"多模态模型训练/预测失败，将自动回退到纯价格模型。错误信息：{e}")
            with st.spinner("正在训练随机森林模型（仅使用价格特征）..."):
                rf = StockForecaster()
                rf_result = rf.predict_future(df, horizon=horizon)
                result = rf_result
                use_hybrid = False

else:  # "自动选择（推荐）"
    if can_use_hybrid:
        try:
            with st.spinner("正在训练『价格 + 新闻』多模态模型 (GRU)..."):
                hf = HybridForecaster()
                hybrid_result = hf.predict_future(
                    df,
                    news_list=news_list,
                    horizon=horizon,
                )
                result = hybrid_result
                use_hybrid = True
        except Exception as e:
            st.error(f"多模态模型训练/预测失败，将自动回退到纯价格模型。错误信息：{e}")
            with st.spinner("正在训练随机森林模型（仅使用价格特征）..."):
                rf = StockForecaster()
                rf_result = rf.predict_future(df, horizon=horizon)
                result = rf_result
                use_hybrid = False
    else:
        # 自动模式下，条件不够就提示原因并使用随机森林
        if len(df) < MIN_SEQ_LEN_FOR_HYBRID:
            st.info(
                f"历史 K 线数据不足 {MIN_SEQ_LEN_FOR_HYBRID} 条，"
                "自动关闭多模态模型，改用仅使用价格特征的随机森林模型。"
            )
        elif len(news_list) < 2:
            st.info(
                "相关新闻条数过少，自动关闭多模态模型，"
                "改用仅使用价格特征的随机森林模型。"
            )
        with st.spinner("正在训练随机森林模型（仅使用价格特征）..."):
            rf = StockForecaster()
            rf_result = rf.predict_future(df, horizon=horizon)
            result = rf_result
            use_hybrid = False

# ========== 4. 展示预测结果 ==========
st.subheader("预测结果")

if use_hybrid:
    # HybridForecaster 里一般是 test_mae
    if getattr(result, "test_mae", None) is not None:
        st.write(f"测试集 MAE（仅参考）：**{result.test_mae:.4f}**")
    st.caption("当前使用模型：价格 + 新闻文本 的序列模型（GRU，多模态）。")
    forecast_df = result.forecast_df
else:
    if getattr(result, "test_mape", None) is not None:
        st.write(f"测试集 MAPE（仅参考）：**{result.test_mape:.2f}%**")
    st.caption("当前使用模型：仅基于价格特征的随机森林回归。")
    forecast_df = result.forecast_df

# 预测曲线：历史 close + 未来预测
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["close"],
        mode="lines",
        name="历史 Close",
    )
)
fig2.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["pred_close"],
        mode="lines+markers",
        name="预测 Close",
    )
)
fig2.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_title="日期",
    yaxis_title="价格",
)
fig2.update_yaxes(autorange=True)

st.plotly_chart(fig2, use_container_width=True)

# 预测数据表
st.dataframe(forecast_df)

# 下载预测 CSV
csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "下载预测结果 CSV",
    data=csv_bytes,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv",
)
