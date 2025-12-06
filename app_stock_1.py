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


st.set_page_config(
    page_title="多模态 K 线预测 Demo",
    layout="wide",
)

st.title("📈 多模态 K 线 + 新闻 预测 Demo")


# ========== yfinance 下载 OHLCV ==========

def load_ohlcv_from_yf(symbol: str, period: str) -> pd.DataFrame:
    data = yf.download(symbol, period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise RuntimeError("无法从 yfinance 获取数据，请检查代码或网络。")

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

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if "date" not in df.columns:
        df.insert(0, "date", pd.to_datetime(df.iloc[:, 0]))
    else:
        df["date"] = pd.to_datetime(df["date"])

    return df[["date", "open", "high", "low", "close", "volume"]]


# ========== 画 K 线 & 预测图 ==========

def plot_candlestick(df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=400,
    )
    return fig


def plot_forecast(df_hist: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_hist["date"],
            y=df_hist["close"],
            mode="lines",
            name="历史收盘价",
        )
    )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["pred_close"],
                mode="lines+markers",
                name="预测收盘价",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title="历史 & 预测",
        xaxis_title="Date",
        yaxis_title="Price",
        height=450,
    )
    return fig


# ========== 侧边栏参数 ==========

with st.sidebar:
    st.header("参数设置")

    ticker = st.text_input("股票代码 / Ticker", value="AAPL")

    period = st.selectbox(
        "历史区间 (yfinance period)",
        ["6mo", "1y", "2y", "5y"],
        index=1,
    )

    horizon = st.slider("预测步数（天）", min_value=1, max_value=30, value=5)

    model_type = st.selectbox(
        "选择模型",
        [
            "RandomForest 数值基线",
            "Hybrid (价格 + 新闻)",
            "Hybrid (价格 + 新闻 + Qwen3-VL K 线图)",
        ],
        index=2,
    )

    run_btn = st.button("开始预测", type="primary")


# ========== 主流程 ==========

if not run_btn:
    st.info("在左侧输入股票代码和参数，然后点击 **开始预测**。")
else:
    try:
        with st.spinner("正在下载 K 线数据..."):
            df_ohlcv = load_ohlcv_from_yf(ticker, period)
        df_ohlcv = df_ohlcv.loc[:, ~df_ohlcv.columns.duplicated()]

        with st.spinner("正在抓取新闻..."):
            news_list = search_stock_news(ticker, days=7, max_results=40)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("历史 K 线")
            st.plotly_chart(
                plot_candlestick(df_ohlcv, title=f"{ticker} OHLC"),
                use_container_width=True,
            )

        with col2:
            st.subheader("最近新闻 (NewsAPI)")
            if not news_list:
                st.write("暂无新闻或未配置 NEWS_API_KEY。")
            else:
                for item in news_list[:10]:
                    st.markdown(
                        f"- **[{item.title}]({item.url})**  \n"
                        f"  {item.published_at.strftime('%Y-%m-%d %H:%M')}  ·  {item.source}"
                    )

        st.markdown("---")

        # 选择模型
        if model_type == "RandomForest 数值基线":
            with st.spinner("使用 RandomForest 基线进行预测..."):
                rf = StockForecaster(horizon=horizon)
                result = rf.forecast(df_ohlcv)

            st.success(f"基线模型 MAPE (验证集) = {result.test_mape:.4f}")
            fig2 = plot_forecast(df_ohlcv, result.forecast_df)
            st.plotly_chart(fig2, use_container_width=True)
            forecast_df = result.forecast_df

        else:
            use_qwen = model_type.endswith("Qwen3-VL K 线图")
            with st.spinner("使用 Hybrid 多模态模型进行预测..."):
                hybrid = HybridForecaster(
                    window=30,
                    horizon=horizon,
                    use_qwen_vl=use_qwen,
                )
                hres = hybrid.forecast(df_ohlcv, news_list)

            if not pd.isna(hres.test_mae):
                st.success(hres.model_info)
            else:
                st.warning("样本太少或 Hybrid 模型未成功训练，结果仅供参考。")

            fig2 = plot_forecast(df_ohlcv, hres.forecast_df)
            st.plotly_chart(fig2, use_container_width=True)
            forecast_df = hres.forecast_df

        # 结果表 + 下载
        st.subheader("预测结果表")
        if forecast_df.empty:
            st.write("暂无预测结果。")
        else:
            st.dataframe(forecast_df)
            csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载预测结果 CSV",
                data=csv_bytes,
                file_name=f"{ticker}_forecast.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"出现错误：{repr(e)}")
