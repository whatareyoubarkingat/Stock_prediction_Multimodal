# app_stock_local.py
import streamlit as st
import plotly.graph_objects as go

from rag_engine_stock_1 import (
    load_ohlcv,
    StockForecaster,
    search_stock_news,
    NewsItem,
)

from stock_engine_hybrid import HybridForecaster


# ====== 免责声明文本 ======
DISCLAIMER_TEXT = """
**【重要声明：使用本系统即视为您已完全理解并接受以下条款】**

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

上記の条項に同意できない場合は、直ちに本システムの利用を中止してください。
"""

# ====== 1) 初始化是否同意免责声明的状态 ======
if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False


# ====== 2) 弹窗(对话框)：用户必须勾选同意 ======
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


# ================================================================
# 先弹出免责声明
# ================================================================
if not st.session_state.accepted_disclaimer:
    disclaimer_dialog()

st.set_page_config(page_title="Stock K-line Forecast (Local)", layout="wide")
st.title("📈 K线预测")
st.title("⚠️ 仅用于学习/演示，不构成任何投资建议。")

with st.sidebar:
    st.header("上传数据")
    uploaded = st.file_uploader(
        "请上传 OHLCV CSV/XLSX（Date/Open/High/Low/Close/Volume）",
        type=["csv", "xlsx", "xls"]
    )

    horizon = st.slider("预测未来天数", 1, 30, 5)
    train_btn = st.button("训练并预测")

    st.markdown("---")
    st.header("新闻参考（可选）")

    # ========= 新增：股票关键词输入 + 搜索按钮 =========
    news_query = st.text_input(
        "股票代码 / 公司名（用于搜索相关新闻）",
        placeholder="例如：600519 或 贵州茅台 或 AAPL"
    )
    news_btn = st.button("搜索相关新闻")

if uploaded is None:
    st.info("请先在左侧上传 CSV/XLSX 数据文件。")
    st.stop()

# 读取数据
try:
    df = load_ohlcv(uploaded)
except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()

# K线图
fig = go.Figure(data=[
    go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="K-line"
    )
])
fig.update_layout(
    height=520,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ========= 新增：在主区域展示新闻 =========
if news_btn:
    if not news_query.strip():
        st.warning("请输入股票代码或公司名后再搜索。")
    else:
        with st.spinner("正在搜索相关新闻（第三方数据源，仅供参考）..."):
            try:
                news_list = search_stock_news(news_query.strip(), max_results=8)
            except Exception as e:
                st.error(f"新闻搜索失败：{e}")
                news_list = []

        if news_list:
            st.subheader("📰 近期相关新闻（仅供参考）")
            st.caption(
                "新闻由第三方数据源提供，可能存在延迟、错误或不完整；"
                "请勿将其视为任何形式的投资建议。"
            )

            for item in news_list:
                # 每条新闻做成一个可展开卡片
                with st.expander(
                    f"{item.title}  —— {item.source}｜{item.published_at.strftime('%Y-%m-%d %H:%M')}"
                ):
                    if item.description:
                        st.write(item.description)
                    st.markdown(f"[🔗 前往原文]({item.url})")
        else:
            st.info("暂未找到相关新闻，可尝试更换关键词或稍后再试。")

st.markdown("---")

# ==========================================================
# 训练 + 预测：优先尝试「价格+新闻」多模态模型，失败则回退
# ==========================================================
if train_btn:
    result = None
    use_hybrid = False   # 标记当前是不是用的多模态模型
    news_list = []

    # 如果用户在侧栏填写了 news_query，就尝试多模态
    if news_query and news_query.strip():
        try:
            with st.spinner("正在获取相关新闻，并训练『价格 + 新闻』多模态模型..."):
                # 1) 搜索新闻（后端还是用你 stock_engine_local 里的 search_stock_news）
                news_list = search_stock_news(news_query.strip(), max_results=100)

                if len(news_list) < 2:
                    st.warning("相关新闻数量过少（<2 条），自动回退到纯价格模型。")
                else:
                    # 2) 训练 + 预测多模态模型
                    hf = HybridForecaster()
                    hybrid_result = hf.predict_future(
                        df,
                        news_list=news_list,
                        horizon=horizon,
                    )
                    result = hybrid_result
                    use_hybrid = True
        except Exception as e:
            st.error(f"多模态模型训练/预测失败，自动回退到纯价格模型。错误信息：{e}")
            use_hybrid = False

    # 如果没填 news_query，或者多模态失败，就使用原来的随机森林模型
    if not use_hybrid:
        with st.spinner("训练随机森林模型（仅使用价格特征）..."):
            rf = StockForecaster()
            rf_result = rf.predict_future(df, horizon=horizon)
            result = rf_result

    # ================== 统一展示预测结果 ==================
    st.subheader("预测结果")

    if use_hybrid:
        # 多模态：显示 MAE
        if result.test_mae is not None:
            st.write(f"测试集 MAE（仅参考）：**{result.test_mae:.4f}**")
        st.caption("当前使用模型：价格 + 新闻文本 的序列模型（GRU）。")
        forecast_df = result.forecast_df
    else:
        # 原模型：显示 MAPE
        if result.test_mape is not None:
            st.write(f"测试集 MAPE（仅参考）：**{result.test_mape:.2f}%**")
        st.caption("当前使用模型：仅基于价格特征的随机森林回归。")
        forecast_df = result.forecast_df

    # 预测曲线图（Close + Pred Close）
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines", name="历史 Close"
    ))
    fig2.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["pred_close"],
        mode="lines+markers",
        name="预测 Close"
    ))
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(forecast_df)

    # 下载预测
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "下载预测结果 CSV",
        data=csv_bytes,
        file_name="forecast.csv",
        mime="text/csv"
    )

else:
    st.info("点击左侧“训练并预测”开始。")
