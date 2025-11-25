import yfinance as yf
import pandas as pd
import time

symbol = "600519.SS"    # 股票代码

# t0 = time.time()

df = yf.download(
    symbol,
    period="1y",
    progress=False,
    threads=False,
    timeout=20,
    auto_adjust=False
)

# --- 新增：转换你要的格式 ---
df2 = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]

# --- 新增：保存文件（你想改名字就在这里改） ---
df2.to_csv("stock_CSV.csv", index=False)

# print("保存成功：maotai_1year.csv")
# print(df2.head())
