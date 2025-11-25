# Stock_prediction_Multimodal
## <span style="color:red">本系统（包括但不限于：股票数据展示、K 线图可视化、趋势分析、价格预测、模型输出结果、相关说明文档等）仅为技术演示、学术研究和个人学习目的而开发，不构成任何形式的投资建议、财务建议、证券交易建议、法律意见或风险提示。本系统的所有内容仅用于展示算法能力，不应用于实际投资决策。</span>

## 0. 环境准备
```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

# 创建 Python 3.10 环境
conda create -n rag310 python=3.10

# 激活环境
conda activate rag310

# 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装多模态模型所需的其他依赖
pip install sentence-transformers tqdm plotly streamlit scikit-learn pandas

# 安装依赖用于画K线预测
pip install pandas numpy scikit-learn streamlit plotly

# 安装能够读xlsx的依赖
pip install openpyxl

# 安装工具
pip install torch sentence-transformers tqdm

```

## 1. 申请钥匙
去 NewsAPI 官网申请一个 key，然后在运行 shell 里：
```bash
export NEWSAPI_API_KEY="你的 key"
```

## 2. 后端rag_engine_stock_1.py，并且，再次基础上，加上一个stock_engine_hybrid.py

## 3. 前端app_stock_1.py
在Ubuntu中，激活rag310，用如下命令运行：
```bash
streamlit run app_stock_1.py
```
# 备注：
## 1. 由于防火墙的原因，需要自己上传CSV（主包给出了一个python获取CSV的代码(getCSV.py)，请自行安装相关依赖）
## 2. 建议在WSL中跑，以用上GPU加速；GPU加速需要CUDA，记得安装

**<span style="color:red">其实现在加上新闻之后，得到的预测图线非常的奇怪，主包还在修改中……</span>**
