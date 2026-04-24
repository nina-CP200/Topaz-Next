# Topaz-Next

> **非商用开源项目** | 基于 LightGBM + 100+技术因子的 A 股量化分析系统
>
> 仅供学习研究使用，**禁止用于任何商业目的或实际资金交易**

---

## 项目简介

Topaz-Next 是一个面向个人研究者的 A 股量化分析系统，以**沪深300成分股**为选股池，通过特征工程构建 100+ 技术因子，输入 LightGBM 模型预测每只股票的短期上涨概率。

### 核心特性

- **沪深300股票池**：覆盖中国A股市场核心蓝筹股
- **100+ 技术因子**：价格位置、动量、波动率、成交量、技术指标等
- **市场环境判断**：自动识别牛市/熊市/震荡/反弹/回调
- **动态仓位建议**：根据市场环境调整建议仓位（20%-95%）
- **Slack推送**：自动发送分析报告到Slack频道

---

## 快速开始

### 一键安装

```bash
# 自动检测网络环境并安装
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash

# 中国大陆用户（强制使用镜像源）
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash -s -- --china
```

### 配置环境

```bash
# 复制环境变量模板
cd ~/topaz-next
cp .env.example .env

# 编辑配置文件
vim .env
```

配置内容：
```
# Slack Bot Token（可选，用于推送报告）
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
SLACK_CHANNEL=#investments
```

### 运行分析

```bash
# 每日分析（首次运行会自动获取历史数据并训练模型）
python3 daily_decision.py

# 使用沪深300分组模型
python3 daily_decision.py --csi300
```

---

## 核心流程

```
09:45  扫描沪深300成分股 → 获取实时行情 → 特征工程（100+因子）
       ↓
10:00  市场环境判断 → LightGBM预测 → Top 5推荐 + Bottom 5回避
       ↓
10:05  Slack推送报告（可选）
```

---

## 主要功能

### 1. 每日分析（daily_decision.py）

- 判断市场环境（bull/bear/sideways/recovery/pullback）
- 计算模型置信度（0.5-0.9）
- 生成仓位建议（20%-95%）
- 输出 Top 5 建议买入股票
- 输出 Bottom 5 建议回避股票

### 2. 股票查询（query_stock.py）

```bash
# 查询特定股票
python3 query_stock.py 600519      # 贵州茅台
python3 query_stock.py 000001.SZ   # 平安银行

# 显示排名前10股票
python3 query_stock.py --top 10
```

### 3. 模型训练（retrain_model.py）

```bash
# Walk-Forward训练（推荐）
python3 retrain_model_walkforward.py

# 基础训练
python3 retrain_model.py
```

### 4. 回测系统（backtest.py）

```bash
# 回测模型表现
python3 backtest.py
```

---

## 技术架构

### 特征工程（100+因子）

`feature_engineer.py` 自动生成以下特征：

- **价格位置因子**（8个）：MA偏离度、价格区间位置
- **动量因子**（12个）：收益率、时间序列动量、加速度
- **波动率因子**（10个）：历史波动率、ATR、偏度、峰度
- **成交量因子**（5个）：量比、OBV、Amihud非流动性
- **技术指标**（12个）：RSI、MACD、KDJ、布林带、CCI
- **价格形态**（5个）：K线形态识别（锤子线、射击之星）
- **统计因子**（6个）：均值回归、最大回撤、夏普比率
- **大盘因子**（9个）：沪深300相关性、Beta系数
- **连续涨跌**（3个）：连续阳线/阴线统计

### 模型

生产环境使用 **LightGBM 分类模型**，输出上涨概率（0-1）。支持按市场环境分组训练多个模型。

---

## 项目结构

```
topaz-next/
├── daily_decision.py          # 主入口：每日分析
├── query_stock.py             # 股票查询工具
├── feature_engineer.py        # 特征工程（100+因子）
├── ensemble_model.py          # Stacking集成模型
├── market_data.py             # 大盘数据与环境判断
├── quantpilot_data_api.py     # 数据获取（腾讯/新浪API）
├── retrain_model.py           # 模型训练
├── retrain_model_walkforward.py  # Walk-Forward训练
├── backtest.py                # 回测系统
├── cache_manager.py           # 特征缓存管理
├── send_report.py             # Slack报告推送
├── fetch_full_history.py      # 历史数据获取
├── csi300_stocks.json         # 沪深300股票列表
├── install.sh                 # 一键安装脚本
├── .env.example               # 环境变量模板
├── requirements.txt           # 依赖列表
├── .gitignore                 # Git忽略配置
└── README.md                  # 项目文档
```

### 数据与模型文件（不在Git中）

以下文件需运行脚本后自动生成，不包含在Git仓库中：

- `ensemble_model.pkl` - 训练后的模型文件
- `ensemble_scaler.pkl` - 特征标准化器
- `csi300_full_history.csv` - 沪深300历史数据
- `cache/features/*.pkl` - 特征缓存文件

---

## 定时任务配置

### Crontab 设置

```bash
crontab -e
```

添加以下内容：

```cron
# Topaz-Next 每日定时任务（工作日，A股交易时间）
# 09:45 生成分析报告
45 9 * * 1-5 /bin/bash ~/topaz-next/daily_report.sh

# 10:00 运行分析并发送Slack
0 10 * * 1-5 /bin/bash ~/topaz-next/daily_decision.sh
```

---

## 依赖说明

**Python依赖**（requirements.txt）：
- requests >= 2.32.3
- lightgbm >= 4.6.0
- scikit-learn >= 1.8.0
- pandas >= 3.0.1
- numpy >= 2.2.3
- joblib >= 1.5.3

**系统要求**：
- Python 3.7+
- 4GB+ 内存
- 稳定网络连接

---

## 投资风险警示

⚠️ **本系统仅供学习研究，不构成任何投资建议**

1. **模型局限**：基于历史数据训练，市场结构变化可能导致模型失效
2. **数据风险**：免费API存在延迟或失效风险
3. **合规声明**：**禁止用于商业目的**（代客理财、收费咨询、基金产品等）

**市场有风险，投资需谨慎**

---

## 开源协议

本项目采用 **GNU General Public License v3.0 (GPL-3.0)** 协议开源。

**主要条款**：
- 自由使用、研究、修改和分发
- 修改后的作品必须同样使用 GPL-3.0 协议
- 分发时必须提供源代码
- 必须保留原作者版权声明

详细协议：https://www.gnu.org/licenses/gpl-3.0.html

---

## 常见问题

### Q: 首次运行如何获取数据？

首次运行 `daily_decision.py` 时，系统会自动：
1. 从腾讯/新浪API获取沪深300历史数据
2. 构建特征并缓存
3. 训练模型（如果没有现成模型）

整个过程约需5-10分钟。

### Q: 模型文件在哪里？

模型文件 `ensemble_model.pkl` 存储在项目根目录，但**不包含在Git仓库中**。需要自行训练或从其他途径获取。

### Q: 如何更新股票池？

编辑 `csi300_stocks.json`，添加或删除股票代码。格式：
```json
{
  "code": "600519.SH",
  "name": "贵州茅台"
}
```

### Q: Slack推送失败？

检查 `.env` 配置：
- SLACK_BOT_TOKEN 是否正确
- SLACK_CHANNEL 是否存在
- Bot是否有频道发送权限

---

**Topaz-Next** — 用数据探索市场，以理性对待投资

*最后更新：2026-04-24*