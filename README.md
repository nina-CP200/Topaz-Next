# Topaz V3 - 智能量化投资分析系统

基于多因子量化评估 + 机器学习的 A 股投资分析与自动交易系统。

## 🎯 系统特点

| 特性 | 说明 |
|------|------|
| 🤖 **ML 集成模型** | XGBoost + LightGBM + CatBoost Stacking，45 个因子 |
| 📊 **多因子评分** | 价值、质量、动量、波动率、技术因子 |
| 💰 **虚拟投资组合** | 100 万初始资金，自动调仓，实时跟踪 |
| ⏰ **定时自动化** | 每日 09:45 自动分析、决策、报告 |
| 📱 **Slack 集成** | 自动推送报告到指定频道/DM |
| 🔄 **OpenClaw 集成** | 通过 Agent 系统自动执行完整流程 |

---

## 📁 项目结构

```
topaz-v3/
├── 📁 核心分析
│   ├── ml_stock_analysis_ensemble.py  # 集成模型分析（主脚本）⭐
│   ├── ml_stock_analysis.py           # 快速模型分析
│   ├── daily_decision.py              # 每日投资决策 ⭐
│   ├── predict_a_share.py             # A股预测
│   └── predict_us.py                  # 美股预测
│
├── 📁 模型文件
│   ├── ensemble_model.pkl             # 集成模型 (6.8MB) ⭐
│   ├── ensemble_scaler.pkl            # 数据标准化器
│   ├── quick_model.pkl                # 快速模型 (224KB)
│   ├── ensemble_model.py              # 模型类定义
│   └── feature_engineer.py            # 特征工程
│
├── 📁 数据获取
│   ├── fetch_a_share_quotes.py        # A股实时行情 ⭐
│   ├── quantpilot_data_api.py         # 数据 API 封装
│   └── fetch_data.py                  # 通用数据获取
│
├── 📁 投资组合
│   ├── virtual_portfolio.json         # 虚拟组合状态 ⭐
│   ├── update_portfolio.py            # 持仓更新
│   └── risk_management.py             # 风险管理
│
├── 📁 定时任务
│   ├── daily_report.sh                # 每日报告脚本 ⭐
│   ├── daily_decision.sh              # 投资决策脚本 ⭐
│   └── crontab_config.txt             # Crontab 配置模板
│
├── 📁 股票列表
│   ├── A股关注股票列表.md             # 25只 A股
│   ├── 美股关注股票列表.md            # 美股列表
│   └── csi300_stocks.json             # 沪深300成分股
│
├── 📁 输出结果
│   ├── predictions_today.json         # 今日预测 ⭐
│   ├── a_share_predictions_today.json # A股预测详情
│   ├── topaz_report.log               # 运行日志
│   └── data/                          # 数据缓存
│
└── 📁 配置
    ├── .env                           # API Keys
    └── README.md                      # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas requests lightgbm scikit-learn numpy xgboost catboost joblib
```

### 2. 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入 API Keys（美股需要，A股免费）
```

### 3. 手动运行分析

```bash
# 运行集成模型分析（推荐）
python ml_stock_analysis_ensemble.py --cn --limit 30

# 运行快速模型
python ml_stock_analysis.py --cn

# 生成投资决策
python daily_decision.py
```

---

## ⏰ 定时任务配置

### 方法 1: Crontab（推荐）

```bash
# 编辑 crontab
crontab -e

# 添加以下内容（已配置在 crontab_config.txt）
45 9 * * 1-5 /home/emmmoji/.openclaw/workspace-topaz/topaz-v3/daily_report.sh
0 10 * * 1-5 /home/emmmoji/.openclaw/workspace-topaz/topaz-v3/daily_decision.sh
```

### 方法 2: OpenClaw Cron（已配置）

OpenClaw 定时任务已配置：
- **09:45** (CST): 运行 `daily_report.sh` → 获取数据 → ML分析 → 生成报告
- **10:00** (CST): 运行 `daily_decision.sh` → 投资决策 → 更新持仓

配置文件：`~/.openclaw/cron/jobs.json`

---

## 🤖 ML 模型说明

### 集成模型 (Ensemble Model)

**模型架构**: Stacking (堆叠)
- **基学习器**: XGBoost + LightGBM + CatBoost + RandomForest + GradientBoosting
- **元学习器**: LogisticRegression
- **特征数量**: 45 个
- **模型大小**: 6.8 MB

**因子体系**:
| 类型 | 具体指标 | 权重 |
|------|---------|------|
| 价值因子 | PE、PB | 20% |
| 质量因子 | ROE、ROA | 25% |
| 动量因子 | 3M/6M/12M 收益率 | 20% |
| 波动率因子 | 历史波动率 | 15% |
| 技术因子 | RSI、MA5/20 | 20% |

### 快速模型 (Quick Model)

- **特征**: 15 个（技术面为主）
- **大小**: 224 KB
- **速度**: <1 秒
- **适用**: 实时快速筛选

---

## 💰 虚拟投资组合

### 挑战设定

- **初始资金**: 100 万人民币
- **目标**: 3 个月内跑赢沪深 300 指数 5%
- **期限**: 2026-03-20 ~ 2026-06-19
- **策略**: Topaz V3 集成模型 + 自动调仓

### 当前持仓（示例）

| 代码 | 名称 | 仓位 | 成本价 | 现价 | 盈亏 |
|------|------|------|--------|------|------|
| 600036 | 招商银行 | 24.8% | 39.92 | 38.92 | -2.5% |
| 600900 | 长江电力 | 19.8% | 27.19 | 26.87 | -1.2% |
| 601888 | 中国中免 | 12.7% | 70.46 | 70.00 | -0.7% |
| 002465 | 海格通信 | 11.6% | 14.11 | 14.18 | +0.5% |
| 000333 | 美的集团 | 9.6% | 73.25 | 73.06 | -0.3% |

**当前状态**:
- 总资产: ~99.5 万
- 累计盈亏: -0.5%
- 仓位: 86.1%
- 现金: 13.9%

---

## 📊 预测信号解读

| 概率 | 信号 | 操作建议 |
|------|------|---------|
| ≥70% | 🟢 强烈买入 | 积极建仓/加仓 |
| 55%-70% | 🟡 买入 | 适量买入 |
| 40%-55% | 🟠 观望 | 持有观望 |
| <40% | 🔴 回避 | 卖出/不买入 |

**预期收益计算**:
```
expected_return = (probability - 0.5) × 20
```

---

## 📱 Slack 集成

### 自动报告内容

每日 09:45 自动发送：
1. 📈 市场概览（上证指数、深证成指、创业板指）
2. 🤖 ML 预测（30 只 A 股，买卖信号、概率、RSI）
3. 💡 投资决策（调仓建议、目标仓位）
4. 📝 交易记录（今日买入/卖出）
5. 💼 组合表现（总值、收益率、仓位）
6. 📋 持仓明细（每只股票的盈亏）
7. 🎯 明日策略

### 配置

```bash
# 报告发送到 Slack DM
channel: slack:dm:U0AGVSHJ08Z

# 或发送到频道
channel: slack:investments
```

---

## 🔧 数据源

| 市场 | 数据源 | 内容 | 费用 |
|------|--------|------|------|
| A股 | 腾讯财经 | 实时价格、PE、PB、ROE | 免费 ✅ |
| A股 | 新浪财经 | 历史 K 线 | 免费 ✅ |
| A股 | 东方财富 | 实时行情 | 免费 ✅ |
| 美股 | FMP | 实时行情、财务数据 | 需 API Key |
| 美股 | Finnhub | 美股数据 | 免费版有限 |

---

## ⚠️ 注意事项

1. **模型局限**: 基于历史数据，不代表未来表现
2. **风险控制**: 建议保留 15-20% 现金应对波动
3. **API 限制**: 东方财富 API 可能有频率限制
4. **数据延迟**: 实时行情可能有 1-5 分钟延迟
5. **投资建议**: 仅供策略验证，不构成投资建议

---

## 📝 更新日志

### 2026-03-24
- ✅ 配置 OpenClaw 定时任务（09:45 自动执行）
- ✅ 集成模型作为默认模型（45 因子）
- ✅ 修复时区问题（UTC → CST）
- ✅ 配置 Slack DM 推送
- ✅ 虚拟投资组合自动调仓

### 2026-03-23
- ✅ 添加每日投资决策脚本
- ✅ 完善虚拟投资组合跟踪
- ✅ 添加风险管理模块

### 2026-02-25
- ✅ 初始版本发布
- ✅ 集成模型训练完成
- ✅ A股/美股双市场支持

---

## 📄 License

MIT License

---

**Topaz V3** - 智能量化投资，让数据驱动决策 🔮
