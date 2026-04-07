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
│   ├── fetch_a_share_quotes_v2.py     # A股实时行情 v2（多源备份）⭐
│   ├── quantpilot_data_api.py         # 数据 API 封装
│   ├── market_data.py                 # 市场数据模块
│   └── fetch_data.py                  # 通用数据获取
│
├── 📁 交易执行
│   ├── execute_trading.py             # 交易执行（带交易日检查）⭐
│   ├── trading_utils.py               # 交易工具（交易日验证）⭐
│   ├── execute_trades.py              # 交易执行辅助
│   └── daily_decision.py              # 每日投资决策
│
├── 📁 投资组合
│   ├── virtual_portfolio.json         # 虚拟组合状态 ⭐
│   ├── update_portfolio.py            # 持仓更新
│   └── risk_management.py             # 风险管理
│
├── 📁 定时任务
│   ├── daily_report.sh                # 每日报告脚本 ⭐
│   ├── daily_decision.sh              # 投资决策脚本 ⭐
│   ├── run_analysis.py                # 分析运行脚本
│   └── crontab_config.txt             # Crontab 配置模板
│
├── 📁 股票列表
│   ├── A股关注股票列表.md             # 25只 A股
│   ├── 美股关注股票列表.md            # 美股列表
│   └── csi300_stocks.json             # 沪深300成分股
│
├── 📁 输出结果
│   ├── predictions_today.json         # 今日预测 ⭐
│   ├── predictions_a_share_today.json # A股预测详情
│   ├── daily_report_*.txt             # 每日报告
│   ├── trading_skip_log.json          # 交易跳过日志
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
- **09:45** (CST, 工作日): 运行 `a-share-daily-analysis` → 获取数据 → ML分析 → 交易决策
- **09:45** (CST, 工作日): 运行 `investment-challenge-tracker` → 更新投资进度

**交易日检查**：
- 自动检查是否为A股交易日（周末+节假日跳过）
- 非交易日自动阻止交易并记录日志
- 配置文件：`trading_utils.py`（节假日数据需定期更新）

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

### 当前持仓（2026-04-03）

| 代码 | 名称 | 持仓 | 成本价 | 现价 | 市值 | 盈亏 |
|------|------|------|--------|------|------|------|
| 601888.SH | 中国中免 | 4,400股 | ¥70.59 | ¥70.05 | ¥308,220 | -0.77% |
| 002465.SZ | 海格通信 | 18,100股 | ¥14.44 | ¥14.55 | ¥263,355 | +0.77% |
| 600030.SH | 中信证券 | 6,600股 | ¥24.22 | ¥24.17 | ¥159,522 | -0.21% |
| 601012.SH | 隆基绿能 | 5,000股 | ¥17.47 | ¥17.36 | ¥86,800 | -0.61% |
| 000701.SZ | 厦门信达 | 7,900股 | ¥5.75 | ¥5.88 | ¥46,452 | +2.24% |
| 688981.SH | 中芯国际 | 300股 | ¥92.40 | ¥93.20 | ¥27,960 | +0.87% |
| 600111.SH | 北方稀土 | 400股 | ¥47.90 | ¥47.58 | ¥19,032 | -0.67% |

**当前状态**:
- 总资产: ¥1,003,692
- 累计盈亏: +¥3,692 (+0.37%)
- 持仓市值: ¥911,341
- 现金: ¥92,351
- 仓位: 90.8%
- 持仓股票: 7只

**投资挑战**:
- 期限: 2026-03-20 ~ 2026-06-19（3个月）
- 目标: 跑赢沪深300指数 5%
- 当前: +0.37% vs 沪深300

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
5. **节假日处理**: 已添加交易日检查，非交易日自动跳过
6. **投资建议**: 仅供策略验证，不构成投资建议

## 🐛 已知问题与修复

### 2026-04-06 节假日交易Bug
**问题**: 清明节假期（4月6日）非交易日触发了交易，导致全仓卖出
**原因**: 
- 定时任务在周日09:45执行
- 获取到的是周五缓存数据，但标注为周日日期
- ML模型检测到数据异常，所有股票概率<20%，触发全仓卖出
**修复**:
- 新增 `trading_utils.py`: 交易日检查、节假日数据
- 新增 `execute_trading.py`: 交易执行前验证交易日
- 更新 `run_analysis.py`: 添加交易日检查提示
- 恢复持仓至4月3日状态

---

## 📝 更新日志

### 2026-04-06
- ✅ 修复节假日交易Bug（4月6日清明节）
- ✅ 新增交易日检查模块（`trading_utils.py`）
- ✅ 新增交易执行脚本（`execute_trading.py`）
- ✅ 恢复持仓至4月3日状态
- ✅ 更新README项目结构

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

## 📊 项目统计

- **代码行数**: ~15,230 行（Python 9,325 行）
- **Python文件**: 29 个
- **模型**: Ensemble (XGBoost + LightGBM + CatBoost)
- **特征**: 45 个因子
- **数据源**: 腾讯财经、新浪财经、东方财富

---

## 📄 License

MIT License

---

**Topaz V3** - 智能量化投资，让数据驱动决策 🔮
