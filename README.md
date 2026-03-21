# QuantPilot - 智能量化投资分析系统

基于多因子量化评估 + 机器学习的股票分析与投资组合管理系统。

## 核心功能

| 功能 | 说明 |
|------|------|
| 🎯 **多因子评分** | 5大维度：价值(20%)、质量(25%)、动量(20%)、波动(15%)、红利(20%) |
| 🤖 **机器学习预测** | LightGBM 集成模型，预测未来收益 |
| 🔮 **收益预测** | 基于因子权重 + ML 模型预测 |
| ⚠️ **风险评估** | 4级风险等级：低/中/高/极高 |
| 💡 **投资建议** | 自动生成买入/持有/回避建议 |
| 💰 **虚拟投资组合** | 模拟投资跟踪，验证策略有效性 |
| 🌏 **双市场支持** | A股(腾讯/新浪财经) + 美股(FMP/Finnhub) |
| 📅 **交易日检测** | 自动跳过周末和节假日 |

## 快速开始

### 安装依赖

```bash
pip install pandas requests lightgbm scikit-learn numpy
```

### 配置 API Keys

复制 `.env.example` 到 `.env` 并填入你的 API Keys：

```bash
cp .env.example .env
```

### 运行分析

```bash
# A股分析（推荐日常使用，免费无需API Key）
python ml_stock_analysis.py --cn

# 美股分析（需要 API Key）
python ml_stock_analysis.py --us

# 启用机器学习预测
python ml_stock_analysis.py --cn --ml

# 限制分析数量（测试用）
python ml_stock_analysis.py --cn --limit 5

# 分析全部市场
python ml_stock_analysis.py
```

### 定时任务

```bash
# 交易日自动检测（推荐）
python run_daily_analysis.py

# Shell 脚本方式
./run_report.sh          # 默认只分析A股
./run_report.sh --us     # 分析美股
./run_report.sh --all    # 分析全部市场
```

## 项目结构

```
quantpilot/
├── 📁 主程序
│   ├── ml_stock_analysis.py    # 主分析脚本（多因子 + ML）
│   ├── run_daily_analysis.py   # 交易日检测包装器
│   ├── predict_a_share.py      # A股预测
│   ├── predict_us.py           # 美股预测
│   └── update_portfolio.py     # 持仓更新
│
├── 📁 模型
│   ├── ensemble_model.py       # 集成模型
│   ├── ensemble_model.pkl      # A股训练模型 (6.8MB)
│   ├── ensemble_scaler.pkl     # 数据标准化器
│   ├── ml_predictor.py         # ML 预测器
│   ├── qlib_predictor.py       # Qlib 预测器
│   └── deep_learning.py        # 深度学习模型
│
├── 📁 数据获取
│   ├── quantpilot_data_api.py  # 数据 API（腾讯/新浪/FMP/Finnhub）
│   ├── fetch_a_share_quotes.py # A股行情获取
│   ├── fetch_data.py           # 通用数据获取
│   └── fetch_training_data.py  # 训练数据收集
│
├── 📁 训练
│   ├── collect_training_data.py # 收集训练数据
│   ├── train_us_model.py        # 训练美股模型
│   ├── ml_sector_trainer.py     # 行业模型训练
│   ├── industry_model.py        # 行业模型
│   └── training_data.csv        # 训练数据 (7.2MB)
│
├── 📁 股票列表
│   ├── A股关注股票列表.md        # 25只 A股
│   ├── 美股关注股票列表.md       # 29只 美股
│   └── 美股测试列表.md
│
├── 📁 输出
│   ├── predictions.csv           # 预测结果
│   ├── predictions_today.json   # 今日预测
│   ├── a_share_predictions_today.json
│   ├── us_predictions_today.json
│   └── virtual_portfolio.json   # 虚拟投资组合
│
├── 📁 Shell 脚本
│   ├── run_report.sh            # 运行报告
│   └── send_slack_report.sh     # 发送 Slack
│
└── 📁 配置
    ├── .env                      # API Keys（不提交）
    └── .env.example              # API Keys 模板
```

## 虚拟投资组合

### 挑战设定

- **初始资金**: 100万人民币
- **目标**: 3个月内跑赢沪深300指数
- **期限**: 2026-03-19 ~ 2026-06-19
- **策略**: Topaz V3 多因子评分 + ML预测

### 数据策略

| 市场 | 更新频率 | 数据源 |
|------|----------|--------|
| A股 | 交易时段每小时 | 腾讯/新浪（免费） |
| 美股 | 不监控 | 节省API费用 |

### 优化计划

| 日期 | 任务 |
|------|------|
| 2026-04-18 | 第一次审视，优化A股参数 |
| 2026-05-18 | 第二次审视，调整策略 |
| 2026-06-19 | 最终复盘，总结经验 |

## 数据源

| 市场 | 数据源 | 获取内容 | API Key |
|------|--------|----------|---------|
| A股 | 腾讯财经 | 实时价格、PE、PB、ROE、市值 | ❌ 免费 |
| A股 | 新浪财经 | 历史K线（ML训练） | ❌ 免费 |
| 美股 | FMP | 实时行情、财务数据 | ✅ 需要 |
| 美股 | Finnhub | 美股数据 | ✅ 免费版有限 |
| 宏观 | 东方财富 | 美元指数、原油、黄金 | ✅ 需要 |

## API Keys 获取

- **Finnhub**: https://finnhub.io/register
- **FMP**: https://financialmodelingprep.com/developer/docs/

## 模型训练

```bash
# 收集训练数据
python collect_training_data.py

# 训练美股模型
python train_us_model.py
```

## 注意事项

1. `.env` 文件包含敏感信息，已添加到 `.gitignore`
2. 日常运行建议使用 `--cn` 参数，A股数据免费
3. 虚拟投资组合仅供策略验证，不构成投资建议

## License

MIT