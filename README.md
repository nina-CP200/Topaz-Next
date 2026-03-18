# Topaz V3 - 智能股票估值分析系统

基于多因子量化评估的股票分析与投资组合管理系统。

## 核心功能

| 功能 | 说明 |
|------|------|
| 🎯 **多因子评分** | 5大维度：价值(20%)、质量(25%)、动量(20%)、波动(15%)、红利(20%) |
| 🔮 **收益预测** | 基于因子权重预测未来收益 |
| ⚠️ **风险评估** | 4级风险等级：低/中/高/极高 |
| 💡 **投资建议** | 自动生成买入/持有/回避建议 |
| 💰 **虚拟投资组合** | 模拟投资跟踪，验证策略有效性 |
| 🌏 **双市场支持** | A股(腾讯财经) + 美股(Finnhub) |

## 快速开始

### 安装依赖

```bash
pip install pandas requests
```

### 配置 API Keys

复制 `.env.example` 到 `.env` 并填入你的 API Keys：

```bash
cp .env.example .env
```

### 运行分析

```bash
# 只分析A股（推荐日常使用，节省API费用）
python ml_stock_analysis.py --cn

# 只分析美股
python ml_stock_analysis.py --us

# 分析全部市场
python ml_stock_analysis.py
```

### 定时任务

```bash
# 默认只分析A股
./run_report.sh

# 分析美股
./run_report.sh --us

# 分析全部市场
./run_report.sh --all
```

## 项目结构

```
topaz-v3/
├── ml_stock_analysis.py      # 主分析脚本（多因子评分）
├── qlib_predictor.py         # Qlib 预测器
├── topaz_data_api.py         # 数据获取 API
├── utils.py                  # 工具函数
├── run_report.sh             # 定时任务脚本
├── send_slack_report.sh      # Slack 报告发送
├── virtual_portfolio.json    # 虚拟投资组合跟踪
├── 美股关注股票列表.md        # 美股股票池
├── A股关注股票列表.md         # A股股票池
├── .env                      # API Keys（不提交到 Git）
└── .env.example              # API Keys 模板
```

## 虚拟投资组合

### 功能说明

- **初始资金**: 100万人民币
- **目标**: 3个月内跑赢沪深300指数
- **期限**: 2026-03-19 ~ 2026-06-19
- **策略**: 基于多因子评分，结合基本面和技术面分析

### 数据策略

| 市场 | 更新频率 |
|------|----------|
| A股 | 交易时段每小时 (9:30-15:00) |
| 美股 | 不监控（节省API费用） |

### 优化计划

| 日期 | 任务 |
|------|------|
| 2026-04-18 | 第一次审视，优化A股参数 |
| 2026-05-18 | 第二次审视，调整策略 |
| 2026-06-19 | 最终复盘，总结经验 |

## 数据源

| 市场 | 数据源 | API |
|------|--------|-----|
| 美股 | Finnhub | FINNHUB_API_KEY |
| 美股 | Financial Modeling Prep | FMP_API_KEY |
| A股 | 腾讯财经 | 免费 |

## API Keys 获取

- **Finnhub**: https://finnhub.io/register
- **FMP**: https://financialmodelingprep.com/developer/docs/

## 注意事项

1. `.env` 文件包含敏感信息，已添加到 `.gitignore`
2. 日常运行建议使用 `--cn` 参数，只分析A股，节省美股API费用
3. 虚拟投资组合仅供策略验证，不构成投资建议

## License

MIT