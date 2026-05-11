# Topaz-Next

> **开源量化分析系统** | A 股量化分析系统 | **自带 Web 仪表盘**

---

## 项目简介

Topaz-Next 是一个面向个人研究者的 A 股量化分析系统，以**沪深300成分股**为选股池，通过特征工程构建 65+ 技术因子，使用集成学习模型预测每只股票的短期上涨概率，并给出具体的买入/卖出操作建议。

自带 **本地 Web 仪表盘**（FastAPI + React + Ant Design），无需部署，浏览器打开即可查看市场总览、成分股评分排名、行业板块分析和组合持仓诊断。

### 核心特性

- **沪深300股票池**：覆盖中国A股市场核心蓝筹股，300只股票全部标注行业分类
- **65+ 技术因子**：价格位置、动量、波动率、成交量、技术指标等
- **集成学习模型**：Stacking 架构融合 XGBoost + LightGBM + CatBoost + RF + GBDT
- **策略层**：综合评分 = ML概率×50% + 技术面×30% + 动量×20%，输出具体理由和仓位建议
- **市场环境判断**：均线位置法（回测准确率80.9%），平滑切换避免一天一周期
- **止盈止损参数**：根据市场环境动态调整（bear -4%/+15%, sideways -6%/+12%, bull -8%/+25%）
- **板块涨幅排名**：按行业聚合，展示近期动量强弱
- **持仓分析**：交互式输入持仓，自动分析盈亏、行业分布、调仓建议
- **Slack推送**：自动发送分析报告到Slack频道

---

## 快速开始

### 一键配置

```bash
# 方式1：远程安装（推荐）
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/setup.sh | bash

# 方式2：本地配置
git clone https://github.com/nina-CP200/Topaz-Next.git
cd Topaz-Next
bash setup.sh

# 方式3：中国镜像源
bash setup.sh --china
```

### 手动配置

```bash
# 1. 安装 uv（https://docs.astral.sh/uv/getting-started/installation/）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆项目
git clone https://github.com/nina-CP200/Topaz-Next.git
cd Topaz-Next

# 3. 创建虚拟环境并安装依赖
uv venv
uv pip install -r requirements.txt

# 4. 准备目录
mkdir -p data/raw data/models data/cache
```

### Web 仪表盘（推荐）

```bash
# 一键启动（自动构建前端 + 启动后端）
./start.sh

# 浏览器打开 http://localhost:8000
```

仪表盘包含 4 个页面：

| 页面 | 功能 |
|------|------|
| **市场总览** | 沪深300实时行情 + 日K线 + 市场情绪 + 仓位建议 |
| **沪深300** | 300只成分股综合评分排名，按建议筛选、搜索、触发重新分析 |
| **行业板块** | 28行业20日/5日动量排行（柱状图 + 表格） |
| **我的组合** | 持仓增删改 + 盈亏计算 + 行业分布饼图 + 买卖建议 |
| **说明** | 系统使用说明和评分体系详解 |

前端技术栈：React 19 + TypeScript + Vite + Ant Design 6 + ECharts。后端自动托管前端静态文件，一个端口搞定。

### CLI 命令

```bash
# 每日分析（首次运行会自动获取历史数据并训练模型）
uv run python -m src.analysis.daily

# 使用沪深300分组模型
uv run python -m src.analysis.daily --csi300

# 交互式持仓分析
./portfolio.sh

# 从文件读取持仓
./portfolio.sh portfolio.txt
```

持仓文件格式（每行一只）：
```
600519 100 1680
000858 200 150
```

---

## 主要功能

### 1. 每日策略分析（src/analysis/daily.py）

- 平滑判断市场环境（bull/bear/sideways/recovery/pullback）
- 综合评分（ML概率 + 技术面 + 动量）
- 输出具体买入理由和仓位建议
- 板块涨幅排名
- 止盈止损参数

输出示例：
```
📊 市场环境: bull
   上涨比例: 58.2%, 20日收益: +3.5%
   模型置信度: 70%

🟢 推荐买入（12只）
  #1 600519.SH 贵州茅台
     综合评分: 0.66 | 建议仓位: 18% | 建议买入
     理由: MACD金叉 + 站稳20日线上方 + 短期动量强

📊 板块涨幅排名
  #1 半导体 20日:+15.0% 5日:+8.0%
  #2 新能源车 20日:+12.0% 5日:+5.0%
```

### 2. 持仓分析（src/portfolio.py）

交互式菜单：
```
1. 添加股票
2. 修改股票
3. 删除股票
4. 分析持仓
5. 退出
```

输出示例：
```
💼 持仓分析报告
📊 持仓汇总
  总市值: ¥182,810
  总盈亏: 📉 ¥-30,189 (-14.17%)

🎯 操作建议
🔴 建议卖出（1只）
  000858.SZ: 清仓，止损离场
🟡 建议减仓（1只）
  600519.SH: 减仓至半仓，降低风险敞口
⚠️ 行业集中度提醒
  白酒 占比 84.5%，建议分散配置
```

### 3. 股票查询（src/analysis/query.py）

```bash
uv run python -m src.analysis.query 600519      # 贵州茅台
uv run python -m src.analysis.query 000001.SZ   # 平安银行
uv run python -m src.analysis.query --top 10    # 排名前10
```

### 4. 模型训练（src/models/）

```bash
uv run python -m src.models.walkforward    # Walk-Forward训练（推荐）
uv run python -m src.models.trainer        # 基础训练
```

### 5. 回测系统（src/backtest/backtest.py）

```bash
uv run python -m src.backtest.backtest
```

---

## 技术架构

### 策略层

综合评分公式：
```
composite_score = ML概率×50% + 技术面×30% + 动量×20%
```

技术面包含：RSI、MACD、均线位置
动量包含：5日/20日收益率

### 市场环境判断（均线位置法，回测准确率80.9%）

```
bull:    price > MA20 > MA60, MA20向上
bear:    price < MA20 < MA60, MA20向下
recovery: price > MA20 但 < MA60（反弹）
pullback: price < MA20 但 > MA60（回调）
sideways: 其他
```

平滑机制：最少持续5天 + 连续3天确认才切换

### 止盈止损参数

| 环境 | 止损 | 止盈 | 持仓天数 |
|------|------|------|---------|
| bear | -4% | 15% | 5天 |
| sideways | -6% | 12% | 7天 |
| bull | -8% | 25% | 10天 |

### 特征工程（65+因子）

`src/features/engineer.py` 自动生成：

- **均线特征**（10个）：MA偏离度、均线斜率、价格相对位置
- **波动率特征**（7个）：历史波动率、EWMA、波动率状态
- **成交量特征**（2个）：量比、成交量均线
- **收益率特征**（4个）：1/5/10/20日收益率
- **时间序列动量**（8个）：多周期动量、均线交叉、趋势强度
- **技术指标**（7个）：RSI、MACD、布林带、KDJ
- **均值回归**（4个）：多周期均值回归、价格分位数
- **尾部风险**（4个）：偏度、峰度、波动率突变
- **指数因子**（9个）：沪深300相关性、Beta系数、相对强弱
- **危机Alpha**（3个）：最大回撤、回撤恢复、夏普比率

### 模型架构

Stacking 集成学习：

```
输入特征 ──► XGBoost ──┐
           ──► LightGBM ─┤
           ──► CatBoost ─┼──► Meta-Learner (逻辑回归) ──► 最终预测
           ──► RF ───────┤
           ──► GBDT ─────┘
```

---

## 项目结构

```
topaz-next/
├── src/                    # 源代码
│   ├── config.py           # 市场周期平滑 + 牛熊市判断
│   ├── strategy.py         # 策略层（综合评分+理由+仓位）
│   ├── sector.py           # 板块涨幅排名
│   ├── portfolio.py        # 持仓分析（交互式）
│   ├── analysis/
│   │   ├── daily.py        # 主分析引擎
│   │   └── query.py        # 股票查询
│   ├── data/
│   │   ├── api.py          # 数据获取（腾讯/新浪）
│   │   ├── market.py       # 阈值配置
│   │   └── cache.py        # 缓存管理
│   ├── features/
│   │   └── engineer.py     # 特征工程
│   ├── models/
│   │   ├── ensemble.py     # 集成模型
│   │   ├── trainer.py      # 模型训练
│   │   └── walkforward.py  # Walk-Forward训练
│   ├── backtest/
│   │   └── backtest.py     # 回测系统
│   ├── reports/
│   │   └── sender.py       # Slack推送
│   └── utils/
│       └── utils.py        # 工具函数
├── backend/                # FastAPI 后端
│   ├── main.py             # 入口 + CORS + 静态文件托管
│   ├── schemas.py          # Pydantic 数据模型
│   └── routers/
│       ├── market.py       # /api/market/overview, /history
│       ├── analysis.py     # /api/analysis/daily, /refresh, /status
│       ├── sectors.py      # /api/sectors
│       └── portfolio.py    # /api/portfolio (CRUD + analyze)
├── frontend/               # React 前端
│   ├── src/
│   │   ├── api/index.ts    # axios 封装 + 类型定义
│   │   ├── components/AppLayout.tsx
│   │   └── pages/
│   │       ├── Dashboard.tsx   # 市场总览
│   │       ├── Stocks.tsx      # 沪深300成分股
│   │       ├── Sectors.tsx     # 行业板块
│   │       ├── Portfolio.tsx   # 我的组合
│   │       └── About.tsx       # 使用说明
│   └── package.json
├── config/
│   ├── csi300_stocks.json          # 沪深300股票列表
│   ├── csi300_industry_map.json    # 行业分类映射（300只）
│   └── .env.example                # 环境变量模板
├── data/                   # 数据目录（运行时生成）
├── start.sh                # Web 仪表盘一键启动脚本
├── portfolio.sh            # 持仓分析入口脚本
├── setup.sh                # 一键配置脚本
├── requirements.txt
└── README.md
```

### 数据与模型文件（不入Git）

- `data/models/ensemble_model.pkl` - 训练后的模型文件
- `data/raw/csi300_full_history.csv` - 沪深300历史数据
- `data/cache/features/*.pkl` - 特征缓存文件
- `.market_state.json` - 市场状态持久化

---

## 行业分类

系统内置300只沪深300成分股的行业分类（28个板块）：

银行(22) | 券商(20) | 保险(5) | 白酒(7) | 医药(22) | 半导体(18) | 新能源车(11) | 光伏/新能源(12) | 电力/公用事业(12) | 家电(4) | 军工(9) | 化工(12) | 有色/矿业(15) | 钢铁(3) | 建材(3) | 建筑(9) | 交运/物流(18) | 食品饮料(6) | 农业(4) | 通信(7) | 传媒(3) | 软件/IT(13) | 地产(3) | 石化/煤炭(9) | 消费电子(9) | 电子(10) | 汽车零部件(3) | 光模块(3)

---

## 定时分析推荐：AstrBot

如果你希望实现**自动化定时分析**并将结果推送到 QQ、微信、Telegram 等即时通讯平台，强烈推荐使用 **AstrBot**：

- **多平台支持**：QQ、微信、Telegram、飞书等
- **插件生态**：丰富的插件系统，可轻松对接本项目的分析脚本
- **定时任务**：内置定时任务调度，无需手动配置 crontab
- **AI 对话**：支持接入各大 LLM，实现智能问答式分析

**GitHub 项目地址**：[https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)

---

## 依赖说明

**Python 后端依赖**：
| 包 | 用途 |
|---|---|
| fastapi / uvicorn | Web 框架和服务器 |
| pydantic | 请求/响应数据校验 |
| requests | HTTP 客户端（数据获取 + Slack） |
| pandas / numpy | 数据处理 |
| scikit-learn / lightgbm | ML 模型 |
| joblib | 模型序列化 |

**Node.js 前端依赖**：
| 包 | 用途 |
|---|---|
| react / react-router-dom | 前端框架 + 路由 |
| antd / @ant-design/icons | UI 组件库 |
| echarts / echarts-for-react | 金融图表（K线/饼图/柱状图） |
| axios | HTTP 请求 |
| dayjs | 日期格式化 |

**系统要求**：
- Python 3.8+
- Node.js 18+（仅开发/构建需要，运行时只需要 dist/）
- 4GB+ 内存
- 稳定网络连接

---

## 投资风险警示

⚠️ **本系统仅供学习研究，不构成任何投资建议**

1. **模型局限**：基于历史数据训练，市场结构变化可能导致模型失效
2. **数据风险**：免费API存在延迟或失效风险
3. **板块分析局限**：存在幸存者偏差，仅供参考

**市场有风险，投资需谨慎**

---

## 开源协议

本项目采用 **GNU General Public License v3.0 (GPL-3.0)** 协议开源。详见 [LICENSE](LICENSE) 文件。

---

**Topaz-Next** — 用数据探索市场，以理性对待投资

*最后更新：2026-05-11*
