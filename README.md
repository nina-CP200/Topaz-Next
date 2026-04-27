# Topaz-Next

> **开源量化分析系统** | 基于 LightGBM + 100+技术因子的 A 股量化分析系统
>
> 采用 GPL-3.0 协议，允许商业使用但衍生作品必须开源

---

## 项目简介

Topaz-Next 是一个面向个人研究者的 A 股量化分析系统，以**沪深300成分股**为选股池，通过特征工程构建 100+ 技术因子，输入 LightGBM 模型预测每只股票的短期上涨概率。

### 核心特性

- **沪深300股票池**：覆盖中国A股市场核心蓝筹股
- **100+ 技术因子**：价格位置、动量、波动率、成交量、技术指标等
- **市场环境判断**：自动识别牛市/熊市/震荡/反弹/回调
- **动态仓位建议**：根据市场环境调整建议仓位（20%-95%）
- **Slack推送**：自动发送分析报告到Slack频道

- 警告：当前项目发现了一个重大训练逻辑问题，目前正在修复，当前输出报告仅供参考。

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

脚本自动完成：
1. 安装 uv（如果未安装）
2. 创建 `.venv` 虚拟环境
3. 安装依赖库
4. 获取沪深300历史数据
5. 训练预测模型
6. 配置 Slack（可选）

### 手动配置

如果你不想用 setup.sh，也可以手动：

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

### 环境配置（可选）

如需 Slack 推送功能，运行后编辑 `.env` 文件：

```bash
vim .env
```

配置内容：
```
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
SLACK_CHANNEL=xxxxxxx
```

配置内容：
```
# Slack Bot Token（可选，用于推送报告）
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxx-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
SLACK_CHANNEL=xxxxxxx
```

### 运行分析

```bash
# 每日分析（首次运行会自动获取历史数据并训练模型）
uv run python -m src.analysis.daily

# 使用沪深300分组模型
uv run python -m src.analysis.daily --csi300
```

---

## 核心流程

```
步骤1：扫描沪深300成分股 → 获取实时行情 → 特征工程（100+因子）
       ↓
步骤2：市场环境判断 → LightGBM预测 → Top 5推荐 + Bottom 5回避
       ↓
步骤3：Slack推送报告（可选）
```

典型定时任务配置（可自定义时间）：
- 开盘后获取数据并分析
- 分析完成后推送报告

---

## 主要功能

### 1. 每日分析（src/analysis/daily.py）

- 判断市场环境（bull/bear/sideways/recovery/pullback）
- 计算模型置信度（0.5-0.9）
- 生成仓位建议（20%-95%）
- 输出 Top 5 建议买入股票
- 输出 Bottom 5 建议回避股票

### 2. 股票查询（src/analysis/query.py）

```bash
# 查询特定股票
uv run python -m src.analysis.query 600519      # 贵州茅台
uv run python -m src.analysis.query 000001.SZ   # 平安银行

# 显示排名前10股票
uv run python -m src.analysis.query --top 10
```

### 3. 模型训练（src/models/）

```bash
# Walk-Forward训练（推荐）
uv run python -m src.models.walkforward

# 基础训练
uv run python -m src.models.trainer
```

### 4. 回测系统（src/backtest/backtest.py）

```bash
# 回测模型表现
uv run python -m src.backtest.backtest
```

---

## 技术架构

### 特征工程（100+因子）

`src/features/engineer.py` 自动生成以下特征：

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
├── src/            # 源代码（分层架构）
│   ├── data/       # 数据层（API、缓存、获取器）
│   ├── features/   # 特征层（工程、验证）
│   ├── models/     # 模型层（训练、推理）
│   ├── analysis/   # 应用层（每日分析、查询）
│   ├── backtest/   # 回测系统
│   └── reports/    # 报告推送
├── scripts/        # Shell脚本（定时任务）
├── config/         # 配置文件（股票列表、环境变量）
├── data/           # 数据目录（运行时生成，不入Git）
├── setup.sh        # 一键配置脚本
├── requirements.txt
└── .venv/          # 虚拟环境（不入Git）
```

详细结构可用 `tree -L 3 src/` 查看。

### 数据与模型文件（不在Git中）

以下文件需运行脚本后自动生成，不包含在Git仓库中：

- `data/models/ensemble_model.pkl` - 训练后的模型文件
- `data/models/ensemble_scaler.pkl` - 特征标准化器
- `data/raw/csi300_full_history.csv` - 沪深300历史数据
- `data/cache/features/*.pkl` - 特征缓存文件

---

## 定时分析推荐：AstrBot

如果你希望实现**自动化定时分析**并将结果推送到 QQ、微信、Telegram 等即时通讯平台，强烈推荐使用 **AstrBot**：

- **多平台支持**：QQ、微信、Telegram、飞书等
- **插件生态**：丰富的插件系统，可轻松对接本项目的分析脚本
- **定时任务**：内置定时任务调度，无需手动配置 crontab
- **AI 对话**：支持接入各大 LLM，实现智能问答式分析

**GitHub 项目地址**：[https://github.com/AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot)

你可以将 Topaz-Next 的分析脚本封装为 AstrBot 插件，实现交易日自动推送分析报告、支持聊天查询股票等功能。

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
- Python 3.8+
- 4GB+ 内存
- 稳定网络连接

---

## 投资风险警示

⚠️ **本系统仅供学习研究，不构成任何投资建议**

1. **模型局限**：基于历史数据训练，市场结构变化可能导致模型失效
2. **数据风险**：免费API存在延迟或失效风险
3. **协议要求**：商业使用需遵守 GPL-3.0（衍生作品必须开源，禁止闭源商业化）

**市场有风险，投资需谨慎**

---

## 开源协议

本项目采用 **GNU General Public License v3.0 (GPL-3.0)** 协议开源。

**主要条款**：
- ✓ 自由使用、研究、修改和分发
- ✓ 允许商业使用（销售、收费服务、商业部署）
- ✗ 禁止闭源商业化（衍生作品必须开源）
- ✗ 禁止私有化修改（必须使用相同协议）

**商业使用说明**：
- 可以：商业部署、收费咨询、代客理财服务、集成到商业产品
- 必须：提供完整源代码、衍生作品使用 GPL-3.0、保留版权声明

详细协议：https://www.gnu.org/licenses/gpl-3.0.html

---

## 常见问题

### Q: 首次运行如何获取数据？

首次运行 `uv run python -m src.analysis.daily` 时，系统会自动：
1. 从腾讯/新浪API获取沪深300历史数据
2. 构建特征并缓存
3. 训练模型（如果没有现成模型）

整个过程约需5-10分钟。

### Q: 模型文件在哪里？

模型文件 `ensemble_model.pkl` 存储在 `data/models/` 目录，但**不包含在Git仓库中**。需要自行训练或从其他途径获取。

### Q: 如何更新股票池？

编辑 `config/csi300_stocks.json`，添加或删除股票代码。格式：
```json
{
  "code": "600519.SH",
  "name": "贵州茅台"
}
```

### Q: Slack推送失败？

检查 `.env` 配置（从 `config/.env.example` 复制）：
- SLACK_BOT_TOKEN 是否正确
- SLACK_CHANNEL 是否存在
- Bot是否有频道发送权限

---

**Topaz-Next** — 用数据探索市场，以理性对待投资

*最后更新：2026-04-25*
