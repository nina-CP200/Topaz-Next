# Topaz-Next

> **非商用开源项目** | 基于 LightGBM + 技术特征因子的 A 股量化分析系统。
>
> 仅供学习研究使用，**禁止用于任何商业目的或实际资金交易**。作者不对任何投资损失承担责任。

---

## 项目简介

Topaz-Next 是一个面向个人研究者的 A 股量化分析系统。以**沪深300成分股**为选股池，通过特征工程构建 100+ 技术因子，输入 LightGBM 模型预测每只股票的短期上涨概率，帮助投资者进行量化分析参考。

## 核心流程

```
09:45  扫描沪深300成分股 → 获取行情 → 特征工程 → LightGBM 预测概率
       └─ 市场环境判断 → Slack 直发：Top 5 推荐 + Bottom 5 回避
```

---

## 模型与特征

### 特征工程（100+ 因子）

`feature_engineer.py` 从 OHLCV 数据自动生成以下特征：

---

#### 1. 价格位置因子（~8个）

衡量当前价格在历史价格区间中的相对位置。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| price_to_ma5 | close / ma5 - 1 | 相对5日均线偏离度 | 短期超买超卖判断 |
| price_to_ma10 | close / ma10 - 1 | 相对10日均线偏离度 | 中短期趋势判断 |
| price_to_ma20 | close / ma20 - 1 | 相对20日均线偏离度 | 中期趋势判断 |
| price_to_ma60 | close / ma60 - 1 | 相对60日均线偏离度 | 长期趋势判断 |
| price_position_10 | (close - low10) / (high10 - low10) | 10日区间位置 | 短期支撑阻力判断 |
| price_position_20 | (close - low20) / (high20 - low20) | 20日区间位置 | 中期支撑阻力判断 |
| close_position | (close - low) / (high - low) | 日内收盘位置 | 当日买卖力量判断 |

**解读示例**：
- price_to_ma20 > 0.05 → 价格高于20日均线5%，短期偏强
- price_position_20 > 0.8 → 价格在20日高位区，注意回调风险

---

#### 2. 动量因子（~12个）

捕捉价格趋势和变化速度。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| return_1d | close[t] / close[t-1] - 1 | 1日收益率 | 短期涨跌判断 |
| return_5d | close[t] / close[t-5] - 1 | 5日收益率 | 周度趋势判断 |
| return_10d | close[t] / close[t-10] - 1 | 10日收益率 | 双周趋势判断 |
| return_20d | close[t] / close[t-20] - 1 | 20日收益率 | 月度趋势判断 |
| tsmom_lb25 | close[t] / close[t-25] - 1 | 25日时间序列动量 | CTA策略核心因子 |
| tsmom_lb60 | close[t] / close[t-60] - 1 | 60日时间序列动量 | 中期趋势强度 |
| momentum_accel_5 | return_5d[t] - return_5d[t-5] | 5日动量加速度 | 趋势加速/减速判断 |
| ma_cross_5_20 | ma5 > ma20 ? 1 : 0 | 5/20日均线交叉 | 金叉死叉信号 |

**解读示例**：
- return_5d > 0.1 → 5日上涨10%，短期强势
- momentum_accel_5 > 0 → 动量正在加速，趋势延续概率高

---

#### 3. 波动率因子（~10个）

衡量价格波动程度和风险水平。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| volatility_5 | std(return_1d, 5) | 5日历史波动率 | 短期风险水平 |
| volatility_10 | std(return_1d, 10) | 10日历史波动率 | 中短期风险水平 |
| volatility_20 | std(return_1d, 20) | 20日历史波动率 | 中期风险水平 |
| volatility_60 | std(return_1d, 60) | 60日历史波动率 | 长期风险水平 |
| vol_ewma | EWMA(return_1d², span=20) | 指数加权波动率 | 近期波动敏感 |
| atr_14 | max(H-L, H-C_prev, L-C_prev) 的14日均值 | 平均真实波幅 | 考虑跳空的波动度量 |
| skewness_20 | skew(return_1d, 20) | 20日收益偏度 | 分布不对称程度 |
| kurtosis_20 | kurtosis(return_1d, 20) | 20日收益峰度 | 尾部风险程度 |

**解读示例**：
- volatility_20 > 0.03 → 20日波动率3%，风险较高
- skewness_20 < 0 → 收益分布左偏，下跌概率较大

---

#### 4. 成交量因子（~5个）

衡量成交活跃度和资金流向。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| volume_ratio | volume[t] / ma(volume, 5) | 量比（相对5日均量） | 放量缩量判断 |
| volume_ma5 | ma(volume, 5) | 5日平均成交量 | 成交基准水平 |
| obv | 累计：上涨加量，下跌减量 | 能量潮指标 | 资金流向判断 |
| amihud_illiq | abs(return_1d) / volume | Amihud非流动性 | 价格对成交敏感度 |
| vol_price_corr | corr(volume, close, 20) | 量价相关性 | 量价配合程度 |

**解读示例**：
- volume_ratio > 2 → 成交量是均量的2倍，异常放量
- obv上升但价格下跌 → 资金流入，可能反转

---

#### 5. 技术指标因子（~12个）

经典技术分析指标的量化实现。

| 因子名称 | 参数 | 含义 | 应用场景 |
|---------|------|------|---------|
| rsi_6 | 6日RSI | 短期相对强弱 | 超买超卖（>70/<30） |
| rsi_14 | 14日RSI | 标准相对强弱 | 中期超买超卖 |
| rsi_24 | 24日RSI | 长期相对强弱 | 长期趋势强度 |
| macd | MACD线（12-26EMA差） | 趋势方向和强度 | 趋势判断 |
| macd_signal | MACD信号线（9日EMA） | MACD平滑线 | 金叉死叉判断 |
| macd_hist | MACD柱状图 | MACD与信号线差 | 趋势加速判断 |
| kdj_k | K值（9日随机指标） | 快速随机值 | 短期超买超卖 |
| kdj_d | D值（K的3日平滑） | 慢速随机值 | 中期超买超卖 |
| bb_position | (close - bb_lower) / (bb_upper - bb_lower) | 布林带位置 | 波动区间判断 |
| cci_20 | (TP - ma(TP,20)) / (0.015 * mad(TP,20)) | 商品通道指标 | 超买超卖（>100/<-100） |
| adx_14 | 平均趋向指标 | 趋势强度（不论方向） | 趋势明显程度 |
| williams_r | (high_14 - close) / (high_14 - low_14) * -100 | Williams %R | 超买超卖（>-20/<-80） |

**解读示例**：
- rsi_14 > 70 → 超买，可能回调
- macd_hist > 0 且上升 → 多头趋势加速
- bb_position > 1 → 突破布林带上轨，强势或反转

---

#### 6. 价格形态因子（~5个）

识别经典K线形态。

| 因子名称 | 计算方法 | 含义 | 应用场景 |
|---------|---------|------|---------|
| candle_body | (close - open) / open | K线实体大小 | 多空力量强度 |
| upper_shadow | (high - max(open,close)) / open | 上影线比例 | 上方压力程度 |
| lower_shadow | (min(open,close) - low) / open | 下影线比例 | 下方支撑程度 |
| hammer | 实体小+下影线长+上影线短 | 锤子线形态 | 底部反转信号 |
| shooting_star | 实体小+上影线长+下影线短 | 射击之星形态 | 顶部反转信号 |

**解读示例**：
- hammer = 1 → 锤子线，可能底部反转
- candle_body > 0.03 → 大阳线，多头强势

---

#### 7. 统计因子（~6个）

统计特征和风险度量。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| mean_reversion_20 | (close - ma20) / std(close, 20) | 20日Z-Score | 均值回归程度 |
| mean_reversion_60 | (close - ma60) / std(close, 60) | 60日Z-Score | 长期均值回归 |
| price_percentile_20 | percentile(close, 20) | 20日价格分位数 | 相对历史位置 |
| price_percentile_60 | percentile(close, 60) | 60日价格分位数 | 长期相对位置 |
| max_drawdown_20 | max(close[i]/close[j] - 1) for i<j in 20d | 20日最大回撤 | 短期风险度量 |
| sharpe_proxy | mean(return_1d) / std(return_1d) * sqrt(252) | 夏普比率代理 | 风险调整收益 |

**解读示例**：
- mean_reversion_20 > 2 → 价格显著高于均值，可能回调
- max_drawdown_20 > -0.1 → 20日最大回撤10%，风险较高

---

#### 8. 大盘指数因子（~9个）

沪深300指数相关因子，衡量个股相对大盘表现。

| 因子名称 | 计算公式 | 含义 | 应用场景 |
|---------|---------|------|---------|
| index_return_1d | 沪深300 1日收益 | 大盘当日涨跌 | 市场整体表现 |
| index_return_5d | 沪深300 5日收益 | 大盘周度表现 | 市场中期趋势 |
| index_ma_position | 沪深300相对均线位置 | 大盘趋势位置 | 市场环境判断 |
| index_volatility | 沪深300波动率 | 大盘风险水平 | 市场整体风险 |
| relative_return | return_1d - index_return_1d | 个股超额收益 | 相对强度判断 |
| beta | corr(return, index_return) * std(return)/std(index) | 个股Beta系数 | 系统性风险暴露 |

---

#### 9. 连续涨跌因子（~3个）

统计连续上涨/下跌天数。

| 因子名称 | 计算方法 | 含义 | 应用场景 |
|---------|---------|------|---------|
| consecutive_up | 连续上涨天数统计 | 连续阳线天数 | 短期强势程度 |
| consecutive_down | 连续下跌天数统计 | 连续阴线天数 | 短期弱势程度 |
| consecutive_count | abs(consecutive_up - consecutive_down) | 连续涨跌幅度 | 趋势强度 |

**解读示例**：
- consecutive_up >= 5 → 连续5日上涨，注意回调
- consecutive_down >= 3 → 连续3日下跌，可能反弹

---

### 因子使用建议

1. **因子筛选**：使用 `model.feature_importance_` 查看因子重要性，保留Top 30-50因子
2. **因子中性化**：对行业/市值中性化，降低共线性
3. **因子更新**：定期（月度/季度）重新训练，适应市场变化
4. **因子组合**：多因子叠加使用，避免单一因子依赖

---

### 模型

生产路径使用 **LightGBM 分类模型**。模型输出每只股票未来上涨的**概率**（0~1），用于分析参考。

模型训练脚本位于 `retrain_model.py` 和 `retrain_model_walkforward.py`（需手动运行）。

---

## 市场环境判断

根据沪深300的 **20日收益** 和 **市场上涨比例** 判断环境：

| 环境 | 条件 | 模型置信度 | 建议仓位 |
|------|------|-----------|---------|
| 反弹 | 上涨>55% 且 20日收益<-2% | 90% | 95% |
| 回调 | 上涨<45% 且 20日收益>2% | 80% | 80% |
| 牛市 | 上涨>55% 且 20日收益>0 | 70% | 70% |
| 熊市 | 上涨<45% 且 20日收益<-2% | 60% | 20% |
| 震荡 | 其他 | 50% | 50% |

---

## 一键安装

```bash
# 自动检测网络环境并安装
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash

# 中国大陆用户（强制使用镜像源）
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash -s -- --china
```

安装完成后，运行分析：
```bash
cd ~/topaz-next && python3 daily_decision.py
```

### OpenCode Prompt

将以下内容发送给 AI Agent：

```
帮我安装 Topaz-Next：
curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash

安装后运行：
cd ~/topaz-next && python3 daily_decision.py

定时任务：
crontab -e
45 9 * * 1-5 /bin/bash ~/topaz-next/daily_report.sh
0 10 * * 1-5 /bin/bash ~/topaz-next/daily_decision.sh
```

---

## 股票查询

运行分析后，可使用查询脚本查看特定股票的评分和排名：

```bash
# 查询特定股票（支持多种格式）
python query_stock.py 600519      # 纯数字代码
python query_stock.py 600519.SH   # 带.SH后缀
python query_stock.py 000001.SZ   # 深圳股票

# 显示排名前N的股票
python query_stock.py --top 10
```

查询结果包含：
- 股票名称、代码
- 总排名和百分位（Top 10%/33%等）
- 上涨概率、预期收益
- 当前价格、涨跌幅
- 风险等级、投资建议

---

## 定时任务配置

### Crontab 配置

```bash
crontab -e
```

添加以下内容（替换为实际项目路径）：

```cron
# Topaz-Next 每日定时任务（工作日，CST）
# 09:45 生成 A 股分析报告
45 9 * * 1-5 /bin/bash /path/to/topaz-next/daily_report.sh

# 10:00 运行分析并发送 Slack
0 10 * * 1-5 /bin/bash /path/to/topaz-next/daily_decision.sh
```

---

## 项目结构

```
topaz-next/
├── daily_decision.py              # 每日分析（主入口）
├── query_stock.py                 # 股票查询脚本
├── feature_engineer.py            # 特征工程（100+因子）
├── ensemble_model.py              # Stacking集成模型
├── retrain_model.py               # 模型训练
├── market_data.py                 # 大盘数据与环境判断
├── quantpilot_data_api.py         # 数据获取（腾讯/新浪API）
├── send_report.py                 # Slack 报告推送
├── cache_manager.py               # 特征缓存管理
├── backtest.py                    # 回测系统
├── fetch_full_history.py          # 历史数据获取
├── setup.sh                       # 启动引导脚本
├── .env.example                   # 环境变量模板
├── csi300_stocks.json             # 沪深300股票列表
├── requirements.txt               # 依赖列表
├── LICENSE                        # GPL-3.0 协议
└── README.md                      # 项目文档
```

---

## 投资风险警示

⚠️ **本系统仅供学习研究，不构成任何投资建议。**

1. **模型局限**：基于历史数据训练，市场结构变化可能导致模型失效。A股免费数据源存在延迟。
2. **技术风险**：API 可能因网络问题或服务商调整而失效。
3. **合规声明**：**不推荐用于任何商业目的**，包括但不限于代客理财、收费咨询、基金产品等。

**市场有风险，投资需谨慎。**

---

## 开源协议

本项目采用 **GNU General Public License v3.0 (GPL-3.0)** 协议开源。

### 主要条款

- **自由使用** — 您可以自由运行、研究、修改和分发本软件
- **衍生作品** — 修改后的作品必须同样使用 GPL-3.0 协议
- **源代码** — 分发时必须提供源代码或获取源代码的方式
- **版权声明** — 必须保留原作者版权声明和协议声明

### 您的权利

1. 运行本软件用于任何目的
2. 研究本软件并修改以满足您的需求
3. 分发本软件副本帮助他人
4. 分发您修改后的版本

### 您的义务

1. 分发时必须附带 GPL-3.0 协议全文
2. 修改后必须标明修改内容并保持 GPL-3.0 协议
3. 分发源代码（或提供获取方式）

详细协议：[https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)

---

**Topaz-Next** — 用数据探索市场，以理性对待投资。

*最后更新：2026-04-23*