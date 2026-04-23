📋 **Topaz-Next 系统工作流变更通知**

## 变更概述

Slack 评分排名报告的发送方式已变更：**不再由你（Agent）中转转发**，改为由 `daily_decision.py` 分析完成后**直接调用 Slack API 发送**。你的职责范围已相应调整。

---

## 你的新职责

### ✅ 你仍然需要做的
1. **触发定时任务**：在 09:45 (CST, 工作日) 运行 `daily_report.sh` 和 `daily_decision.sh`
2. **转发剩余报告内容**：`daily_decision.py` 打印到 stdout 的完整报告中，**"买谁卖谁"和"操作完结果怎么样"** 这两部分仍需要你读取并转发到 Slack
3. **错误通知**：如果看到输出中包含 `🚨 AGENT_NOTIFY:` 标记，立即通知我（见下方说明）

### ❌ 你**不再**需要做的
- **不要**再尝试从日志中 grep/解析评分排名信息并发送到 Slack
- `daily_decision.py` 现在会在分析完成的瞬间（约 09:47）自动将 **Top 5 评分最高** 和 **Bottom 7 评分最低** 发送到 Slack，比你中转更快

---

## 你需要知道的细节

### 1. 评分排名直发机制
- **发送方**：`daily_decision.py` 内部调用 `send_report.send_score_ranking()`
- **发送时机**：`analyze_stocks()` 返回结果后、生成交易决策**之前**
- **消息格式**：Slack Block Kit 富格式（带分区、emoji、分栏）
- **消息内容**：
  - 市场环境（regime / 置信度 / 上涨比例）
  - 🟢 Top 5 评分最高（代码、名称、概率、预期收益、建议）
  - 🔴 Bottom 7 评分最低（建议回避）
  - 风险提示 + 时间戳

### 2. 失败通知机制（重要）
如果代码直发 Slack 失败（网络问题、Token 失效等），`daily_decision.py` 会输出以下格式的日志：

```
🚨 AGENT_NOTIFY: Slack 评分排名发送失败
   原因: <具体错误信息>
   请检查 .env 中的 SLACK_BOT_TOKEN 和网络连接
```

**你的处理规则**：
- 如果看到 `🚨 AGENT_NOTIFY:`，**立即通知我**，告诉我失败原因
- 这是代码直发失败后的兜底机制，你需要帮我确认是否需要手动补发

### 3. Token 已迁移
- Slack Token 已从代码硬编码迁移到 `.env` 文件中的 `SLACK_BOT_TOKEN` 和 `SLACK_CHANNEL`
- 你不需要知道 Token 的具体值，代码自己会读取
- 如果 Token 失效，你会看到 `AGENT_NOTIFY` 通知

---

## 示例时间线

```
09:45:00  你触发 daily_decision.sh
09:47:00  daily_decision.py 完成股票分析
09:47:01  ⭐ 代码直发 Slack: Top 5 + Bottom 7 评分排名（你已不需要处理）
09:47:05  daily_decision.py 继续生成买卖决策、更新持仓
09:47:10  daily_decision.py 打印完整报告到 stdout
09:47:15  你读取 stdout，转发"买谁卖谁"和"持仓汇总"到 Slack
```

---

## 如果出现问题

| 现象 | 你的操作 |
|------|---------|
| 看到 `AGENT_NOTIFY` | 立即通知我，附上报错原因 |
| Slack 没收到评分排名 | 先检查 daily_decision.py 日志是否有 `AGENT_NOTIFY`；如果没有，检查脚本是否正常执行 |
| 收到两份评分排名 | 说明你**仍**在转发，请停止转发评分排名部分 |

---

**总结**：评分排名直发已上线，你的延迟消除了。你只需要关注兜底通知和剩余报告的转发。有任何异常立即告诉我。
