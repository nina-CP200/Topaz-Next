#!/usr/bin/env python3
"""
================================================================================
模块名称: send_report.py
功能描述: 发送交易报告到 Slack 频道

通过 Slack Bot API 直接发送消息，支持 Block Kit 富格式消息，绕过 Agent 中转
以降低延迟，实现实时报告推送。

================================================================================
Slack 配置说明
================================================================================

1. 配置文件位置
   在项目根目录创建 .env 文件，包含以下配置：
   
   SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
   SLACK_CHANNEL=C0XXXXXXXX

2. 配置项说明
   - SLACK_BOT_TOKEN: Slack Bot 的 OAuth Token，用于 API 认证
   - SLACK_CHANNEL: 目标频道 ID（可以是公开频道、私有频道或用户 DM）

================================================================================
Slack Bot Token 获取步骤
================================================================================

步骤 1: 创建 Slack App
   - 访问 https://api.slack.com/apps
   - 点击 "Create New App"
   - 选择 "From scratch"
   - 输入 App 名称（如 "Topaz-Next Bot"）和选择工作区

步骤 2: 配置 Bot 权限（OAuth & Permissions）
   在 "Bot Token Scopes" 中添加以下权限：
   - chat:write      - 发送消息到频道
   - chat:write.public - 发送消息到公开频道（无需加入）
   - files:write     - 上传文件（可选）
   - users:read      - 读取用户信息（可选）

步骤 3: 安装 App 到工作区
   - 在 "OAuth & Permissions" 页面点击 "Install to Workspace"
   - 授权后，复制 "Bot User OAuth Token"（以 xoxb- 开头）
   - 将 Token 保存到 .env 文件的 SLACK_BOT_TOKEN 变量

步骤 4: 获取频道 ID
   方法一：在 Slack 中右键频道 → 复制链接 → 从 URL 中提取
   方法二：在 Slack 中输入 /list 命令查看
   方法三：使用 Slack API 测试工具

步骤 5: 邀请 Bot 到频道（私有频道需要）
   在目标频道中输入: /invite @Topaz-Next-Bot

================================================================================
"""

import sys
import json
import requests
import os
from datetime import datetime
from typing import Dict, List, Optional


# ==============================================================================
# 环境变量加载
# ==============================================================================
#
# .env 配置文件格式说明
# ----------------------
# .env 文件是一个简单的键值对配置文件，每行一个配置项：
#
# 示例 .env 文件内容:
#   # 这是注释行，会被忽略
#   SLACK_BOT_TOKEN=xoxb-1234567890-1234567890-abcdefabcdef1234567890
#   SLACK_CHANNEL=C1234567890
#
# 格式规则:
#   1. 每行格式: KEY=VALUE
#   2. 以 # 开头的行视为注释
#   3. 空行会被忽略
#   4. 等号左右可以有空白字符（会被自动去除）
#   5. 值不需要引号包裹（引号会被视为值的一部分）
#   6. 不支持多行值，每行必须是一个完整的配置项
#
# 优先级规则:
#   系统环境变量 > .env 文件中的配置
#   这意味着可以通过设置系统环境变量来覆盖 .env 中的配置
#
# ==============================================================================


def load_env(env_path: str = ".env") -> Dict[str, str]:
    """
    轻量级 .env 文件解析器
    
    此函数实现了一个简单的配置文件解析器，不依赖 python-dotenv 库，
    适合在没有额外依赖的环境中运行。
    
    参数:
        env_path: .env 文件的相对路径，默认为 ".env"（相对于脚本所在目录）
    
    返回:
        Dict[str, str]: 配置项字典，键值对均为字符串
    
    解析规则:
        1. 忽略空行和以 # 开头的注释行
        2. 以 = 分隔键值，等号左侧为键，右侧为值
        3. 自动去除键值两端的空白字符
        4. 系统环境变量会覆盖 .env 文件中的同名配置
    
    示例:
        >>> env = load_env()
        >>> token = env.get("SLACK_BOT_TOKEN")
    """
    env_vars = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, env_path)

    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    for key in ["SLACK_BOT_TOKEN", "SLACK_CHANNEL"]:
        if os.environ.get(key):
            env_vars[key] = os.environ.get(key)

    return env_vars


_ENV = load_env()


def get_slack_config() -> tuple:
    """
    获取 Slack 配置参数
    
    从已加载的环境变量中提取 Slack Bot Token 和目标频道 ID。
    
    返回:
        tuple: (token, channel)
            - token: Slack Bot OAuth Token（以 xoxb- 开头）
            - channel: 目标频道 ID，默认为用户 DM 频道
    
    默认值说明:
        如果未配置 SLACK_CHANNEL，默认为空字符串，
        此时需要在调用时手动指定频道，否则发送会失败。
    """
    token = _ENV.get("SLACK_BOT_TOKEN", "")
    channel = _ENV.get("SLACK_CHANNEL", "")
    return token, channel


# ==============================================================================
# 底层 Slack API 调用
# ==============================================================================
#
# Slack Web API 说明
# -------------------
# Slack 提供了一套 RESTful Web API，允许应用程序与 Slack 进行交互。
# 本模块使用 chat.postMessage API 端点发送消息到 Slack 频道。
#
# API 端点: https://slack.com/api/chat.postMessage
# 认证方式: Bearer Token（在 Authorization 头中传递）
# 请求格式: JSON
# 响应格式: JSON
#
# 常见错误码:
#   - invalid_auth: Token 无效或已过期
#   - channel_not_found: 频道不存在或 Bot 无访问权限
#   - not_in_channel: Bot 未加入该频道（私有频道）
#   - missing_scope: Token 缺少必要的 OAuth scope
#   - rate_limited: 请求频率超出限制
#
# ==============================================================================


def _send_slack_api(payload: Dict) -> tuple:
    """
    调用 Slack chat.postMessage API 发送消息
    
    此函数是所有 Slack 消息发送的底层实现，使用 requests 库调用 Slack API。
    
    参数:
        payload: Slack API 请求体，是一个字典，包含以下字段：
            - channel: 目标频道 ID（必需）
            - text: 消息文本内容（纯文本消息时必需）
            - blocks: Block Kit 消息块（富格式消息时使用）
            - username: 发送者显示名称（可选，Bot 默认使用 App 名称）
            - icon_emoji: 发送者头像 emoji（可选）
    
    返回:
        tuple: (success: bool, error_message: str)
            - success: True 表示发送成功
            - error_message: 错误信息，成功时为空字符串
    
    API 响应格式:
        成功响应: {"ok": true, "ts": "1234567890.123456", ...}
        失败响应: {"ok": false, "error": "error_code"}
    
    超时设置:
        API 请求超时时间为 30 秒，足够应对网络延迟。
    
    示例:
        >>> payload = {"channel": "C12345", "text": "Hello!"}
        >>> success, error = _send_slack_api(payload)
        >>> if success:
        ...     print("消息发送成功")
    """
    token, _ = get_slack_config()

    if not token:
        return False, "SLACK_BOT_TOKEN 未配置，请检查 .env 文件"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        try:
            resp = response.json()
        except json.JSONDecodeError:
            return False, f"Slack API 返回非 JSON: {response.text[:200]}"

        if resp.get("ok"):
            return True, ""
        else:
            error = resp.get("error", "unknown_error")
            return False, f"Slack API 错误: {error}"

    except requests.Timeout:
        return False, "Slack API 请求超时（30秒）"
    except requests.RequestException as e:
        return False, f"网络请求异常: {str(e)}"
    except Exception as e:
        return False, f"发送异常: {str(e)}"


# ==============================================================================
# Block Kit 消息构建
# ==============================================================================
#
# Slack Block Kit 说明
# ---------------------
# Block Kit 是 Slack 提供的一种消息构建框架，允许创建丰富的可视化消息。
# 通过组合不同类型的 "Block"（块），可以构建复杂的消息布局。
#
# 官方文档: https://api.slack.com/block-kit
# Block Kit Builder: https://app.slack.com/block-kit-builder
#
# 常用 Block 类型:
#   - header: 标题块，显示大号加粗文本
#   - section: 区块，可包含文本或字段列表
#   - divider: 分隔线
#   - context: 上下文块，显示小号灰色文本
#   - actions: 操作块，包含按钮等交互元素
#
# 文本格式类型:
#   - plain_text: 纯文本，不支持格式化
#   - mrkdwn: Markdown 格式，支持 *粗体*、`代码`、_斜体_ 等
#
# Section 字段布局:
#   Section 块的 fields 属性可包含最多 10 个字段，按两列排列。
#   每个 field 的 text 使用 mrkdwn 格式。
#
# 示例 Block 结构:
#   {
#       "type": "section",
#       "text": {
#           "type": "mrkdwn",
#           "text": "*标题*\\n内容"
#       }
#   }
#
# ==============================================================================


def build_score_ranking_blocks(
    results: List[Dict],
    market_regime: str,
    model_confidence: float,
    advance_ratio: float,
) -> List[Dict]:
    """
    构建评分排名的 Slack Block Kit 消息块
    
    此函数将股票分析结果转换为 Slack Block Kit 格式，生成包含以下内容的
    富格式消息：
        1. 标题块：显示报告日期
        2. 市场环境块：显示市场状态、置信度、上涨比例
        3. Top 5 评分最高块：显示评分最高的 5 只股票
        4. Bottom 7 评分最低块：显示评分最低的 7 只股票
        5. 风险提示块：显示免责声明和发送时间
    
    参数:
        results: 股票分析结果列表，每个元素为字典，包含以下字段：
            - symbol: 股票代码（如 "AAPL"）
            - name: 股票名称（如 "苹果公司"）
            - probability: 上涨概率（0-1 的浮点数）
            - predicted_return: 预期收益率（百分比）
            - advice: 投资建议（如 "强烈买入"）
        market_regime: 市场环境类型，可选值：
            - "bull": 牛市
            - "bear": 熊市
            - "sideways": 震荡市
            - "recovery": 复苏期
            - "pullback": 回调期
        model_confidence: 模型置信度（0-1），表示模型对预测的信心程度
        advance_ratio: 市场上涨比例（0-1），当日上涨股票占比
    
    返回:
        List[Dict]: Block Kit 消息块列表，可直接用于 Slack API 的 blocks 参数
    
    消息结构示例:
        ┌──────────────────────────────────────┐
        │ 📊 Topaz-Next 评分排名 - 2024-01-15  │  <- Header
        ├──────────────────────────────────────┤
        │ 市场环境: sideways    模型置信度: 65% │  <- Section (fields)
        │ 上涨比例: 52%         环境评估: ✅    │
        ├──────────────────────────────────────┤  <- Divider
        │ 🟢 评分最高 Top 5                     │  <- Section
        │ #1 `AAPL` *苹果*                      │
        │   概率: 78% | 预期收益: 📈 +5.2%      │
        │ ...                                  │
        ├──────────────────────────────────────┤
        │ 🔴 评分最低 Bottom 7                  │  <- Section
        │ #1 `XYZ` *某公司*                    │
        │ ...                                  │
        ├──────────────────────────────────────┤
        │ ⚠️ 风险提示：本分析仅供参考...        │  <- Context
        └──────────────────────────────────────┘
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    regime_effect = {
        "recovery": "✅ 最佳环境（IC=0.15）",
        "pullback": "✅ 有效环境（IC=0.11）",
        "bull": "✅ 有效环境（IC=0.10）",
        "bear": "⚠️ 熊市保守（IC=0.06）",
        "sideways": "📊 震荡分散（IC=0.03）",
    }
    effect_note = regime_effect.get(market_regime, "未知")

    blocks = []

    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📊 Topaz-Next 评分排名 - {today_str}",
                "emoji": True,
            },
        }
    )

    blocks.append(
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*市场环境:*\n{market_regime}"},
                {"type": "mrkdwn", "text": f"*模型置信度:*\n{model_confidence:.0%}"},
                {"type": "mrkdwn", "text": f"*上涨比例:*\n{advance_ratio:.1%}"},
                {"type": "mrkdwn", "text": f"*环境评估:*\n{effect_note}"},
            ],
        }
    )

    blocks.append({"type": "divider"})

    sorted_by_prob = sorted(
        results, key=lambda x: x.get("probability", 0), reverse=True
    )
    top5 = sorted_by_prob[:5]

    blocks.append(
        {"type": "section", "text": {"type": "mrkdwn", "text": "🟢 *评分最高 Top 5*"}}
    )

    for i, stock in enumerate(top5, 1):
        symbol = stock.get("symbol", "")
        name = stock.get("name", "")
        prob = stock.get("probability", 0)
        pred_ret = stock.get("predicted_return", 0)
        advice = stock.get("advice", "")

        ret_emoji = "📈" if pred_ret >= 0 else "📉"

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"#{i} `{symbol}` *{name}*\n"
                        f"  概率: *{prob:.1%}* | 预期收益: {ret_emoji} *{pred_ret:+.1f}%* | {advice}"
                    ),
                },
            }
        )

    blocks.append({"type": "divider"})

    bottom7 = sorted_by_prob[-7:][::-1]

    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "🔴 *评分最低 Bottom 7（建议回避）*"},
        }
    )

    for i, stock in enumerate(bottom7, 1):
        symbol = stock.get("symbol", "")
        name = stock.get("name", "")
        prob = stock.get("probability", 0)
        pred_ret = stock.get("predicted_return", 0)
        advice = stock.get("advice", "")

        ret_emoji = "📈" if pred_ret >= 0 else "📉"

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"#{i} `{symbol}` *{name}*\n"
                        f"  概率: *{prob:.1%}* | 预期收益: {ret_emoji} *{pred_ret:+.1f}%* | {advice}"
                    ),
                },
            }
        )

    blocks.append({"type": "divider"})

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"⚠️ 风险提示：本分析仅供参考，不构成投资建议。"
                        f"  发送时间: {time_str}"
                    ),
                }
            ],
        }
    )

    return blocks


# ==============================================================================
# 公开 API
# ==============================================================================
#
# 本模块提供以下公开函数供外部调用：
#
# 1. send_score_ranking()
#    发送评分排名报告（Block Kit 富格式），直接调用 Slack API
#    适用场景：daily_decision.py 分析完成后立即调用
#
# 2. send_slack_message()
#    发送纯文本消息（简单格式），保留旧版兼容性
#    适用场景：发送简单的文本通知或交易执行报告
#
# 3. parse_report()
#    解析报告输出文本，生成可读的消息格式
#    适用场景：处理 Agent 生成的报告文本
#
# ==============================================================================


def send_score_ranking(
    results: List[Dict],
    market_regime: str = "sideways",
    model_confidence: float = 0.5,
    advance_ratio: float = 0.5,
) -> bool:
    """
    发送评分排名报告（Top 5 + Bottom 7）到 Slack
    
    此函数是本模块的核心功能，用于发送包含股票评分排名的富格式消息。
    在 daily_decision.py 分析完成后立即调用，绕过 Agent 中转以降低延迟。
    
    使用流程:
        1. 在 daily_decision.py 中调用此函数
        2. 传入股票分析结果列表和市场环境参数
        3. 函数自动构建 Block Kit 消息并发送到 Slack
    
    参数:
        results: 股票分析结果列表，每个元素为包含以下字段的字典：
            - symbol: str - 股票代码（如 "AAPL"）
            - name: str - 股票名称（如 "苹果公司"）
            - probability: float - 上涨概率（0-1）
            - predicted_return: float - 预期收益率（百分比）
            - advice: str - 投资建议（如 "强烈买入"、"观望"）
        market_regime: 市场环境类型，可选值：
            - "bull": 牛市 - 整体上涨趋势
            - "bear": 熊市 - 整体下跌趋势
            - "sideways": 震荡市 - 无明显趋势
            - "recovery": 复苏期 - 从熊市转向牛市
            - "pullback": 回调期 - 牛市中的短期调整
            默认值: "sideways"
        model_confidence: 模型置信度（0-1），表示模型对预测的信心程度
            - 0.8+ 高置信度，建议参考
            - 0.5-0.8 中等置信度
            - <0.5 低置信度，谨慎参考
            默认值: 0.5
        advance_ratio: 市场上涨比例（0-1），当日上涨股票数量占比
            用于判断市场整体表现
            默认值: 0.5
    
    返回:
        bool: 发送是否成功
            - True: 消息已成功发送到 Slack
            - False: 发送失败，错误信息会打印到控制台
    
    消息格式:
        消息使用 Slack Block Kit 构建，包含以下部分：
        1. 标题：报告日期
        2. 市场环境：环境类型、置信度、上涨比例、环境评估
        3. Top 5：评分最高的 5 只股票（按概率降序）
        4. Bottom 7：评分最低的 7 只股票（按概率升序）
        5. 风险提示：免责声明和发送时间
    
    示例:
        >>> results = [
        ...     {"symbol": "AAPL", "name": "苹果", "probability": 0.78, 
        ...      "predicted_return": 5.2, "advice": "强烈买入"},
        ...     {"symbol": "MSFT", "name": "微软", "probability": 0.72,
        ...      "predicted_return": 4.1, "advice": "买入"},
        ... ]
        >>> success = send_score_ranking(
        ...     results, 
        ...     market_regime="bull",
        ...     model_confidence=0.85,
        ...     advance_ratio=0.65
        ... )
    """
    if not results:
        print("⚠️ send_score_ranking: 无分析结果，跳过发送")
        return False

    _, channel = get_slack_config()
    blocks = build_score_ranking_blocks(
        results, market_regime, model_confidence, advance_ratio
    )

    payload = {
        "channel": channel,
        "blocks": blocks,
        "text": f"Topaz-Next 评分排名 - {datetime.now().strftime('%Y-%m-%d')}",
    }

    success, error = _send_slack_api(payload)

    if success:
        print("✓ 评分排名已直接发送至 Slack（绕过 Agent）")
        return True
    else:
        print(f"🚨 AGENT_NOTIFY: Slack 评分排名发送失败")
        print(f"   原因: {error}")
        print(f"   请检查 .env 中的 SLACK_BOT_TOKEN 和网络连接")
        return False


def send_slack_message(text: str, channel: str = None) -> bool:
    """
    发送纯文本 Slack 消息（保留旧版 API 兼容性）
    
    此函数用于发送简单的文本消息，不支持 Block Kit 富格式。
    主要用于发送交易执行报告等由 Agent 处理的完整报告。
    
    参数:
        text: 消息文本内容，支持 Slack mrkdwn 格式：
            - *粗体*: 用星号包裹
            - _斜体_: 用下划线包裹
            - `代码`: 用反引号包裹
            - ~删除线~: 用波浪线包裹
            - > 引用: 以大于号开头
        channel: 目标频道 ID，如果为 None 则使用 .env 中的默认频道
    
    返回:
        bool: 发送是否成功
    
    消息格式:
        发送的消息会附加以下自定义属性：
        - username: 显示名称为 "佩丽卡"
        - icon_emoji: 头像使用 🔮 emoji
    
    示例:
        >>> success = send_slack_message("*交易完成*\\n已买入 AAPL")
        >>> success = send_slack_message("测试消息", channel="C12345")
    """
    if channel is None:
        _, channel = get_slack_config()

    payload = {
        "channel": channel,
        "text": text,
        "username": "佩丽卡",
        "icon_emoji": ":crystal_ball:",
    }

    success, error = _send_slack_api(payload)
    if not success:
        print(f"⚠️ Slack 消息发送失败: {error}")
    return success


def parse_report(output: str, report_type: str) -> str:
    """
    解析报告输出文本，生成可读的 Slack 消息格式
    
    此函数用于解析 daily_decision.py 生成的报告文本，提取关键信息
    并格式化为适合 Slack 显示的消息格式。
    
    参数:
        output: 原始报告文本，包含以下部分（以 === 分隔）：
            - 大盘环境: 市场状态和趋势分析
            - 建议买入: 推荐买入的股票列表
            - 建议卖出: 推荐卖出的股票列表
            - 持仓汇总: 当前持仓和资金状态
        report_type: 报告类型，影响消息标题和结尾：
            - "execute": 下午交易执行报告（已执行）
            - "preview": 明日交易预告（未执行）
    
    返回:
        str: 格式化后的消息文本，适合直接发送到 Slack
    
    消息结构:
        📊 下午交易执行报告 / 明日交易预告
        
        📈 大盘环境
          [市场环境信息]
        
        ✅ 买入
          [买入建议]
        
        ❌ 卖出
          [卖出建议]
        
        💼 当前持仓
          [持仓信息]
        
        ✅ 交易已执行，持仓已更新 / 📌 这是预告，明日开盘后才会执行交易
    
    示例:
        >>> output = "=== 大盘环境 ===\\n上涨趋势\\n=== 建议买入 ===\\n1. AAPL"
        >>> msg = parse_report(output, "preview")
        >>> print(msg)
        📊 明日交易预告
        
        📈 大盘环境
          上涨趋势
        ...
    """
    lines = output.split("\n")

    market_info = []
    buy_info = []
    sell_info = []
    holdings_info = []

    section = None
    for line in lines:
        if "大盘环境" in line:
            section = "market"
        elif "建议买入" in line:
            section = "buy"
        elif "建议卖出" in line:
            section = "sell"
        elif "持仓汇总" in line:
            section = "holdings"
        elif line.startswith("==="):
            section = None
        elif section == "market" and line.strip():
            market_info.append(line)
        elif section == "buy" and line.strip() and line.strip()[0].isdigit():
            buy_info.append(line)
        elif section == "sell" and line.strip() and line.strip()[0].isdigit():
            sell_info.append(line)
        elif (
            section == "holdings"
            and line.strip()
            and ("现金" in line or "持仓" in line or "总资" in line or "盈亏" in line)
        ):
            holdings_info.append(line)

    if report_type == "execute":
        title = "📊 下午交易执行报告"
        footer = "✅ 交易已执行，持仓已更新"
    else:
        title = "📊 明日交易预告"
        footer = "📌 这是预告，明日开盘后才会执行交易"

    msg_parts = [title, ""]

    if market_info:
        msg_parts.append("📈 大盘环境")
        for line in market_info[:4]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    if buy_info:
        msg_parts.append("✅ 买入")
        for line in buy_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    if sell_info:
        msg_parts.append("❌ 卖出")
        for line in sell_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    if holdings_info:
        msg_parts.append("💼 当前持仓")
        for line in holdings_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    msg_parts.append(footer)

    return "\n".join(msg_parts)


def main():
    """
    命令行入口函数
    
    支持通过命令行直接调用此脚本发送报告。
    
    用法:
        python send_report.py <execute|preview> <输出内容或文件路径>
    
    参数:
        execute: 发送交易执行报告（已执行）
        preview: 发送交易预告（未执行）
    
    示例:
        # 从文件读取并发送执行报告
        python send_report.py execute /tmp/report.txt
        
        # 直接传入文本并发送预告
        python send_report.py preview "=== 大盘环境 ===\\n上涨趋势"
    """
    if len(sys.argv) < 3:
        print("用法: send_report.py <execute|preview> <输出内容>")
        sys.exit(1)

    report_type = sys.argv[1]
    output_file = sys.argv[2]

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            output = f.read()
    else:
        output = output_file

    message = parse_report(output, report_type)

    if send_slack_message(message):
        print("✓ 报告已发送")
    else:
        print("✗ 发送失败")


# ==============================================================================
# 模块使用示例
# ==============================================================================
#
# 1. 作为命令行工具使用:
#    python send_report.py execute report.txt
#
# 2. 作为模块导入使用:
#    from send_report import send_score_ranking, send_slack_message
#    
#    # 发送评分排名（推荐方式）
#    results = [
#        {"symbol": "AAPL", "name": "苹果", "probability": 0.78, 
#         "predicted_return": 5.2, "advice": "强烈买入"},
#    ]
#    send_score_ranking(results, market_regime="bull", model_confidence=0.85)
#    
#    # 发送简单文本消息
#    send_slack_message("*交易完成*\\n已买入 AAPL")
#
# 3. 自定义消息格式:
#    使用 build_score_ranking_blocks() 函数可以自定义消息内容，
#    然后通过 _send_slack_api() 发送自定义的 payload。
#
# ==============================================================================

if __name__ == "__main__":
    main()
