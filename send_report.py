#!/usr/bin/env python3
"""
发送交易报告到 Slack
通过 Slack Bot API 直接发送，支持 Block Kit 富格式
"""

import sys
import json
import subprocess
import os
from datetime import datetime
from typing import Dict, List, Optional


# ========== 环境变量加载 ==========


def load_env(env_path: str = ".env") -> Dict[str, str]:
    """轻量级 .env 文件解析器（不依赖 python-dotenv）"""
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

    # 环境变量优先级高于 .env 文件
    for key in ["SLACK_BOT_TOKEN", "SLACK_CHANNEL"]:
        if os.environ.get(key):
            env_vars[key] = os.environ.get(key)

    return env_vars


_ENV = load_env()


def get_slack_config() -> tuple:
    """获取 Slack 配置，返回 (token, channel)"""
    token = _ENV.get("SLACK_BOT_TOKEN", "")
    channel = _ENV.get("SLACK_CHANNEL", "U0AGVSHJ08Z")
    return token, channel


# ========== 底层 Slack API 调用 ==========


def _send_slack_api(payload: Dict) -> tuple:
    """
    调用 Slack chat.postMessage API
    返回 (success: bool, error_message: str)
    """
    token, _ = get_slack_config()

    if not token:
        return False, "SLACK_BOT_TOKEN 未配置，请检查 .env 文件"

    cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        "https://slack.com/api/chat.postMessage",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Content-Type: application/json; charset=utf-8",
        "-d",
        json.dumps(payload, ensure_ascii=False),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"curl 执行失败: {result.stderr}"

        try:
            resp = json.loads(result.stdout)
        except json.JSONDecodeError:
            return False, f"Slack API 返回非 JSON: {result.stdout[:200]}"

        if resp.get("ok"):
            return True, ""
        else:
            error = resp.get("error", "unknown_error")
            return False, f"Slack API 错误: {error}"

    except subprocess.TimeoutExpired:
        return False, "Slack API 请求超时（30秒）"
    except Exception as e:
        return False, f"发送异常: {str(e)}"


# ========== Block Kit 消息构建 ==========


def build_score_ranking_blocks(
    results: List[Dict],
    market_regime: str,
    model_confidence: float,
    advance_ratio: float,
) -> List[Dict]:
    """构建评分排名的 Slack Block Kit 消息块"""

    today_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    # 环境说明映射
    regime_effect = {
        "recovery": "✅ 最佳环境（IC=0.15）",
        "pullback": "✅ 有效环境（IC=0.11）",
        "bull": "✅ 有效环境（IC=0.10）",
        "bear": "⚠️ 熊市保守（IC=0.06）",
        "sideways": "📊 震荡分散（IC=0.03）",
    }
    effect_note = regime_effect.get(market_regime, "未知")

    blocks = []

    # 1. Header
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

    # 2. 市场环境信息
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

    # 3. Top 5 评分最高
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

        # 预期收益颜色标记
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

    # 4. Bottom 7 评分最低
    bottom7 = sorted_by_prob[-7:][::-1]  # 反转，让最低的在最前面

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

    # 5. 风险提示 + 时间戳
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


# ========== 公开 API ==========


def send_score_ranking(
    results: List[Dict],
    market_regime: str = "sideways",
    model_confidence: float = 0.5,
    advance_ratio: float = 0.5,
) -> bool:
    """
    发送评分排名报告（Top 5 + Bottom 7）到 Slack

    在 daily_decision.py 分析完成后立即调用，绕过 Agent 中转以降低延迟。

    Args:
        results: 股票分析结果列表，每个元素为包含 symbol, name, probability, predicted_return, advice 的字典
        market_regime: 市场环境 (bull/bear/sideways/recovery/pullback)
        model_confidence: 模型置信度 (0-1)
        advance_ratio: 市场上涨比例 (0-1)

    Returns:
        bool: 发送是否成功
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
    用于发送交易执行报告等完整报告（仍由 Agent 处理的部分）
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
    """解析报告输出，生成可读消息（保留旧版功能）"""

    lines = output.split("\n")

    # 提取关键信息
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

    # 构建消息
    if report_type == "execute":
        title = "📊 下午交易执行报告"
        footer = "✅ 交易已执行，持仓已更新"
    else:
        title = "📊 明日交易预告"
        footer = "📌 这是预告，明日开盘后才会执行交易"

    msg_parts = [title, ""]

    # 大盘环境
    if market_info:
        msg_parts.append("📈 大盘环境")
        for line in market_info[:4]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    # 买入
    if buy_info:
        msg_parts.append("✅ 买入")
        for line in buy_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    # 卖出
    if sell_info:
        msg_parts.append("❌ 卖出")
        for line in sell_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    # 持仓
    if holdings_info:
        msg_parts.append("💼 当前持仓")
        for line in holdings_info[:5]:
            msg_parts.append(f"  {line.strip()}")
        msg_parts.append("")

    msg_parts.append(footer)

    return "\n".join(msg_parts)


def main():
    if len(sys.argv) < 3:
        print("用法: send_report.py <execute|preview> <输出内容>")
        sys.exit(1)

    report_type = sys.argv[1]
    output_file = sys.argv[2]

    # 读取输出内容
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            output = f.read()
    else:
        output = output_file  # 直接传入内容

    # 解析并生成消息
    message = parse_report(output, report_type)

    # 发送
    if send_slack_message(message):
        print("✓ 报告已发送")
    else:
        print("✗ 发送失败")


if __name__ == "__main__":
    main()
