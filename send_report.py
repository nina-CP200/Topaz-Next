#!/usr/bin/env python3
"""
发送交易报告到 Slack
通过 OpenClaw 的消息系统发送，显示为佩丽卡
"""

import sys
import json
import subprocess
import os

def send_slack_message(text: str, channel: str = "U0AGVSHJ08Z"):
    """
    发送 Slack 消息
    使用 Slack API 直接发送
    """
    token = "xoxb-10554594661939-10572535571923-MTxMlnckpizh9iZtlSyDpOT8"
    
    # 使用 curl 发送
    cmd = [
        'curl', '-s', '-X', 'POST',
        'https://slack.com/api/chat.postMessage',
        '-H', f'Authorization: Bearer {token}',
        '-H', 'Content-Type: application/json; charset=utf-8',
        '-d', json.dumps({
            'channel': channel,
            'text': text,
            'username': '佩丽卡',
            'icon_emoji': ':crystal_ball:'
        })
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def parse_report(output: str, report_type: str) -> str:
    """解析报告输出，生成可读消息"""
    
    lines = output.split('\n')
    
    # 提取关键信息
    market_info = []
    buy_info = []
    sell_info = []
    holdings_info = []
    
    section = None
    for line in lines:
        if '大盘环境' in line:
            section = 'market'
        elif '建议买入' in line:
            section = 'buy'
        elif '建议卖出' in line:
            section = 'sell'
        elif '持仓汇总' in line:
            section = 'holdings'
        elif line.startswith('==='):
            section = None
        elif section == 'market' and line.strip():
            market_info.append(line)
        elif section == 'buy' and line.strip() and line.strip()[0].isdigit():
            buy_info.append(line)
        elif section == 'sell' and line.strip() and line.strip()[0].isdigit():
            sell_info.append(line)
        elif section == 'holdings' and line.strip() and ('现金' in line or '持仓' in line or '总资' in line or '盈亏' in line):
            holdings_info.append(line)
    
    # 构建消息
    if report_type == 'execute':
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
    
    return '\n'.join(msg_parts)


def main():
    if len(sys.argv) < 3:
        print("用法: send_report.py <execute|preview> <输出内容>")
        sys.exit(1)
    
    report_type = sys.argv[1]
    output_file = sys.argv[2]
    
    # 读取输出内容
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
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


if __name__ == '__main__':
    main()