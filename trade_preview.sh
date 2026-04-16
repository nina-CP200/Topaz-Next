#!/bin/bash
# Topaz 下午预告计划
# 每个交易日 17:00 运行，只预告决策，不执行交易
# 执行后发送预告报告到 Slack DM（显示为佩丽卡）

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/../topaz_report.log"
OUTPUT_FILE=/tmp/trade_preview_output.txt

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "$(date): [预告计划] 生成明日决策预告（不执行交易）" >> $LOG_FILE

# 运行投资决策脚本（预告模式）
OUTPUT=$(/home/emmmoji/myenv/bin/python daily_decision.py --preview 2>&1)
echo "$OUTPUT" >> $LOG_FILE
echo "$OUTPUT" > $OUTPUT_FILE

echo "$(date): [预告完成] 决策预告已生成" >> $LOG_FILE

# 发送预告报告到 Slack（显示为佩丽卡）
/home/emmmoji/myenv/bin/python send_report.py preview "$OUTPUT_FILE"

echo "预告报告已发送"