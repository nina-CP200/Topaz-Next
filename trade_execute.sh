#!/bin/bash
# Topaz 下午交易执行
# 每个交易日 14:30 运行，执行交易决策
# 执行后发送报告到 Slack DM（显示为佩丽卡）

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/../topaz_report.log"
OUTPUT_FILE=/tmp/trade_execute_output.txt

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "$(date): [下午交易] 执行交易决策" >> $LOG_FILE

# 运行投资决策脚本（执行模式）
OUTPUT=$(/home/emmmoji/myenv/bin/python daily_decision.py --execute 2>&1)
echo "$OUTPUT" >> $LOG_FILE
echo "$OUTPUT" > $OUTPUT_FILE

echo "$(date): [交易完成] 交易决策已执行" >> $LOG_FILE

# 发送交易报告到 Slack（显示为佩丽卡）
/home/emmmoji/myenv/bin/python send_report.py execute "$OUTPUT_FILE"

echo "交易报告已发送"