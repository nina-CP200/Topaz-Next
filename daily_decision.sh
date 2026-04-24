#!/bin/bash
# Topaz 每日股票分析
# 每个交易日运行

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/../topaz_report.log"

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "$(date): [股票分析] 生成分析报告" >> $LOG_FILE

# 运行分析脚本
python3 daily_decision.py >> $LOG_FILE 2>&1

echo "$(date): [分析完成] 报告生成完成" >> $LOG_FILE
