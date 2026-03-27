#!/bin/bash
# Topaz 每日投资决策与持仓更新
# 每个交易日 10:00 运行（在报告生成后）

cd /home/emmmoji/.openclaw/workspace-topaz/topaz-v3

LOG_FILE=/home/emmmoji/.openclaw/workspace-topaz/topaz_report.log

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "$(date): [投资决策] 生成投资建议并更新持仓" >> $LOG_FILE

# 运行投资决策脚本
/home/emmmoji/myenv/bin/python daily_decision.py >> $LOG_FILE 2>&1

echo "$(date): [决策完成] 投资建议生成完成" >> $LOG_FILE
