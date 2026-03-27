#!/bin/bash
# Topaz 每日 A 股分析报告
# 每个交易日 9:45 运行

cd /home/emmmoji/.openclaw/workspace-topaz/topaz-v3

LOG_FILE=/home/emmmoji/.openclaw/workspace-topaz/topaz_report.log

# 检查是否是交易日（周末跳过）
WEEKDAY=$(date +%w)
if [ "$WEEKDAY" -eq 0 ] || [ "$WEEKDAY" -eq 6 ]; then
    echo "$(date): 周末不是交易日，跳过" >> $LOG_FILE
    exit 0
fi

# 检查 A 股休市（元旦、春节、清明、劳动节、端午、中秋、国庆等）
# 简化处理：只检查周末，节假日需要手动维护列表
HOLIDAY_FILE="/home/emmmoji/.openclaw/workspace-topaz/topaz-v3/holidays.txt"
TODAY=$(date +%Y-%m-%d)
if [ -f "$HOLIDAY_FILE" ] && grep -q "$TODAY" "$HOLIDAY_FILE"; then
    echo "$(date): 今天是节假日 ($TODAY)，跳过" >> $LOG_FILE
    exit 0
fi

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "$(date): [生成报告] A 股 ML 分析报告" >> $LOG_FILE

# 运行 ML 分析
/home/emmmoji/myenv/bin/python ml_stock_analysis_ensemble.py --cn >> $LOG_FILE 2>&1

echo "$(date): [报告完成] A 股 ML 分析完成" >> $LOG_FILE
