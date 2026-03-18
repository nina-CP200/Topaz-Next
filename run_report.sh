#!/bin/bash
# Topaz 定时任务脚本
# 每天8点生成多因子分析报告
# 9点发送报告到Slack

cd /home/emmmoji/.openclaw/workspace-topaz/topaz-v3

# 加载环境变量
source .env 2>/dev/null
export FINNHUB_API_KEY=${FINNHUB_API_KEY:-d6ffmvhr01qjq8n1fpigd6ffmvhr01qjq8n1fpj0}
export FMP_API_KEY=${FMP_API_KEY:-FbzgSY9UNbuZGVidK40ZHdWti09pTGAT}
export TIINGO_API_KEY=${TIINGO_API_KEY:-cd797cc697fd32ab64c912ba75536679aee6b722}

# 获取当前时间
HOUR=$(date +%-H)
MINUTE=$(date +%-M)

# 检查是否是交易日
YESTERDAY_WEEKDAY=$(date -d "yesterday" +%w)
if [ "$YESTERDAY_WEEKDAY" -eq 0 ] || [ "$YESTERDAY_WEEKDAY" -eq 6 ]; then
    echo "$(date): 周末不是交易日，跳过" >> /home/emmmoji/.openclaw/workspace-topaz/topaz_report.log
    exit 0
fi

LOG_FILE=/home/emmmoji/.openclaw/workspace-topaz/topaz_report.log

case $HOUR in
    8)
        echo "$(date): [生成报告] 多因子分析报告" >> $LOG_FILE
        # 运行多因子分析（不使用ML）
        /home/emmmoji/myenv/bin/python ml_stock_analysis.py >> $LOG_FILE 2>&1
        ;;
    9)
        if [ "$MINUTE" -ge 0 ] && [ "$MINUTE" -lt 30 ]; then
            echo "$(date): [发送报告] 发送报告到Slack" >> $LOG_FILE
            # 调用Slack发送脚本
            if [ -f /home/emmmoji/.openclaw/workspace-topaz/topaz-v3/send_slack_report.sh ]; then
                /home/emmmoji/.openclaw/workspace-topaz/topaz-v3/send_slack_report.sh >> $LOG_FILE 2>&1
            fi
        fi
        ;;
    *)
        echo "$(date): 非任务时间 (hour=$HOUR)" >> $LOG_FILE
        ;;
esac
