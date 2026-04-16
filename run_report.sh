#!/bin/bash
# Topaz 定时任务脚本
# 支持 A股/美股 分开运行，节省 API 费用
#
# 用法:
#   ./run_report.sh       # 只分析A股（默认，节省API）
#   ./run_report.sh --cn  # 只分析A股
#   ./run_report.sh --us  # 只分析美股
#   ./run_report.sh --all # 分析全部市场

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# 加载环境变量
source .env 2>/dev/null
export FINNHUB_API_KEY=${FINNHUB_API_KEY:-d6ffmvhr01qjq8n1fpigd6ffmvhr01qjq8n1fpj0}
export FMP_API_KEY=${FMP_API_KEY:-FbzgSY9UNbuZGVidK40ZHdWti09pTGAT}
export TIINGO_API_KEY=${TIINGO_API_KEY:-cd797cc697fd32ab64c912ba75536679aee6b722}

# 获取当前时间
HOUR=$(date +%-H)
MINUTE=$(date +%-M)

# 检查是否是交易日（周末跳过）
YESTERDAY_WEEKDAY=$(date -d "yesterday" +%w)
if [ "$YESTERDAY_WEEKDAY" -eq 0 ] || [ "$YESTERDAY_WEEKDAY" -eq 6 ]; then
    echo "$(date): 周末不是交易日，跳过" >> "$SCRIPT_DIR/../topaz_report.log"
    exit 0
fi

LOG_FILE="$SCRIPT_DIR/../topaz_report.log"

# 处理参数
case "$1" in
    --us)
        MARKET="--us"
        ;;
    --all)
        MARKET=""
        ;;
    *)
        MARKET="--cn"  # 默认只分析A股
        ;;
esac

case $HOUR in
    8)
        echo "$(date): [生成报告] 多因子分析报告 (market=$MARKET)" >> $LOG_FILE
        # 运行多因子分析
        /home/emmmoji/myenv/bin/python ml_stock_analysis.py $MARKET >> $LOG_FILE 2>&1
        ;;
    9)
        if [ "$MINUTE" -ge 0 ] && [ "$MINUTE" -lt 30 ]; then
            echo "$(date): [发送报告] 发送报告到Slack" >> $LOG_FILE
            # 调用Slack发送脚本
            if [ -f "$SCRIPT_DIR/send_slack_report.sh" ]; then
                "$SCRIPT_DIR/send_slack_report.sh" >> $LOG_FILE 2>&1
            fi
        fi
        ;;
    *)
        echo "$(date): 非任务时间 (hour=$HOUR)" >> $LOG_FILE
        ;;
esac