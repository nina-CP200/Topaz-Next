#!/bin/bash
# 发送完整报告到 Slack 频道（Agent 处理部分）
# Token 从 .env 读取，不再硬编码

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/../topaz_report.log"

# 从 .env 加载 Slack 配置（如果存在）
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

SLACK_TOKEN="${SLACK_BOT_TOKEN:-}"
CHANNEL="${SLACK_CHANNEL:-#investments}"

if [ -z "$SLACK_TOKEN" ]; then
    echo "错误: SLACK_BOT_TOKEN 未配置，请检查 .env 文件"
    exit 1
fi

sleep 1

# 1. 标题
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
-d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "📊 Topaz-Next A股分析报告 - $(date +%Y-%m-%d)" | jq -Rs .)}"

sleep 1

# 2. A股汇总
CN_SUMMARY=$(grep -A 8 "【A 股】共" "$LOG_FILE" | head -6 | tr '\n' ' ')
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "🇨🇳 *A股* $CN_SUMMARY" | jq -Rs .)}"

sleep 1

# 3. A股推荐 - 从日志提取
CN_RECO=$(grep -E "贵州茅台|五粮液|长江电力|东方财富|中信证券|宁德时代|立讯精密" "$LOG_FILE" | grep "因子" | head -7 | grep -oE "【[^】]+】" | sed 's/【//g' | sed 's/】//g' | tr '\n' ' ')
if [ -n "$CN_RECO" ]; then
    curl -s -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer $SLACK_TOKEN" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "✅ *A股推荐*: $CN_RECO" | jq -Rs .)}"
    sleep 1
fi

# 4. 风险提示
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "⚠️ *风险提示*: 本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。" | jq -Rs .)}"

echo "完整报告已发送"
