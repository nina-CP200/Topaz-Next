#!/bin/bash
# 发送完整报告到 Slack #investments 频道

LOG_FILE=/home/emmmoji/.openclaw/workspace-topaz/topaz_report.log
SLACK_TOKEN="xoxb-10554594661939-10572535571923-MTxMlnckpizh9iZtlSyDpOT8"
CHANNEL="#investments"

sleep 1

# 1. 标题
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "📊 Topaz V3 多因子分析报告 - $(date +%Y-%m-%d)" | jq -Rs .)}"

sleep 1

# 2. 美股汇总
US_SUMMARY=$(grep -A 8 "【美股】共" "$LOG_FILE" | head -6 | tr '\n' ' ')
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "🇺🇸 *美股* $US_SUMMARY" | jq -Rs .)}"

sleep 1

# 3. A股汇总  
CN_SUMMARY=$(grep -A 8 "【A 股】共" "$LOG_FILE" | head -6 | tr '\n' ' ')
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "🇨🇳 *A股* $CN_SUMMARY" | jq -Rs .)}"

sleep 1

# 4. 美股推荐 - 用关键词
US_RECO="AMZN GOOGL META TSLA AMD MSFT"
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "✅ *美股推荐*: $US_RECO" | jq -Rs .)}"

sleep 1

# 5. A股推荐 - 从日志提取
CN_RECO=$(grep -E "贵州茅台|五粮液|长江电力|东方财富|中信证券|宁德时代|立讯精密" "$LOG_FILE" | grep "因子" | head -7 | grep -oE "【[^】]+】" | sed 's/【//g' | sed 's/】//g' | tr '\n' ' ')
if [ -n "$CN_RECO" ]; then
    curl -s -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer $SLACK_TOKEN" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "✅ *A股推荐*: $CN_RECO" | jq -Rs .)}"
    sleep 1
fi

# 6. 风险提示
curl -s -X POST https://slack.com/api/chat.postMessage \
    -H "Authorization: Bearer $SLACK_TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "{\"channel\": \"$CHANNEL\", \"text\": $(echo "⚠️ *风险提示*: 本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。" | jq -Rs .)}"

echo "完整报告已发送"
