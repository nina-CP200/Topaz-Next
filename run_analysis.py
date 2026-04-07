#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, '.')
from ml_stock_analysis_ensemble import MLStockAnalyzer
from utils import parse_stock_list
from trading_utils import check_trading_status, validate_market_data
from datetime import datetime

# 检查交易日
print("=" * 60)
print("交易日检查")
print("=" * 60)
trading_status = check_trading_status()
print(f"状态: {trading_status['message']}")

if not trading_status['can_trade']:
    print(f"\n⚠️ 警告: 今天不是交易日或不在交易时段")
    print(f"   最近交易日: {trading_status['last_trading_day']}")
    print(f"   分析将继续，但交易决策应跳过")
    print("=" * 60)

# 手动运行分析
analyzer = MLStockAnalyzer(batch=True, limit=30)
a_stocks_file = 'A股关注股票列表.md'
stocks = parse_stock_list(a_stocks_file)
print(f'\n分析 {len(stocks)} 只股票')

# 获取历史数据
analyzer.fetch_history_data(stocks)
analyzer.fetch_current_data(stocks)

# 分析每只股票
results = []
for symbol, name, market in stocks:
    if symbol in analyzer.current_data and symbol in analyzer.history_data:
        result = analyzer.analyze_stock(
            symbol,
            analyzer.current_data[symbol],
            analyzer.history_data[symbol]
        )
        if result:
            results.append({
                'symbol': result['symbol'],
                'name': result['name'],
                'close': result['current_price'],
                'pred': 1 if result['advice'] in ['建议买入', '建议持有'] else 0,
                'probability': result['probability'],
                'expected_return': result['predicted_return'],
                'risk': result['risk_level'],
                'advice': result['advice']
            })

# 保存结果
with open('predictions_a_share_today.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'保存了 {len(results)} 只股票的预测结果')
for r in results[:5]:
    print(f"{r['symbol']}: {r['name']} - {r['advice']} (概率: {r['probability']:.2%})")
