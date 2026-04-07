#!/usr/bin/env python3
"""
基于ML信号执行交易决策
"""
import json
from datetime import datetime

# 读取预测结果
with open('predictions_a_share_today.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

# 读取当前持仓
with open('virtual_portfolio.json', 'r', encoding='utf-8') as f:
    portfolio = json.load(f)

# 当前日期
today = datetime.now().strftime('%Y-%m-%d')
time_str = datetime.now().strftime('%H:%M')

# 构建预测字典
pred_dict = {p['symbol']: p for p in predictions}

# 交易记录
trades_executed = []
holdings_to_remove = []

# 风控参数
MAX_POSITION_PCT = 0.20  # 单股最大仓位20%
MIN_CASH_PCT = 0.10      # 最小现金比例10%
MAX_TOTAL_POSITION = 0.90 # 最大总仓位90%

print("=" * 80)
print(f"A股每日交易执行报告 - {today} {time_str}")
print("=" * 80)

# 1. 评估现有持仓 - 根据ML信号卖出
print("\n【持仓评估 - 基于ML信号】")
for code, holding in portfolio['holdings'].items():
    # 标准化代码
    std_code = code
    if '.' not in code:
        if code.startswith('6'):
            std_code = f"{code}.SH"
        else:
            std_code = f"{code}.SZ"

    pred = pred_dict.get(std_code)
    if pred:
        prob = pred['probability']
        advice = pred['advice']
        current_price = pred['close']

        # 更新当前价格
        holding['current_price'] = current_price

        # ML建议回避或观望，执行卖出
        if prob < 0.50:
            shares = holding['shares']
            sell_amount = shares * current_price
            cost_basis = shares * holding['cost_price']
            pnl = sell_amount - cost_basis

            trade = {
                'date': today,
                'time': time_str,
                'type': 'sell',
                'code': code,
                'name': holding['name'],
                'shares': shares,
                'price': current_price,
                'amount': sell_amount,
                'pnl': pnl,
                'reason': f"ML建议{advice}(概率{prob:.1%},预测收益{pred['expected_return']:+.1f}%)，执行风控卖出"
            }
            trades_executed.append(trade)
            holdings_to_remove.append(code)

            portfolio['cash'] += sell_amount
            print(f"  卖出 {code} {holding['name']}: {shares}股 @ ¥{current_price:.2f}, 金额¥{sell_amount:,.0f}, 盈亏¥{pnl:,.0f}")
        else:
            print(f"  持有 {code} {holding['name']}: {holding['shares']}股 - ML建议{advice}(概率{prob:.1%})")
    else:
        print(f"  持有 {code} {holding['name']}: 无ML预测数据，保持持仓")

# 移除已卖出的持仓
for code in holdings_to_remove:
    del portfolio['holdings'][code]

# 2. 评估买入机会
print("\n【买入评估 - 基于ML信号】")
buy_candidates = []
for pred in predictions:
    if pred['probability'] >= 0.60:  # 只买入概率>=60%的股票
        buy_candidates.append(pred)

# 按概率排序
buy_candidates.sort(key=lambda x: x['probability'], reverse=True)

# 计算可用资金
total_value = portfolio['cash'] + sum(h['shares'] * h['current_price'] for h in portfolio['holdings'].values())
available_cash = portfolio['cash']
max_position_value = total_value * MAX_POSITION_PCT

for pred in buy_candidates:
    code = pred['symbol']
    name = pred['name']
    price = pred['close']
    prob = pred['probability']

    # 标准化代码
    code_short = code.split('.')[0]

    # 检查是否已有持仓
    existing_value = 0
    for c, h in portfolio['holdings'].items():
        if c.replace('.SH', '').replace('.SZ', '') == code_short:
            existing_value = h['shares'] * h['current_price']
            break

    # 计算可买入金额
    target_value = max_position_value
    can_buy_value = min(target_value - existing_value, available_cash * 0.3)  # 每次最多用30%现金

    if can_buy_value > 10000:  # 最小交易金额1万
        shares = int(can_buy_value / price / 100) * 100  # 100股整数倍
        if shares >= 100:
            buy_amount = shares * price

            trade = {
                'date': today,
                'time': time_str,
                'type': 'buy',
                'code': code_short,
                'name': name,
                'shares': shares,
                'price': price,
                'amount': buy_amount,
                'reason': f"ML建议买入(概率{prob:.1%},预测收益{pred['expected_return']:+.1f}%)"
            }
            trades_executed.append(trade)

            # 更新持仓
            portfolio['holdings'][code_short] = {
                'name': name,
                'shares': shares,
                'cost_price': price,
                'current_price': price
            }
            portfolio['cash'] -= buy_amount
            available_cash = portfolio['cash']

            print(f"  买入 {code_short} {name}: {shares}股 @ ¥{price:.2f}, 金额¥{buy_amount:,.0f}")

# 3. 更新组合价值
holdings_value = sum(h['shares'] * h['current_price'] for h in portfolio['holdings'].values())
portfolio['holdings_value'] = holdings_value
portfolio['total_value'] = portfolio['cash'] + holdings_value
portfolio['pnl'] = portfolio['total_value'] - portfolio['initial_capital']
portfolio['pnl_pct'] = portfolio['pnl'] / portfolio['initial_capital'] * 100

# 添加交易记录
portfolio['trades'].extend(trades_executed)

# 添加每日净值记录
daily_record = {
    'date': today,
    'time': time_str,
    'cash': portfolio['cash'],
    'holdings_value': holdings_value,
    'total_value': portfolio['total_value'],
    'pnl': portfolio['pnl'],
    'pnl_pct': portfolio['pnl_pct'],
    'benchmark_value': 0,  # 待更新
    'benchmark_pct': 0
}
portfolio['daily_values'].append(daily_record)

# 保存更新后的持仓
with open('virtual_portfolio.json', 'w', encoding='utf-8') as f:
    json.dump(portfolio, f, ensure_ascii=False, indent=2)

# 输出汇总
print("\n" + "=" * 80)
print("【交易执行汇总】")
print("=" * 80)
print(f"执行交易数量: {len(trades_executed)}")
for trade in trades_executed:
    print(f"  {trade['type'].upper()}: {trade['code']} {trade['name']} - {trade['shares']}股 @ ¥{trade['price']:.2f}")

print("\n【最新持仓】")
print(f"  现金: ¥{portfolio['cash']:,.0f}")
print(f"  持仓市值: ¥{holdings_value:,.0f}")
print(f"  总资产: ¥{portfolio['total_value']:,.0f}")
print(f"  盈亏: ¥{portfolio['pnl']:,.0f} ({portfolio['pnl_pct']:+.2f}%)")

print("\n持仓明细:")
for code, h in portfolio['holdings'].items():
    value = h['shares'] * h['current_price']
    pct = value / portfolio['total_value'] * 100
    print(f"  {code} {h['name']}: {h['shares']}股 @ ¥{h['current_price']:.2f} = ¥{value:,.0f} ({pct:.1f}%)")

# 生成ML信号汇总
print("\n" + "=" * 80)
print("【ML信号汇总】")
print("=" * 80)
print(f"分析股票数: {len(predictions)}")
print(f"建议买入: {sum(1 for p in predictions if p['probability'] >= 0.60)} 只")
print(f"建议持有: {sum(1 for p in predictions if 0.50 <= p['probability'] < 0.60)} 只")
print(f"建议观望: {sum(1 for p in predictions if 0.40 <= p['probability'] < 0.50)} 只")
print(f"建议回避: {sum(1 for p in predictions if p['probability'] < 0.40)} 只")

print("\nTop 5 买入信号:")
buy_signals = [p for p in predictions if p['probability'] >= 0.50]
buy_signals.sort(key=lambda x: x['probability'], reverse=True)
for p in buy_signals[:5]:
    print(f"  {p['symbol']} {p['name']}: 概率{p['probability']:.1%}, 预测收益{p['expected_return']:+.1f}%")

if not buy_signals:
    print("  无买入信号（所有股票概率均低于50%）")

print("\n" + "=" * 80)
print("组合已更新并保存到 virtual_portfolio.json")
print("=" * 80)
