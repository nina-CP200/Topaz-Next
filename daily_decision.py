#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 每日投资决策系统
根据 ML 分析结果生成投资建议并更新虚拟投资组合
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemble_model import EnsembleModel
from feature_engineer import FeatureEngineer
from quantpilot_data_api import get_history_data, get_stock_data
from utils import parse_stock_list


def load_portfolio(portfolio_file: str) -> Dict:
    """加载投资组合"""
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'initial_capital': 1000000,
        'cash': 1000000,
        'holdings': {},
        'trades': [],
        'daily_values': []
    }


def save_portfolio(portfolio: Dict, portfolio_file: str):
    """保存投资组合"""
    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)


def analyze_stocks(stock_list_file: str) -> List[Dict]:
    """分析股票列表"""
    ensemble = EnsembleModel(model_dir='.')
    fe = FeatureEngineer()
    
    stocks = parse_stock_list(stock_list_file)
    results = []
    
    for symbol, name, category in stocks[:25]:  # 限制 25 只
        try:
            # 获取数据
            history = get_history_data(symbol, 'A股', days=60)
            current = get_stock_data(symbol, 'A股', name)
            
            if history is None or not current:
                print(f"  ⚠️ {symbol} {name}: 数据获取失败，跳过")
                continue
            
            # 生成特征
            history['code'] = symbol
            df_features = fe.generate_all_features(history)
            df_features = df_features.fillna(0)
            
            # 预测
            feature_cols = ensemble.feature_cols
            latest = df_features.iloc[-1:][feature_cols]
            X = latest.values
            
            pred = ensemble.predict(X)
            proba = pred['probability'][0]
            
            # 计算预期收益
            expected_return = (proba - 0.5) * 20  # 转换为百分比
            
            # 风险等级
            if proba >= 0.7:
                risk_level = '低风险'
            elif proba >= 0.55:
                risk_level = '中风险'
            elif proba >= 0.4:
                risk_level = '高风险'
            else:
                risk_level = '极高风险'
            
            # 投资建议
            if proba >= 0.7:
                advice = '建议买入'
            elif proba >= 0.55:
                advice = '建议持有'
            elif proba >= 0.4:
                advice = '建议观望'
            else:
                advice = '建议回避'
            
            results.append({
                'symbol': symbol,
                'name': name,
                'current_price': current.get('current_price', 0),
                'change_pct': current.get('change', 0),
                'pe_ratio': current.get('pe_ratio', 0),
                'pb_ratio': current.get('pb_ratio', 0),
                'roe': current.get('roe', 0),
                'probability': proba,
                'predicted_return': expected_return,
                'risk_level': risk_level,
                'advice': advice
            })
            
        except Exception as e:
            print(f"分析 {symbol} 失败：{e}")
            continue
    
    return results


def generate_decision(results: List[Dict], portfolio: Dict) -> Dict:
    """生成投资决策"""
    # 筛选建议买入的股票
    buy_candidates = [r for r in results if r['advice'] == '建议买入']
    buy_candidates.sort(key=lambda x: x['probability'], reverse=True)
    
    # 筛选现有持仓
    holdings = portfolio.get('holdings', {})
    hold_candidates = [r for r in results if r['symbol'] in holdings]
    
    # 决策
    decisions = {
        'buy': [],
        'sell': [],
        'hold': []
    }
    
    # 买入决策：选择前 3 只高概率股票
    cash = portfolio.get('cash', 0)
    for stock in buy_candidates[:3]:
        if cash > 50000:  # 至少 5 万现金
            amount = min(cash * 0.15, 200000)  # 每只最多 20 万或 15% 现金
            shares = int(amount / stock['current_price'] / 100) * 100  # 整百股
            if shares > 0:
                decisions['buy'].append({
                    'symbol': stock['symbol'],
                    'name': stock['name'],
                    'shares': shares,
                    'price': stock['current_price'],
                    'amount': shares * stock['current_price'],
                    'probability': stock['probability'],
                    'reason': f"ML 概率{stock['probability']:.1%}, 预测收益{stock['predicted_return']:.1f}%"
                })
                cash -= shares * stock['current_price']
    
    # 卖出决策：持仓中建议回避的股票
    for stock in hold_candidates:
        if stock['advice'] == '建议回避' and stock['symbol'] in holdings:
            holding = holdings[stock['symbol']]
            decisions['sell'].append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'shares': holding['shares'],
                'price': stock['current_price'],
                'amount': holding['shares'] * stock['current_price'],
                'reason': f"ML 概率{stock['probability']:.1%}, 建议回避"
            })
    
    # 持有决策：其他持仓
    for stock in hold_candidates:
        if stock['symbol'] not in [s['symbol'] for s in decisions['sell']]:
            decisions['hold'].append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'shares': holdings[stock['symbol']]['shares'],
                'price': stock['current_price'],
                'reason': f"ML 概率{stock['probability']:.1%}, {stock['advice']}"
            })
    
    return decisions


def update_portfolio(portfolio: Dict, decisions: Dict) -> Dict:
    """更新投资组合"""
    holdings = portfolio.get('holdings', {})
    trades = portfolio.get('trades', [])
    cash = portfolio.get('cash', 0)
    
    today = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M')
    
    # 执行买入
    for buy in decisions['buy']:
        symbol = buy['symbol']
        if symbol in holdings:
            # 加仓
            old_shares = holdings[symbol]['shares']
            old_cost = holdings[symbol]['cost_price']
            new_shares = old_shares + buy['shares']
            new_cost = (old_cost * old_shares + buy['price'] * buy['shares']) / new_shares
            
            holdings[symbol]['shares'] = new_shares
            holdings[symbol]['cost_price'] = new_cost
        else:
            # 新建仓
            holdings[symbol] = {
                'name': buy['name'],
                'shares': buy['shares'],
                'cost_price': buy['price'],
                'current_price': buy['price']
            }
        
        cash -= buy['amount']
        trades.append({
            'date': today,
            'time': time_str,
            'type': 'buy',
            'code': symbol,
            'name': buy['name'],
            'shares': buy['shares'],
            'price': buy['price'],
            'amount': buy['amount'],
            'reason': buy['reason']
        })
    
    # 执行卖出
    for sell in decisions['sell']:
        symbol = sell['symbol']
        if symbol in holdings:
            holding = holdings[symbol]
            cash += sell['amount']
            trades.append({
                'date': today,
                'time': time_str,
                'type': 'sell',
                'code': symbol,
                'name': sell['name'],
                'shares': sell['shares'],
                'price': sell['price'],
                'amount': sell['amount'],
                'reason': sell['reason']
            })
            del holdings[symbol]
    
    # 更新持仓现价
    for symbol in holdings:
        try:
            current = get_stock_data(symbol, 'A股', holdings[symbol]['name'])
            if current:
                holdings[symbol]['current_price'] = current.get('current_price', holdings[symbol]['cost_price'])
        except Exception as e:
            print(f"  ⚠️ 更新 {symbol} 价格失败: {e}")
    
    # 计算总资产
    holdings_value = sum(h['shares'] * h.get('current_price', h['cost_price']) for h in holdings.values())
    total_value = cash + holdings_value
    pnl = total_value - portfolio['initial_capital']
    pnl_pct = pnl / portfolio['initial_capital'] * 100
    
    # 记录每日净值
    portfolio['daily_values'].append({
        'date': today,
        'time': time_str,
        'cash': round(cash, 2),
        'holdings_value': round(holdings_value, 2),
        'total_value': round(total_value, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2)
    })
    
    portfolio['cash'] = round(cash, 2)
    portfolio['holdings'] = holdings
    portfolio['trades'] = trades
    portfolio['holdings_value'] = round(holdings_value, 2)
    portfolio['total_value'] = round(total_value, 2)
    portfolio['pnl'] = round(pnl, 2)
    portfolio['pnl_pct'] = round(pnl_pct, 2)
    
    return portfolio


def print_report(decisions: Dict, portfolio: Dict):
    """打印报告"""
    print("\n" + "=" * 80)
    print("📊 Topaz 每日投资决策报告")
    print("=" * 80)
    print(f"报告时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 买入决策
    if decisions['buy']:
        print("\n✅ 建议买入")
        for buy in decisions['buy']:
            print(f"  {buy['symbol']} {buy['name']}: {buy['shares']}股 @ ¥{buy['price']:.2f} = ¥{buy['amount']:,.0f}")
            print(f"    理由：{buy['reason']}")
    
    # 卖出决策
    if decisions['sell']:
        print("\n❌ 建议卖出")
        for sell in decisions['sell']:
            print(f"  {sell['symbol']} {sell['name']}: {sell['shares']}股 @ ¥{sell['price']:.2f} = ¥{sell['amount']:,.0f}")
            print(f"    理由：{sell['reason']}")
    
    # 持有决策
    if decisions['hold']:
        print("\n📌 继续持有")
        for hold in decisions['hold']:
            print(f"  {hold['symbol']} {hold['name']}: {hold['shares']}股 @ ¥{hold['price']:.2f}")
            print(f"    理由：{hold['reason']}")
    
    # 持仓汇总
    print("\n" + "=" * 80)
    print("💼 持仓汇总")
    print("=" * 80)
    print(f"现金：¥{portfolio['cash']:,.2f}")
    print(f"持仓市值：¥{portfolio['holdings_value']:,.2f}")
    print(f"总资产：¥{portfolio['total_value']:,.2f}")
    print(f"累计盈亏：¥{portfolio['pnl']:,.2f} ({portfolio['pnl_pct']:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("风险提示：本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")
    print("=" * 80)


def find_stock_list_file(base_dir: str, prefix: str) -> str:
    """查找股票列表文件"""
    for f in os.listdir(base_dir):
        if f.startswith(prefix) and f.endswith('.md'):
            return os.path.join(base_dir, f)
    return None


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    portfolio_file = os.path.join(base_dir, 'virtual_portfolio.json')
    stock_list_file = find_stock_list_file(base_dir, 'A股')
    
    if not stock_list_file or not os.path.exists(stock_list_file):
        print(f"❌ 未找到 A股列表文件")
        return
    
    # 加载投资组合
    print("📂 加载投资组合...")
    portfolio = load_portfolio(portfolio_file)
    
    # 分析股票
    print("📈 分析 A股...")
    results = analyze_stocks(stock_list_file)
    print(f"  完成 {len(results)} 只股票分析")
    
    # 生成决策
    print("🤖 生成投资决策...")
    decisions = generate_decision(results, portfolio)
    
    # 更新投资组合
    print("💼 更新投资组合...")
    portfolio = update_portfolio(portfolio, decisions)
    
    # 保存
    save_portfolio(portfolio, portfolio_file)
    print(f"✓ 投资组合已保存：{portfolio_file}")
    
    # 打印报告
    print_report(decisions, portfolio)


if __name__ == '__main__':
    main()
