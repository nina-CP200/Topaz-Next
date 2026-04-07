#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 每日投资决策系统
根据 ML 分析结果生成投资建议并更新虚拟投资组合
支持大盘环境判断和条件策略
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
from market_data import (
    get_index_data, 
    get_market_sentiment, 
    judge_market_environment,
    get_market_adjusted_thresholds
)


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
    """生成投资决策（结合大盘环境）"""
    
    # ========== 大盘环境判断 ==========
    print("\n📊 判断大盘环境...")
    try:
        index_data = get_index_data()
        sentiment = get_market_sentiment()
        market_env = judge_market_environment(index_data, sentiment)
        market_thresholds = get_market_adjusted_thresholds(market_env)
        
        print(f"  沪深300: {index_data['price']:.2f} ({index_data['change_pct']:+.2f}%)")
        if sentiment:
            print(f"  市场情绪: 上涨{sentiment['up_count']}家, 下跌{sentiment['down_count']}家, 上涨比例{sentiment['advance_ratio']:.1%}")
        print(f"  环境判断: {market_env} - {market_thresholds['description']}")
    except Exception as e:
        print(f"  大盘判断失败: {e}，使用默认参数")
        market_env = 'sideways'
        market_thresholds = get_market_adjusted_thresholds('sideways')
    
    # 从大盘阈值中获取参数
    BUY_THRESHOLD = market_thresholds['buy_threshold']
    SELL_THRESHOLD = market_thresholds['sell_threshold']
    POSITION_MAX = market_thresholds['position_max']
    SINGLE_MAX = market_thresholds['single_max']
    
    # ========== 综合评分（建议1：概率 × 预测收益）==========
    def calc_score(stock):
        """综合评分 = 概率 × (1 + 预测收益/10)"""
        prob = stock['probability']
        ret = stock.get('predicted_return', 0)
        return prob * (1 + ret / 10)
    
    for r in results:
        r['score'] = calc_score(r)
    
    # 筛选建议买入的股票（根据大盘环境调整）
    if market_env == 'bull':
        # 牛市：买入建议买入和持有的股票
        buy_candidates = [r for r in results if r['advice'] in ['建议买入', '建议持有']]
    elif market_env == 'bear':
        # 熊市：只买入强烈推荐的股票
        buy_candidates = [r for r in results if r['advice'] == '建议买入' and r['probability'] > 0.8]
    else:
        # 震荡/反弹：买入建议买入的股票
        buy_candidates = [r for r in results if r['advice'] == '建议买入']
    
    buy_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 筛选现有持仓
    holdings = portfolio.get('holdings', {})
    hold_candidates = [r for r in results if r['symbol'] in holdings]
    
    # 决策
    decisions = {
        'buy': [],
        'sell': [],
        'hold': [],
        'market_info': {
            'environment': market_env,
            'index_price': index_data.get('price', 0) if index_data else 0,
            'index_change': index_data.get('change_pct', 0) if index_data else 0,
            'advance_ratio': sentiment.get('advance_ratio', 0.5) if sentiment else 0.5,
            'description': market_thresholds['description']
        }
    }
    
    # ========== 卖出决策 ==========
    
    # 参数（根据大盘环境调整）
    STOP_LOSS_PCT = -0.08 if market_env != 'bear' else -0.05  # 熊市止损更严格
    TAKE_PROFIT_PCT = 0.15 if market_env == 'bull' else 0.12  # 牛市止盈更高
    AVOID_THRESHOLD = SELL_THRESHOLD
    SWAP_PROFIT_THRESHOLD = 0.05
    SWAP_SCORE_THRESHOLD = 0.65
    SWAP_SCORE_GAP = 0.15 if market_env != 'bull' else 0.20  # 牛市换仓门槛更高
    
    # 计算总资产（用于持仓集中度检查）
    total_value = portfolio.get('total_value', portfolio.get('cash', 0))
    for symbol, h in holdings.items():
        total_value += h['shares'] * h.get('current_price', h['cost_price'])
    
    for stock in hold_candidates:
        symbol = stock['symbol']
        if symbol not in holdings:
            continue
            
        holding = holdings[symbol]
        cost_price = holding['cost_price']
        current_price = stock['current_price']
        pnl_pct = (current_price - cost_price) / cost_price
        prob = stock['probability']
        score = stock['score']
        position_value = holding['shares'] * current_price
        position_pct = position_value / total_value if total_value > 0 else 0
        
        sell_reason = None
        
        # 条件1: 止损
        if pnl_pct < STOP_LOSS_PCT:
            sell_reason = f"止损: 亏损{pnl_pct:.1%} (阈值{STOP_LOSS_PCT:.0%})"
        
        # 条件2: 止盈 + 评分下降
        elif pnl_pct > TAKE_PROFIT_PCT and prob < 0.7:
            sell_reason = f"止盈: 盈利{pnl_pct:.1%}, 评分{prob:.1%}"
        
        # 条件3: ML 建议"回避"
        elif stock['advice'] == '建议回避' or prob < AVOID_THRESHOLD:
            sell_reason = f"ML建议回避(概率{prob:.1%})"
        
        # 条件4: 换仓（盈利 + 评分下降 + 有更好机会）
        elif pnl_pct > SWAP_PROFIT_THRESHOLD and prob < SWAP_SCORE_THRESHOLD:
            for candidate in buy_candidates[:5]:
                if candidate['score'] - score > SWAP_SCORE_GAP:
                    sell_reason = f"换仓: 盈利{pnl_pct:.1%}, 评分{prob:.1%} → {candidate['name']}评分{candidate['probability']:.1%}"
                    break
        
        # 条件5: 持仓集中度处理（综合判断）
        if position_pct > 0.25 and not sell_reason:
            # 集中度高，但需要综合看
            if prob < 0.5 and pnl_pct < 0:
                # 低评分 + 亏损 → 减仓
                sell_reason = f"风控减仓: 持仓{position_pct:.1%}, 评分{prob:.1%}, 亏损{pnl_pct:.1%}"
            # 其他情况：持有观察，不盲目减仓
        
        if sell_reason:
            decisions['sell'].append({
                'symbol': symbol,
                'name': stock['name'],
                'shares': holding['shares'],
                'price': current_price,
                'amount': holding['shares'] * current_price,
                'probability': prob,
                'score': score,
                'reason': sell_reason
            })
    
    # ========== 买入决策 ==========
    
    # 计算可用现金（当前现金 + 预计卖出金额）
    cash = portfolio.get('cash', 0)
    sell_amount = sum(s['amount'] for s in decisions['sell'])
    available_cash = cash + sell_amount
    
    # 建议2：动态仓位分配（结合大盘环境）
    def get_position_pct(stock, is_adding_position=False, pnl_pct=0):
        """根据评分和大盘环境动态分配仓位
        
        Args:
            stock: 股票信息
            is_adding_position: 是否加仓
            pnl_pct: 当前盈亏比例
        """
        score = stock['score']
        
        if is_adding_position:
            # 加仓逻辑：抄底（亏损时加仓）
            if market_env == 'bear':
                # 熊市不加仓
                return 0
            elif pnl_pct < -0.05 and score > 0.7:
                return min(0.15, SINGLE_MAX)
            elif pnl_pct < 0 and score > 0.6:
                return min(0.08, SINGLE_MAX * 0.5)
            else:
                return 0
        else:
            # 新建仓逻辑（根据大盘环境调整）
            if market_env == 'bull':
                # 牛市：可以更激进
                if score > 0.85:
                    return min(0.25, SINGLE_MAX)
                elif score > 0.75:
                    return min(0.20, SINGLE_MAX)
                elif score > 0.65:
                    return min(0.15, SINGLE_MAX)
                else:
                    return min(0.10, SINGLE_MAX)
            elif market_env == 'bear':
                # 熊市：保守
                if score > 0.85:
                    return min(0.12, SINGLE_MAX)
                elif score > 0.75:
                    return min(0.08, SINGLE_MAX)
                else:
                    return 0
            else:
                # 震荡/反弹：中性
                if score > 0.85:
                    return min(0.20, SINGLE_MAX)
                elif score > 0.75:
                    return min(0.15, SINGLE_MAX)
                elif score > 0.65:
                    return min(0.10, SINGLE_MAX)
                else:
                    return min(0.06, SINGLE_MAX)
    
    # 优先处理：抄底加仓（已有持仓 + 亏损 + 高评分）
    for stock in hold_candidates:
        symbol = stock['symbol']
        if symbol in [s['symbol'] for s in decisions['sell']]:
            continue  # 已计划卖出
        if symbol not in holdings:
            continue
            
        holding = holdings[symbol]
        cost_price = holding['cost_price']
        current_price = stock['current_price']
        pnl_pct = (current_price - cost_price) / cost_price
        
        # 只在亏损时考虑加仓（抄底）
        if pnl_pct >= 0:
            continue  # 盈利不加仓
        
        score = stock['score']
        if score < 0.6:
            continue  # 评分太低不加仓
        
        if available_cash < 30000:
            break
        
        position_pct = get_position_pct(stock, is_adding_position=True, pnl_pct=pnl_pct)
        if position_pct <= 0:
            continue
        
        amount = min(available_cash * position_pct, 150000)  # 加仓上限 15 万
        shares = int(amount / current_price / 100) * 100
        
        if shares > 0:
            decisions['buy'].append({
                'symbol': symbol,
                'name': stock['name'],
                'shares': shares,
                'price': current_price,
                'amount': shares * current_price,
                'probability': stock['probability'],
                'score': score,
                'position_pct': position_pct,
                'reason': f"抄底加仓: 亏损{pnl_pct:.1%}, 评分{stock['probability']:.1%}, 加仓{position_pct:.0%}"
            })
            available_cash -= shares * current_price
    
    # 新建仓（非持仓股票）
    for stock in buy_candidates[:5]:
        # 跳过已在持仓中的股票（已在上面的抄底加仓处理）
        if stock['symbol'] in holdings:
            continue
            
        if available_cash < 30000:
            break
        
        position_pct = get_position_pct(stock)
        # 根据大盘环境调整最大金额
        max_amount = 250000 if market_env == 'bull' else (150000 if market_env == 'bear' else 200000)
        amount = min(available_cash * position_pct, max_amount)
        shares = int(amount / stock['current_price'] / 100) * 100
        
        if shares > 0:
            decisions['buy'].append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'shares': shares,
                'price': stock['current_price'],
                'amount': shares * stock['current_price'],
                'probability': stock['probability'],
                'score': stock['score'],
                'position_pct': position_pct,
                'reason': f"评分{stock['score']:.2f}(概率{stock['probability']:.1%}, 预测收益{stock['predicted_return']:.1f}%), 仓位{position_pct:.0%}"
            })
            available_cash -= shares * stock['current_price']
    
    # ========== 持有决策 ==========
    
    for stock in hold_candidates:
        if stock['symbol'] not in [s['symbol'] for s in decisions['sell']]:
            decisions['hold'].append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'shares': holdings[stock['symbol']]['shares'],
                'price': stock['current_price'],
                'probability': stock['probability'],
                'score': stock['score'],
                'reason': f"评分{stock['score']:.2f}, {stock['advice']}"
            })
    
    return decisions


def update_portfolio(portfolio: Dict, decisions: Dict) -> Dict:
    """更新投资组合"""
    holdings = portfolio.get('holdings', {})
    trades = portfolio.get('trades', [])
    cash = portfolio.get('cash', 0)
    
    today = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M')
    
    # 先执行卖出，获得现金
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
    
    # 执行买入（严格校验现金）
    for buy in decisions['buy']:
        # 【关键】现金校验：不允许孖展交易
        if cash < buy['amount']:
            print(f"  ⚠️ 现金不足，跳过买入 {buy['symbol']} {buy['name']} (需要 ¥{buy['amount']:,.0f}，可用 ¥{cash:,.0f})")
            continue
        
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
    
    # 大盘环境信息
    if 'market_info' in decisions:
        mi = decisions['market_info']
        print(f"\n📈 大盘环境")
        print(f"  沪深300: {mi['index_price']:.2f} ({mi['index_change']:+.2f}%)")
        print(f"  上涨比例: {mi['advance_ratio']:.1%}")
        print(f"  环境判断: {mi['environment']} - {mi['description']}")
    
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
