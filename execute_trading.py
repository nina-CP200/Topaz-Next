#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行脚本 - 带交易日检查
基于ML信号执行交易决策
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, '.')
from trading_utils import check_trading_status, validate_market_data, is_trading_day


def load_portfolio() -> Dict:
    """加载投资组合"""
    portfolio_path = 'virtual_portfolio.json'
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'initial_capital': 1000000,
        'cash': 1000000,
        'holdings': {},
        'holdings_value': 0,
        'total_value': 1000000,
        'pnl': 0,
        'pnl_pct': 0,
        'trades': [],
        'daily_values': []
    }


def load_predictions() -> List[Dict]:
    """加载ML预测结果"""
    pred_path = 'predictions_a_share_today.json'
    if not os.path.exists(pred_path):
        print(f"❌ 预测文件不存在: {pred_path}")
        return []
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_market_data() -> Dict:
    """加载市场数据"""
    data_path = 'data/a_share_quotes_today.json'
    if not os.path.exists(data_path):
        return {}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def execute_trading():
    """执行交易决策"""
    print("=" * 80)
    print("A股交易执行系统")
    print("=" * 80)
    
    # ===== 步骤1: 交易日检查 =====
    print("\n【步骤1】交易日检查")
    trading_status = check_trading_status()
    print(f"  状态: {trading_status['message']}")
    
    if not trading_status['can_trade']:
        print(f"\n⚠️ 交易已阻止: {trading_status['message']}")
        print(f"   今天是: {datetime.now().strftime('%Y-%m-%d %A')}")
        print(f"   最近交易日: {trading_status['last_trading_day']}")
        
        # 记录日志
        log_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'action': 'SKIPPED',
            'reason': trading_status['message'],
            'status': trading_status
        }
        
        log_path = 'trading_skip_log.json'
        logs = []
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 已记录跳过日志")
        return {'success': False, 'reason': trading_status['message']}
    
    # ===== 步骤2: 数据验证 =====
    print("\n【步骤2】市场数据验证")
    market_data = load_market_data()
    validation = validate_market_data(market_data)
    
    if not validation['valid']:
        print(f"  ⚠️ 数据验证警告:")
        for issue in validation['issues']:
            print(f"    - {issue}")
        
        # 如果数据日期不是今天，询问是否继续
        if validation['data_date'] != datetime.now().strftime('%Y-%m-%d'):
            print(f"\n  ⚠️ 警告: 数据日期({validation['data_date']})与今天不符")
            print(f"     这可能导致错误的交易决策（如4月6日的情况）")
            # 在非交易日，阻止交易
            if not is_trading_day():
                print(f"\n  ❌ 交易已阻止: 非交易日使用历史数据")
                return {'success': False, 'reason': '非交易日使用历史数据'}
    else:
        print(f"  ✓ 数据验证通过")
        print(f"  数据日期: {validation['data_date']}")
    
    # ===== 步骤3: 加载预测结果 =====
    print("\n【步骤3】加载ML预测结果")
    predictions = load_predictions()
    if not predictions:
        print("  ❌ 无预测结果")
        return {'success': False, 'reason': '无预测结果'}
    
    print(f"  ✓ 加载 {len(predictions)} 只股票预测")
    
    # 统计预测分布
    buy_count = sum(1 for p in predictions if p['advice'] == '建议买入')
    hold_count = sum(1 for p in predictions if p['advice'] == '建议持有')
    avoid_count = sum(1 for p in predictions if p['advice'] == '建议回避')
    print(f"  买入: {buy_count}, 持有: {hold_count}, 回避: {avoid_count}")
    
    # 检查是否全部建议回避（异常信号）
    if avoid_count == len(predictions):
        print(f"\n  ⚠️ 警告: 所有股票均建议回避")
        print(f"     这可能是数据异常导致的，建议人工复核")
    
    # ===== 步骤4: 加载当前持仓 =====
    print("\n【步骤4】加载当前持仓")
    portfolio = load_portfolio()
    holdings = portfolio.get('holdings', {})
    cash = portfolio.get('cash', 1000000)
    
    print(f"  现金: ¥{cash:,.2f}")
    print(f"  持仓: {len(holdings)} 只股票")
    for code, info in holdings.items():
        print(f"    - {code}: {info.get('shares', 0)} 股")
    
    # ===== 步骤5: 生成交易建议（不实际执行） =====
    print("\n【步骤5】生成交易建议")
    print("  （实际交易已禁用，仅生成建议）")
    
    suggestions = []
    
    # 检查持仓股票
    for code, info in holdings.items():
        # 查找预测
        pred = next((p for p in predictions if p['symbol'].replace('.SH', '').replace('.SZ', '') == code.replace('.SH', '').replace('.SZ', '')), None)
        if pred:
            if pred['advice'] == '建议回避' and pred['probability'] < 0.2:
                suggestions.append({
                    'action': 'SELL',
                    'code': code,
                    'name': pred['name'],
                    'shares': info.get('shares', 0),
                    'reason': f"ML建议回避(概率{pred['probability']:.1%},预期收益{pred['expected_return']:.1f}%)",
                    'confidence': 'high' if pred['probability'] < 0.15 else 'medium'
                })
    
    # 检查买入机会
    for pred in predictions:
        if pred['advice'] == '建议买入' and pred['probability'] > 0.6:
            code = pred['symbol'].replace('.SH', '').replace('.SZ', '')
            if code not in holdings:
                suggestions.append({
                    'action': 'BUY',
                    'code': code,
                    'name': pred['name'],
                    'price': pred['close'],
                    'reason': f"ML建议买入(概率{pred['probability']:.1%},预期收益{pred['expected_return']:.1f}%)",
                    'confidence': 'high' if pred['probability'] > 0.75 else 'medium'
                })
    
    print(f"\n  生成 {len(suggestions)} 条交易建议:")
    for s in suggestions:
        emoji = "🔴" if s['action'] == 'SELL' else "🟢"
        print(f"    {emoji} {s['action']}: {s['code']} ({s['name']})")
        print(f"       原因: {s['reason']}")
        print(f"       置信度: {s['confidence']}")
    
    # ===== 步骤6: 保存建议 =====
    print("\n【步骤6】保存交易建议")
    suggestion_file = f"trading_suggestions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(suggestion_file, 'w', encoding='utf-8') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'trading_status': trading_status,
            'data_validation': validation,
            'suggestions': suggestions,
            'portfolio_summary': {
                'cash': cash,
                'holdings_count': len(holdings),
                'total_value': portfolio.get('total_value', 0)
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 建议已保存: {suggestion_file}")
    
    print("\n" + "=" * 80)
    print("交易执行完成")
    print("=" * 80)
    
    return {
        'success': True,
        'suggestions': suggestions,
        'trading_status': trading_status
    }


if __name__ == '__main__':
    result = execute_trading()
    sys.exit(0 if result['success'] else 1)
