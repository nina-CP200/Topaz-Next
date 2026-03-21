#!/usr/bin/env python3
"""
A股预测脚本
使用训练好的 A股模型进行预测
读取 A股关注股票列表.md
"""

import numpy as np
import pandas as pd
import os
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quantpilot_predictor import QuantPilotPredictor
from quantpilot_data_api import get_history_data


def parse_a_share_list(md_file: str) -> dict:
    """
    从 A股关注股票列表.md 解析股票代码和名称
    
    Args:
        md_file: md文件路径
        
    Returns:
        {股票代码: {'name': 名称, 'sector': 板块}} 字典
    """
    stock_map = {}
    
    if not os.path.exists(md_file):
        print(f"⚠️ A股列表文件不存在: {md_file}")
        return stock_map
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 当前板块名
    current_sector = 'Other'
    
    for line in content.split('\n'):
        # 检测标题行 ## 板块名
        if line.startswith('## '):
            current_sector = line[3:].strip()
            continue
        
        # 跳过表头和分隔符
        if line.startswith('| 股票代码') or line.startswith('|---') or line.startswith('| ---'):
            continue
        
        # 解析表格行 | 600519.SH | 贵州茅台 | 白酒 |
        if line.startswith('|') and '|' in line[1:]:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                code = parts[1].strip()  # 如 600519.SH
                name = parts[2].strip()
                # 验证是有效股票代码（6位数字.交易所）
                if code and re.match(r'^\d{6}\.[A-Z]+$', code):
                    stock_map[code] = {
                        'name': name,
                        'sector': current_sector
                    }
    
    print(f"📋 从 {md_file} 加载 {len(stock_map)} 只 A股")
    return stock_map


def fetch_a_share_data(code: str, days: int = 120) -> pd.DataFrame:
    """
    获取 A股历史数据
    
    Args:
        code: 股票代码 (如 600519.SH, 000001.SZ, 或纯代码)
        days: 天数
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # 提取纯代码（去掉交易所后缀）
        pure_code = code.split('.')[0] if '.' in code else code
        
        df = get_history_data(pure_code, days=days)
        if df is None or len(df) < 30:
            return None
        return df
    except Exception as e:
        print(f"  获取 {code} 数据失败: {e}")
        return None


def predict_a_share_stocks(stock_list_file: str = None, output_file: str = None):
    """
    预测 A股关注列表中的所有股票
    
    Args:
        stock_list_file: 股票列表文件路径
        output_file: 输出文件路径
    """
    print("\n" + "="*70)
    print("QuantPilot A股预测系统")
    print("="*70)
    
    # 加载股票列表
    if stock_list_file is None:
        stock_list_file = os.path.join(os.path.dirname(__file__), 'A股关注股票列表.md')
    
    stock_map = parse_a_share_list(stock_list_file)
    
    if not stock_map:
        print("❌ 没有可预测的股票")
        return
    
    # 初始化预测器
    predictor = QuantPilotPredictor()
    
    # 批量预测
    results = []
    symbols = list(stock_map.keys())
    
    print(f"\n预测 {len(symbols)} 只股票...")
    print("-"*70)
    
    for i, code in enumerate(symbols, 1):
        info = stock_map[code]
        print(f"[{i}/{len(symbols)}] {code} {info['name']:<8} ", end='', flush=True)
        
        try:
            # 获取数据
            df = fetch_a_share_data(code)
            if df is None:
                print("❌ 数据不足")
                continue
            
            # 预测
            result = predictor.predict(code, df)
            
            if result and 'prediction' in result:
                pred_label = result.get('prediction', 1)
                confidence = result.get('confidence', 0)
                
                label_map = {0: '下跌', 1: '横盘', 2: '上涨'}
                pred_name = label_map.get(pred_label, '未知')
                
                results.append({
                    'code': code,
                    'name': info['name'],
                    'sector': info['sector'],
                    'prediction': pred_name,
                    'prediction_code': pred_label,
                    'confidence': confidence,
                    'current_price': result.get('current_price', 0),
                    'change_pct': result.get('change_pct', 0)
                })
                
                emoji = '📈' if pred_label == 2 else ('📉' if pred_label == 0 else '➡️')
                print(f"{emoji} {pred_name} ({confidence:.1%})")
            else:
                print("❌ 预测失败")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 汇总统计
    print("\n" + "="*70)
    print("预测汇总")
    print("="*70)
    
    if results:
        up = sum(1 for r in results if r['prediction_code'] == 2)
        down = sum(1 for r in results if r['prediction_code'] == 0)
        flat = sum(1 for r in results if r['prediction_code'] == 1)
        
        print(f"📈 上涨: {up} 只 ({up/len(results):.1%})")
        print(f"📉 下跌: {down} 只 ({down/len(results):.1%})")
        print(f"➡️ 横盘: {flat} 只 ({flat/len(results):.1%})")
        
        # 高置信度预测
        high_conf = [r for r in results if r['confidence'] > 0.6]
        if high_conf:
            print(f"\n🎯 高置信度预测 (>60%):")
            for r in sorted(high_conf, key=lambda x: -x['confidence'])[:5]:
                print(f"  {r['code']} {r['name']}: {r['prediction']} ({r['confidence']:.1%})")
    
    # 保存结果
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), 'a_share_predictions_today.json')
    
    output_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'total': len(results),
        'predictions': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A股预测')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--list', type=str, help='股票列表文件')
    args = parser.parse_args()
    
    predict_a_share_stocks(
        stock_list_file=args.list,
        output_file=args.output
    )