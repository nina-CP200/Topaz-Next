#!/usr/bin/env python3
"""
获取沪深300完整历史数据（2020年至今）
使用东方财富接口支持更长历史
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import parse_stock_list
from feature_engineer import FeatureEngineer


def get_eastmoney_history(symbol: str, start_date: str = '20200101') -> pd.DataFrame:
    """
    从东方财富获取A股历史数据（支持更长历史）
    
    Args:
        symbol: 股票代码（如 000001.SZ）
        start_date: 开始日期
    
    Returns:
        DataFrame
    """
    # 确定市场代码
    if symbol.endswith('.SH'):
        secid = '1.' + symbol.replace('.SH', '')
    elif symbol.endswith('.SZ'):
        secid = '0.' + symbol.replace('.SZ', '')
    else:
        secid = '1.' + symbol if symbol.startswith('6') else '0.' + symbol
    
    url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': secid,
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57',
        'klt': '101',
        'fqt': '1',
        'beg': start_date,
        'end': '20301231',
    }
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, params=params, headers=headers, timeout=15)
    data = response.json()
    
    if data.get('data') and data['data'].get('klines'):
        klines = data['data']['klines']
        records = []
        for item in klines:
            parts = item.split(',')
            if len(parts) >= 6:
                records.append({
                    'date': parts[0],
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'volume': float(parts[5]),
                    'code': symbol
                })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    return None


def fetch_csi300_full_history(start_year: int = 2020, output_file: str = 'csi300_full_history.csv', limit: int = None):
    """
    获取沪深300完整历史数据
    
    Args:
        start_year: 开始年份
        output_file: 输出文件
        limit: 限制股票数量（用于测试）
    """
    print("=" * 60)
    print("获取沪深300完整历史数据（东方财富接口）")
    print("=" * 60)
    
    stock_list_file = 'csi300_stock_list.csv'
    stocks = parse_stock_list(stock_list_file)
    
    if limit:
        stocks = stocks[:limit]
    
    print(f"股票数量: {len(stocks)}")
    
    start_date = f"{start_year}0101"
    all_data = []
    failed_stocks = []
    
    for i, (symbol, name, category) in enumerate(stocks):
        print(f"\n[{i+1}/{len(stocks)}] {symbol} {name}")
        
        try:
            df = get_eastmoney_history(symbol, start_date)
            
            if df is not None and len(df) > 0:
                df['name'] = name
                all_data.append(df)
                print(f"  ✓ {len(df)} 条 ({df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')})")
            else:
                print(f"  ✗ 获取失败")
                failed_stocks.append(symbol)
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            failed_stocks.append(symbol)
            continue
    
    print(f"\n成功获取: {len(all_data)} 只股票")
    print(f"失败股票: {len(failed_stocks)} 只")
    
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)
        
        print(f"\n合并数据: {len(df_all)} 条, {df_all['code'].nunique()} 只股票")
        print(f"时间范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
        
        df_all.to_csv(output_file, index=False)
        print(f"原始数据保存: {output_file}")
        
        return df_all
    
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='获取沪深300完整历史数据')
    parser.add_argument('--start-year', type=int, default=2020, help='开始年份')
    parser.add_argument('--output', type=str, default='csi300_full_history.csv', help='输出文件')
    parser.add_argument('--limit', type=int, default=None, help='限制股票数量（测试用）')
    
    args = parser.parse_args()
    
    fetch_csi300_full_history(args.start_year, args.output, args.limit)