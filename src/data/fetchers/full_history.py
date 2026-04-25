#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300完整历史数据获取模块
使用腾讯财经和新浪财经API（免费、稳定）

数据源：
  - 主源：腾讯财经（支持长历史）
  - 备源：新浪财经

使用示例：
  python fetch_full_history.py
  python fetch_full_history.py --limit 10  # 测试模式
"""

import pandas as pd
import requests
import time
import json
import os
from src.utils.utils import load_stock_list_from_json


def get_tencent_history(symbol: str, days: int = 500) -> pd.DataFrame:
    """
    从腾讯财经获取A股历史K线数据
    
    接口地址：
https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    
    参数：
        symbol: 股票代码（如 000001.SZ, 600000.SH）
        days: 获取天数（默认500天，约2年）
    
    返回：
        DataFrame 或 None
    """
    try:
        # 代码格式转换
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
        
        # 腾讯财经K线接口（前复权）
        url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={api_symbol},day,,,{days},qfq'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stock.finance.qq.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        if data.get('data') and data['data'].get(api_symbol):
            stock_data = data['data'][api_symbol]
            klines = stock_data.get('qfqday', stock_data.get('day', []))
            
            if klines:
                records = []
                for item in klines:
                    # 格式: [日期, 开盘, 收盘, 最低, 最高, 成交量]
                    if len(item) >= 6:
                        records.append({
                            'date': item[0],
                            'open': float(item[1]),
                            'close': float(item[2]),
                            'low': float(item[3]),
                            'high': float(item[4]),
                            'volume': float(item[5]),
                            'code': symbol
                        })
                
                df = pd.DataFrame(records)
                return df
        
        return None
    except Exception as e:
        print(f"  腾讯获取失败: {e}")
        return None


def get_sina_history(symbol: str) -> pd.DataFrame:
    """
    从新浪财经获取A股历史K线数据（备用）
    
    接口地址：
https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
    
    返回：
        DataFrame 或 None
    """
    try:
        # 代码格式转换
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
        
        # 新浪财经K线接口
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={api_symbol}&scale=240&ma=5&mab=10"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        if data and isinstance(data, list):
            records = []
            for item in data:
                if 'day' in item:
                    records.append({
                        'date': item['day'],
                        'open': float(item['open']),
                        'close': float(item['close']),
                        'low': float(item['low']),
                        'high': float(item['high']),
                        'volume': float(item['volume']),
                        'code': symbol
                    })
            
            if records:
                return pd.DataFrame(records)
        
        return None
    except Exception as e:
        print(f"  新浪获取失败: {e}")
        return None


def fetch_csi300_full_history(output_file: str = 'csi300_full_history.csv', limit: int = None, stock_list_file: str = None):
    """
    获取沪深300成分股完整历史数据
    
    流程：
        1. 加载股票列表
        2. 腾讯API获取数据（主源）
        3. 新浪API获取数据（备源）
        4. 合并保存
    
    参数：
        output_file: 输出文件名
        limit: 限制股票数量（测试用）
        stock_list_file: 股票列表文件路径
    """
    print("=" * 60)
    print("获取沪深300完整历史数据")
    print("=" * 60)
    print("数据源: 腾讯财经（主） + 新浪财经（备）")
    
    # 加载股票列表
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if stock_list_file is None:
        stock_list_file = os.path.join(base_dir, 'config/csi300_stocks.json')
    
    stocks = load_stock_list_from_json(stock_list_file)
    
    if limit:
        stocks = stocks[:limit]
    
    print(f"股票数量: {len(stocks)}")
    
    all_data = []
    failed_stocks = []
    
    for i, (symbol, name, category) in enumerate(stocks):
        print(f"\n[{i+1}/{len(stocks)}] {symbol} {name}")
        
        # 尝试腾讯API
        df = get_tencent_history(symbol, days=500)
        
        # 如果腾讯失败，尝试新浪
        if df is None:
            print("  腾讯失败，尝试新浪...")
            df = get_sina_history(symbol)
        
        if df is not None and len(df) > 0:
            df['name'] = name
            all_data.append(df)
            print(f"  ✓ 获取 {len(df)} 条数据")
            print(f"    时间: {df['date'].min()} ~ {df['date'].max()}")
        else:
            failed_stocks.append(symbol)
            print(f"  ✗ 获取失败")
        
        # 请求间隔（避免限流）
        time.sleep(0.5)
    
    # 统计结果
    print("\n" + "=" * 60)
    print(f"成功获取: {len(all_data)} 只股票")
    print(f"失败股票: {len(failed_stocks)} 只")
    
    if all_data:
        # 合并数据
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)
        
        print(f"\n合并数据: {len(df_all)} 条")
        print(f"时间范围: {df_all['date'].min()} ~ {df_all['date'].max()}")
        
        # 保存
        df_all.to_csv(output_file, index=False)
        print(f"\n✓ 数据保存: {output_file}")
        
        return df_all
    else:
        print("\n✗ 未获取到任何数据")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='获取沪深300历史数据')
    parser.add_argument('--output', default='data/raw/csi300_full_history.csv', help='输出文件名')
    parser.add_argument('--limit', type=int, default=None, help='限制股票数量（测试用）')
    
    args = parser.parse_args()
    
    fetch_csi300_full_history(args.output, args.limit)