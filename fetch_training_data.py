#!/usr/bin/env python3
"""
批量获取沪深300历史数据用于训练
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

def get_kline(code, start='20200101', end='20260319'):
    """从东方财富获取K线数据"""
    market = 1 if code.startswith('6') else 0
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        'secid': f'{market}.{code}',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': 101,  # 日K
        'fqt': 1,    # 前复权
        'beg': start,
        'end': end,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data and 'data' in data and data['data'] and 'klines' in data['data']:
            klines = data['data']['klines']
            rows = []
            for k in klines:
                parts = k.split(',')
                rows.append({
                    'code': code,
                    'date': parts[0],
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'volume': float(parts[5]),
                    'amount': float(parts[6]),
                })
            return pd.DataFrame(rows)
    except Exception as e:
        pass
    return None

def calculate_features(df):
    """计算技术指标特征"""
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 移动平均
    df['ma_5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma_10'] = df.groupby('code')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma_20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma_60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60).mean())
    
    # RSI
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.001)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('code')['close'].transform(calc_rsi)
    df['rsi_6'] = df.groupby('code')['close'].transform(lambda x: calc_rsi(x, 6))
    
    # MACD
    df['ema_12'] = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12).mean())
    df['ema_26'] = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26).mean())
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('code')['macd'].transform(lambda x: x.ewm(span=9).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 价格变化
    df['price_change_5d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(5))
    df['price_change_10d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(10))
    df['price_change_20d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change(20))
    
    # 成交量比
    df['volume_ma_5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma_5'].replace(0, 1)
    
    # 波动率
    df['volatility_10d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(10).std())
    df['volatility_20d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
    
    # 目标变量：未来5天涨跌
    df['future_return'] = df.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)
    df['target'] = (df['future_return'] > 0).astype(int)
    
    return df

def main():
    print("=" * 60)
    print("批量获取沪深300历史数据")
    print("=" * 60)
    
    # 读取股票列表
    with open('csi300_stocks_with_industry.json') as f:
        stocks = json.load(f)
    
    industry_map = {s['code']: s.get('industry', '其他') for s in stocks}
    
    # 检查已有数据
    cache_file = 'csi300_raw_data.pkl'
    if Path(cache_file).exists():
        print(f"加载缓存数据: {cache_file}")
        all_data = pd.read_pickle(cache_file)
        print(f"缓存数据: {len(all_data)} 行")
    else:
        all_data = []
        success = 0
        
        for i, stock in enumerate(stocks):
            code = stock['code']
            
            df = get_kline(code)
            if df is not None and len(df) > 0:
                df['industry'] = industry_map.get(code, '其他')
                all_data.append(df)
                success += 1
            
            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{len(stocks)}, 成功: {success}")
                time.sleep(0.5)
        
        all_data = pd.concat(all_data, ignore_index=True)
        
        # 缓存原始数据
        all_data.to_pickle(cache_file)
        print(f"\n✓ 原始数据保存到 {cache_file}")
    
    # 计算特征
    print("\n计算技术指标...")
    df = calculate_features(all_data)
    
    # 保存特征数据
    df.to_pickle('csi300_features.pkl')
    df.to_csv('csi300_features.csv', index=False)
    
    print(f"\n✓ 特征数据保存完成")
    print(f"  总行数: {len(df)}")
    print(f"  股票数: {df['code'].nunique()}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  特征数: {len(df.columns)}")
    
    return df

if __name__ == '__main__':
    main()