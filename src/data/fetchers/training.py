#!/usr/bin/env python3
"""
================================================================================
模块名称：fetch_training_data.py
================================================================================
功能描述：
    批量获取沪深300成分股的历史K线数据，并计算技术指标特征，用于机器学习模型训练。

主要功能：
    1. 从腾讯/新浪财经API获取股票日K线数据（开盘价、收盘价、最高价、最低价、成交量、成交额）
       - 优先使用腾讯API（支持更长历史数据）
       - 腾讯失败时自动fallback到新浪API
    2. 计算多种技术指标特征：
       - 移动平均线（MA5/MA10/MA20/MA60）
       - 相对强弱指数（RSI6/RSI14）
       - 指数平滑异同移动平均线（MACD及其信号线、柱状图）
       - 价格变化率（5/10/20日）
       - 成交量比率
       - 波动率（10/20日）
    3. 生成目标变量：未来5日涨跌标签

数据来源：
    腾讯财经（qt.gtimg.cn）优先，新浪财经备用

使用场景：
    1. 构建股票预测模型的数据集
    2. 回测量化交易策略的特征工程
    3. 研究技术指标与股价走势的相关性

输出文件：
    - csi300_raw_data.pkl：原始K线数据缓存文件
    - csi300_features.pkl：包含技术指标的特征数据（pickle格式）
    - csi300_features.csv：包含技术指标的特征数据（CSV格式）

依赖说明：
    - requests：HTTP请求库，用于调用API
    - pandas：数据处理库，用于数据清洗和特征计算
    - csi300_stocks.json：沪深300成分股列表文件

注意事项：
    - API调用有频率限制，代码中已添加延时控制
    - 建议首次运行后使用缓存数据，避免重复请求
    - 目标变量基于未来5日收益率，数据末尾5日target为NaN

作者：（请补充）
创建日期：（请补充）
最后修改：（请补充）
================================================================================
"""

import requests
import pandas as pd
import json
import time
import os
from pathlib import Path
from datetime import datetime

from src.data.api import get_history_data


def get_kline(code, start='20200101', end='20260319'):
    """
    从腾讯/新浪财经API获取单只股票的日K线数据
    
    数据源优先级：
        1. 腾讯财经API（支持更长历史数据）
        2. 新浪财经API（腾讯失败时备用）
    
    参数说明：
        code (str): 股票代码，6位数字字符串
                    - 上海市场：以'6'开头（如'600000'）
                    - 深圳市场：以'0'或'3'开头（如'000001'）
        start (str): 起始日期，格式'YYYYMMDD'，默认'20200101'
        end (str): 结束日期，格式'YYYYMMDD'，默认'20260319'
    
    返回值：
        pandas.DataFrame: 包含K线数据的DataFrame，列包括：
            - code: 股票代码
            - date: 交易日期
            - open: 开盘价
            - close: 收盘价
            - high: 最高价
            - low: 最低价
            - volume: 成交量（手）
        如果请求失败或无数据，返回None
    
    使用场景：
        1. 获取单只股票的历史行情数据
        2. 批量获取多只股票数据时的底层函数
    
    示例：
        >>> df = get_kline('600000', start='20230101', end='20231231')
        >>> print(df.head())
    
    注意事项：
        - 网络异常时返回None，调用方需处理空值
        - 自动添加交易所后缀（.SH/.SZ）
    """
    # 添加交易所后缀
    if code.startswith('6'):
        symbol = f"{code}.SH"
    else:
        symbol = f"{code}.SZ"
    
    # 计算需要的天数
    start_date = datetime.strptime(start, '%Y%m%d')
    end_date = datetime.strptime(end, '%Y%m%d')
    days = (end_date - start_date).days
    
    # 使用统一的数据获取接口（腾讯优先，新浪备用）
    df = get_history_data(symbol, 'A股', days=days)
    
    if df is not None and len(df) > 0:
        # 转换格式以匹配原有代码
        result = pd.DataFrame({
            'code': code,
            'date': df['date'].values if 'date' in df.columns else df.index.strftime('%Y-%m-%d'),
            'open': df['open'].values,
            'close': df['close'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'volume': df['volume'].values,
        })
        return result
    
    return None

def calculate_features(df):
    """
    计算技术指标特征，用于机器学习模型训练
    
    功能说明：
        对原始K线数据进行特征工程，计算多种常用的技术分析指标，
        并生成目标变量（未来5日涨跌标签）。
    
    参数说明：
        df (pandas.DataFrame): 原始K线数据，必须包含以下列：
            - code: 股票代码
            - date: 交易日期
            - close: 收盘价
            - volume: 成交量
    
    返回值：
        pandas.DataFrame: 添加了技术指标特征和目标变量的DataFrame，新增列包括：
            移动平均线：
                - ma_5: 5日移动平均线
                - ma_10: 10日移动平均线
                - ma_20: 20日移动平均线（月线）
                - ma_60: 60日移动平均线（季线）
            RSI指标：
                - rsi_14: 14日相对强弱指数
                - rsi_6: 6日相对强弱指数
            MACD指标：
                - ema_12: 12日指数移动平均
                - ema_26: 26日指数移动平均
                - macd: MACD线（快线）
                - macd_signal: MACD信号线（慢线）
                - macd_hist: MACD柱状图
            价格变化：
                - price_change_5d: 5日价格变化率
                - price_change_10d: 10日价格变化率
                - price_change_20d: 20日价格变化率
            成交量指标：
                - volume_ma_5: 5日成交量均值
                - volume_ratio: 量比（当日成交量/5日均值）
            波动率指标：
                - volatility_10d: 10日波动率（标准差）
                - volatility_20d: 20日波动率（标准差）
            目标变量：
                - future_return: 未来5日收益率
                - target: 涨跌标签（1=上涨，0=下跌）
    
    使用场景：
        1. 为股票涨跌预测模型准备训练特征
        2. 研究技术指标与未来收益的关系
        3. 构建量化因子库
    
    注意事项：
        - 数据按股票代码和日期排序后计算，确保特征正确
        - 移动平均类指标在前N日会有NaN值
        - 目标变量在数据末尾5行为NaN（无法获取未来数据）
        - RSI计算时对除零进行了保护（替换为0.001）
    """
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
    """
    主函数：执行完整的数据获取和处理流程
    
    执行流程：
        1. 读取沪深300成分股列表（csi300_stocks.json）
        2. 检查是否存在缓存数据：
           - 存在：直接加载缓存
           - 不存在：逐只股票获取K线数据并缓存
        3. 计算技术指标特征
        4. 保存特征数据（pickle和CSV格式）
    
    输入文件：
        csi300_stocks.json: JSON格式，包含沪深300成分股列表
        示例格式：
        [
            {"code": "600000", "name": "浦发银行", "industry": "银行"},
            {"code": "600004", "name": "白云机场", "industry": "交通运输"},
            ...
        ]
    
    输出文件：
        csi300_raw_data.pkl: 原始K线数据缓存，避免重复请求API
        csi300_features.pkl: 特征数据，pandas DataFrame格式
        csi300_features.csv: 特征数据，CSV格式，便于查看和分析
    
    使用场景：
        1. 首次训练模型时获取完整历史数据
        2. 定期更新训练数据集
        3. 特征工程实验和调试
    
    运行示例：
        $ python fetch_training_data.py
        
    输出示例：
        ============================================================
        批量获取沪深300历史数据
        ============================================================
        进度: 20/300, 成功: 20
        ...
        ✓ 原始数据保存到 csi300_raw_data.pkl
        
        计算技术指标...
        
        ✓ 特征数据保存完成
          总行数: 120000
          股票数: 300
          日期范围: 20200102 ~ 20260319
          特征数: 25
    
    注意事项：
        - 首次运行需要较长时间（约5-10分钟）获取所有股票数据
        - 后续运行会使用缓存，速度较快
        - 如需更新数据，删除缓存文件后重新运行
    """
    print("=" * 60)
    print("批量获取沪深300历史数据")
    print("=" * 60)
    
    # 读取股票列表
    with open('config/csi300_stocks.json') as f:
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