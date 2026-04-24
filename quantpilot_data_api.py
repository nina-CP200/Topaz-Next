#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
QuantPilot 数据获取 API 模块
================================================================================

【模块说明】
本模块提供A股市场实时行情和历史K线数据的获取功能，是QuantPilot量化分析系统的
数据基础层。通过封装第三方财经数据源API，为上层策略分析提供统一的数据接口。

【数据源说明】
1. 腾讯财经 (qt.gtimg.cn)
   - 用途：A股实时行情数据、历史K线数据
   - 特点：响应速度快、数据字段丰富（含PE、PB、ROE等）、支持长历史数据
   - 限制：非官方API，可能存在稳定性风险

2. 新浪财经 (money.finance.sina.com.cn)
   - 用途：A股历史K线数据（腾讯失败时备用）
   - 特点：数据完整、支持多种周期
   - 限制：非官方API，请求频率需控制

【数据获取策略】
- 历史数据：优先腾讯API（支持更长历史），失败时fallback到新浪API
- 实时数据：使用腾讯API

【数据延迟说明】
- 实时行情：延迟约3-5秒（非交易时段可能更长）
- 历史数据：每日收盘后更新，盘中数据可能不完整
- 宏观指标：延迟约1-2分钟

【频率限制说明】
- 建议：每秒不超过3次请求，避免IP被封禁
- 高频场景：建议使用本地缓存或专业数据服务
- 最佳实践：批量请求时添加time.sleep(0.3)间隔

【备用数据源建议】
如需更稳定的数据服务，可考虑以下方案：
1. 免费方案：Tushare Pro (https://tushare.pro) - 需注册获取token
2. 免费方案：AkShare (https://github.com/akfamily/akshare) - 开源财经数据接口库
3. 付费方案：Wind、同花顺iFinD - 专业金融数据终端
4. 开源方案：本地数据库 + 日志采集（适合历史回测）

【添加新数据源指南】
1. 创建新的数据获取函数，遵循以下命名规范：get_<数据源>_<数据类型>
2. 返回格式与本模块现有函数保持一致
3. 在get_stock_data/get_history_data函数中添加fallback逻辑
4. 添加适当的异常处理和日志记录

【版本】
- v1.0.0: 初始版本，支持A股实时行情和历史数据
- 作者：QuantPilot Team
- 许可证：MIT License

================================================================================
"""

import os
import requests
import re
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta


def get_tencent_china_stock(symbol: str, symbol_name: str = None) -> Dict:
    """
    从腾讯财经获取A股实时行情数据
    
    【接口地址】
    https://qt.gtimg.cn/q={symbol}
    
    【参数说明】
    symbol : str
        股票代码，支持以下格式：
        - '600000' 或 '600000.SH'（上海证券交易所）
        - '000001' 或 '000001.SZ'（深圳证券交易所）
        - 6开头的代码自动识别为上交所，其他为深交所
    symbol_name : str, optional
        股票名称，如不提供则从API获取
    
    【返回格式】
    Dict 包含以下字段：
    - symbol: str, 股票代码（如 '600000.SH'）
    - name: str, 股票名称
    - current_price: float, 当前价格（元）
    - prev_close: float, 昨收价（元）
    - change: float, 涨跌幅（%）
    - change_amount: float, 涨跌额（元）
    - open: float, 开盘价（元）
    - high: float, 最高价（元）
    - low: float, 最低价（元）
    - volume: float, 成交量（手）
    - turnover: float, 成交额（万元）
    - pe_ratio: float, 市盈率（动态）
    - pb_ratio: float, 市净率
    - roe: float, 净资产收益率（%）
    - dividend_yield: float, 股息率（%）- 当前固定为0
    - currency: str, 交易货币（'CNY'）
    
    【错误处理】
    - 网络超时：返回空字典 {}
    - 股票代码无效：返回空字典 {}
    - API返回异常数据：返回空字典 {}
    - 注意：异常信息被静默处理，调用方需检查返回是否为空
    
    【数据延迟】
    交易时段延迟约3-5秒，非交易时段显示最后收盘价
    
    【频率限制】
    建议每次请求间隔至少0.3秒，高频请求可能导致IP临时封禁
    """
    try:
        # 统一转换为大写格式
        symbol = symbol.upper()
        
        # 股票代码格式转换：将Wind格式转换为腾讯API格式
        # 腾讯API格式：sh600000（上交所）或 sz000001（深交所）
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            # 根据股票代码首字母判断交易所：6开头为上交所，其他为深交所
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
            
        # 发送HTTP请求获取数据
        url = f"https://qt.gtimg.cn/q={api_symbol}"
        response = requests.get(url, timeout=10)
        data = response.text
        
        # 验证响应格式：腾讯API返回数据以 'v_' 开头
        if not data.startswith('v_'):
            return {}

        # 解析返回数据：腾讯API使用'~'作为字段分隔符
        # 字段索引参考：
        # [1]股票名称 [3]当前价 [4]昨收 [5]今开
        # [31]涨跌额 [32]涨跌幅 [33]最高 [34]最低
        # [36]成交量 [37]成交额 [43]市盈率 [46]市净率 [52]ROE
        parts = data.split('~')
        if len(parts) >= 55:
            return {
                'symbol': symbol,
                'name': symbol_name or parts[1],
                'current_price': float(parts[3]),
                'prev_close': float(parts[4]),
                'change': float(parts[32]) if parts[32] else 0,
                'change_amount': float(parts[31]) if parts[31] else 0,
                'open': float(parts[5]),
                'high': float(parts[33]),
                'low': float(parts[34]),
                'volume': float(parts[36]),
                'turnover': float(parts[37]),
                'pe_ratio': float(parts[43]) * 10 if parts[43] else 0,  # 腾讯返回的PE需要乘10
                'pb_ratio': float(parts[46]) if parts[46] else 0,
                'roe': float(parts[52]) if parts[52] else 0,
                'dividend_yield': 0,  # 腾讯API暂不提供股息率
                'currency': 'CNY'
            }
        return {}
    except Exception as e:
        # 静默处理异常，返回空字典
        # 生产环境建议添加日志记录：logger.error(f"获取腾讯股票数据失败: {e}")
        return {}


def get_stock_data(symbol: str, market: str = 'A股', name: str = None) -> Optional[Dict]:
    """
    获取股票实时数据（统一入口函数）
    
    【接口说明】
    本函数作为数据获取的统一入口，便于后续扩展多数据源支持。
    当前仅支持A股市场，调用腾讯财经API获取数据。
    
    【参数说明】
    symbol : str
        股票代码（如 '600000', '000001.SZ'）
    market : str, default 'A股'
        市场类型，当前仅支持 'A股'
        预留扩展：'港股'、'美股' 等
    name : str, optional
        股票名称，用于覆盖API返回的名称
    
    【返回格式】
    Dict: 成功时返回股票数据字典，详见 get_tencent_china_stock 函数文档
    None: 失败或市场不支持时返回 None
    
    【扩展指南】
    如需添加新市场支持：
    1. 在本函数中添加市场判断分支
    2. 实现对应市场的数据获取函数
    3. 建议实现fallback机制：主数据源失败时尝试备用数据源
    
    示例代码：
        if market == '港股':
            data = get_hk_stock_data(symbol, name)
            if data:
                return data
            # 尝试备用数据源
            return get_hk_stock_backup(symbol, name)
    """
    return get_tencent_china_stock(symbol, name)


def get_tencent_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    从新浪财经获取A股历史K线数据
    
    【接口地址】
    https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
    
    【参数说明】
    symbol : str
        股票代码，支持格式同 get_tencent_china_stock
    days : int, default 60
        请求的历史数据天数
        注意：新浪API单次请求返回的数据量有限，建议不超过100天
    
    【返回格式】
    pandas.DataFrame，包含以下列：
    - date: str, 日期时间字符串
    - datetime: datetime, 日期时间索引
    - open: float, 开盘价
    - close: float, 收盘价
    - high: float, 最高价
    - low: float, 最低价
    - volume: float, 成交量
    - symbol: str, 股票代码
    - code: str, 股票代码（同symbol）
    
    失败时返回 None
    
    【错误处理】
    - 网络超时：打印错误信息，返回 None
    - 数据解析失败：打印错误信息，返回 None
    - API返回空数据：返回 None
    
    【数据延迟】
    每日收盘后更新，盘中数据可能不完整或缺失
    
    【注意事项】
    1. 此接口实际调用新浪财经API（函数名可能造成误导）
    2. scale=240 表示日线数据（240分钟交易时间）
    3. ma=5, mab=10 为均线参数，不影响基础K线数据
    """
    try:
        # 股票代码格式转换
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
        
        # 构建请求URL
        # scale: K线周期（240=日线）
        # ma, mab: 均线参数
        scale = 240
        url = f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={api_symbol}&scale={scale}&ma=5&mab=10"
        
        # 发送请求（添加User-Agent避免被拦截）
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        
        # 解析JSON数据并转换为DataFrame
        if data and isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            
            # 处理日期字段（新浪API返回的日期字段名为'day'）
            if 'day' in df.columns:
                df['date'] = df['day']
            
            df['code'] = symbol
            df['date'] = df['date'].astype(str)
            
            # 提取日期部分（去除时间部分），设置为索引
            df['datetime'] = pd.to_datetime(df['date'].str.split(' ').str[0])
            df.set_index('datetime', inplace=True)
            
            # 类型转换：确保数值字段为float类型
            df['open'] = df['open'].astype(float)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['symbol'] = symbol
            
            return df
        return None
    except Exception as e:
        print(f"获取 A 股 {symbol} 历史数据失败：{e}")
        return None


def get_qq_history(symbol: str, days: int = 500) -> Optional[pd.DataFrame]:
    """
    从腾讯财经获取A股历史K线数据（支持更长历史周期）
    
    【接口地址】
https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    
    【参数说明】
    symbol : str
        股票代码，支持格式同 get_tencent_china_stock
    days : int, default 500
        请求的历史数据天数
        腾讯API支持更长周期，可获取数年历史数据
    
    【返回格式】
    pandas.DataFrame，包含以下列：
    - date: str, 日期字符串（如 '2024-01-15'）
    - datetime: datetime, 日期时间索引
    - open: float, 开盘价
    - close: float, 收盘价
    - high: float, 最高价
    - low: float, 最低价
    - volume: float, 成交量
    - symbol: str, 股票代码
    - code: str, 股票代码
    
    失败时返回 None
    
    【错误处理】
    - 网络超时（15秒）：打印错误信息，返回 None
    - API返回空数据：返回 None
    - JSON解析失败：抛出异常，打印错误信息，返回 None
    
    【数据延迟】
    每日收盘后更新，盘中可能不更新
    
    【特点】
    1. 支持前复权数据（qfq参数）
    2. 单次请求可获取更多历史数据
    3. 数据字段包括：日期、开、收、低、高、成交量
    
    【API参数详解】
    - param: {code},day,,,{days},qfq
      - code: 股票代码（如 sh600000）
      - day: 日线数据（可选：week周线、month月线）
      - days: 返回数据条数
      - qfq: 前复权（可选：hfq后复权、空不复权）
    """
    try:
        # 股票代码格式转换
        if symbol.endswith('.SH'):
            code = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            code = 'sz' + symbol.replace('.SZ', '')
        else:
            code = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
        
        # 构建请求URL（使用前复权数据）
        url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq'
        
        # 设置请求头（模拟浏览器访问）
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stock.finance.qq.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        # 解析返回数据结构
        # data -> data -> {code} -> qfqday/day -> [klines]
        if data.get('data') and data['data'].get(code):
            stock_data = data['data'][code]
            # 优先使用前复权数据，如无则使用原始数据
            klines = stock_data.get('qfqday', stock_data.get('day', []))
            
            if klines:
                records = []
                # 解析K线数据：[日期, 开, 收, 低, 高, 成交量]
                for item in klines:
                    if len(item) >= 6:
                        records.append({
                            'date': item[0],
                            'open': float(item[1]),
                            'close': float(item[2]),
                            'low': float(item[3]),
                            'high': float(item[4]),
                            'volume': float(item[5])
                        })
                
                # 构建DataFrame
                df = pd.DataFrame(records)
                df['datetime'] = pd.to_datetime(df['date'])
                df.set_index('datetime', inplace=True)
                df['symbol'] = symbol
                df['code'] = symbol
                
                return df
        return None
    except Exception as e:
        print(f"获取腾讯历史数据 {symbol} 失败：{e}")
        return None


def get_history_data(symbol: str, market: str = 'A股', days: int = 60) -> Optional[pd.DataFrame]:
    """
    获取历史K线数据（统一入口函数）
    
    【接口说明】
    本函数根据请求的历史天数自动选择最优数据源：
    - days > 100：优先使用腾讯财经API（支持更长历史）
    - days <= 100：使用新浪财经API（响应更快）
    
    【参数说明】
    symbol : str
        股票代码（如 '600000', '000001.SZ'）
    market : str, default 'A股'
        市场类型，当前仅支持 'A股'
    days : int, default 60
        历史数据天数
    
    【返回格式】
    pandas.DataFrame，格式详见 get_qq_history 或 get_tencent_history
    失败时返回 None
    
    【数据源选择策略】
    1. 长周期（>100天）：腾讯财经API
       - 优点：单次请求可获取更多数据
       - 缺点：响应稍慢
    2. 短周期（<=100天）：新浪财经API
       - 优点：响应快
       - 缺点：单次请求数据量有限
    
    【扩展建议】
    可添加更多数据源fallback：
        if days > 100:
            data = get_qq_history(symbol, days)
            if data is not None:
                return data
            # 尝试备用数据源
            data = get_tushare_history(symbol, days)  # 需自行实现
            if data is not None:
                return data
        return get_tencent_history(symbol, days)
    """
    if days > 100:
        data = get_qq_history(symbol, days)
        if data is not None:
            return data
    return get_tencent_history(symbol, days)


def get_macro_indicators() -> Dict:
    """
    【已弃用】获取宏观经济指标
    
    原东方财富API已不可用，此函数返回空字典。
    如需宏观指标数据，请使用专业数据源（如Tushare Pro、Wind等）。
    """
    return {}