#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topaz 数据获取 API
从 topaz-v3 项目整合的数据获取模块
支持：A 股（腾讯财经）+ 美股（Finnhub API）
"""

import os
import requests
import re
from typing import Dict, Optional
from dotenv import load_dotenv
import pandas as pd

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

load_dotenv()

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "")


def get_finnhub_us_stock(symbol: str, symbol_name: str = None) -> Dict:
    """从 Finnhub 获取美股实时行情"""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'c' in data and data['c'] > 0:
            return {
                'symbol': symbol,
                'name': symbol_name or symbol,
                'current_price': data.get('c', 0),
                'prev_close': data.get('pc', 0),
                'open': data.get('o', 0),
                'change': data.get('dp', 0),
                'change_amount': data.get('d', 0),
                'high': data.get('h', 0),
                'low': data.get('l', 0),
                'volume': data.get('v', 0),
                'currency': 'USD'
            }
        return {}
    except Exception as e:
        print(f"获取美股 {symbol} 失败：{e}")
        return {}


def get_finnhub_company_info(symbol: str) -> Dict:
    """从 Finnhub 获取公司基本面信息"""
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data:
            return {
                'market_cap': data.get('marketCapitalization', 0) * 1e6,
                'pe_ratio': data.get('peForward', 0) or data.get('peTrailing', 0),
                'pb_ratio': data.get('pbRatio', 0),
                'dividend_yield': data.get('lastDividend', 0),
                'roe': data.get('roe', 0) * 100 if data.get('roe') else 0,
                'exchange': data.get('exchange', ''),
                'industry': data.get('finnhubIndustry', ''),
            }
        return {}
    except Exception as e:
        return {}


def get_tencent_china_stock(symbol: str, symbol_name: str = None) -> Dict:
    """从腾讯财经获取 A 股实时行情"""
    try:
        symbol = symbol.upper()
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
            
        url = f"http://qt.gtimg.cn/q={api_symbol}"
        response = requests.get(url, timeout=10)
        data = response.text
        
        if not data.startswith('v_'):
            return {}

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
                'pe_ratio': float(parts[43]) * 10 if parts[43] else 0,
                'pb_ratio': float(parts[46]) if parts[46] else 0,
                'roe': float(parts[52]) if parts[52] else 0,
                'dividend_yield': 0,
                'currency': 'CNY'
            }
        return {}
    except Exception as e:
        return {}


def get_macro_indicators() -> Dict:
    """获取宏观经济指标"""
    indicators = {}
    try:
        # 美元指数
        url = 'https://push2.eastmoney.com/api/qt/ulist.np/get'
        params = {'fltt': 2, 'invt': 2, 'fields': 'f1,f2,f3,f4,f12,f13,f14', 'secids': '1.1000163'}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get('data') and data['data']['diff']:
            item = data['data']['diff'][0]
            indicators['DXY'] = {
                'symbol': 'DXY',
                'name': '美元指数',
                'current_price': item.get('f2', 0),
                'change': item.get('f3', 0)
            }
        
        # 10 年期美债收益率
        params['secids'] = '1.1000257'
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        if data.get('data') and data['data']['diff']:
            item = data['data']['diff'][0]
            indicators['US10Y'] = {
                'symbol': 'US10Y',
                'name': '10 年期美债收益率',
                'current_price': item.get('f2', 0),
                'change': item.get('f3', 0)
            }
    except Exception as e:
        print(f"获取宏观数据失败：{e}")
    return indicators


def get_stock_data(symbol: str, market: str = 'A 股', name: str = None) -> Optional[Dict]:
    """统一接口：获取股票数据"""
    if market == '美股':
        data = get_finnhub_us_stock(symbol, name)
        if data:
            info = get_finnhub_company_info(symbol)
            data.update(info)
        return data
    else:  # A 股
        return get_tencent_china_stock(symbol, name)


def get_finnhub_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    从 Finnhub 获取美股历史数据
    注意：Finnhub 免费 API 不支持历史数据，此函数将返回 None

    Parameters
    ----------
    symbol : str
        股票代码
    days : int
        获取天数

    Returns
    -------
    pd.DataFrame
        历史数据 DataFrame（免费 API 返回 None）
    """
    import pandas as pd
    from datetime import datetime, timedelta

    # Finnhub 免费 API 不支持历史数据
    if not FINNHUB_API_KEY or len(FINNHUB_API_KEY) < 10:
        return None

    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        url = f"https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': int(start_time.timestamp()),
            'to': int(end_time.timestamp()),
            'token': FINNHUB_API_KEY
        }

        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        # 403 表示免费 API 不支持
        if response.status_code == 403:
            return None

        if data.get('s') == 'ok' and data.get('c'):
            df = pd.DataFrame({
                'open': data.get('o', []),
                'high': data.get('h', []),
                'low': data.get('l', []),
                'close': data.get('c', []),
                'volume': data.get('v', []),
            })
            df['datetime'] = pd.to_datetime(data.get('t', []), unit='s')
            df.set_index('datetime', inplace=True)
            df['symbol'] = symbol
            return df

        return None
    except Exception as e:
        return None


def get_fmp_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    从 Financial Modeling Prep (FMP) 获取美股历史数据
    
    Parameters
    ----------
    symbol : str
        股票代码
    days : int
        获取天数
        
    Returns
    -------
    pd.DataFrame
        历史数据 DataFrame
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    if not FMP_API_KEY or len(FMP_API_KEY) < 5:
        return None
    
    try:
        # FMP API 端点
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d'),
            'apikey': FMP_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        # 检查是否有有效数据
        if data.get('historical'):
            historical = data['historical']
            df = pd.DataFrame(historical)
            
            # 重命名列以匹配项目格式
            df = df.rename(columns={
                'date': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adjClose': 'close',  # 使用复权收盘价
                'volume': 'volume'
            })
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            df['symbol'] = symbol
            
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol']]
        
        return None
    except Exception as e:
        print(f"获取 FMP {symbol} 历史数据失败：{e}")
        return None



def get_tiingo_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    从 Tiingo 获取美股历史数据
    
    Parameters
    ----------
    symbol : str
        股票代码
    days : int
        获取天数
    
    Returns
    -------
    pd.DataFrame
        历史数据 DataFrame
    """
    import pandas as pd
    
    if not TIINGO_API_KEY or len(TIINGO_API_KEY) < 5:
        print("警告：未配置有效的 TIINGO_API_KEY")
        return None
    
    try:
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {
            'startDate': (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d'),
            'endDate': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'resampleFreq': 'daily'
        }
        headers = {
            'Authorization': f'Token {TIINGO_API_KEY}'
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    'date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
                df['date'] = pd.to_datetime(df['date'])
                return df
        else:
            print(f"Tiingo API error: {response.status_code}")
    except Exception as e:
        print(f"获取 Tiingo 历史数据失败: {e}")
    return None


def get_tencent_history(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    从新浪财经获取 A 股历史数据
    
    Parameters
    ----------
    symbol : str
        股票代码
    days : int
        获取天数
        
    Returns
    -------
    pd.DataFrame
        历史数据 DataFrame
    """
    import pandas as pd
    
    try:
        # 转换股票代码格式
        if symbol.endswith('.SH'):
            api_symbol = 'sh' + symbol.replace('.SH', '')
        elif symbol.endswith('.SZ'):
            api_symbol = 'sz' + symbol.replace('.SZ', '')
        else:
            api_symbol = 'sh' + symbol if symbol.startswith('6') else 'sz' + symbol
        
        # 新浪财经历史数据 API (使用 money.finance.sina.com.cn)
        # scale 参数是分钟数：日线=240（一天交易4小时）
        scale = 240  # 日线
        url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={api_symbol}&scale={scale}&ma=5&mab=10"
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            # API 返回的字段是 day, open, high, low, close, volume
            df = pd.DataFrame(data)
            # 重命名字段（兼容性）
            if 'day' in df.columns:
                df['date'] = df['day']
            # 添加股票代码列
            df['code'] = symbol
            # 解析日期（格式：2026-01-20 10:30:00 或 2026-01-20）
            df['date'] = df['date'].astype(str)
            df['datetime'] = pd.to_datetime(df['date'].str.split(' ').str[0])
            df.set_index('datetime', inplace=True)
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


def get_history_data(symbol: str, market: str = 'A 股', days: int = 60) -> Optional[pd.DataFrame]:
    """
    统一接口：获取历史数据

    Parameters
    ----------
    symbol : str
        股票代码
    market : str
        'A 股' 或 '美股'
    days : int
        获取天数

    Returns
    -------
    pd.DataFrame
        历史数据 DataFrame，包含 open, high, low, close, volume
    """
    import pandas as pd

    if market == '美股':
        # 优先使用 Tiingo（如果有 API Key）
        if TIINGO_API_KEY and len(TIINGO_API_KEY) >= 5:
            data = get_tiingo_history(symbol, days)
            if data is not None:
                return data
        # 其次使用 FMP（如果有 API Key）
        if FMP_API_KEY and len(FMP_API_KEY) >= 5:
            data = get_fmp_history(symbol, days)
            if data is not None:
                return data
        # 回退到 Finnhub（免费 API 不支持历史数据）
        return get_finnhub_history(symbol, days)
    else:
        return get_tencent_history(symbol, days)
