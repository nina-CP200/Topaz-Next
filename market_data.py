#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大盘数据获取模块
获取沪深300指数、市场情绪等数据
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def get_index_data(index_code: str = '000300.SH') -> Optional[Dict]:
    """
    获取指数数据（使用腾讯财经接口）
    
    Args:
        index_code: 指数代码，默认沪深300
    
    Returns:
        指数数据字典
    """
    try:
        # 转换代码格式（腾讯格式）
        if index_code.endswith('.SH'):
            api_code = f"sh{index_code.replace('.SH', '')}"
        elif index_code.endswith('.SZ'):
            api_code = f"sz{index_code.replace('.SZ', '')}"
        else:
            api_code = f"sh{index_code}" if index_code.startswith('0') else f"sh{index_code}"
        
        # 腾讯财经接口
        url = f"http://qt.gtimg.cn/q={api_code}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.text
        
        if not data or 'v_' not in data:
            return None
        
        parts = data.split('~')
        if len(parts) >= 45:
            return {
                'code': index_code,
                'name': parts[1],
                'price': float(parts[3]) if parts[3] else 0,
                'prev_close': float(parts[4]) if parts[4] else 0,
                'change_pct': float(parts[32]) if parts[32] else 0,
                'change_amount': float(parts[31]) if parts[31] else 0,
                'high': float(parts[33]) if parts[33] else 0,
                'low': float(parts[34]) if parts[34] else 0,
                'open': float(parts[5]) if parts[5] else 0,
                'volume': float(parts[36]) if parts[36] else 0,
                'amount': float(parts[37]) if parts[37] else 0,
            }
        return None
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return None


def get_index_history(index_code: str = '000300.SH', days: int = 60) -> Optional[pd.DataFrame]:
    """
    获取指数历史数据
    
    Args:
        index_code: 指数代码
        days: 获取天数
    
    Returns:
        历史数据 DataFrame
    """
    try:
        # 转换代码格式
        if index_code.endswith('.SH'):
            api_code = f"sh{index_code.replace('.SH', '')}"
        elif index_code.endswith('.SZ'):
            api_code = f"sz{index_code.replace('.SZ', '')}"
        else:
            api_code = f"sh{index_code}"
        
        url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        params = {
            "symbol": api_code,
            "scale": 240,  # 日线
            "ma": 5
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['day'].str.split(' ').str[0])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 计算均线
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            
            # 计算涨跌幅
            df['pct_change'] = df['close'].pct_change() * 100
            
            # 取最近 days 天
            df = df.tail(days).reset_index(drop=True)
            
            return df
        return None
    except Exception as e:
        print(f"获取指数历史数据失败: {e}")
        return None


def get_market_sentiment() -> Optional[Dict]:
    """
    获取市场情绪数据（涨跌家数、涨停跌停等）
    通过统计全市场股票涨跌情况
    
    Returns:
        市场情绪字典
    """
    try:
        # 东方财富A股列表接口 - 不排序，获取样本
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        
        # 多取几页来获得更准确的统计
        up_count = 0
        down_count = 0
        flat_count = 0
        limit_up = 0
        limit_down = 0
        
        for pn in range(1, 5):  # 取前4页，每页500条
            params = {
                "pn": pn,
                "pz": 500,
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f12",  # 按代码排序，而不是涨跌幅
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",  # A股
                "fields": "f2,f3,f4,f12,f14"
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://quote.eastmoney.com/'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if not data.get('data') or not data['data'].get('diff'):
                break
            
            stocks = data['data']['diff']
            
            for stock in stocks:
                change_pct = stock.get('f3', None)
                if change_pct is None:
                    continue
                
                try:
                    change_pct = float(change_pct)
                except:
                    continue
                
                if change_pct > 0.01:
                    up_count += 1
                    if change_pct >= 9.5:
                        limit_up += 1
                elif change_pct < -0.01:
                    down_count += 1
                    if change_pct <= -9.5:
                        limit_down += 1
                else:
                    flat_count += 1
        
        total = up_count + down_count + flat_count
        
        if total == 0:
            return None
        
        return {
            'sample_stocks': total,
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'advance_ratio': up_count / total,
            'limit_up_ratio': limit_up / total,
            'limit_down_ratio': limit_down / total,
            'sentiment_score': (up_count - down_count) / total,
        }
    except Exception as e:
        print(f"获取市场情绪失败: {e}")
        return None


def judge_market_environment(index_data: Dict = None, sentiment: Dict = None) -> str:
    """
    判断大盘环境
    
    Args:
        index_data: 指数数据
        sentiment: 市场情绪
    
    Returns:
        'bull': 牛市
        'bear': 熊市
        'recovery': 反弹
        'sideways': 震荡
    """
    try:
        # 获取数据
        if index_data is None:
            index_data = get_index_data()
        if sentiment is None:
            sentiment = get_market_sentiment()
        
        # 获取历史数据判断趋势
        index_history = get_index_history(days=30)
        
        if index_data is None or index_history is None:
            return 'sideways'  # 默认震荡
        
        current_price = index_data['price']
        change_pct = index_data['change_pct']
        
        # 计算20日涨跌幅
        if len(index_history) >= 20:
            price_20d_ago = index_history.iloc[-20]['close']
            return_20d = (current_price - price_20d_ago) / price_20d_ago
        else:
            return_20d = 0
        
        # 计算均线位置
        ma20 = index_history['ma20'].iloc[-1] if 'ma20' in index_history.columns else current_price
        ma5 = index_history['ma5'].iloc[-1] if 'ma5' in index_history.columns else current_price
        
        # 获取情绪数据
        advance_ratio = sentiment.get('advance_ratio', 0.5) if sentiment else 0.5
        
        # 判断逻辑
        if current_price > ma20 and return_20d > 0.05 and advance_ratio > 0.6:
            return 'bull'  # 牛市：价格高于20日线，20日涨幅>5%，上涨家数>60%
        
        elif current_price < ma20 and return_20d < -0.05 and advance_ratio < 0.4:
            return 'bear'  # 熊市：价格低于20日线，20日跌幅>5%，上涨家数<40%
        
        elif ma5 > ma20 and return_20d < 0:
            return 'recovery'  # 反弹：短期均线穿过长期，但整体还在跌
        
        else:
            return 'sideways'  # 震荡
    
    except Exception as e:
        print(f"判断市场环境失败: {e}")
        return 'sideways'


def get_market_adjusted_thresholds(market_env: str) -> Dict:
    """
    根据市场环境返回调整后的交易阈值
    
    Args:
        market_env: 市场环境
    
    Returns:
        阈值字典
    """
    thresholds = {
        'bull': {
            'buy_threshold': 0.65,      # 牛市降低买入门槛
            'sell_threshold': 0.35,     # 牛市提高卖出门槛
            'position_max': 0.95,       # 允许更高仓位
            'single_max': 0.25,         # 单股最大仓位
            'description': '牛市环境，积极做多'
        },
        'bear': {
            'buy_threshold': 0.80,      # 熊市提高买入门槛
            'sell_threshold': 0.50,     # 熊市降低卖出门槛
            'position_max': 0.60,       # 降低仓位上限
            'single_max': 0.15,         # 单股最大仓位降低
            'description': '熊市环境，保守防守'
        },
        'recovery': {
            'buy_threshold': 0.75,
            'sell_threshold': 0.40,
            'position_max': 0.80,
            'single_max': 0.20,
            'description': '反弹环境，精选抄底'
        },
        'sideways': {
            'buy_threshold': 0.70,
            'sell_threshold': 0.40,
            'position_max': 0.85,
            'single_max': 0.20,
            'description': '震荡环境，中性策略'
        }
    }
    
    return thresholds.get(market_env, thresholds['sideways'])


def test_market_data():
    """测试大盘数据获取"""
    print("=" * 60)
    print("大盘数据测试")
    print("=" * 60)
    
    # 1. 获取指数数据
    print("\n1. 沪深300指数:")
    index_data = get_index_data()
    if index_data:
        print(f"   价格: {index_data['price']:.2f}")
        print(f"   涨跌: {index_data['change_pct']:.2f}%")
        print(f"   最高: {index_data['high']:.2f}")
        print(f"   最低: {index_data['low']:.2f}")
    
    # 2. 获取市场情绪
    print("\n2. 市场情绪:")
    sentiment = get_market_sentiment()
    if sentiment:
        print(f"   上涨家数: {sentiment['up_count']}")
        print(f"   下跌家数: {sentiment['down_count']}")
        print(f"   涨停家数: {sentiment['limit_up']}")
        print(f"   跌停家数: {sentiment['limit_down']}")
        print(f"   上涨比例: {sentiment['advance_ratio']:.1%}")
    
    # 3. 判断市场环境
    print("\n3. 市场环境判断:")
    market_env = judge_market_environment(index_data, sentiment)
    thresholds = get_market_adjusted_thresholds(market_env)
    print(f"   环境: {market_env}")
    print(f"   说明: {thresholds['description']}")
    print(f"   买入阈值: {thresholds['buy_threshold']:.0%}")
    print(f"   卖出阈值: {thresholds['sell_threshold']:.0%}")
    print(f"   最大仓位: {thresholds['position_max']:.0%}")
    
    return market_env, thresholds


if __name__ == '__main__':
    test_market_data()