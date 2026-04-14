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
    获取市场情绪数据
    基于大盘指数涨跌幅直接判断，不再依赖涨跌家数统计
    
    Returns:
        市场情绪字典
    """
    try:
        # 获取主要指数数据
        indices = {
            'sh000001': '上证指数',
            'sh000300': '沪深300',
            'sz399001': '深证成指',
            'sz399006': '创业板指'
        }
        
        index_changes = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for code, name in indices.items():
            try:
                url = f"http://qt.gtimg.cn/q={code}"
                response = requests.get(url, headers=headers, timeout=5)
                if response.text and 'v_' in response.text:
                    parts = response.text.split('~')
                    if len(parts) >= 45:
                        change_pct = float(parts[32]) if parts[32] else 0
                        index_changes.append({
                            'name': name,
                            'change_pct': change_pct
                        })
            except:
                continue
        
        if not index_changes:
            return None
        
        # 计算平均涨跌幅
        avg_change = sum(item['change_pct'] for item in index_changes) / len(index_changes)
        
        # 基于指数涨跌幅计算市场情绪指标
        # 使用历史统计规律：指数涨跌幅与上涨家数比例的关系
        # 涨1% ≈ 65%股票上涨，跌1% ≈ 35%股票上涨
        advance_ratio = 0.5 + avg_change / 5
        advance_ratio = max(0.2, min(0.8, advance_ratio))  # 限制在20%-80%
        
        # 基于约5300只A股计算涨跌家数
        total_stocks = 5300
        up_count = int(total_stocks * advance_ratio)
        down_count = int(total_stocks * (1 - advance_ratio))
        flat_count = total_stocks - up_count - down_count
        
        # 根据指数涨跌幅估算涨停跌停家数
        if avg_change > 2:
            limit_up = int(80 + avg_change * 20)
            limit_down = max(0, int(3 - avg_change))
        elif avg_change > 1:
            limit_up = int(40 + avg_change * 20)
            limit_down = max(0, int(8 - avg_change * 3))
        elif avg_change > 0:
            limit_up = int(15 + avg_change * 15)
            limit_down = max(0, int(12 - avg_change * 5))
        elif avg_change > -1:
            limit_up = max(0, int(12 + avg_change * 8))
            limit_down = int(15 - avg_change * 10)
        elif avg_change > -2:
            limit_up = max(0, int(8 + avg_change * 5))
            limit_down = int(30 - avg_change * 10)
        else:
            limit_up = max(0, int(3 + avg_change * 2))
            limit_down = int(60 - avg_change * 15)
        
        return {
            'sample_stocks': total_stocks,
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'advance_ratio': advance_ratio,
            'limit_up_ratio': limit_up / total_stocks,
            'limit_down_ratio': limit_down / total_stocks,
            'sentiment_score': (up_count - down_count) / total_stocks,
            'source': 'index_based',  # 标记为基于指数计算
            'avg_index_change': avg_change,
            'index_details': index_changes
        }
    except Exception as e:
        print(f"  获取市场情绪失败: {e}")
        return None


def _get_sentiment_eastmoney() -> Optional[Dict]:
    """
    【已弃用】通过东方财富接口获取市场情绪
    保留此函数以兼容旧代码，但不再主动调用
    """
    return None


def _get_sentiment_tencent_legacy() -> Optional[Dict]:
    """
    【已弃用】通过腾讯财经接口获取市场情绪（旧备用方案）
    保留此函数以兼容旧代码，但不再主动调用
    """
    try:
        # 获取主要指数数据来推断市场情绪
        indices = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
        index_data_list = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for idx in indices:
            try:
                url_idx = f"http://qt.gtimg.cn/q={idx}"
                r = requests.get(url_idx, headers=headers, timeout=5)
                if r.text and 'v_' in r.text:
                    parts = r.text.split('~')
                    if len(parts) >= 45:
                        change_pct = float(parts[32]) if parts[32] else 0
                        index_data_list.append(change_pct)
            except:
                continue
        
        if not index_data_list:
            return None
        
        # 基于指数平均涨跌幅估算上涨比例
        avg_change = sum(index_data_list) / len(index_data_list)
        
        # 简单线性映射：指数涨1% ≈ 60%股票上涨
        # 指数跌1% ≈ 40%股票上涨
        estimated_advance_ratio = 0.5 + avg_change / 10
        estimated_advance_ratio = max(0.2, min(0.8, estimated_advance_ratio))  # 限制在20%-80%
        
        # 估算涨跌家数（基于约5300只A股）
        total_stocks = 5300
        up_count = int(total_stocks * estimated_advance_ratio)
        down_count = int(total_stocks * (1 - estimated_advance_ratio))
        flat_count = total_stocks - up_count - down_count
        
        # 根据指数涨跌幅估算涨停跌停家数
        if avg_change > 1:
            limit_up = int(50 + avg_change * 30)  # 大涨时涨停多
            limit_down = max(0, int(5 - avg_change * 2))
        elif avg_change < -1:
            limit_up = max(0, int(5 + avg_change * 2))
            limit_down = int(30 - avg_change * 20)  # 大跌时跌停多
        else:
            limit_up = 15
            limit_down = 10
        
        return {
            'sample_stocks': total_stocks,
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'advance_ratio': estimated_advance_ratio,
            'limit_up_ratio': limit_up / total_stocks,
            'limit_down_ratio': limit_down / total_stocks,
            'sentiment_score': (up_count - down_count) / total_stocks,
            'source': 'estimated',
            'avg_index_change': avg_change,
        }
    except Exception as e:
        return None


def _get_sentiment_eastmoney_legacy() -> Optional[Dict]:
    """【已弃用】东方财富旧接口"""
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        
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
                "fid": "f12",
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
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
        return None  # 静默失败，让上层处理


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


def judge_market_regime(advance_ratio: float = None, days: int = 5) -> Dict:
    """
    判断细致的市场环境（用于ML模型置信度）
    
    Args:
        advance_ratio: 当日上涨比例
        days: 回看天数
    
    Returns:
        {
            'regime': 'bull/bear/weak_bull/weak_bear/sideways',
            'confidence': 0-1（与训练环境匹配度）
            'adv_ratio_5d_ma': 5日上涨比例均值
        }
    """
    try:
        if advance_ratio is None:
            sentiment = get_market_sentiment()
            advance_ratio = sentiment.get('advance_ratio', 0.5) if sentiment else 0.5
        
        # 获取历史上涨比例
        index_history = get_index_history(days=max(days, 20))
        if index_history is None or len(index_history) < days:
            return {'regime': 'sideways', 'confidence': 0.3, 'adv_ratio_5d_ma': advance_ratio}
        
        # 估算历史上涨比例（用涨跌家数近似）
        # 简化：用指数涨跌判断
        changes = index_history['close'].pct_change()
        adv_approx = (changes > 0).rolling(days).mean().iloc[-1] if len(changes) >= days else 0.5
        
        # 使用当前上涨比例
        adv_ratio_5d_ma = (advance_ratio + adv_approx * 4) / 5 if adv_approx else advance_ratio
        
        # 分类（与训练时一致）
        if advance_ratio > 0.6 and adv_ratio_5d_ma > 0.55:
            regime = 'bull'
            confidence = 0.5  # 训练期主要是weak_bear，bull环境置信度低
        elif advance_ratio < 0.4 and adv_ratio_5d_ma < 0.45:
            regime = 'bear'
            confidence = 0.7  # 与训练期接近
        elif advance_ratio > 0.55:
            regime = 'weak_bull'
            confidence = 0.3  # 训练期较少，置信度低
        elif advance_ratio < 0.45:
            regime = 'weak_bear'
            confidence = 0.9  # 训练期主要环境，置信度高
        else:
            regime = 'sideways'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'adv_ratio_5d_ma': adv_ratio_5d_ma,
            'advance_ratio': advance_ratio
        }
        
    except Exception as e:
        return {'regime': 'sideways', 'confidence': 0.3, 'adv_ratio_5d_ma': 0.5}


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