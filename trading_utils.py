#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易工具模块
包含交易日检查、数据验证等功能
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests


def is_trading_day(date: datetime = None) -> bool:
    """
    检查给定日期是否为A股交易日
    
    参数:
        date: 要检查的日期，默认为今天
    
    返回:
        True如果是交易日，False如果不是
    """
    if date is None:
        date = datetime.now()
    
    # 获取日期信息
    weekday = date.weekday()  # 0=周一, 6=周日
    date_str = date.strftime('%Y-%m-%d')
    
    # 周末不是交易日
    if weekday >= 5:  # 周六或周日
        return False
    
    # 检查是否是节假日（使用本地缓存的节假日数据）
    if is_holiday(date_str):
        return False
    
    return True


def is_holiday(date_str: str) -> bool:
    """
    检查给定日期是否为节假日
    使用本地缓存的节假日数据，如果缓存不存在则使用简单规则
    
    参数:
        date_str: 日期字符串，格式 'YYYY-MM-DD'
    
    返回:
        True如果是节假日，False如果不是
    """
    # 2026年A股节假日（需要定期更新）
    holidays_2026 = [
        # 元旦
        '2026-01-01', '2026-01-02',
        # 春节
        '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20',
        # 清明节
        '2026-04-04', '2026-04-05', '2026-04-06',  # 4月6日是清明节假期
        # 劳动节
        '2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05',
        # 端午节
        '2026-06-19', '2026-06-20', '2026-06-22',
        # 中秋节
        '2026-09-25', '2026-09-26', '2026-09-27',
        # 国庆节
        '2026-10-01', '2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08',
    ]
    
    return date_str in holidays_2026


def get_last_trading_day(date: datetime = None) -> datetime:
    """
    获取最近一个交易日
    
    参数:
        date: 参考日期，默认为今天
    
    返回:
        最近一个交易日的日期
    """
    if date is None:
        date = datetime.now()
    
    # 向前查找最近交易日
    current = date
    while not is_trading_day(current):
        current -= timedelta(days=1)
    
    return current


def validate_market_data(data: Dict) -> Dict:
    """
    验证市场数据的有效性
    
    参数:
        data: 市场数据字典
    
    返回:
        验证结果 {'valid': bool, 'issues': List[str], 'data_date': str}
    """
    issues = []
    data_date = None
    
    if not data:
        return {'valid': False, 'issues': ['数据为空'], 'data_date': None}
    
    # 检查数据日期
    if 'date' in data:
        data_date = data['date']
        today = datetime.now().strftime('%Y-%m-%d')
        
        if data_date != today:
            # 数据日期不是今天，可能是非交易日使用了旧数据
            last_trading = get_last_trading_day().strftime('%Y-%m-%d')
            if data_date == last_trading:
                issues.append(f'数据为上一交易日({data_date})，非交易日使用历史数据')
            else:
                issues.append(f'数据日期异常: {data_date}，今天: {today}')
    
    # 检查价格数据
    quotes = data.get('quotes', [])
    if not quotes:
        issues.append('没有获取到行情数据')
    else:
        # 检查价格是否异常（比如全为0或全相同）
        prices = [q.get('price', 0) for q in quotes if q.get('price', 0) > 0]
        if len(prices) == 0:
            issues.append('所有股票价格异常（全为0）')
        elif len(set(prices)) == 1:
            issues.append('所有股票价格相同，可能是缓存数据')
    
    # 检查数据时间
    if 'time' in data:
        data_time = data['time']
        try:
            hour = int(data_time.split(':')[0])
            if hour < 9 or hour > 15:
                issues.append(f'数据时间异常: {data_time}，不在交易时段')
        except:
            pass
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'data_date': data_date
    }


def check_trading_status() -> Dict:
    """
    检查当前交易状态
    
    返回:
        {
            'can_trade': bool,      # 是否可以交易
            'is_trading_day': bool, # 是否是交易日
            'is_trading_hours': bool, # 是否在交易时段
            'last_trading_day': str,  # 最近交易日
            'message': str            # 状态说明
        }
    """
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    
    # 检查是否是交易日
    trading_day = is_trading_day(now)
    
    # 检查是否在交易时段 (9:30-11:30, 13:00-15:00)
    hour = now.hour
    minute = now.minute
    time_val = hour * 100 + minute
    
    trading_hours = (
        (930 <= time_val <= 1130) or  # 上午
        (1300 <= time_val <= 1500)    # 下午
    )
    
    # 获取最近交易日
    last_trading = get_last_trading_day(now).strftime('%Y-%m-%d')
    
    # 确定状态和消息
    if not trading_day:
        can_trade = False
        message = f'今天({today_str})不是交易日，最近交易日: {last_trading}'
    elif not trading_hours:
        can_trade = False
        message = f'当前不在交易时段，交易时间: 9:30-11:30, 13:00-15:00'
    else:
        can_trade = True
        message = '可以正常交易'
    
    return {
        'can_trade': can_trade,
        'is_trading_day': trading_day,
        'is_trading_hours': trading_hours,
        'last_trading_day': last_trading,
        'message': message
    }


def safe_execute_trading(func):
    """
    交易函数装饰器：在执行交易前检查交易日
    
    用法:
        @safe_execute_trading
        def my_trading_function():
            ...
    """
    def wrapper(*args, **kwargs):
        status = check_trading_status()
        
        if not status['can_trade']:
            print(f"⚠️ 交易已阻止: {status['message']}")
            return {
                'success': False,
                'error': status['message'],
                'status': status
            }
        
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("交易日检查工具测试")
    print("=" * 60)
    
    # 检查今天
    today = datetime.now()
    print(f"\n今天: {today.strftime('%Y-%m-%d')} ({['周一', '周二', '周三', '周四', '周五', '周六', '周日'][today.weekday()]})")
    print(f"是否是交易日: {is_trading_day(today)}")
    
    # 检查交易状态
    status = check_trading_status()
    print(f"\n交易状态:")
    print(f"  是否可以交易: {status['can_trade']}")
    print(f"  是否是交易日: {status['is_trading_day']}")
    print(f"  是否在交易时段: {status['is_trading_hours']}")
    print(f"  最近交易日: {status['last_trading_day']}")
    print(f"  状态说明: {status['message']}")
