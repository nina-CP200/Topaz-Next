#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
============

本模块提供股票分析系统的通用工具函数，包括：
- 股票列表加载：从 JSON 文件读取股票代码和名称
- 价格格式化：支持人民币（CNY）和美元（USD）格式化
- 百分比格式化：自动添加正负号

依赖：
- json: JSON 文件解析
- os: 文件系统操作
- typing: 类型注解支持

作者：Topaz Team
创建日期：2024
"""

import json
import os
from typing import List, Tuple


def load_stock_list_from_json(file_path: str) -> List[Tuple[str, str, str]]:
    """
    从 JSON 文件加载股票列表
    
    JSON 格式说明
    -------------
    输入的 JSON 文件应包含股票对象数组，每个对象包含以下字段：
    
    示例：
    [
        {"code": "600519", "name": "贵州茅台"},
        {"code": "000001", "name": "平安银行"},
        {"code": "300750", "name": "宁德时代"}
    ]
    
    必需字段：
    - code: 股票代码（6位数字字符串）
    - name: 股票名称（中文名称）
    
    代码转换规则
    ------------
    根据股票代码首位数字自动判断交易所并添加后缀：
    
    +----------+------------------+----------+
    | 首位数字 | 交易所           | 后缀     |
    +==========+==================+==========+
    | 6        | 上海证券交易所   | .SH      |
    +----------+------------------+----------+
    | 0        | 深圳证券交易所   | .SZ      |
    +----------+------------------+----------+
    | 3        | 创业板（深交所） | .SZ      |
    +----------+------------------+----------+
    | 其他     | 保持原样         | 无       |
    +----------+------------------+----------+
    
    转换示例：
    - "600519" -> "600519.SH"（上证）
    - "000001" -> "000001.SZ"（深证）
    - "300750" -> "300750.SZ"（创业板）
    
    Parameters
    ----------
    file_path : str
        JSON 文件路径
        
    Returns
    -------
    List[Tuple[str, str, str]]
        股票列表，每个元素为 (symbol, name, category) 元组：
        - symbol: 带交易所后缀的股票代码，如 "600519.SH"
        - name: 股票名称，如 "贵州茅台"
        - category: 分类标签，当前默认为空字符串
        
    Raises
    ------
    无显式异常，文件不存在时返回空列表
    
    Examples
    --------
    >>> stocks = load_stock_list_from_json("stocks.json")
    >>> print(stocks[0])
    ('600519.SH', '贵州茅台', '白酒')
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        stocks_data = json.load(f)
    
    # 加载行业分类映射
    industry_map = {}
    industry_file = os.path.join(os.path.dirname(file_path), "csi300_industry_map.json")
    if os.path.exists(industry_file):
        with open(industry_file, 'r', encoding='utf-8') as f:
            industry_map = json.load(f)
    
    stocks = []
    for s in stocks_data:
        code = s.get("code", "")
        name = s.get("name", "")
        if code:
            # 如果已有后缀，直接使用
            if '.SH' in code.upper() or '.SZ' in code.upper():
                symbol = code.upper()
            elif code.startswith("6"):
                symbol = f"{code}.SH"
            elif code.startswith("0") or code.startswith("3"):
                symbol = f"{code}.SZ"
            else:
                symbol = code
            industry = industry_map.get(symbol, industry_map.get(code, ""))
            stocks.append((symbol, name, industry))
    
    return stocks


def format_price(price: float, currency: str = 'CNY') -> str:
    """
    格式化价格显示
    
    格式说明
    --------
    将数值价格转换为带货币符号的字符串格式，保留两位小数。
    
    输出格式：{货币符号}{价格:.2f}
    
    货币符号对照表
    -------------
    +----------+------------------+----------+
    | 货币代码 | 货币名称         | 符号     |
    +==========+==================+==========+
    | CNY      | 人民币           | ¥        |
    +----------+------------------+----------+
    | USD      | 美元             | $        |
    +----------+------------------+----------+
    | 其他     | 默认视为人民币   | ¥        |
    +----------+------------------+----------+
    
    Parameters
    ----------
    price : float
        价格数值（必须为数值类型）
    currency : str, optional
        货币类型，默认为 'CNY'（人民币）
        - 'CNY': 人民币，符号 ¥
        - 'USD': 美元，符号 $
        
    Returns
    -------
    str
        格式化后的价格字符串，如 "¥168.50" 或 "$100.00"
        
    Examples
    --------
    >>> format_price(168.5)
    '¥168.50'
    >>> format_price(100.0, 'USD')
    '$100.00'
    >>> format_price(-5.25)
    '¥-5.25'
    """
    symbol = '$' if currency == 'USD' else '¥'
    return f"{symbol}{price:.2f}"


def format_pct(value: float) -> str:
    """
    格式化百分比显示
    
    格式说明
    --------
    将数值转换为百分比字符串，保留两位小数。
    正数自动添加 '+' 符号，便于直观显示涨跌。
    
    Parameters
    ----------
    value : float
        百分比数值
        
    Returns
    -------
    str
        格式化后的百分比字符串，如 "+5.23%" 或 "-2.10%"
        
    Examples
    --------
    >>> format_pct(5.23)
    '+5.23%'
    >>> format_pct(-2.1)
    '-2.10%'
    >>> format_pct(0)
    '0.00%'
    """
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.2f}%"


# ============================================================================
# 扩展建议：可添加的新函数
# ============================================================================
#
# 1. format_volume(value: int) -> str
#    功能：格式化成交量，添加万、亿单位
#    示例：12345678 -> "1234.57万", 123456789 -> "1.23亿"
#
# 2. format_amount(value: float) -> str
#    功能：格式化成交金额，添加万、亿单位
#    示例：1234567890.0 -> "12.35亿"
#
# 3. validate_stock_code(code: str) -> bool
#    功能：验证股票代码格式是否有效（6位数字）
#    示例：validate_stock_code("600519") -> True
#
# 4. get_exchange(code: str) -> str
#    功能：根据股票代码返回交易所名称
#    返回：'上海证券交易所'、'深圳证券交易所' 或 '未知'
#
# 5. format_change(current: float, previous: float) -> Tuple[str, str]
#    功能：计算并格式化价格变动
#    返回：(变动额字符串, 变动百分比字符串)
#    示例：format_change(105.0, 100.0) -> ('+5.00', '+5.00%')
#
# 6. save_stocks_to_json(stocks: List[Tuple], file_path: str) -> None
#    功能：将股票列表保存为 JSON 文件（load_stock_list_from_json 的逆向操作）
#
# 7. format_timestamp(ts: int, fmt: str = '%Y-%m-%d %H:%M:%S') -> str
#    功能：将时间戳转换为可读字符串
#    示例：format_timestamp(1704067200) -> "2024-01-01 00:00:00"
#
# 8. calculate_ma(prices: List[float], period: int) -> List[float]
#    功能：计算移动平均线
#    示例：calculate_ma([10, 11, 12, 13, 14], 3) -> [None, None, 11.0, 12.0, 13.0]
# ============================================================================
