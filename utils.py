#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import re
from typing import List, Tuple


def parse_stock_list(file_path: str) -> List[Tuple[str, str, str]]:
    """
    解析股票列表 Markdown 文件
    
    Parameters
    ----------
    file_path : str
        Markdown 文件路径
        
    Returns
    -------
    List[Tuple[str, str, str]]
        [(symbol, name, category), ...]
    """
    stocks = []
    
    if not os.path.exists(file_path):
        return stocks
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = content.split('## ')
    for section in sections:
        if not section.strip():
            continue
        lines = section.split('\n')
        category = lines[0].strip() if lines else 'Unknown'

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('|') and '---' in line:
                continue
            if '代码' in line or '名称' in line:
                continue
            if line.startswith('|') and '|' in line[1:]:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3 and parts[1] and (parts[1][0].isdigit() or parts[1].isalpha()):
                    stocks.append((parts[1], parts[2], category))

    return stocks


def format_price(price: float, currency: str = 'CNY') -> str:
    """格式化价格显示"""
    symbol = '$' if currency == 'USD' else '¥'
    return f"{symbol}{price:.2f}"


def format_pct(value: float) -> str:
    """格式化百分比"""
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.2f}%"
